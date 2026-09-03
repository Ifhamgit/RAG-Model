"""Embedders — the dense arm's representation (DESIGN.md §c.1).

Three backends behind one protocol, because the constraint here is unusual: the
brief supplies a single key whose provider is not fixed, and neither Anthropic
nor OpenRouter exposes an embeddings endpoint. A design that hard-codes a hosted
embedder has a real chance of not running at all on the grading machine, and
"must actually run without additional setup" is an explicit requirement.

    fastembed  primary   BAAI/bge-small-en-v1.5, 384-dim, ONNX on CPU, no torch
    local      fallback  TF-IDF -> truncated SVD, pure NumPy, zero dependencies
    openai     opt-in    text-embedding-3-small, never auto-selected

Every backend returns L2-normalised vectors, so the cosine similarity the store
computes is a plain dot product (§c.3).
"""

from __future__ import annotations

import logging
import pathlib
import re
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)

# bge models are trained with an asymmetric objective: queries carry a task
# instruction, passages do not. Applying it to both sides — or to neither —
# measurably degrades retrieval, so the asymmetry is deliberate, not an oversight.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-/.,]*", re.IGNORECASE)


def l2_normalise(m: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so a dot product is cosine similarity.

    Zero rows are left alone rather than producing NaN — an empty or
    entirely-out-of-vocabulary chunk should score 0 against everything, not
    poison the whole matrix.
    """
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        n = float(np.linalg.norm(m))
        return m if n == 0.0 else (m / n).astype(np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (m / norms).astype(np.float32)


@runtime_checkable
class Embedder(Protocol):
    """The one interface the rest of the system knows about."""

    name: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


# --------------------------------------------------------------------------
# primary: fastembed / bge-small-en-v1.5
# --------------------------------------------------------------------------


class FastEmbedEmbedder:
    """Pretrained sentence embeddings, CPU-only, no torch (§c.1).

    This is the primary because the corpus needs *paraphrase*: the documents are
    written in defined policy terms ("Withdrawal Request", "Refund Day Count")
    and users ask in everyday words ("can I quit and get my money back?"). Exact
    codes and figures are BM25's job. Only a pretrained model knows "quit" is
    near "withdraw" before it has seen this corpus.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported lazily: heavy, optional

        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed_query("dimension probe").shape[0])
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = np.array(list(self._model.embed(list(texts))), dtype=np.float32)
        return l2_normalise(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        vec = next(iter(self._model.query_embed([BGE_QUERY_PREFIX + text])))
        return l2_normalise(np.asarray(vec, dtype=np.float32))


# --------------------------------------------------------------------------
# fallback: corpus-fitted TF-IDF -> truncated SVD
# --------------------------------------------------------------------------


class LsaEmbedder:
    """Zero-dependency fallback, fitted from the corpus itself (§c.1).

    The first fastembed run needs network access to fetch the model. If the
    machine is offline or the download is blocked, the system must still answer
    questions — so this exists, in pure NumPy, and fits the whole corpus in
    tens of milliseconds.

    Collapsing TF-IDF to a low-rank space puts "fee", "tuition", "cost" and
    "INR" on shared axes *because this corpus uses them together*. That is a
    real if shallow semantic signal. Its ceiling, stated plainly: it cannot
    bridge a vocabulary gap it has never seen. When this backend is active the
    system leans harder on BM25 and on query expansion, and every trace records
    `embedding_backend` so a weaker run is explained rather than mysterious.

    Corpus-fitted also means *not* per-chunk: changing the corpus changes every
    vector, which is why the ingest pipeline cannot do an incremental update on
    this backend.
    """

    name = "local:tfidf-svd"

    def __init__(self, dim: int = 128):
        self._target_dim = dim
        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None  # (vocab, dim)

    # -- fitting ----------------------------------------------------------
    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return [t.lower() for t in _TOKEN_RE.findall(text)]

    def _counts(self, texts: Sequence[str]) -> np.ndarray:
        m = np.zeros((len(texts), len(self._vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in self._tokenise(text):
                j = self._vocab.get(tok)
                if j is not None:
                    m[i, j] += 1.0
        return m

    def fit(self, texts: Sequence[str]) -> None:
        if not texts:
            raise ValueError("LsaEmbedder.fit needs at least one document")

        vocab: dict[str, int] = {}
        for text in texts:
            for tok in self._tokenise(text):
                vocab.setdefault(tok, len(vocab))
        self._vocab = vocab

        tf = self._counts(texts)
        df = (tf > 0).sum(axis=0).astype(np.float32)
        # Smoothed idf, the standard form: never divides by zero, never negative.
        self._idf = (np.log((1.0 + len(texts)) / (1.0 + df)) + 1.0).astype(np.float32)

        weighted = l2_normalise(tf * self._idf)

        # The rank of an (n_docs x vocab) matrix cannot exceed n_docs, so asking
        # for more components than documents yields noise axes.
        k = int(min(self._target_dim, min(weighted.shape) - 1))
        k = max(k, 2)
        _u, _s, vt = np.linalg.svd(weighted, full_matrices=False)
        self._components = np.ascontiguousarray(vt[:k].T.astype(np.float32))

    @property
    def is_fitted(self) -> bool:
        return self._components is not None

    @property
    def dim(self) -> int:
        return 0 if self._components is None else int(self._components.shape[1])

    # -- embedding --------------------------------------------------------
    def _project(self, texts: Sequence[str]) -> np.ndarray:
        if self._components is None or self._idf is None:
            raise RuntimeError("LsaEmbedder used before fit(); ingest must run first")
        weighted = l2_normalise(self._counts(texts) * self._idf)
        return l2_normalise(weighted @ self._components)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._project(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._project([text])[0]

    # -- persistence ------------------------------------------------------
    # The fitted state must survive the process: ingest fits it, but `--query`
    # runs later and still has to project the question into the same space.
    def save(self, path: pathlib.Path) -> None:
        if self._components is None or self._idf is None:
            raise RuntimeError("nothing to save; fit() first")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vocab=np.array(list(self._vocab.keys()), dtype=object),
            idf=self._idf,
            components=self._components,
        )

    def load(self, path: pathlib.Path) -> bool:
        if not path.exists():
            return False
        data = np.load(path, allow_pickle=True)
        self._vocab = {tok: i for i, tok in enumerate(data["vocab"].tolist())}
        self._idf = data["idf"].astype(np.float32)
        self._components = data["components"].astype(np.float32)
        return True


# --------------------------------------------------------------------------
# opt-in: OpenAI
# --------------------------------------------------------------------------


class OpenAIEmbedder:
    """Hosted embeddings. Never auto-selected — see get_embedder."""

    def __init__(self, model: str, api_key: str, base_url: str = ""):
        from openai import OpenAI

        self.name = f"openai:{model}"
        self._model = model
        self._client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed_query("dimension probe").shape[0])
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):  # batch to stay inside request limits
            resp = self._client.embeddings.create(model=self._model, input=list(texts[i : i + 128]))
            out.extend(d.embedding for d in resp.data)
        return l2_normalise(np.array(out, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=[text])
        return l2_normalise(np.array(resp.data[0].embedding, dtype=np.float32))


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


class EmbeddingBackendError(RuntimeError):
    """An explicitly requested backend could not be constructed."""


def get_embedder(settings: Settings) -> Embedder:
    """Resolve EMBEDDING_BACKEND to a concrete embedder.

    "auto" degrades quietly but *loudly logged*: fastembed first, LSA if it is
    absent or its download fails. An explicit value never degrades — being
    silently downgraded to a weaker representation is the kind of thing that
    makes an eval result inexplicable a week later.
    """
    backend = settings.embedding_backend

    if backend in ("auto", "fastembed"):
        try:
            emb = FastEmbedEmbedder(settings.fastembed_model)
            log.info("embedding backend: %s (%d dim)", emb.name, emb.dim)
            return emb
        except Exception as exc:
            if backend == "fastembed":
                raise EmbeddingBackendError(
                    f"EMBEDDING_BACKEND=fastembed but it could not be loaded: {exc}\n"
                    "Install it with `pip install fastembed`, or set EMBEDDING_BACKEND=local "
                    "to use the offline fallback."
                ) from exc
            log.warning(
                "fastembed unavailable (%s); falling back to the corpus-fitted LSA embedder. "
                "Retrieval on paraphrased questions will be weaker — see DESIGN.md §c.1.",
                exc,
            )

    if backend == "openai":
        # Never reached from "auto": the configured key may be an Anthropic or
        # OpenRouter key, and neither provider serves an embeddings endpoint.
        if not settings.has_api_key:
            raise EmbeddingBackendError(
                "EMBEDDING_BACKEND=openai requires an OpenAI API key in SCALER_LLM_API_KEY."
            )
        if settings.resolved_provider != "openai":
            raise EmbeddingBackendError(
                f"EMBEDDING_BACKEND=openai, but the configured key resolves to provider "
                f"'{settings.resolved_provider}', which has no embeddings endpoint. "
                "Use fastembed or local instead."
            )
        emb = OpenAIEmbedder(
            settings.openai_embedding_model, settings.llm_api_key.get_secret_value()
        )
        log.info("embedding backend: %s", emb.name)
        return emb

    lsa = LsaEmbedder(dim=128)
    if lsa.load(settings.lsa_state_path):
        log.info("embedding backend: %s (%d dim, restored)", lsa.name, lsa.dim)
    else:
        log.info("embedding backend: %s (unfitted; ingest will fit it)", lsa.name)
    return lsa
