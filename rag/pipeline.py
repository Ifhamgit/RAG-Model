"""Pipeline — orchestration for ingest (and, from Step 9, ask).

    corpus/ -> load_corpus -> chunk_documents -> embed -> Store

Everything interesting lives in the modules this calls; the job here is
sequencing, timing, and the idempotency decision (DESIGN.md §a.2 step 4).
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import pathlib
import time
from typing import Optional

import numpy as np

from .chunker import chunk_documents, summarise
from .config import Settings
from .embeddings import Embedder, LsaEmbedder, get_embedder
from .loaders import load_corpus
from .models import AnswerResult, Chunk
from .store import Store

log = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class IngestReport:
    """What the ingest did, for the CLI, `/health`, and the eval record."""

    files: int
    raw_docs: int
    chunks: int
    embedded: int
    reused: int
    backend: str
    dim: int
    incremental: bool
    corpus_fingerprint: str
    elapsed_s: float
    distribution: str

    def render(self) -> str:
        mode = (
            f"incremental — {self.embedded} embedded, {self.reused} reused"
            if self.incremental
            else f"full rebuild — {self.embedded} embedded"
        )
        return "\n".join(
            [
                f"Ingested {self.files} files ({self.raw_docs} raw docs) in {self.elapsed_s:.1f}s",
                f"  embedder    : {self.backend} ({self.dim} dim)",
                f"  strategy    : {mode}",
                f"  fingerprint : {self.corpus_fingerprint[:16]}",
                "",
                self.distribution,
            ]
        )


def corpus_fingerprint(corpus_dir: pathlib.Path) -> str:
    """Hash of filenames, sizes and mtimes.

    Cheap enough to compute every run, and it changes whenever a document is
    added, removed, or edited — which is what `/health` needs to answer "is the
    index built from the corpus that is on disk right now?"
    """
    h = hashlib.sha256()
    for p in sorted(pathlib.Path(corpus_dir).iterdir(), key=lambda p: p.name.lower()):
        if p.is_file() and not p.name.startswith("."):
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()


def _embed_incrementally(
    chunks: list[Chunk], embedder: Embedder, store: Store
) -> tuple[np.ndarray, int, int]:
    """Re-embed only chunks whose content hash changed (§a.2 step 4).

    Valid only for per-chunk backends: a fastembed or OpenAI vector depends on
    that chunk's text alone, so an unchanged chunk's vector is still correct.
    """
    old_hashes = store.content_hashes()
    try:
        old_matrix, old_ids = store._load_matrix()
        old_row = {cid: i for i, cid in enumerate(old_ids)}
    except (FileNotFoundError, RuntimeError):
        old_matrix, old_row = None, {}

    reusable: dict[int, np.ndarray] = {}
    todo: list[int] = []
    for i, c in enumerate(chunks):
        row = old_row.get(c.chunk_id)
        if (
            old_matrix is not None
            and row is not None
            and old_hashes.get(c.chunk_id) == c.content_hash
            and old_matrix.shape[1] == embedder.dim
        ):
            reusable[i] = old_matrix[row]
        else:
            todo.append(i)

    matrix = np.zeros((len(chunks), embedder.dim), dtype=np.float32)
    for i, vec in reusable.items():
        matrix[i] = vec
    if todo:
        fresh = embedder.embed_documents([chunks[i].text for i in todo])
        for slot, i in enumerate(todo):
            matrix[i] = fresh[slot]
    return matrix, len(todo), len(reusable)


def ingest(
    settings: Settings, force: bool = False, embedder: Optional[Embedder] = None
) -> IngestReport:
    """Build the index from the corpus. Safe to re-run."""
    t0 = time.perf_counter()
    settings.ensure_dirs()

    docs = load_corpus(settings.corpus_dir)
    chunks = chunk_documents(docs, settings, verbose=True)
    if not chunks:
        raise ValueError(f"No chunks produced from {settings.corpus_dir}.")

    embedder = embedder or get_embedder(settings)

    with Store(settings.db_path, settings.vectors_path) as store:
        if force:
            store.reset()

        # The LSA fallback is fitted from the corpus, so every vector depends on
        # every document: adding one chunk changes the vocabulary, the idf, and
        # the SVD basis. Incremental reuse is therefore not merely an
        # optimisation that is skipped — it would be *wrong*, mixing vectors
        # from two different spaces. It refits and re-vectorises every time,
        # which costs well under a second at this corpus size.
        is_corpus_fitted = isinstance(embedder, LsaEmbedder)

        if is_corpus_fitted:
            assert isinstance(embedder, LsaEmbedder)
            embedder.fit([c.text for c in chunks])
            embedder.save(settings.lsa_state_path)
            matrix = embedder.embed_documents([c.text for c in chunks])
            embedded, reused, incremental = len(chunks), 0, False
            log.info("corpus-fitted backend: refitted and re-vectorised all %d chunks", len(chunks))
        elif force or store.count() == 0:
            matrix = embedder.embed_documents([c.text for c in chunks])
            embedded, reused, incremental = len(chunks), 0, False
        else:
            matrix, embedded, reused = _embed_incrementally(chunks, embedder, store)
            incremental = True
            log.info("per-chunk backend: %d embedded, %d reused unchanged", embedded, reused)

        store.add_chunks(chunks, matrix)

        fingerprint = corpus_fingerprint(settings.corpus_dir)
        store.set_meta("embedding_backend", embedder.name)
        store.set_meta("embedding_dim", int(embedder.dim))
        store.set_meta("chunk_count", len(chunks))
        store.set_meta("corpus_fingerprint", fingerprint)
        store.set_meta("built_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        store.set_meta("source_files", sorted({d.source_file for d in docs}))
        store.set_meta("chunking", {
            "max_chars": settings.max_chars,
            "overlap_chars": settings.overlap_chars,
            "min_chars": settings.min_chars,
        })

    return IngestReport(
        files=len({d.source_file for d in docs}),
        raw_docs=len(docs),
        chunks=len(chunks),
        embedded=embedded,
        reused=reused,
        backend=embedder.name,
        dim=int(embedder.dim),
        incremental=incremental,
        corpus_fingerprint=fingerprint,
        elapsed_s=time.perf_counter() - t0,
        distribution=summarise(chunks),
    )


# ===========================================================================
# query path
# ===========================================================================


class QueryEngine:
    """Holds the expensive objects so a request does not rebuild them.

    The embedder, the index and the HTTP client are all constructed once — at
    CLI start or, for the API, at application startup. Doing this per request
    would put ~700 ms of model load into every query.
    """

    def __init__(self, settings: Settings, require_llm: bool = True):
        from .answerer import QueryExpander
        from .llm import LLMClient, LLMError
        from .retriever import Retriever
        from .tracing import Tracer

        self.s = settings
        settings.ensure_dirs()

        self.store = Store(settings.db_path, settings.vectors_path)
        self.embedder = get_embedder(settings)
        self.retriever = Retriever(self.store, self.embedder, settings)
        self.tracer = Tracer(self.store, settings)

        self.client: Optional[LLMClient] = None
        try:
            self.client = LLMClient(settings)
        except LLMError:
            # Retrieval-only use is legitimate and must not require a key.
            if require_llm:
                raise
            log.warning("no LLM client; retrieval-only mode")
        self.expander = QueryExpander(self.client, settings)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "QueryEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ask(self, question: str) -> "AnswerResult":
        """Expand -> retrieve -> answer -> verify -> trace.

        Exactly one trace row is written per call, on every path including
        failure: a failure must never be a gap in the log (§e.1).
        """
        from .answerer import answer as generate

        t0 = time.perf_counter()
        trace_id = self.tracer.new_trace_id()
        retrieval = None
        result = None
        prompt_meta: dict = {}
        expand_ms = 0.0

        try:
            # Passed as a callable, not a value: the retriever decides whether
            # the expansion is worth an LLM call at all. On a hard miss it is
            # not — see Retriever.retrieve.
            retrieval = self.retriever.retrieve(
                question, expand_fn=lambda: self.expander.expand(question)
            )
            expand_ms = retrieval.latency_expand_ms
            result, prompt_meta = generate(question, retrieval, self.client, self.s)

            result.trace_id = trace_id
            result.latency_expand_ms = expand_ms
            result.embedding_backend = self.embedder.name
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        except Exception as exc:
            stage = getattr(exc, "stage", "pipeline")
            elapsed = (time.perf_counter() - t0) * 1000
            self.tracer.write(
                self.tracer.build_row(
                    trace_id, question, retrieval, result, prompt_meta, elapsed,
                    latency_expand_ms=expand_ms,
                    embedding_backend=self.embedder.name,
                    error=f"{type(exc).__name__}: {exc}",
                    error_stage=stage,
                )
            )
            raise

        finally:
            if result is not None:
                self.tracer.write(
                    self.tracer.build_row(
                        trace_id, question, retrieval, result, prompt_meta,
                        result.latency_ms,
                        latency_expand_ms=expand_ms,
                        embedding_backend=self.embedder.name,
                    )
                )


def ask(question: str, settings: Settings) -> "AnswerResult":
    """One-shot convenience wrapper. The CLI and API hold a QueryEngine instead."""
    with QueryEngine(settings) as engine:
        return engine.ask(question)
