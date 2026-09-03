"""Retriever — hybrid dense + BM25, fused by rank (DESIGN.md §c.3–c.6).

Both arms are load-bearing, and each covers the other's blind spot. The corpus
is full of exact tokens that carry the whole meaning — `ASD-501`, `INR
7,00,000`, `Day 31 to Day 60`, `MA-REFUND-2026-01` — where semantic search is
weak and BM25 is excellent. It is equally full of policy vocabulary the user
will never type: "Withdrawal Request" for "can I quit and get my money back",
where the reverse holds.

Deliberately has no dependency on the LLM. Retrieval failures are the majority
of RAG failures and no amount of prompt work rescues them, so this is
inspectable on its own via `python main.py --search`.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from typing import Optional

from .config import Settings
from .embeddings import Embedder
from .models import Chunk, RetrievedChunk
from .store import Store

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


@dataclasses.dataclass(slots=True)
class RetrievalResult:
    """Chunks plus the raw evidence behind them.

    The arm signals are carried out of here rather than collapsed into a single
    score because §d.3 decides abstention on them and §e.2 debugs with them.
    """

    chunks: list[RetrievedChunk]

    top_score: Optional[float]          # best raw dense cosine
    score_gap: Optional[float]          # top1 - top2 dense cosine
    top_bm25: Optional[float]

    dense_miss: bool
    sparse_miss: bool
    hard_miss: bool

    query: str
    expanded_query: str
    content_terms: list[str]            # query terms after stopword removal
    matched_terms: list[str]            # of those, the ones present in the index

    latency_embed_ms: float = 0.0
    latency_retrieve_ms: float = 0.0

    def signals(self) -> dict[str, object]:
        return {
            "top_score": self.top_score,
            "score_gap": self.score_gap,
            "top_bm25": self.top_bm25,
            "dense_miss": self.dense_miss,
            "sparse_miss": self.sparse_miss,
            "hard_miss": self.hard_miss,
            "content_terms": self.content_terms,
            "matched_terms": self.matched_terms,
        }


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.s = settings

    # ------------------------------------------------------------------ arms
    def _content_terms(self, query: str) -> list[str]:
        """Query words that would actually prove a match.

        The domain stopwords matter as much as the English ones: nearly every
        chunk contains "Meridian" or "program", so without removing them the
        sparse miss-condition could never fire and no question would ever be
        detected as out-of-corpus (§c.4).
        """
        stop = self.s.stopwords
        return [w for w in (m.group().lower() for m in _WORD_RE.finditer(query)) if w not in stop]

    # ----------------------------------------------------------------- fusion
    def _fuse(
        self,
        dense: list[tuple[str, float]],
        sparse: list[tuple[str, float]],
        chunks: dict[str, Chunk],
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion.

        The arms are not comparable by magnitude — cosine is bounded in [0,1],
        BM25 is unbounded and corpus-dependent — and normalising them needs
        weights there is no principled way to pick and no data to fit. RRF uses
        only rank, so it is parameter-light, robust to one arm being badly
        calibrated, and rewards chunks both arms agree on.
        """
        k = self.s.rrf_k
        dense_rank = {cid: i + 1 for i, (cid, _) in enumerate(dense)}
        sparse_rank = {cid: i + 1 for i, (cid, _) in enumerate(sparse)}
        dense_score = dict(dense)
        sparse_score = dict(sparse)

        out: list[RetrievedChunk] = []
        for cid in dict.fromkeys([c for c, _ in dense] + [c for c, _ in sparse]):
            chunk = chunks.get(cid)
            if chunk is None:  # index/db drift; skip rather than crash a query
                continue
            score = 0.0
            if cid in dense_rank:
                score += 1.0 / (k + dense_rank[cid])
            if cid in sparse_rank:
                score += 1.0 / (k + sparse_rank[cid])
            out.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=dense_score.get(cid),
                    dense_rank=dense_rank.get(cid),
                    bm25_score=sparse_score.get(cid),
                    bm25_rank=sparse_rank.get(cid),
                    rrf_score=score,
                    rrf_score_prefused=score,
                )
            )
        out.sort(key=lambda r: -r.rrf_score)
        return out

    def _apply_authority(self, fused: list[RetrievedChunk]) -> None:
        """Demote a chunk that points at a document already in the result set (§c.6).

        This exists because of a measured failure: on "What is the refund
        policy?" the dense arm ranks the FAQ's summary (0.732) above the
        authoritative refund schedule (0.724). Both are relevant; only one is
        safe to quote, and the FAQ's own text says so.

        Two properties this must keep. It **reorders, never filters** — the
        summary stays retrievable and citable, because sometimes it is the best
        answer. And "overlapping topic" is never judged by code: the only signal
        is a document ID the author explicitly wrote, so the rule cannot
        misfire on a guess about meaning.
        """
        present = {r.chunk.doc_id for r in fused}
        for r in fused:
            if r.chunk.defers_to and any(d in present for d in r.chunk.defers_to):
                r.rrf_score *= self.s.authority_penalty
                r.authority_demoted = True
        fused.sort(key=lambda r: -r.rrf_score)

    # ------------------------------------------------------------- entrypoint
    def retrieve(self, query: str, expanded_query: Optional[str] = None) -> RetrievalResult:
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(query)  # ORIGINAL query, never the expansion
        t_embed = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        n = self.s.candidates_per_arm

        dense = self.store.dense_search(qvec, n)
        # §c.5: the expansion feeds the sparse arm only. The dense arm sees the
        # user's own words, so an expansion that drifts off-target cannot poison
        # both arms at once.
        sparse_query = expanded_query or query
        sparse = self.store.bm25_search(sparse_query, n)

        chunks = self.store.get_chunks({c for c, _ in dense} | {c for c, _ in sparse})
        fused = self._fuse(dense, sparse, chunks)
        self._apply_authority(fused)

        top = fused[: self.s.top_k]
        for i, r in enumerate(top, start=1):
            r.final_rank = i

        # ----- miss detection (§c.4) -------------------------------------
        # Note what is deliberately absent: a floor on the fused RRF score.
        # Both arms always return their top-N regardless of relevance, so a
        # fused score measures agreement between arms, not absolute relevance —
        # a threshold on it cannot tell a good match from the best of a bad lot.
        # Detection therefore reads each arm's RAW signal, before fusion.
        top_cosine = dense[0][1] if dense else None
        gap = (dense[0][1] - dense[1][1]) if len(dense) > 1 else None
        top_bm25 = sparse[0][1] if sparse else None

        content_terms = self._content_terms(sparse_query)
        matched = self.store.matching_terms(content_terms)

        dense_miss = top_cosine is None or top_cosine < self.s.miss_dense_cosine
        sparse_miss = (not matched) or (top_bm25 is None) or (top_bm25 < self.s.miss_bm25_floor)

        return RetrievalResult(
            chunks=top,
            top_score=top_cosine,
            score_gap=gap,
            top_bm25=top_bm25,
            dense_miss=dense_miss,
            sparse_miss=sparse_miss,
            # Both must hold. Either alone is a routine, survivable weakness:
            # a paraphrase the embedder handles but BM25 cannot, or an exact
            # code BM25 finds and the embedder blurs. Only when neither arm has
            # any real purchase is the corpus genuinely silent.
            hard_miss=dense_miss and sparse_miss,
            query=query,
            expanded_query=expanded_query or "",
            content_terms=content_terms,
            matched_terms=matched,
            latency_embed_ms=t_embed,
            latency_retrieve_ms=(time.perf_counter() - t1) * 1000,
        )
