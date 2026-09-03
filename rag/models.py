"""The pipeline's shared vocabulary.

Four dataclasses carry everything between stages:

    RawDoc         a loader's output; format-agnostic         (§b.4)
    Chunk          a chunker's output; the indexed unit       (§b.1-b.3)
    RetrievedChunk a Chunk plus how it scored, in each arm    (§c.3-c.6)
    AnswerResult   the full outcome of one question           (§d)

The loader boundary is the important one: after a file becomes a `RawDoc`,
nothing downstream knows or cares whether it started as a PDF, and that is what
keeps the chunker from growing per-format branches.

No I/O and no business logic here — these are contracts, not behaviour.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional

DocType = Literal["pdf", "text", "faq", "yaml"]

# §c.6. Three tiers, not two:
#   authoritative — self-declares as governing, or is the sole source for its
#                   content (refund_terms, placement_policy, program_pricing,
#                   eligibility_criteria)
#   reference     — accurate, but points elsewhere for the binding version
#                   (course_brochure)
#   summary       — explicitly says not to quote it (the FAQ's refund section)
Authority = Literal["authoritative", "reference", "summary"]

CitationIntegrity = Literal["ok", "violated"]


@dataclasses.dataclass(frozen=True, slots=True)
class RawDoc:
    """One loadable unit of source text, normalised out of its original format.

    PDFs produce one RawDoc per page so page numbers survive as citation
    anchors; text files produce one; the YAML pricing file produces one per
    top-level record, already rendered to prose.
    """

    text: str
    source_file: str
    doc_title: str
    doc_id: str
    doc_type: DocType
    authority: Authority = "authoritative"

    # Document IDs this document defers to, read from its own header or
    # "next steps" section — e.g. the brochure points at the pricing,
    # eligibility, and placement documents for the binding versions. Drives the
    # §c.6 reordering, which only fires when the named document is actually
    # present in the same result set.
    defers_to: tuple[str, ...] = ()

    page: Optional[int] = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class Chunk:
    """One indexed, retrievable, citable unit.

    `text` and `body` differ deliberately. `text` carries the breadcrumb header
    described in §b.2 and is what gets embedded and BM25-indexed — it is what
    repairs the context that splitting destroyed. `body` is the bare content,
    and is what the model is shown and what `content_hash` covers, so a
    formatting change to the header never triggers a spurious re-embed.
    """

    chunk_id: str
    text: str
    body: str

    source_file: str
    doc_title: str
    doc_id: str
    doc_type: DocType
    section: str
    authority: Authority
    defers_to: tuple[str, ...]
    content_hash: str

    page: Optional[int] = None
    char_len: int = 0

    def provenance(self) -> str:
        """The `[Sn]` header line the model sees (§d.1). Authority is included
        because system-prompt rule 5 resolves conflicts by reading it."""
        parts = [self.source_file, self.doc_title, self.doc_id]
        loc = f"§ {self.section}" if self.section else ""
        if self.page is not None:
            loc = f"{loc} | page {self.page}" if loc else f"page {self.page}"
        if loc:
            parts.append(loc)
        parts.append(f"authority: {self.authority}")
        return " | ".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(slots=True)
class RetrievedChunk:
    """A `Chunk` with the evidence for why it was retrieved.

    Mutable by design: the §c.6 authority pass rescales `rrf_score` after
    fusion, and `final_rank` is assigned after that.

    Both arms' scores *and* ranks are carried, not just the fused result. That
    is what lets §e.2 step 3 localise a failure to a specific arm — a chunk with
    a good `bm25_rank` and a bad `dense_rank` points at the embedder, and the
    reverse points at vocabulary mismatch. A field is None when that arm did not
    return this chunk at all, which is itself the signal.
    """

    chunk: Chunk

    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None

    rrf_score: float = 0.0
    rrf_score_prefused: Optional[float] = None  # before the authority penalty
    authority_demoted: bool = False
    final_rank: Optional[int] = None

    def to_dict(self, preview_chars: int = 240) -> dict[str, Any]:
        """Trace/API shape (§e.1). Carries a preview rather than full text — the
        full text is one join away in the `chunks` table."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc": self.chunk.source_file,
            "doc_id": self.chunk.doc_id,
            "section": self.chunk.section,
            "page": self.chunk.page,
            "authority": self.chunk.authority,
            "dense_score": self.dense_score,
            "dense_rank": self.dense_rank,
            "bm25_score": self.bm25_score,
            "bm25_rank": self.bm25_rank,
            "rrf_score": self.rrf_score,
            "authority_demoted": self.authority_demoted,
            "final_rank": self.final_rank,
            "text_preview": self.chunk.body[:preview_chars],
        }


@dataclasses.dataclass(slots=True)
class AnswerResult:
    """The complete outcome of one question, as returned by the API and traced.

    An abstention is a *successful* result with `sufficient_context=False` and
    `escalate=True` — never an error (§d.3). `escalate` is the field a real
    support tool routes on.
    """

    question: str
    answer: str
    citations: list[str] = dataclasses.field(default_factory=list)
    sufficient_context: bool = False
    reasoning_note: str = ""
    sources: list[RetrievedChunk] = dataclasses.field(default_factory=list)

    # §d.2 verification verdicts, set server-side after the model responds.
    citation_integrity: CitationIntegrity = "ok"
    invalid_citations: list[str] = dataclasses.field(default_factory=list)
    unsupported_claim: bool = False

    # §d.3 routing.
    no_context: bool = False
    escalate: bool = False

    trace_id: str = ""
    expanded_query: str = ""

    latency_ms: float = 0.0
    latency_expand_ms: float = 0.0
    latency_embed_ms: float = 0.0
    latency_retrieve_ms: float = 0.0
    latency_llm_ms: float = 0.0
    latency_verify_ms: float = 0.0

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    embedding_backend: str = ""

    error: Optional[str] = None
    error_stage: Optional[str] = None

    @property
    def top_score(self) -> Optional[float]:
        """Best raw dense cosine. Logged per query so the §c.4 miss threshold can
        be recalibrated from real traffic rather than guessed twice."""
        return self.sources[0].dense_score if self.sources else None

    @property
    def score_gap(self) -> Optional[float]:
        """top1 − top2 dense cosine. A cheap confidence proxy: a flat
        distribution across five chunks means the retriever is guessing."""
        if len(self.sources) < 2:
            return None
        a, b = self.sources[0].dense_score, self.sources[1].dense_score
        return None if a is None or b is None else a - b

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "sufficient_context": self.sufficient_context,
            "reasoning_note": self.reasoning_note,
            "sources": [s.to_dict() for s in self.sources],
            "citation_integrity": self.citation_integrity,
            "invalid_citations": self.invalid_citations,
            "unsupported_claim": self.unsupported_claim,
            "no_context": self.no_context,
            "escalate": self.escalate,
            "trace_id": self.trace_id,
            "expanded_query": self.expanded_query,
            "latency_ms": self.latency_ms,
            "latency_breakdown": {
                "expand_ms": self.latency_expand_ms,
                "embed_ms": self.latency_embed_ms,
                "retrieve_ms": self.latency_retrieve_ms,
                "llm_ms": self.latency_llm_ms,
                "verify_ms": self.latency_verify_ms,
            },
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "embedding_backend": self.embedding_backend,
            "top_score": self.top_score,
            "score_gap": self.score_gap,
            "error": self.error,
            "error_stage": self.error_stage,
        }
