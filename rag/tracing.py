"""Tracing — one structured record per query (DESIGN.md §e).

The design principle, from §e.2: **a trace must contain enough to reproduce the
failure without the user.** Everything in the schema earns its place against
that test.

Two sinks, deliberately. SQLite because it lives beside the chunks, so
investigating a bad answer is a join from trace to retrieved chunk to chunk
text. JSONL because it greps, tails, and ships to a log pipeline unchanged.

Two fields deserve a note because they are the ones people leave out:

  * **per-arm ranks as well as scores.** §e.2 step 3 localises a failure by
    comparing `dense_rank` against `bm25_rank`. A good BM25 rank with a bad
    dense rank means the embedder missed; the reverse means vocabulary
    mismatch; bad in both means the chunk itself is wrong. Without both, that
    step is guesswork.
  * **`prompt_sha256`.** The full prompt is large and repetitive, so it is
    logged only under TRACE_FULL_PROMPT. The hash always is: it settles "is
    this a retrieval bug or model nondeterminism?" in one query, by proving
    whether two runs sent an identical prompt.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
import uuid
from typing import Any, Optional

from .config import Settings
from .models import AnswerResult
from .retriever import RetrievalResult
from .store import Store

log = logging.getLogger(__name__)

# Approximate USD per 1M tokens, for the cost figure in the trace. Used to
# answer "what would this cost at 500 tickets/day", not to reconcile a bill —
# gateway pricing varies and these are refreshed by hand.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-5": (1.25, 10.00),
}
_DEFAULT_PRICE = (2.00, 10.00)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Best-effort cost. Matches on the model name's tail so an OpenRouter slug
    like "anthropic/claude-sonnet-5" resolves to the same entry."""
    name = (model or "").split("/")[-1].strip()
    inp, out = PRICING.get(name, _DEFAULT_PRICE)
    return round((input_tokens * inp + output_tokens * out) / 1_000_000, 6)


@dataclasses.dataclass(slots=True)
class Tracer:
    """Builds and persists one trace row per query."""

    store: Store
    settings: Settings

    def new_trace_id(self) -> str:
        return str(uuid.uuid4())

    def build_row(
        self,
        trace_id: str,
        question: str,
        retrieval: Optional[RetrievalResult],
        result: Optional[AnswerResult],
        prompt_meta: Optional[dict[str, Any]],
        latency_ms: float,
        latency_expand_ms: float = 0.0,
        embedding_backend: str = "",
        error: Optional[str] = None,
        error_stage: Optional[str] = None,
    ) -> dict[str, Any]:
        pm = prompt_meta or {}
        retrieved = [r.to_dict() for r in (result.sources if result else [])]

        row: dict[str, Any] = {
            "trace_id": trace_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "query": question,
            "expanded_query": retrieval.expanded_query if retrieval else None,
            "retrieved": retrieved,
            # Which code path actually ran — essential when behaviour differs
            # between machines because the embedder auto-selected differently.
            "retrieval_mode": _mode(retrieval),
            "embedding_backend": embedding_backend,
            "prompt_token_estimate": pm.get("prompt_token_estimate"),
            "prompt_char_len": pm.get("prompt_char_len"),
            "prompt_sha256": pm.get("prompt_sha256"),
            "prompt_full": pm.get("prompt_full"),
            "n_sources": pm.get("n_sources", len(retrieved)),
            "latency_ms": round(latency_ms, 2),
            "latency_expand_ms": round(latency_expand_ms, 2),
            "error": error,
            "error_stage": error_stage,
        }

        if retrieval is not None:
            row.update({
                "top_score": retrieval.top_score,
                "score_gap": retrieval.score_gap,
                "latency_embed_ms": round(retrieval.latency_embed_ms, 2),
                "latency_retrieve_ms": round(retrieval.latency_retrieve_ms, 2),
            })

        if result is not None:
            row.update({
                "answer": result.answer,
                "citations": result.citations,
                "sufficient_context": result.sufficient_context,
                "reasoning_note": result.reasoning_note,
                "citation_integrity": result.citation_integrity,
                "invalid_citations": result.invalid_citations,
                "unsupported_claim": result.unsupported_claim,
                "latency_llm_ms": round(result.latency_llm_ms, 2),
                "latency_verify_ms": round(result.latency_verify_ms, 2),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "model": result.model,
                "est_cost_usd": estimate_cost_usd(
                    result.model, result.input_tokens, result.output_tokens
                ),
                "no_context": result.no_context,
                "escalate": result.escalate,
            })
        return row

    def write(self, row: dict[str, Any]) -> None:
        """Persist to both sinks.

        Tracing must never break a query: an observability failure is not a
        reason to fail a user's request, so every write is guarded and a failure
        is logged and swallowed.
        """
        try:
            self.store.write_trace(row)
        except Exception as exc:
            log.warning("failed to write trace to sqlite: %s", exc)

        try:
            path = pathlib.Path(self.settings.traces_jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        except Exception as exc:
            log.warning("failed to append trace to jsonl: %s", exc)


def _mode(retrieval: Optional[RetrievalResult]) -> str:
    """A compact label for how retrieval went, for filtering traces in bulk."""
    if retrieval is None:
        return "none"
    if retrieval.hard_miss:
        return "hard_miss"
    parts = ["hybrid"]
    if retrieval.expanded_query:
        parts.append("expanded")
    if retrieval.dense_miss:
        parts.append("weak_dense")
    if retrieval.sparse_miss:
        parts.append("weak_sparse")
    return "+".join(parts)
