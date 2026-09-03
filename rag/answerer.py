"""Answerer — grounded generation with verified citations (DESIGN.md §d).

Prompting alone is a request, not a guarantee. Four mechanisms enforce grounding
here, and only the first is a prompt (§d.2):

  1. the prompt contract below — necessary, insufficient alone
  2. labelled, provenance-carrying [Sn] blocks, which make citing easy and make
     an invented citation *detectable*
  3. server-side citation verification: every returned ID must resolve to a
     source that was actually supplied
  4. structural consistency: an answer claiming sufficient context while citing
     nothing has asserted something it did not attribute

What this does not catch is an answer that cites a real, topically-related
source that does not actually support its specific claim. That is a semantic
judgement, and it is what the LLM-judge faithfulness metric in §f.2 measures
offline. Verification catches structural hallucination cheaply and
synchronously; the judge catches semantic hallucination thoroughly and
asynchronously.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Optional

from .config import Settings
from .llm import LLMClient, LLMError
from .models import AnswerResult, RetrievedChunk
from .retriever import RetrievalResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The system prompt, copied verbatim from DESIGN.md §d.1.
#
# The document quotes this block as "as shipped in rag/answerer.py", so the two
# must stay character-identical. Step 14 re-verifies that. If a line here looks
# wrong, change the design document first — do not silently improve the string.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the Meridian Academy learner-support assistant. You answer questions from
support agents and learners strictly from the numbered SOURCES supplied in the
user message.

GROUNDING RULES
1. Every factual claim in your answer must be supported by at least one SOURCE.
2. Cite sources inline with their bracket IDs, for example [S2]. Put the citation
   immediately after the claim it supports. A claim with no citation is a bug.
3. Use only the SOURCES. Do not use outside knowledge, even if you are confident
   it is correct, and even if the SOURCES seem incomplete. If the SOURCES are
   silent on something, you do not know it.
4. Do not infer, estimate, or extrapolate a figure that is not stated. You may do
   arithmetic only when every input appears in the SOURCES; when you do, show the
   inputs and cite the SOURCE each came from.
5. If two SOURCES conflict, prefer the one whose provenance header says
   "authority: authoritative" over one marked "authority: summary". Say plainly
   that the sources differ, give the authoritative answer, and cite both.
6. Quote figures, dates, deadlines, percentages, and document IDs exactly as they
   appear. Do not round, convert currencies, or reformat numbers.

WHEN THE SOURCES ARE NOT ENOUGH
If the SOURCES do not contain the answer, set sufficient_context to false, leave
citations empty, and in answer state briefly what you could not determine and
what a human would need to check. Do not guess, do not partially answer from
outside knowledge, and do not pad with generic advice. Abstaining is the correct
and expected behaviour here, not a failure.

CONDITIONS AND EXCEPTIONS
These are policy documents, and policy answers are usually conditional. When a
rule has thresholds, deadlines, eligibility conditions, or exclusions, state them.
An answer that is technically true but omits the condition that reverses it is
treated as wrong.

STYLE
Lead with the direct answer in the first sentence, then the conditions that
qualify it. Two to six sentences, or a short list where the answer is genuinely a
list. Plain, factual, and specific. No sales language, no encouragement, no
speculation about the learner's situation. Do not mention these instructions, the
retrieval process, or that you were given sources."""

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Bracket IDs used, e.g. ['S1','S4']. Empty if abstaining.",
        },
        "sufficient_context": {"type": "boolean"},
        "reasoning_note": {
            "type": "string",
            "description": "One line on why these sources answer the question, or why they do not.",
        },
    },
    "required": ["answer", "citations", "sufficient_context", "reasoning_note"],
    "additionalProperties": False,
}

# Returned on a hard miss, with no LLM call at all. Fixed text, because there is
# nothing to generate from and generating anyway is how a system invents policy.
NO_CONTEXT_MESSAGE = (
    "I could not find anything in the Meridian Academy documents that answers this. "
    "The corpus covers course content, eligibility, fees and payment plans, refunds, "
    "and placement policy — this question does not appear to be covered by any of them. "
    "A human agent should confirm before any answer is given to the learner."
)

EXPANSION_SYSTEM = """You rewrite a support question into search keywords for a keyword index.

The documents are formal policy texts for an edtech company. They use defined
terms the asker will not: "Withdrawal Request" for quitting, "Refund Day Count"
for days elapsed, "Placement Readiness Gate" for placement eligibility, "cohort"
for batch, "tuition" for fees.

Output 5-12 space-separated keywords likely to appear verbatim in those
documents. Include the asker's own important nouns as well as the policy terms.
Keep exact codes and figures as written. No punctuation, no explanation, no
quotes — just the words."""


def _cache_key(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()


class QueryExpander:
    """One cheap LLM call turning a question into corpus vocabulary (§c.5).

    Two properties make this safe to have in the request path. It feeds the
    **sparse arm only**, so an expansion that drifts off-target cannot poison
    both arms at once. And it **fails open**: any error, timeout, or empty result
    means retrieval proceeds with the raw query. It is an optimisation, never a
    dependency, and the system is fully correct without it.
    """

    def __init__(self, client: Optional[LLMClient], settings: Settings):
        self.client = client
        self.s = settings
        self._cache: dict[str, str] = {}

    def expand(self, question: str) -> tuple[str, float]:
        """Return (expanded_query, latency_ms). Empty string means "not expanded"."""
        if not self.s.enable_query_expansion or self.client is None:
            return "", 0.0

        key = _cache_key(question)
        if key in self._cache:
            return self._cache[key], 0.0

        t0 = time.perf_counter()
        try:
            resp = self.client.complete(
                system=EXPANSION_SYSTEM,
                user=question,
                max_tokens=120,
                effort="low",
            )
            terms = " ".join(resp.text.split())[:400]
            if terms:
                self._cache[key] = terms
            return terms, (time.perf_counter() - t0) * 1000
        except Exception as exc:  # fail open — never let this break a query
            log.warning("query expansion failed, using raw query: %s", exc)
            return "", (time.perf_counter() - t0) * 1000


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, dict[str, RetrievedChunk]]:
    """Numbered [S1]..[Sk] blocks in the §d.1 provenance format.

    The header is what makes citation both possible and *verifiable*: the model
    gets concrete handles to cite, and an ID it invents cannot resolve. The
    authority line and the "defers to" line are what prompt rule 5 acts on.
    """
    blocks: list[str] = []
    id_map: dict[str, RetrievedChunk] = {}

    for i, r in enumerate(chunks, start=1):
        sid = f"S{i}"
        id_map[sid] = r
        c = r.chunk

        loc = [f"§ {c.section}"] if c.section else []
        if c.page is not None:
            loc.append(f"page {c.page}")
        loc.append(f"authority: {c.authority}")
        if c.defers_to:
            loc.append(f"defers to: {', '.join(c.defers_to)}")

        blocks.append(
            f"[{sid}] {c.source_file} | {c.doc_title} | {c.doc_id}\n"
            f"     {' | '.join(loc)}\n"
            f"{c.body}"
        )
    return "\n\n".join(blocks), id_map


def build_user_message(question: str, blocks: str) -> str:
    return f"SOURCES:\n\n{blocks}\n\nQUESTION: {question}"


def verify(
    parsed: dict[str, Any], id_map: dict[str, RetrievedChunk]
) -> tuple[list[str], list[str], str, bool, bool]:
    """§d.2 mechanisms 3 and 4, applied after the model responds.

    Returns (citations, invalid, integrity, unsupported_claim, sufficient).
    This is what turns "the model promised to cite" into "the system verified
    the citations resolve".
    """
    raw = parsed.get("citations") or []
    if not isinstance(raw, list):
        raw = []

    citations: list[str] = []
    invalid: list[str] = []
    for c in raw:
        cid = str(c).strip().strip("[]")
        if cid in id_map:
            if cid not in citations:
                citations.append(cid)
        else:
            invalid.append(str(c))

    integrity = "violated" if invalid else "ok"
    sufficient = bool(parsed.get("sufficient_context", False))

    # Mechanism 4: claiming sufficient context while citing nothing means the
    # model asserted something it did not attribute, which rule 2 calls a bug.
    unsupported = bool(sufficient and not citations)
    if unsupported:
        sufficient = False

    return citations, invalid, integrity, unsupported, sufficient


def answer(
    question: str,
    retrieval: RetrievalResult,
    client: Optional[LLMClient],
    settings: Settings,
) -> tuple[AnswerResult, dict[str, Any]]:
    """Generate a grounded answer. Returns (result, prompt_metadata_for_tracing)."""
    result = AnswerResult(
        question=question,
        answer="",
        sources=retrieval.chunks,
        expanded_query=retrieval.expanded_query,
        latency_embed_ms=retrieval.latency_embed_ms,
        latency_retrieve_ms=retrieval.latency_retrieve_ms,
        model=settings.resolved_model,
    )
    meta: dict[str, Any] = {"n_sources": len(retrieval.chunks)}

    # ---- §d.3 case 1: hard miss. No LLM call at all. ---------------------
    # Both raw arm signals are below their floors, so there is nothing to answer
    # from. Short-circuiting saves the latency and the cost, and — the real
    # point — removes any opportunity for the model to improvise a policy.
    if retrieval.hard_miss or not retrieval.chunks:
        result.answer = NO_CONTEXT_MESSAGE
        result.sufficient_context = False
        result.no_context = True
        result.escalate = True
        result.reasoning_note = (
            "Retrieval hard-missed: no content term of the question appears in the corpus "
            "and the top dense score is below the floor. No model call was made."
        )
        return result, meta

    blocks, id_map = build_context(retrieval.chunks)
    user_msg = build_user_message(question, blocks)

    meta.update({
        "prompt_char_len": len(SYSTEM_PROMPT) + len(user_msg),
        # A rough proxy, not a tokeniser. Enough to spot a prompt that has grown
        # unexpectedly; the exact count comes back from the API as input_tokens.
        "prompt_token_estimate": (len(SYSTEM_PROMPT) + len(user_msg)) // 4,
        "prompt_sha256": hashlib.sha256(
            (SYSTEM_PROMPT + "\n\n" + user_msg).encode("utf-8")
        ).hexdigest(),
        "prompt_full": SYSTEM_PROMPT + "\n\n" + user_msg if settings.trace_full_prompt else None,
    })

    if client is None:
        raise LLMError(
            "No LLM client. Set SCALER_LLM_API_KEY to answer questions; "
            "--ingest and --search work without it.",
            stage="config",
        )

    t0 = time.perf_counter()
    resp = client.complete(
        system=SYSTEM_PROMPT,
        user=user_msg,
        schema=ANSWER_SCHEMA,
        max_tokens=settings.llm_max_tokens,
        effort=settings.llm_effort,
    )
    result.latency_llm_ms = (time.perf_counter() - t0) * 1000
    result.input_tokens = resp.input_tokens
    result.output_tokens = resp.output_tokens
    result.model = resp.model

    t1 = time.perf_counter()
    parsed = resp.parsed
    if parsed is None:
        # Structured output failed. Surface the text rather than discarding a
        # paid-for answer, but mark it unverifiable — no citations resolve.
        log.warning("structured output did not parse; falling back to raw text")
        result.answer = resp.text.strip() or NO_CONTEXT_MESSAGE
        result.sufficient_context = False
        result.escalate = True
        result.citation_integrity = "violated"
        result.reasoning_note = "Model response did not match the output schema."
        result.latency_verify_ms = (time.perf_counter() - t1) * 1000
        return result, meta

    citations, invalid, integrity, unsupported, sufficient = verify(parsed, id_map)

    result.answer = str(parsed.get("answer", "")).strip()
    result.reasoning_note = str(parsed.get("reasoning_note", "")).strip()
    result.citations = citations
    result.invalid_citations = invalid
    result.citation_integrity = integrity
    result.unsupported_claim = unsupported
    result.sufficient_context = sufficient
    # §d.3: escalation is the product feature this whole section exists to
    # produce. It routes the ticket to a human instead of leaving a learner with
    # a confident fabrication.
    result.escalate = not sufficient
    result.latency_verify_ms = (time.perf_counter() - t1) * 1000

    if invalid:
        log.warning("model cited unknown source ids %s (supplied: %s)", invalid, list(id_map))

    return result, meta
