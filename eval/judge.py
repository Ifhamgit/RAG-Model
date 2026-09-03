"""LLM judge for the three semantic metrics (DESIGN.md §f.2).

The design rule that matters here: **each judge sees only what it needs.**

The faithfulness judge is never shown `expected_facts`, because it is not being
asked "is this right" — it is asked "is this supported by the context the model
was actually given". Those come apart, and keeping them apart is the whole point
of measuring them separately: a system can be perfectly faithful to
badly-retrieved context and still be confidently wrong, and only two metrics
side by side reveal that.

Every judgement returns a one-line justification, written into
`eval/results/`, so a surprising score can be inspected rather than trusted.
§f.3 is explicit that this judge shares a model family with the system under
test and can therefore share its blind spots.
"""

from __future__ import annotations

from typing import Any

from rag.llm import LLMClient

# --------------------------------------------------------------------------
# 1. Answer correctness — does the answer assert each required fact?
# --------------------------------------------------------------------------

CORRECTNESS_SYSTEM = """You check whether an ANSWER asserts each of a list of REQUIRED FACTS.

Judge meaning, not wording. A fact counts as present if the answer states it in
any phrasing, including numerals written differently.

  fact "40%"                     answer "forty percent of tuition"       -> present
  fact "GST is not refunded"     answer "GST is excluded from the refund" -> present
  fact "GST is not refunded"     answer "GST is refunded in full"         -> ABSENT (it states the opposite)
  fact "21 business days"        answer "about three weeks"               -> ABSENT (not the stated figure)

Some facts are NEGATIVE, e.g. "does not apply the 10% upfront discount". Such a
fact is present when the answer's reasoning is consistent with it — either by
saying so, or by producing a figure that only holds if the discount was not
applied. It is absent if the answer applies the thing it should not have.

Return one verdict per fact, in the same order, each with a one-line reason."""

CORRECTNESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "present": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["fact", "present", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------
# 2. Faithfulness — is every claim supported by the retrieved context?
# --------------------------------------------------------------------------

FAITHFULNESS_SYSTEM = """You check whether an ANSWER is supported by the CONTEXT it was given.

Split the answer into its distinct factual claims and classify each one:

  supported      the context states it, or it follows directly from arithmetic
                 on figures the context states
  contradicted   the context states something incompatible with it
  not_in_context the context neither states nor contradicts it — the claim came
                 from somewhere else

Worked example. CONTEXT: "Day 31 to Day 60: 40% tuition refunded, GST NOT
refunded." ANSWER: "You get 40% back [S1], GST is not refunded [S1], and
processing takes 21 business days."
  -> "40% refunded"          supported
  -> "GST is not refunded"   supported
  -> "21 business days"      not_in_context (true elsewhere, but not in THIS context)

Judge only against the context shown. Do not use outside knowledge, and do not
reward a claim for being true in the real world. An answer that correctly
declines to answer ("the sources do not cover this") has no factual claims to
check and is trivially faithful — return an empty list.

Extract only claims ABOUT THE SUBJECT MATTER. Statements about the answering
process itself are not factual claims and must be skipped entirely, not
classified as not_in_context:

  "the sources do not state a placement rate for 2026"   skip (about the sources)
  "a human would need to check the 2026 outcomes report" skip (a recommendation)
  "I could not determine the exact figure"               skip (about the answer)

This matters because the assistant is instructed to say what it could not
determine and what a human should check. Counting that instruction-following as
an ungrounded claim would penalise the answer for behaving correctly, and would
make careful abstentions score worse than confident ones."""

FAITHFULNESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "contradicted", "not_in_context"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["claim", "verdict", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------
# 3. Context precision — was each retrieved chunk actually useful?
# --------------------------------------------------------------------------

PRECISION_SYSTEM = """You judge whether each retrieved SOURCE was useful for answering the QUESTION.

Useful means the source contains information a correct answer would draw on or
would need in order to state a condition, exception, or figure. Being on the
same broad topic is not enough.

  question "refund on day 45"   source: the refund day-count table        -> useful
  question "refund on day 45"   source: GST treatment by day count        -> useful
  question "refund on day 45"   source: how to contact support            -> not useful
  question "refund on day 45"   source: the dispute/arbitration clause    -> not useful

Return one verdict per source, in the order given, each with a one-line reason."""

PRECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "useful": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["source_id", "useful", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


class Judge:
    """Wraps the three judged metrics. effort='low' and a strict schema, never
    a temperature — current Anthropic models reject sampling parameters."""

    def __init__(self, client: LLMClient):
        self.client = client

    def _ask(self, system: str, user: str, schema: dict) -> dict[str, Any]:
        resp = self.client.complete(
            system=system, user=user, schema=schema, max_tokens=2000, effort="low"
        )
        return resp.parsed or {}

    # -- metric 2 ---------------------------------------------------------
    def correctness(self, question: str, answer: str, facts: list[str]) -> dict[str, Any]:
        if not facts:
            return {"score": 1.0, "verdicts": [], "note": "no required facts"}
        listed = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
        out = self._ask(
            CORRECTNESS_SYSTEM,
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nREQUIRED FACTS:\n{listed}",
            CORRECTNESS_SCHEMA,
        )
        v = out.get("verdicts", [])
        present = sum(1 for x in v if x.get("present"))
        return {
            "score": present / len(facts) if facts else 1.0,
            "present": present,
            "total": len(facts),
            "verdicts": v,
        }

    # -- metric 3 ---------------------------------------------------------
    def faithfulness(self, answer: str, context: str) -> dict[str, Any]:
        # Note what is NOT passed: the question's expected facts. This judge is
        # asked whether the answer is grounded in what it was shown, not whether
        # it is correct (§f.2).
        out = self._ask(
            FAITHFULNESS_SYSTEM,
            f"CONTEXT:\n{context}\n\nANSWER:\n{answer}",
            FAITHFULNESS_SCHEMA,
        )
        claims = out.get("claims", [])
        if not claims:
            return {"score": 1.0, "contradicted": 0, "claims": [], "note": "no factual claims"}
        supported = sum(1 for c in claims if c.get("verdict") == "supported")
        contradicted = sum(1 for c in claims if c.get("verdict") == "contradicted")
        return {
            "score": supported / len(claims),
            "supported": supported,
            "contradicted": contradicted,
            "not_in_context": len(claims) - supported - contradicted,
            "total": len(claims),
            "claims": claims,
        }

    # -- metric 4 ---------------------------------------------------------
    def context_precision(self, question: str, sources: list[tuple[str, str]]) -> dict[str, Any]:
        if not sources:
            return {"score": 1.0, "verdicts": [], "note": "no sources retrieved"}
        blocks = "\n\n".join(f"[{sid}]\n{text[:900]}" for sid, text in sources)
        out = self._ask(
            PRECISION_SYSTEM,
            f"QUESTION:\n{question}\n\nSOURCES:\n{blocks}",
            PRECISION_SCHEMA,
        )
        v = out.get("verdicts", [])
        useful = sum(1 for x in v if x.get("useful"))
        return {
            "score": useful / len(sources) if sources else 1.0,
            "useful": useful,
            "total": len(sources),
            "verdicts": v,
        }
