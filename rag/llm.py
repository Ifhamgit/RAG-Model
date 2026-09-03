"""LLM client — thin, provider-agnostic, no framework (DESIGN.md §a.3 decision 8).

One `complete()` call used by three callers: the answerer (§d), query expansion
(§c.5), and the evaluation judge (§f.2). Everything provider-specific is
confined here.

**No temperature, anywhere in the interface.** Current Anthropic models reject
sampling parameters with a 400, so no code path may send one. Determinism where
it matters — the judge — comes from a fixed rubric, low effort, and a strict
output schema, not from a sampling knob.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
import re
import time
from typing import Any, Optional

from .config import Settings

log = logging.getLogger(__name__)

# Retry only on failures that a retry can plausibly fix. A 400 means the request
# is wrong and will be wrong again; retrying it wastes time and money and buries
# the real error behind a timeout.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Any LLM failure, carrying the stage that failed so traces can localise it."""

    def __init__(self, message: str, stage: str = "llm", status: Optional[int] = None):
        super().__init__(message)
        self.stage = stage
        self.status = status


@dataclasses.dataclass(slots=True)
class LLMResponse:
    text: str
    parsed: Optional[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    latency_ms: float
    finish_reason: str = ""


def _status_of(exc: Exception) -> Optional[int]:
    """Best-effort HTTP status from an SDK exception, without importing either SDK."""
    for attr in ("status_code", "status", "http_status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    v = getattr(resp, "status_code", None)
    return v if isinstance(v, int) else None


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Parse a JSON object out of a model response.

    Structured output should make this unnecessary, but not every model on every
    gateway honours a schema strictly, and an answer is too expensive to discard
    over a stray code fence. Falls back to the outermost brace-balanced span.
    """
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class LLMClient:
    """Resolves a provider from the key, then speaks whichever protocol it needs."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.provider = settings.resolved_provider
        self.model = settings.resolved_model

        if not settings.has_api_key:
            raise LLMError(
                "No API key. Set SCALER_LLM_API_KEY in .env (see .env.example).\n"
                "Retrieval-only commands (--ingest, --search) work without it.",
                stage="config",
            )

        key = settings.llm_api_key.get_secret_value()
        base_url = settings.resolved_base_url

        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=key, timeout=settings.llm_timeout_s, max_retries=0
            )
        else:
            # OpenAI and OpenRouter share a wire protocol; only the host differs.
            from openai import OpenAI

            self._client = OpenAI(
                api_key=key,
                timeout=settings.llm_timeout_s,
                max_retries=0,
                **({"base_url": base_url} if base_url else {}),
            )

        # Never log the key — only what it resolved to.
        log.info(
            "llm: provider=%s model=%s%s",
            self.provider,
            self.model,
            f" base_url={base_url}" if base_url else "",
        )

    # ---------------------------------------------------------------- public
    def complete(
        self,
        system: str,
        user: str,
        schema: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        effort: Optional[str] = None,
    ) -> LLMResponse:
        """One request. Retries are bounded and only on retryable failures."""
        max_tokens = max_tokens or self.s.llm_max_tokens
        effort = effort or self.s.llm_effort

        last: Optional[Exception] = None
        for attempt in range(self.s.llm_max_retries + 1):
            t0 = time.perf_counter()
            try:
                if self.provider == "anthropic":
                    return self._anthropic(system, user, schema, max_tokens, effort, t0)
                return self._openai(system, user, schema, max_tokens, effort, t0)
            except LLMError:
                raise
            except Exception as exc:
                status = _status_of(exc)
                retryable = status is None or status in RETRYABLE_STATUS
                if not retryable or attempt == self.s.llm_max_retries:
                    raise LLMError(
                        f"{type(exc).__name__}: {exc}", stage="llm", status=status
                    ) from exc
                # Exponential backoff with jitter: synchronised retries from
                # parallel eval cases would re-hit the same rate limit together.
                delay = (2**attempt) + random.uniform(0, 0.5)
                log.warning(
                    "llm attempt %d/%d failed (status=%s); retrying in %.1fs: %s",
                    attempt + 1, self.s.llm_max_retries + 1, status, delay, exc,
                )
                time.sleep(delay)
                last = exc
        raise LLMError(f"exhausted retries: {last}", stage="llm")  # pragma: no cover

    # ------------------------------------------------------------- providers
    def _anthropic(self, system, user, schema, max_tokens, effort, t0) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            # Adaptive thinking is the current form; budget_tokens is removed on
            # these models and returns a 400. Depth is controlled by effort.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
        if schema is not None:
            kwargs["output_config"]["format"] = {
                "type": "json_schema",
                "schema": schema,
            }
        # Deliberately absent: temperature/top_p/top_k (400 on current models),
        # and assistant prefill (also rejected).

        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

        # A refusal arrives as HTTP 200 with stop_reason "refusal", so it must be
        # checked before reading content — otherwise it looks like an empty answer.
        if getattr(resp, "stop_reason", "") == "refusal":
            detail = getattr(resp, "stop_details", None)
            raise LLMError(
                f"model declined the request (category={getattr(detail, 'category', None)})",
                stage="llm_refusal",
            )

        return LLMResponse(
            text=text,
            parsed=_extract_json(text) if schema is not None else None,
            input_tokens=int(getattr(resp.usage, "input_tokens", 0)),
            output_tokens=int(getattr(resp.usage, "output_tokens", 0)),
            model=getattr(resp, "model", self.model),
            provider=self.provider,
            latency_ms=(time.perf_counter() - t0) * 1000,
            finish_reason=getattr(resp, "stop_reason", "") or "",
        )

    def _openai(self, system, user, schema, max_tokens, effort, t0) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            }
        if self.provider == "openrouter":
            # OpenRouter's own reasoning control, passed through to whichever
            # upstream model is selected. `extra_body` because it is not part of
            # the OpenAI schema the SDK validates against.
            kwargs["extra_body"] = {"reasoning": {"effort": effort}}

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)

        return LLMResponse(
            text=text,
            parsed=_extract_json(text) if schema is not None else None,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            model=getattr(resp, "model", self.model),
            provider=self.provider,
            latency_ms=(time.perf_counter() - t0) * 1000,
            finish_reason=getattr(choice, "finish_reason", "") or "",
        )
