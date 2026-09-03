"""Settings — one source of truth for every tunable in the pipeline.

Values and their justifications come from DESIGN.md; the section reference on
each field points at the paragraph that argues for it. Nothing here reads a
document, calls a model, or touches the index — this module is configuration
only, so importing it is always cheap and always safe.

Precedence: process environment > .env > the defaults below.
"""

from __future__ import annotations

import pathlib
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root. Relative paths in settings resolve against this rather than
# the process working directory, so `python main.py` behaves identically whether
# it is run from the repo root or anywhere else.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

Provider = Literal["auto", "anthropic", "openai", "openrouter"]
EmbeddingBackend = Literal["auto", "fastembed", "local", "openai"]
Effort = Literal["low", "medium", "high"]

# Default model per resolved provider. Anthropic direct is the path DESIGN.md
# documents; the other two exist so the service runs against whatever key the
# machine actually has (DESIGN.md §a.3 decision 8).
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-5",
    # Verified against OpenRouter's live model catalogue. Sonnet-tier is the
    # right default for grounded extraction over ~1,500 tokens of supplied
    # context: the task is faithfulness to given text, not open-ended
    # reasoning. Override with LLM_MODEL=anthropic/claude-opus-4.8 if the
    # faithfulness metric (§f.2) says otherwise.
    "openrouter": "anthropic/claude-sonnet-5",
}

# OpenRouter speaks the OpenAI wire protocol at its own host, so the OpenAI SDK
# drives it unchanged once base_url is redirected.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Terms so common in this corpus that matching them proves nothing. Without
# this list the sparse miss-condition in §c.4 would never fire: virtually every
# chunk contains "Meridian" or "program", so every query would "match
# something" and no question would ever be detected as out-of-corpus.
DOMAIN_STOPWORDS: frozenset[str] = frozenset(
    {
        "meridian",
        "academy",
        "program",
        "programs",
        "programme",
        "course",
        "courses",
        "learner",
        "learners",
        "offer",
        "offers",
        "offered",
        "support",
    }
)

# Ordinary English stopwords, stripped before the sparse miss-check for the
# same reason. Deliberately short — this is a match-significance filter, not a
# linguistics exercise, and FTS5 does its own tokenising.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "doing", "have", "has", "had", "having",
        "i", "you", "he", "she", "it", "we", "they", "me", "my", "your",
        "and", "or", "but", "if", "then", "than", "so", "as", "of", "at",
        "by", "for", "with", "about", "into", "to", "from", "in", "on",
        "can", "could", "will", "would", "should", "may", "might", "must",
        "what", "when", "where", "who", "whom", "which", "why", "how",
        "there", "here", "this", "that", "these", "those", "any", "some",
        "get", "got", "no", "not", "yes", "please", "tell",
    }
)


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------- LLM ---
    # SecretStr so the value never appears in a repr, a log line, a pydantic
    # validation error, or a serialised trace. Read it deliberately with
    # `settings.llm_api_key.get_secret_value()` at the one call site that needs
    # it (rag/llm.py) and nowhere else.
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="SCALER_LLM_API_KEY",
        description="The single key the brief provides. Answering, expansion, and the eval judge.",
    )
    llm_provider: Provider = "auto"
    llm_model: str = ""
    llm_base_url: str = ""

    # Reasoning effort for the answering call. DESIGN.md §d is an extraction
    # task over supplied context, so "medium" is the cost/quality sweet spot.
    # Raise if the §f.2 faithfulness metric drops.
    llm_effort: Effort = "medium"
    llm_max_tokens: int = 2048
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # Deliberately absent: temperature. Current Anthropic models reject sampling
    # parameters outright, so no code path may send one. Determinism where it
    # matters (the eval judge, §f.2) comes from structured output plus a rubric.

    # --------------------------------------------------------- embeddings ---
    # §c.1. "auto" prefers fastembed (bge-small-en-v1.5) and falls back to the
    # dependency-free TF-IDF/LSA embedder. An explicit value fails loudly rather
    # than silently downgrading.
    embedding_backend: EmbeddingBackend = "auto"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"

    # -------------------------------------------------------------- paths ---
    corpus_dir: pathlib.Path = pathlib.Path("corpus")
    index_dir: pathlib.Path = pathlib.Path("index")
    logs_dir: pathlib.Path = pathlib.Path("logs")

    # ----------------------------------------------------------- chunking ---
    # §b.3. Reasoned from the content: one complete policy clause plus its
    # worked example is the atomic unit of an answer here.
    max_chars: int = 1200
    overlap_chars: int = 180
    min_chars: int = 220

    # ---------------------------------------------------------- retrieval ---
    # §c.4.
    candidates_per_arm: int = 20
    top_k: int = 5
    rrf_k: int = 60

    # Miss detection. There is deliberately no MIN_FUSED_SCORE: both arms always
    # return their top-N regardless of relevance, so a fused RRF score measures
    # agreement between arms, never absolute relevance. Detection therefore
    # inspects each arm's raw signal before fusion, and a hard miss requires
    # BOTH conditions to hold.
    #
    # PROVISIONAL values, calibrated in Step 13 from logged `top_score` on the
    # §f.1 abstention cases versus the answerable ones.
    miss_dense_cosine: float = 0.55
    miss_bm25_floor: float = 1.0

    # §c.6. Multiplier on the fused score of a chunk that explicitly defers to a
    # document already present in the same result set. Reorders, never filters.
    authority_penalty: float = 0.8

    # -------------------------------------------------------------- flags ---
    enable_query_expansion: bool = True
    trace_full_prompt: bool = False

    # ------------------------------------------------------- derived paths ---
    @property
    def db_path(self) -> pathlib.Path:
        """Chunks, the FTS5 index, and traces — one file, one join (§c.2)."""
        return self.index_dir / "rag.db"

    @property
    def vectors_path(self) -> pathlib.Path:
        return self.index_dir / "vectors.npy"

    @property
    def lsa_state_path(self) -> pathlib.Path:
        """Fitted vocabulary/idf/components for the corpus-derived fallback embedder."""
        return self.index_dir / "lsa_state.npz"

    @property
    def traces_jsonl_path(self) -> pathlib.Path:
        return self.logs_dir / "traces.jsonl"

    # --------------------------------------------------- resolved LLM view ---
    @property
    def resolved_provider(self) -> str:
        """Which API this key belongs to.

        DESIGN.md §a.3 decision 8 specifies a two-way split on the key prefix.
        A third branch is required in practice: an OpenRouter key (`sk-or-`)
        speaks the OpenAI protocol but against a different host, so routing it
        to the plain OpenAI client would fail at connection time.
        """
        if self.llm_provider != "auto":
            return self.llm_provider
        key = self.llm_api_key.get_secret_value()
        if key.startswith("sk-ant-"):
            return "anthropic"
        if key.startswith("sk-or-"):
            return "openrouter"
        return "openai"

    @property
    def resolved_model(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.resolved_provider]

    @property
    def resolved_base_url(self) -> str:
        """Empty string means "use the SDK's own default host"."""
        if self.llm_base_url:
            return self.llm_base_url
        return OPENROUTER_BASE_URL if self.resolved_provider == "openrouter" else ""

    @property
    def has_api_key(self) -> bool:
        return bool(self.llm_api_key.get_secret_value())

    @property
    def stopwords(self) -> frozenset[str]:
        """Union used by the §c.4 sparse miss-condition."""
        return DOMAIN_STOPWORDS | ENGLISH_STOPWORDS

    # ---------------------------------------------------------- validation ---
    @field_validator("corpus_dir", "index_dir", "logs_dir")
    @classmethod
    def _anchor_to_project_root(cls, v: pathlib.Path) -> pathlib.Path:
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @model_validator(mode="after")
    def _check_coherent(self) -> Settings:
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        if self.min_chars >= self.max_chars:
            raise ValueError("min_chars must be smaller than max_chars")
        if self.top_k > self.candidates_per_arm:
            raise ValueError("top_k cannot exceed candidates_per_arm")
        if not 0.0 < self.authority_penalty <= 1.0:
            raise ValueError("authority_penalty must be in (0, 1]; it demotes, never promotes")
        return self

    # -------------------------------------------------------------- utils ---
    def ensure_dirs(self) -> None:
        """Create the writable directories. Never touches corpus_dir, which is read-only."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, object]:
        """Safe-to-log snapshot for `--check-llm`, `/health`, and trace headers.

        Reports only whether a key is present, never any part of its value.
        """
        return {
            "provider": self.resolved_provider,
            "model": self.resolved_model,
            "base_url": self.resolved_base_url or "<sdk default>",
            "effort": self.llm_effort,
            "api_key_present": self.has_api_key,
            "embedding_backend": self.embedding_backend,
            "corpus_dir": str(self.corpus_dir),
            "index_dir": str(self.index_dir),
            "chunking": {
                "max_chars": self.max_chars,
                "overlap_chars": self.overlap_chars,
                "min_chars": self.min_chars,
            },
            "retrieval": {
                "top_k": self.top_k,
                "candidates_per_arm": self.candidates_per_arm,
                "rrf_k": self.rrf_k,
                "miss_dense_cosine": self.miss_dense_cosine,
                "miss_bm25_floor": self.miss_bm25_floor,
                "authority_penalty": self.authority_penalty,
            },
            "query_expansion": self.enable_query_expansion,
        }


settings = Settings()
