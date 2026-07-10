"""Single source of truth for all tunable parameters, model IDs, thresholds, and paths.

Nothing elsewhere in the package should hardcode a model name, a top-N, a threshold, or a
path — import it from here. Retrieval tuning (§9) means changing values here and re-measuring
against the eval set, so every knob is centralized and attributable.

Secrets and per-environment overrides load from the environment / .env via pydantic-settings;
tuning constants have sensible defaults but can be overridden with SENTINEL_* env vars.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Paths (repo-root relative; indexes/ and *.db are gitignored) ---
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.jsonl"
INDEXES_DIR = REPO_ROOT / "indexes"
FAISS_DIR = INDEXES_DIR / "faiss"
BM25_PATH = INDEXES_DIR / "bm25.pkl"
CHUNKS_PATH = INDEXES_DIR / "chunks.jsonl"          # canonical chunk store (shared chunk IDs)
EVAL_DB_PATH = REPO_ROOT / "eval_runs.db"           # SQLite, gitignored
DASHBOARD_DATA_DIR = REPO_ROOT / "dashboard" / "data"


class Settings(BaseSettings):
    """Runtime configuration. Reads SENTINEL_* env vars and .env; secrets have no defaults."""

    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------------- Secrets
    # These use the provider SDKs' own env var names (no SENTINEL_ prefix).
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    jwt_secret: str = Field(default="dev-insecure-change-me", validation_alias="SENTINEL_JWT_SECRET")
    demo_user: str = "demo"
    demo_password: str = "change-me"

    # ---------------------------------------------------------------- Models
    # Local (₹0) models — verified stable HF model IDs.
    embedding_model: str = "BAAI/bge-small-en-v1.5"          # dense embeddings
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # cross-encoder rerank
    # LLM providers for product generation + the Ragas judge. Live free-tier quotas were probed
    # against the API 2026-07-09 (do not invent IDs): EVERY Gemini free model here caps at
    # ~20 requests/DAY (gemini-3.5-flash, -2.5-flash-lite) or 0 (gemini-2.0-flash) — far too few
    # for a 48-item eval (~300 judge calls) or the per-PR CI gate. Groq's free tier gives
    # llama-3.3-70b ~1,000 req/day, which covers both. So generation + judge run on Groq; Gemini
    # stays wired as an alternate provider (sentinel/llm.py) for anyone who enables billing.
    # Generator and judge are DIFFERENT models (separate Groq free-tier quota pools) so the judge
    # never grades its own model's output (self-preference bias; calibrated Day 10). Both are
    # non-deprecated GPT-OSS on Groq's free tier, run with reasoning_effort="low" (set in
    # sentinel/llm.py) — at default effort these reasoning models burn their whole token budget
    # "thinking" and return EMPTY answers (generation) or reasoning-polluted output that Ragas
    # can't parse (judge); "low" makes them terse and reliable.
    #   generation = openai/gpt-oss-20b  — clean cited answers, fast.
    #   judge      = openai/gpt-oss-120b — larger; produces valid Ragas structured output.
    # (The provider-agnostic factory also allows a paid openai gpt-4o-mini judge via one config
    # line — the Ragas-native default — if a stronger/faster judge is wanted.)
    generation_provider: str = "groq"          # "groq" | "google"
    generation_model: str = "openai/gpt-oss-20b"
    judge_provider: str = "groq"
    judge_model: str = "openai/gpt-oss-120b"

    # ---------------------------------------------------------------- Chunking (§8)
    # Record which strategy produced an index so eval runs are attributable (§8).
    chunk_strategy: str = "recursive"      # "recursive" | "semantic"
    chunk_tokens: int = 512
    chunk_overlap_pct: float = 0.12        # ~10–15% overlap

    # ---------------------------------------------------------------- Retrieval (§9)
    dense_top_n: int = 50                  # FAISS first-stage
    sparse_top_n: int = 50                 # BM25 first-stage
    rrf_k: int = 60                        # Reciprocal Rank Fusion constant
    rerank_top_k: int = 5                  # cross-encoder output (fused top-N -> top-K)

    # ---------------------------------------------------------------- Eval + gate (§12/§13)
    faithfulness_threshold: float = 0.80   # CI gate fails below this
    ci_eval_subset_size: int = 25          # representative subset run in CI (cost/time bound)
    # A per-item score below this flags the item for failure attribution (§12).
    low_score_threshold: float = 0.60
    # context_recall below this => classify as a retrieval failure (else generation failure).
    retrieval_fail_recall_threshold: float = 0.50

    # ---------------------------------------------------------------- Service (§11)
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    rate_limit: str = "20/minute"          # slowapi per-IP limit; a test must prove it fires
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---------------------------------------------------------------- Resilience (§5)
    # We do NOT storm-retry: each attempt is one real hit AFTER waiting out the window the server
    # tells us to (RetryInfo.retryDelay — e.g. "retry in 45s" for a 10-RPM free-tier burst), so
    # retries are genuine, quota-respectful attempts, not amplification. max cap must exceed a
    # full RPM window (~60s) or we'd wake early and 429 again.
    judge_max_retries: int = 8             # attempts around 429s (tenacity, honoring retryDelay)
    judge_backoff_seconds: float = 2.0     # exponential base when the server gives no retryDelay
    judge_backoff_max_seconds: float = 65.0  # cap per wait — long enough to clear a per-minute window
    judge_call_timeout: float = 90.0       # per-metric-call ceiling (a hung judge can't stall a run)
    # Proactive pacing between eval items. Both generation (~4K tokens/call, 12K TPM) and the
    # judge (~6 calls/item, 30K TPM) are token-heavy on the free tier; a real gap keeps a
    # sequential run under those TPM limits so backoff only has to absorb true bursts (§12).
    # 25s keeps generation (llama-3.3-70b, 12K TPM) comfortably under its per-minute window.
    eval_item_pause_seconds: float = 25.0


settings = Settings()
