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
    # GOOGLE_API_KEY has no SENTINEL_ prefix (that's the name the Google SDK expects).
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    jwt_secret: str = Field(default="dev-insecure-change-me", validation_alias="SENTINEL_JWT_SECRET")
    demo_user: str = "demo"
    demo_password: str = "change-me"

    # ---------------------------------------------------------------- Models
    # Local (₹0) models — verified stable HF model IDs.
    embedding_model: str = "BAAI/bge-small-en-v1.5"          # dense embeddings
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # cross-encoder rerank
    # Gemini free tier — ONLY for generation + Ragas judge. Verified stable 2026-07-06.
    # Fallback for higher free-tier RPD if 3.5-flash quota is tight: gemini-2.5-flash-lite.
    generation_model: str = "gemini-3.5-flash"
    judge_model: str = "gemini-3.5-flash"

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
    judge_max_retries: int = 6             # tenacity attempts around 429s
    judge_backoff_seconds: float = 2.0     # exponential base


settings = Settings()
