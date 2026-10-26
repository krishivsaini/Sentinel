"""Single source of truth for all tunable parameters, model IDs, thresholds, and paths.

Nothing elsewhere in the package should hardcode a model name, a top-N, a threshold, or a
path — import it from here. Retrieval tuning (§9) means changing values here and re-measuring
against the eval set, so every knob is centralized and attributable.

Secrets and per-environment overrides load from the environment / .env via pydantic-settings;
tuning constants have sensible defaults but can be overridden with SENTINEL_* env vars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Paths (repo-root relative; indexes*/ and *.db are gitignored) ---
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.jsonl"
# The service's index: the real RFC corpus only.
INDEXES_DIR = REPO_ROOT / "indexes"
# Real corpus + the labelled adversarial documents, for the guardrail eval only. A sibling of
# indexes/, not a subdirectory, so rebuilding or caching the service index can never touch it
# and nothing that reads indexes/ can ever see a poisoned chunk.
ADVERSARIAL_INDEX_DIR = REPO_ROOT / "indexes_adversarial"
ADVERSARIAL_DIR = DATA_DIR / "adversarial"          # poisoned docs + attack questions (labelled)
# The independent public attack set (BIPIA), in its own sibling index so adding it can't shift
# retrieval — BM25 statistics included — for the 50 hand-written attacks.
BIPIA_INDEX_DIR = REPO_ROOT / "indexes_bipia"
BIPIA_DIR = DATA_DIR / "bipia"
GUARDRAILS_DIR = DATA_DIR / "guardrails"            # screening cache, PII labels, eval reports
EVAL_DB_PATH = REPO_ROOT / "eval_runs.db"           # SQLite, gitignored
DASHBOARD_DATA_DIR = REPO_ROOT / "dashboard" / "data"


@dataclass(frozen=True)
class IndexPaths:
    """Where one index's files live. Every index directory has the same layout."""

    root: Path

    @property
    def chunks(self) -> Path:
        return self.root / "chunks.jsonl"   # canonical chunk store (shared chunk IDs)

    @property
    def faiss(self) -> Path:
        return self.root / "faiss" / "index.faiss"

    @property
    def bm25(self) -> Path:
        return self.root / "bm25.pkl"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"


# Back-compat aliases for the service index.
CHUNKS_PATH = IndexPaths(INDEXES_DIR).chunks


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
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    # OpenAI-compatible endpoints used by the "nvidia" and "deepseek" providers.
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    deepseek_base_url: str = "https://api.deepseek.com"
    # OpenAI reasoning models (gpt-5.x / gpt-6.x) reject temperature=0, so their behaviour is set
    # with these instead. "none" = no hidden reasoning tokens: cheapest, and nothing to leak into
    # output the Ragas parser must read. "flex" = half-price, higher-latency tier; fine for the
    # judge (offline eval + CI), never used for user-facing generation.
    openai_reasoning_effort: str = "none"
    openai_service_tier: str = "flex"
    # Output cap sent on every Groq call. Groq's own default for gpt-oss is 3,072, which truncated
    # judge verdicts on long answers (see sentinel.llm). The model allows far more.
    groq_max_output_tokens: int = 16384
    jwt_secret: str = Field(default="dev-insecure-change-me", validation_alias="SENTINEL_JWT_SECRET")
    demo_user: str = "demo"
    demo_password: str = "change-me"

    # ---------------------------------------------------------------- Models
    # Local (₹0) models — verified stable HF model IDs.
    embedding_model: str = "BAAI/bge-small-en-v1.5"          # dense embeddings
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # cross-encoder rerank
    # Which reranker the pipeline uses (SENTINEL_RERANKER): "baseline" is reranker_model, on
    # which every published number was measured; "finetuned" is finetuned_reranker — a local
    # path (relative to the repo root) or a Hugging Face model id — trained by
    # scripts/train_reranker.py. The default changes only if the fine-tuned model wins on the
    # ground truth (docs/reranker-finetune.md).
    reranker: Literal["baseline", "finetuned"] = "baseline"
    finetuned_reranker: str = "runs/main/best"
    # Device for the local embedder + reranker; None = sentence-transformers' pick (CUDA, else
    # Apple MPS, else CPU). Defaults to CPU: on a Mac, MPS memory grew across a long eval and a
    # 200-query latency benchmark until both died "MPS backend out of memory" — a long-running
    # service would too. CPU held up against the last MPS benchmark that completed (rerank p50 954
    # vs 879 ms, p95 1,021 vs 1,928 ms; retrieve p95 14 vs 379 ms). CI runs on CPU as well. Override with SENTINEL_TORCH_DEVICE.
    torch_device: str | None = "cpu"
    # LLM providers for product generation + the Ragas judge. Live free-tier quotas were probed
    # against the API 2026-07-09 (do not invent IDs): EVERY Gemini free model here caps at
    # ~20 requests/DAY (gemini-3.5-flash, -2.5-flash-lite) or 0 (gemini-2.0-flash) — far too few
    # for a 48-item eval (~300 judge calls) or the per-PR CI gate. Groq's free tier gives
    # llama-3.3-70b ~1,000 req/day, which covers both. So generation + judge run on Groq; NVIDIA
    # and Gemini stay wired as alternate providers (sentinel/llm.py).
    # Generator and judge are DIFFERENT models (separate Groq free-tier quota pools) so the judge
    # never grades its own model's output (self-preference bias; calibrated Day 10). Both are
    # non-deprecated GPT-OSS on Groq's free tier, run with reasoning_effort="low" (set in
    # sentinel/llm.py) — at default effort these reasoning models burn their whole token budget
    # "thinking" and return EMPTY answers (generation) or reasoning-polluted output that Ragas
    # can't parse (judge); "low" makes them terse and reliable.
    #   generation = openai/gpt-oss-20b  — clean cited answers, fast.
    #   judge      = openai/gpt-oss-120b — larger; produces valid Ragas structured output.
    # Do NOT swap judge_provider without re-running scripts/judge_calibration.py: calibration is
    # host-specific, not just model-specific. Serving this same judge from NVIDIA drops it from
    # r=+0.90 to r=+0.54 (see data/ground_truth_audit.md and sentinel/llm.py for the measurements).
    generation_provider: str = "groq"          # "groq" | "deepseek" | "nvidia" | "google"
    generation_model: str = "openai/gpt-oss-20b"
    judge_provider: str = "groq"
    judge_model: str = "openai/gpt-oss-120b"

    # ---------------------------------------------------------------- Chunking (§8)
    # Record which strategy produced an index so eval runs are attributable (§8).
    chunk_strategy: str = "recursive"      # "recursive" | "semantic"
    chunk_tokens: int = 512
    chunk_overlap_pct: float = 0.12        # ~10–15% overlap

    # ---------------------------------------------------------------- Retrieval (§9)
    # Which index retrieval reads. The adversarial eval points this at ADVERSARIAL_INDEX_DIR.
    index_dir: Path = INDEXES_DIR
    # Document scopes a caller may retrieve when none are given (v2-R4). Every real RFC chunk is
    # "public"; the adversarial corpus is "adversarial" and only its eval asks for it.
    default_scopes: tuple[str, ...] = ("public",)
    dense_top_n: int = 50                  # FAISS first-stage
    sparse_top_n: int = 50                 # BM25 first-stage
    rrf_k: int = 60                        # Reciprocal Rank Fusion constant
    # Fused candidates the cross-encoder re-scores; its cost is linear in this and dominates query
    # latency. Kept at 50 for the best retrieval quality. Measured option
    # (data/rerank_pool_sweep.json): 20 keeps hit@5 (0.896) and nearly all recall@5 (0.812 vs
    # 0.816) at 2.6x less rerank time, but changes the top-5 for 13 of 48 questions; below 20
    # quality falls (10: recall@5 0.743).
    rerank_pool_size: int = 50
    rerank_top_k: int = 5                  # cross-encoder output (fused top-N -> top-K)

    # ---------------------------------------------------------------- Eval + gate (§12/§13)
    faithfulness_threshold: float = 0.80   # CI gate fails below this
    # Representative subset run in CI (faithfulness only — see run_eval.run): every other item of
    # the 48. At 10 items the mean moved ~0.08 between runs of the same code (0.967 vs 0.891), too
    # noisy for a 0.80 line; 24 halves the standard error. Needs Groq's paid Developer tier
    # (250K tokens/min) — the free tier's 200K tokens/day covers about two 10-item runs.
    ci_eval_subset_size: int = 24
    # A gate needs enough data to mean something: if fewer than this many items actually score
    # (throttle/skip), the gate fails as "insufficient coverage" rather than passing on thin data.
    ci_min_items: int = 18
    # Relative gate: a PR also fails if its mean faithfulness falls more than this margin below
    # main's committed per-question baseline (data/faithfulness_baseline.json). The margin is the
    # larger of a floor and k standard deviations of the baseline's run-to-run mean, so ordinary
    # generation noise doesn't fail main while moderate regressions — which the absolute 0.80 line
    # misses — do fail. Calibrated on the committed baseline (4 runs: mean 0.951, sd 0.034), k=2
    # puts the line at 0.883: below every main run on record (lowest 0.904), above both subtle
    # regressions (0.855, 0.865). k=2.5 (line 0.866) would pass the 0.865 one by noise alone.
    faithfulness_regression_margin: float = 0.05
    faithfulness_regression_sd_k: float = 2.0
    # A per-item score below this flags the item for failure attribution (§12).
    low_score_threshold: float = 0.60
    # context_recall below this => classify as a retrieval failure (else generation failure).
    retrieval_fail_recall_threshold: float = 0.50

    # ---------------------------------------------------------------- Guardrails (v2)
    # Prompt-injection screening at index build (sentinel.guardrails.injection). Verdicts are
    # cached in data/guardrails/screening_cache.jsonl (committed), so rebuilding an unchanged
    # corpus makes no API calls. A chunk is quarantined if ANY detector flags it.
    injection_screening: bool = True
    # Meta's purpose-built injection/jailbreak classifier: returns P(malicious). Fast (~200 ms),
    # but missed a politely-worded "note for AI assistants" in probing.
    promptguard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    promptguard_threshold: float = 0.5
    promptguard_rpm: int = 95             # Groq allows 100 requests/min for it, even on paid tiers
    # A reasoning classifier that applies a written policy (injection.POLICY). Caught the polite
    # injection Prompt Guard missed, and left RFC text *about* injection attacks alone.
    safeguard_model: str = "openai/gpt-oss-safeguard-20b"
    screening_workers: int = 8             # parallel classifier calls for uncached chunks
    # Screen the user's question before retrieval (sentinel.guardrails.input). A flagged question
    # is refused. Fails closed unless input_guard_fail_open: the generator shares the provider.
    input_guard: bool = True
    input_guard_detectors: tuple[str, ...] = ("policy",)
    input_guard_fail_open: bool = False
    # Fence retrieved passages as untrusted data in the generation prompt (sentinel.generate).
    prompt_hardening: bool = True
    # Redact PII in retrieved passages before generation (sentinel.guardrails.pii).
    redact_context_pii: bool = True
    # Sanitise answers before they leave the service (sentinel.guardrails.output): only links to
    # these hosts (or their subdomains) survive; images and PII are always removed. The RFC 2606
    # documentation domains are allowed because RFC answers quote their example URLs — but not
    # the bare ".example" TLD, which the adversarial corpus uses for its phishing links.
    output_guard: bool = True
    answer_link_hosts: tuple[str, ...] = (
        "rfc-editor.org", "ietf.org", "iana.org", "w3.org",
        "example.com", "example.net", "example.org",
    )
    # Check the finished answer for carried-out passage instructions (sentinel.guardrails.answer);
    # a flagged answer is withheld. Fails closed unless answer_check_fail_open.
    answer_check: bool = True
    answer_check_fail_open: bool = False
    # Second CI gate (v2-R5): the share of the 40 adversarial attacks allowed to succeed with every
    # defence on. Set from measurement — see data/guardrails/report.json.
    injection_asr_threshold: float = 0.05

    # ---------------------------------------------------------------- Service (§11)
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    rate_limit: str = "20/minute"          # slowapi per-IP limit; a test must prove it fires
    # A public deployment: turn /query away once the day's metered LLM spend reaches this
    # many USD (None = no cap), and the browser origins allowed to call the API (the dashboard).
    daily_spend_limit_usd: float | None = None
    cors_origins: tuple[str, ...] = ()
    api_host: str = "0.0.0.0"
    # Cloud Run sets PORT; locally SENTINEL_API_PORT or the default.
    api_port: int = Field(default=8000, validation_alias=AliasChoices("SENTINEL_API_PORT", "PORT"))

    # ---------------------------------------------------------------- Resilience (§5)
    # We do NOT storm-retry: each attempt is one real hit AFTER waiting out the window the server
    # tells us to (RetryInfo.retryDelay — e.g. "retry in 45s" for a 10-RPM free-tier burst), so
    # retries are genuine, quota-respectful attempts, not amplification. max cap must exceed a
    # full RPM window (~60s) or we'd wake early and 429 again.
    # A judged item fires ~6 calls (~18K tokens) that overrun the 8K-TPM free window mid-item, so
    # later calls must WAIT OUT the per-minute window; a generous retry budget lets each call grind
    # through rather than skipping the item. Only engages on throttle — normal/paid runs hit attempt 1.
    judge_max_retries: int = 15            # attempts around 429s (tenacity, honoring retryDelay)
    judge_backoff_seconds: float = 2.0     # exponential base when the server gives no retryDelay
    judge_backoff_max_seconds: float = 65.0  # cap per wait — long enough to clear a per-minute window
    judge_call_timeout: float = 90.0       # per-metric-call ceiling (a hung judge can't stall a run)
    # Whole-item wall-clock ceiling. A hung HTTP stream on the generation path has no other
    # timeout and once stalled a run indefinitely; this turns a hang into a skip (resume retries
    # it). Generous vs. an observed ~2-3 min throttled item, but far below an indefinite hang.
    eval_item_timeout: float = 600.0
    # Proactive pacing between eval items. None is needed on Groq's paid tier (250K tokens/min);
    # on the free tier (8K tokens/min) set SENTINEL_EVAL_ITEM_PAUSE_SECONDS=25 so backoff only has
    # to absorb true bursts rather than every item.
    eval_item_pause_seconds: float = 0.0
    # Stop a run once its metered LLM spend reaches this many USD (sentinel.spend). None = no cap.
    spend_limit_usd: float | None = None


settings = Settings()
