"""Provider-agnostic chat-model factory (§10/§12).

Product generation and the Ragas judge both need a LangChain chat model. Every backend here
returns a `BaseChatModel`, so callers stay provider-agnostic: only `config` picks the
provider + model, and this factory builds it.

Providers:
  "groq"   — DEFAULT. Free tier shares one quota pool across roles (hence the pacing constants
             in config), but it is enormously faster per call, which is what actually bounds a
             run (see the measurement below).
  "nvidia" — build.nvidia.com, OpenAI-compatible. Free tier is metered per model (40 RPM each),
             so roles get independent quota. Kept wired as a fallback / for long offline runs.
  "google" — Gemini; free tier is ~20 requests/DAY/model, so it's an alternate, not a default.

Provider choice is measured, not assumed (probed 2026-09-02, same trivial JSON prompt, warm):

    model                                   Groq            NVIDIA
    openai/gpt-oss-120b (judge)             0.45-0.79s      12.6-40.8s
    openai/gpt-oss-20b  (generation)        0.31-0.41s      -
    nvidia/nemotron-3.5-lightning-30b       -               26-57s
    nvidia/nemotron-3-ultra-550b-a55b       -               13.4s + 2/3 InternalServerError

Groq is ~30-60x faster on the *identical* model. That reverses the case for switching: NVIDIA's
per-model 40 RPM is unreachable when a single call takes ~25s (that is ~2.4 RPM sequential), so
latency, not quota, is the binding constraint. Several large NVIDIA judges are also unusable:
nemotron-ultra-253b / nemotron-70b / nemotron-4-340b return 404 (listed but not deployed for the
account), and nemotron-3-super-120b leaks raw reasoning into content — the exact pollution the
judge must avoid. Hence: Groq stays the default; NVIDIA stays wired because the abstraction makes
it free to keep, and it is the escape hatch if Groq's terms change.

`max_retries=0` is the right default here — the callers (service SSE, eval loop) own their own
backoff, and stacking the SDK's retries on top only amplifies free-tier throttling.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from sentinel.config import settings


@lru_cache(maxsize=8)
def chat_model(
    provider: str, model: str, *, temperature: float = 0.0, max_retries: int = 0
) -> BaseChatModel:
    """Build (and cache) a chat model for the given provider + model id.

    provider: "nvidia" | "groq" | "google". Cached by (provider, model, temperature,
    max_retries) so each distinct configuration loads once."""
    if provider == "nvidia":
        from langchain_openai import ChatOpenAI

        # build.nvidia.com speaks the OpenAI API, so a ChatOpenAI pointed at its base URL is the
        # whole integration — no extra SDK. Only the gpt-oss family needs the reasoning-effort
        # clamp (see below); nemotron models are asked to run at their default effort so we can
        # test whether a larger judge produces valid Ragas structured output without the hack.
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            api_key=settings.nvidia_api_key or "not-set",
            base_url=settings.nvidia_base_url,
            **({"reasoning_effort": "low"} if "gpt-oss" in model else {}),
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        extra: dict = {}
        # GPT-OSS are reasoning models: at the default effort they can burn the whole token
        # budget "thinking" and return EMPTY content (generation) or reasoning-polluted output
        # that breaks Ragas' structured parsing (judge). "low" keeps them terse and on-task.
        if "gpt-oss" in model:
            extra["reasoning_effort"] = "low"
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            api_key=settings.groq_api_key or None,
            **extra,
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            google_api_key=settings.google_api_key or None,
        )
    raise ValueError(
        f"unknown LLM provider: {provider!r} (expected 'nvidia', 'groq', or 'google')"
    )
