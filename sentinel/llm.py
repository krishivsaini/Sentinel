"""Provider-agnostic chat-model factory (§10/§12).

Product generation and the Ragas judge both need a LangChain chat model. Every backend here
returns a `BaseChatModel`, so callers stay provider-agnostic: only `config` picks the
provider + model, and this factory builds it.

Providers:
  "nvidia" — build.nvidia.com, OpenAI-compatible. Free tier is per-model (40 RPM each), so
             generation, judge, and (v2) a guardrails classifier get INDEPENDENT quota instead
             of sharing one pool. Hosts very large judges (nemotron-ultra-253b) that Groq's free
             tier doesn't.
  "groq"   — free tier, fast, but one shared quota pool across roles.
  "google" — Gemini; free tier is ~20 requests/DAY/model, so it's an alternate, not a default.

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
