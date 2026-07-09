"""Provider-agnostic chat-model factory (§10/§12).

Product generation and the Ragas judge both need a LangChain chat model, but the free Gemini
tier caps at ~20 requests/day/model — unusable for a 48-item eval or the per-PR CI gate. So both
run on **Groq's free tier** (llama-3.3-70b, ~1,000 req/day), with Gemini kept as an alternate
provider. `ChatGroq` and `ChatGoogleGenerativeAI` are both `BaseChatModel`s, so every caller
stays provider-agnostic: only `config` picks the provider + model, and this factory builds it.

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

    provider: "groq" | "google". Cached by (provider, model, temperature, max_retries) so each
    distinct configuration loads once."""
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            api_key=settings.groq_api_key or None,
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            google_api_key=settings.google_api_key or None,
        )
    raise ValueError(f"unknown LLM provider: {provider!r} (expected 'groq' or 'google')")
