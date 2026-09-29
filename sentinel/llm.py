"""Provider-agnostic chat-model factory (§10/§12).

Product generation and the Ragas judge both need a LangChain chat model. Every backend here
returns a `BaseChatModel`, so callers stay provider-agnostic: only `config` picks the
provider + model, and this factory builds it.

Providers:
  "groq"   — DEFAULT. Free tier shares one quota pool across roles (hence the pacing constants
             in config), but it is enormously faster per call, which is what actually bounds a
             run (see the measurement below).
  "openai" — paid. Intended for the Ragas judge (gpt-6-luna, flex tier). Its reasoning models
             reject temperature=0, so they're configured via reasoning_effort instead — see below.
  "deepseek" — paid, direct DeepSeek API (OpenAI-compatible). Intended for the Ragas judge:
             cheap (off-peak rates are half of peak), no free-tier throttling, and a different
             lab from the gpt-oss generator, so the judge never grades its own family's output.
             Thinking mode is forced off — see the branch below.
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


_OPENAI_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def accepts_temperature(provider: str, model: str) -> bool:
    """Whether per-call `temperature` may be sent to this model.

    Matters for Ragas: its LLM wrapper sets temperature=0.01 on EVERY judge call unless told not
    to (`bypass_temperature=True`). OpenAI reasoning models reject any non-default temperature once
    reasoning is enabled — a hard 400 (verified 2026-09-28 on gpt-6-luna) — but accept it at
    reasoning_effort="none". Every Ragas caller consults this so eval, CI and calibration agree."""
    if provider == "openai" and model.startswith(_OPENAI_REASONING_PREFIXES):
        return settings.openai_reasoning_effort == "none"
    return True


@lru_cache(maxsize=8)
def chat_model(
    provider: str, model: str, *, temperature: float = 0.0, max_retries: int = 0
) -> BaseChatModel:
    """Build (and cache) a chat model for the given provider + model id.

    provider: "openai" | "deepseek" | "nvidia" | "groq" | "google". Cached by (provider, model,
    temperature, max_retries) so each distinct configuration loads once."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict = {}
        if model.startswith(_OPENAI_REASONING_PREFIXES):
            # Reasoning models: `temperature` other than the default is a hard 400 error
            # (verified 2026-09-28 on gpt-6-luna), so it is deliberately NOT sent. Output is
            # steered with reasoning_effort instead; run-to-run repeatability is measured
            # rather than assumed (scripts/judge_calibration.py --repeat).
            kwargs["reasoning_effort"] = settings.openai_reasoning_effort
        else:
            kwargs["temperature"] = temperature
        return ChatOpenAI(
            model=model,
            max_retries=max_retries,
            api_key=settings.openai_api_key or "not-set",
            service_tier=settings.openai_service_tier,
            **kwargs,
        )
    if provider == "deepseek":
        from langchain_openai import ChatOpenAI

        # DeepSeek models run in thinking mode by default. That's wrong for both of our roles:
        # the chain-of-thought is billed as output, it's the same failure class that broke Ragas
        # parsing with gpt-oss, and in thinking mode the API ignores `temperature` — the judge
        # needs temperature=0 for repeatable scores. So thinking is always disabled here.
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            api_key=settings.deepseek_api_key or "not-set",
            base_url=settings.deepseek_base_url,
            extra_body={"thinking": {"type": "disabled"}},
        )
    if provider == "nvidia":
        from langchain_openai import ChatOpenAI

        # build.nvidia.com speaks the OpenAI API, so a ChatOpenAI pointed at its base URL is the
        # whole integration — no extra SDK. Only the gpt-oss family needs the reasoning-effort
        # clamp (same reason as the groq branch below); nemotron models run at default effort,
        # which probe_judge.py confirmed is safe for the two that work.
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
        f"unknown LLM provider: {provider!r} "
        "(expected 'openai', 'deepseek', 'nvidia', 'groq', or 'google')"
    )
