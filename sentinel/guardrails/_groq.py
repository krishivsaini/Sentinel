"""Minimal Groq REST client for the guardrail classifiers.

Not the LangChain factory in sentinel.llm: that sends max_tokens=16384 on every call, and Llama
Prompt Guard 2 is a 512-token classifier that rejects any request asking for more than it has.
Retries use the shared policy (sentinel.retry): honour the server's own wait hint on a
429, back off on transient errors, and never retry a per-day quota, which can't clear in time.
"""

from __future__ import annotations

import threading
import time
from functools import lru_cache

import httpx
import tenacity

from sentinel.config import settings
from sentinel.retry import backoff_wait, is_transient, log_retry

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GuardrailAPIError(RuntimeError):
    """A classifier call that failed for a non-transient reason (bad request, auth, quota)."""


@lru_cache(maxsize=1)
def _client() -> httpx.Client:
    if not settings.groq_api_key:
        raise GuardrailAPIError(
            "GROQ_API_KEY is not set: guardrail classifiers need it (or a complete screening "
            "cache — see sentinel.guardrails.injection)."
        )
    return httpx.Client(
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


class _RateLimiter:
    """Spaces calls evenly under a requests-per-minute cap, across threads. Groq's 429s are
    honoured by the retry policy anyway; this keeps a parallel screening run from triggering
    hundreds of them in the first place (Prompt Guard allows 100 RPM even on the paid tier)."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / rpm
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        time.sleep(max(0.0, slot - now))


_limiters: dict[str, _RateLimiter] = {}
_limiters_lock = threading.Lock()


def _limiter(model: str, rpm: int) -> _RateLimiter:
    with _limiters_lock:
        return _limiters.setdefault(model, _RateLimiter(rpm))


# Groq answers some gpt-oss calls with a 400 when the model's own output failed to parse
# ("Parsing failed ... failed_generation"). The model is stochastic, so a fresh call usually
# succeeds; it is retried like a rate limit rather than failing a whole screening run.
_SERVER_PARSE_FAILURE = ("parsing failed", "failed_generation")


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if not isinstance(exc, GuardrailAPIError):
        return False
    msg = str(exc).lower()
    return is_transient(exc) or any(sig in msg for sig in _SERVER_PARSE_FAILURE)


@tenacity.retry(
    retry=tenacity.retry_if_exception(_retryable),
    wait=backoff_wait,
    stop=tenacity.stop_after_attempt(settings.judge_max_retries),
    before_sleep=log_retry,
    reraise=True,
)
def chat(model: str, messages: list[dict], *, rpm: int | None = None, **params) -> str:
    """One chat completion; returns the message content. `rpm` caps this model's request rate
    across threads. Raises GuardrailAPIError on any API error (its message carries the server's
    text, which the retry policy reads)."""
    if rpm:
        _limiter(model, rpm).acquire()
    resp = _client().post(GROQ_CHAT_URL, json={"model": model, "messages": messages, **params})
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if resp.status_code != 200 or "choices" not in body:
        detail = body.get("error", {}).get("message") or resp.text[:300]
        raise GuardrailAPIError(f"{resp.status_code} {model}: {detail}")
    return body["choices"][0]["message"].get("content") or ""
