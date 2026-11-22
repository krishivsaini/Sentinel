"""Spend meter — every paid LLM call is priced, and a run can be capped in dollars.

The Groq key funds both local measurement and the CI gates, under a fixed monthly budget, so a
run that overshoots its estimate costs the next PR its gate. Every Groq call records the token
counts the API reports (generation and the Ragas judge through a LangChain callback, the
guardrail classifiers in sentinel.guardrails._groq), priced from PRICES. With
config.spend_limit_usd set, the next call after the limit raises BudgetExceeded instead of
being sent: a run stops within the cost of the calls already in flight.

    SENTINEL_SPEND_LIMIT_USD=0.25 python -m ...     # or a runner's --max-spend flag

The deployed API also has a per-day budget (config.daily_spend_limit_usd), checked when a request
starts rather than per call, so a request either runs to completion or is turned away whole.
Its tally is kept in Redis when the service has one (RedisDayTotal), so a cold start or a
redeploy doesn't reset it; without Redis it lives in the process.
Each request also gets its own tally (start_request_cost), so the query log can say what that
one answer cost.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from contextvars import ContextVar
from datetime import UTC, date, datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from sentinel.config import settings

# USD per million tokens (input, output), from console.groq.com/docs/models, 2026-10-24.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-safeguard-20b": (0.075, 0.30),
    "meta-llama/llama-prompt-guard-2-86m": (0.04, 0.04),
}
# A model missing from the table is priced at the dearest rate rather than as free.
_FALLBACK = max(PRICES.values(), key=lambda p: p[0] + p[1])


# The running cost of the request being served, if any. A one-element list rather than a float so
# the threads a request runs in (the SSE stream is iterated in a threadpool, each step in a copy
# of the request's context) all add to the same tally.
_request_cost: ContextVar[list[float] | None] = ContextVar("request_cost", default=None)


def start_request_cost() -> list[float]:
    """Start a cost tally for the current request; every call recorded under it adds to it."""
    tally = [0.0]
    _request_cost.set(tally)
    return tally


class BudgetExceeded(RuntimeError):
    """The run reached config.spend_limit_usd; no further paid calls are sent."""


class RedisDayTotal:
    """Each UTC day's spend in Redis, shared by every process and kept across restarts. Short
    timeouts and no exceptions: if Redis is unreachable the meter falls back to its own count."""

    def __init__(self, client: Any, prefix: str = "sentinel:spend:") -> None:
        import redis as redis_lib
        self._r = client
        self._prefix = prefix
        self._errors: tuple[type[BaseException], ...] = (redis_lib.RedisError, OSError)

    @classmethod
    def from_url(cls, url: str) -> RedisDayTotal:
        import redis as redis_lib
        return cls(redis_lib.Redis.from_url(url, socket_connect_timeout=0.5, socket_timeout=0.5))

    def add(self, day: date, usd: float) -> None:
        key = f"{self._prefix}{day.isoformat()}"
        try:
            pipe = self._r.pipeline()
            pipe.incrbyfloat(key, usd)
            pipe.expire(key, 3 * 86400)
            pipe.execute()
        except self._errors:
            pass

    def get(self, day: date) -> float | None:
        try:
            value = self._r.get(f"{self._prefix}{day.isoformat()}")
        except self._errors:
            return None
        return float(value) if value is not None else 0.0


class _Meter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: RedisDayTotal | None = None
        self.reset()

    def use_store(self, store: RedisDayTotal | None) -> None:
        """Keep the daily tally in `store` as well (see RedisDayTotal)."""
        self._store = store

    def reset(self) -> None:
        self.usd = 0.0
        self.calls: dict[str, int] = defaultdict(int)
        self.tokens: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self._day: date | None = None
        self._usd_day = 0.0

    def spent_today(self) -> float:
        """USD recorded since 00:00 UTC: by every process when a store is attached (or by this
        one, if the store can't be read or is behind), else by this process."""
        today = datetime.now(UTC).date()
        with self._lock:
            local = self._usd_day if self._day == today else 0.0
        shared = self._store.get(today) if self._store is not None else None
        return max(local, shared) if shared is not None else local

    def check(self) -> None:
        limit = settings.spend_limit_usd
        if limit is not None and self.usd >= limit:
            raise BudgetExceeded(f"spend limit reached: ${self.usd:.4f} of ${limit:.2f} spent")

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        p_in, p_out = PRICES.get(model, _FALLBACK)
        cost = (input_tokens * p_in + output_tokens * p_out) / 1e6
        today = datetime.now(UTC).date()
        with self._lock:
            if self._day != today:
                self._day, self._usd_day = today, 0.0
            self._usd_day += cost
            self.usd += cost
            if (tally := _request_cost.get()) is not None:
                tally[0] += cost
            self.calls[model] += 1
            self.tokens[model][0] += input_tokens
            self.tokens[model][1] += output_tokens
        if self._store is not None and cost:
            self._store.add(today, cost)

    def summary(self) -> dict:
        with self._lock:
            return {
                "usd": round(self.usd, 4),
                "by_model": {
                    m: {"calls": self.calls[m], "input_tokens": t[0], "output_tokens": t[1]}
                    for m, t in sorted(self.tokens.items())
                },
            }


meter = _Meter()


class SpendCallback(BaseCallbackHandler):
    """Meters a LangChain chat model: refuses to start a call past the limit, records usage at
    the end. Streaming included — Groq reports usage in the stream's final chunk."""

    raise_error = True   # let BudgetExceeded stop the call instead of being logged and dropped

    def __init__(self, model: str) -> None:
        self.model = model

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        meter.check()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None)
                if usage:
                    meter.record(self.model, usage.get("input_tokens", 0),
                                 usage.get("output_tokens", 0))
                    return
        usage = (response.llm_output or {}).get("token_usage") or {}
        meter.record(self.model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
