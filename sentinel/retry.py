"""Retry policy shared by every rate-limited API call (the eval judge, generation, guardrail
classifiers). Built for free-tier quotas and kept because it's simply correct on paid ones:

  * retry only what's transient — rate limits, timeouts, 5xx, dropped connections;
  * wait exactly as long as the server asks (its retry-after hint), else back off exponentially;
  * never retry a per-day quota: it can't clear inside any backoff window, so retrying only burns
    time (five such items once overran a CI job's limit);
  * log every backoff, so a throttled run can be diagnosed from its log.
"""

from __future__ import annotations

import re

from sentinel.config import settings
from sentinel.logging_config import log
from sentinel.spend import BudgetExceeded

# Error-message fragments that mark a *transient* failure worth backing off and retrying.
# Anything else re-raises immediately — we don't want to spin on a real bug.
TRANSIENT_SIGNATURES = (
    "429",
    "rate limit",
    "resourceexhausted",
    "resource exhausted",
    "quota",
    "deadline",
    "timeout",
    "timed out",
    "500",
    "502",
    "503",
    "504",
    "bad gateway",
    "gateway timeout",
    "unavailable",
    "internal error",
    "connection error",   # Groq drops connections under sustained load — transient, retry it
    "connection reset",
    "connection aborted",
)

# A *daily* quota (Groq: "... tokens per day (TPD): Limit 200000 ...").
DAILY_QUOTA_SIGNATURES = ("per day", "(tpd)", "(rpd)")

# 429s carry the exact wait the server wants — Gemini as prose ("Please retry in 45.2s") and
# RetryInfo ('retryDelay': '45s'); Groq as "Please try again in 2.4s", or "in 7m12.5s" once the
# wait passes a minute. Honoring it beats a blind exponential: we wake right after the window
# clears instead of guessing and 429-ing again.
_RETRY_PROSE_RE = re.compile(
    r"(?:retry|try again) in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", re.IGNORECASE
)
_RETRY_INFO_RE = re.compile(r"retryDelay'?:?\s*'?(\d+)s")


def _text(exc: BaseException) -> str:
    return f"{type(exc).__name__} {exc}".lower()


def is_daily_quota(exc: BaseException) -> bool:
    return any(t in _text(exc) for t in DAILY_QUOTA_SIGNATURES)


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, BudgetExceeded):   # its message can contain "500" ("$0.2500")
        return False
    return any(t in _text(exc) for t in TRANSIENT_SIGNATURES) and not is_daily_quota(exc)


def server_retry_delay(exc: BaseException | None) -> float | None:
    """The wait the server asked for, in seconds, if its error message says."""
    if exc is None:
        return None
    s = str(exc)
    if m := _RETRY_PROSE_RE.search(s):
        h, mins, secs = m.groups()
        return int(h or 0) * 3600 + int(mins or 0) * 60 + float(secs)
    if m := _RETRY_INFO_RE.search(s):
        return float(m.group(1))
    return None


def backoff_wait(retry_state) -> float:
    """tenacity wait: the server-requested delay (+1s cushion) when present, else exponential.
    Capped at judge_backoff_max_seconds so a mis-parsed hint can't hang a run."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    server = server_retry_delay(exc)
    if server is not None:
        wait = server + 1.0
    else:
        wait = settings.judge_backoff_seconds * (2 ** max(retry_state.attempt_number - 1, 0))
    return min(wait, settings.judge_backoff_max_seconds)


def log_retry(retry_state) -> None:
    """tenacity before_sleep: record each backoff with the server's message."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    log.warning(
        "retry",
        attempt=retry_state.attempt_number,
        wait_s=round(retry_state.next_action.sleep, 1) if retry_state.next_action else None,
        error=str(exc)[:240],
    )
