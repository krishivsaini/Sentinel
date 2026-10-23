"""Small-sample statistics for the guardrail report.

With 40 attacks, "0 succeeded" is weaker evidence than it reads: the true success rate could
still be several percent. Every proportion in the report therefore carries an exact 95%
Clopper–Pearson interval — exact rather than a normal approximation, which collapses to
[0, 0] at zero successes and claims certainty the data can't give.
"""

from __future__ import annotations

from math import comb


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _bisect(f, lo: float = 0.0, hi: float = 1.0, iters: int = 80) -> float:
    """Root of a function decreasing in p on [lo, hi]."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval for a binomial proportion k/n."""
    if n <= 0:
        return (0.0, 1.0)
    tail = (1 - confidence) / 2
    # Lower bound: the p at which seeing k or more successes has probability `tail`.
    lower = 0.0 if k == 0 else _bisect(lambda p: tail - (1 - _binom_cdf(k - 1, n, p)))
    # Upper bound: the p at which seeing k or fewer successes has probability `tail`.
    upper = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p) - tail)
    return (lower, upper)


def proportion(k: int, n: int) -> dict:
    """k/n with its rate and 95% interval, the shape every rate in the report uses."""
    lo, hi = clopper_pearson(k, n)
    return {
        "k": k,
        "n": n,
        "rate": round(k / n, 4) if n else None,
        "ci95": [round(lo, 4), round(hi, 4)],
    }
