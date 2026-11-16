"""Operations view of the live service, from the structured logs it already writes.

The service logs one event per request (`query`, `query.refused`, `query.budget_exhausted`)
and one per rating (`feedback`), as JSON that Cloud Run ships to Cloud Logging. This module reads
those events back and summarises them: traffic and outcomes, per-stage and end-to-end latency
percentiles, Groq cost per answer, and reader feedback. No metrics backend, no agent, nothing
new to run or pay for; the logs are the telemetry.

    python scripts/ops_report.py --gcloud --project <gcp-project> --days 30
    python scripts/ops_report.py --log-file service.log
"""

from __future__ import annotations

import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ANSWER_EVENTS = ("query", "query.refused")
ALL_EVENTS = (*ANSWER_EVENTS, "query.budget_exhausted", "feedback")
# Per-stage timings, in pipeline order, then the two end-to-end ones a reader feels.
STAGES = ("plan", "retrieve", "rerank", "generate", "guard", "answer_check")
END_TO_END = ("first_token", "request")


def read_cloud_logging(project: str, days: int, events=ALL_EVENTS,
                       service: str = "sentinel-api") -> list[dict]:
    """The service's structured events from Cloud Logging (free to read at this volume)."""
    names = " OR ".join(f'"{e}"' for e in events)
    flt = (f'resource.type="cloud_run_revision" AND resource.labels.service_name="{service}" '
           f"AND jsonPayload.event=({names})")
    proc = subprocess.run(
        ["gcloud", "logging", "read", flt, "--project", project, f"--freshness={days}d",
         "--format=json", "--limit=100000"],
        capture_output=True, text=True, check=True)
    out = []
    for entry in json.loads(proc.stdout or "[]"):
        if "jsonPayload" in entry:
            ev = dict(entry["jsonPayload"])
            ev.setdefault("timestamp", entry.get("timestamp"))
            out.append(ev)
    return out


def read_log_file(path: Path) -> list[dict]:
    """JSON log lines from a file (e.g. local uvicorn output); other lines are skipped."""
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def latency_summary(name: str, samples_ms: list[float]) -> dict:
    """Same shape and interpolation as scripts/bench_latency.py, so both read alike."""
    s = sorted(samples_ms)

    def pct(p: float) -> float:
        if not s:
            return 0.0
        k = (len(s) - 1) * p / 100.0
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return {
        "stage": name,
        "n": len(s),
        "mean_ms": round(statistics.fmean(s), 1) if s else 0.0,
        "p50_ms": round(pct(50), 1),
        "p95_ms": round(pct(95), 1),
        "p99_ms": round(pct(99), 1),
        "min_ms": round(s[0], 1) if s else 0.0,
        "max_ms": round(s[-1], 1) if s else 0.0,
    }


def _outcome(ev: dict) -> str:
    return {"query.refused": "refused", "query.budget_exhausted": "budget_exhausted"}.get(
        ev["event"], ev.get("outcome") or "answered")


def summarise(events: list[dict], source: str) -> dict:
    """Aggregate the events into the operations report. Holds counts and timings only: no
    questions, answers or comments, so it can be committed and shown on the dashboard."""
    requests = [e for e in events if e.get("event") in (*ANSWER_EVENTS, "query.budget_exhausted")]
    queries = [e for e in requests if e["event"] == "query"]
    outcomes = Counter(_outcome(e) for e in requests)
    n = len(requests)

    samples: dict[str, list[float]] = defaultdict(list)
    for e in queries:
        for stage, ms in (e.get("latency_ms") or {}).items():
            if stage in STAGES and ms is not None:
                samples[stage].append(float(ms))
        for key in END_TO_END:
            if e.get(f"{key}_ms") is not None:
                samples[key].append(float(e[f"{key}_ms"]))

    costed = [float(e["cost_usd"]) for e in requests if e.get("cost_usd") is not None]
    ratings = {}
    for e in sorted((e for e in events if e.get("event") == "feedback"),
                    key=lambda e: e.get("timestamp") or ""):
        ratings[e.get("answer_id")] = e.get("rating")   # a reader's latest rating wins
    outcome_of = {e.get("answer_id"): _outcome(e) for e in requests if e.get("answer_id")}
    by_outcome: dict[str, Counter] = defaultdict(Counter)
    for aid, rating in ratings.items():
        by_outcome[outcome_of.get(aid, "unknown")][rating] += 1

    daily: dict[str, dict] = defaultdict(lambda: {"requests": 0, "cost_usd": 0.0, "request_ms": []})
    for e in requests:
        day = (e.get("timestamp") or "")[:10] or "unknown"
        daily[day]["requests"] += 1
        daily[day]["cost_usd"] += float(e.get("cost_usd") or 0.0)
        if e["event"] == "query" and e.get("request_ms") is not None:
            daily[day]["request_ms"].append(float(e["request_ms"]))

    stamps = sorted(e["timestamp"] for e in events if e.get("timestamp"))
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "window": {"from": stamps[0] if stamps else None, "to": stamps[-1] if stamps else None},
        "requests": n,
        "outcomes": {k: {"n": v, "rate": round(v / n, 4) if n else None}
                     for k, v in sorted(outcomes.items())},
        "latency": [latency_summary(s, samples[s]) for s in (*STAGES, *END_TO_END) if samples[s]],
        "cost_usd": {
            "total": round(sum(costed), 6),
            "per_request_mean": round(statistics.fmean(costed), 6) if costed else None,
            "requests_with_cost": len(costed),
        },
        "feedback": {
            "up": sum(r == "up" for r in ratings.values()),
            "down": sum(r == "down" for r in ratings.values()),
            "by_outcome": {k: dict(v) for k, v in sorted(by_outcome.items())},
        },
        "daily": [
            {"date": day, "requests": d["requests"], "cost_usd": round(d["cost_usd"], 6),
             "request_p95_ms": latency_summary("request", d["request_ms"])["p95_ms"]
             if d["request_ms"] else None}
            for day, d in sorted(daily.items())
        ],
    }
