"""Check the semantic cache on a deployed Sentinel: four questions in a row.

  1. a fresh question               -> answered normally (retrieval, rerank, generation)
  2. the exact repeat               -> must come from the cache
  3. a rewording                    -> should come from the cache
  4. a near-miss (DoT -> DoH)       -> must NOT come from the cache

Costs real model calls for the two fresh answers (about $0.001 in total at Groq's prices).
Run it against an instance whose cache doesn't already hold these questions (a fresh deploy or a
cold start), or the first question will hit too.

    python scripts/cache_live_check.py --project <gcp-project>   # data/cache_live_check.json

With --project, each answer's in-service time and model cost are read back from the service's
own `query` log events in Cloud Logging (needs gcloud); without it, only client-side times.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "cache_live_check.json"
DEFAULT_API = "https://sentinel-api-oh6bzi2ava-uc.a.run.app"
QUESTIONS = [
    ("fresh", "What transport port does DNS over TLS (DoT) use by default?"),
    ("repeat", "What transport port does DNS over TLS (DoT) use by default?"),
    ("reworded", "By default, which transport port does DNS over TLS (DoT) use?"),
    ("near-miss", "What transport port does DNS over HTTPS (DoH) use by default?"),
]


def ask(api: str, headers: dict, question: str) -> tuple[dict, float]:
    t = time.perf_counter()
    event, done = None, None
    with httpx.stream("POST", f"{api}/query", json={"question": question}, headers=headers,
                      timeout=180) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event == "done":
                done = json.loads(line[5:])
    if done is None:
        raise RuntimeError(f"no 'done' event for {question!r}")
    return done, time.perf_counter() - t


def logged(project: str, answer_id: str) -> dict:
    """The service's own `query` log event for one answer (in-service time, model cost)."""
    for _ in range(6):                       # log ingestion lags by a few seconds
        out = subprocess.run(
            ["gcloud", "logging", "read",
             f'jsonPayload.event="query" AND jsonPayload.answer_id="{answer_id}"',
             "--project", project, "--freshness", "1h", "--limit", "1", "--format", "json"],
            capture_output=True, text=True, check=True).stdout
        rows = json.loads(out or "[]")
        if rows:
            return rows[0]["jsonPayload"]
        time.sleep(5)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api", default=os.environ.get("SENTINEL_API", DEFAULT_API))
    ap.add_argument("--user", default=os.environ.get("SENTINEL_USER", "demo"))
    ap.add_argument("--password", default=os.environ.get("SENTINEL_PASSWORD", "change-me"))
    ap.add_argument("--project", help="GCP project, to read in-service time and cost from logs")
    ap.add_argument("--out", type=Path, default=OUT_PATH, help="where to write the report")
    args = ap.parse_args()

    tok = httpx.post(f"{args.api}/token", data={"username": args.user,
                                                 "password": args.password}, timeout=60)
    tok.raise_for_status()
    headers = {"Authorization": f"Bearer {tok.json()['access_token']}"}

    rows = []
    for label, question in QUESTIONS:
        done, wall = ask(args.api, headers, question)
        row = {"label": label, "question": question,
               "cache_hit": done.get("cached_from") is not None,
               "cached_from": done.get("cached_from"), "similarity": done.get("cache_similarity"),
               "wall_s": round(wall, 2),
               "stages_ms": {k: round(v, 1)
                             for k, v in (done.get("latency_ms") or {}).items() if v},
               "answer_start": done.get("text", "")[:60], "answer_id": done["answer_id"]}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("label", "cache_hit", "similarity", "wall_s")}))
    if args.project:
        for row in rows:
            ev = logged(args.project, row.pop("answer_id"))
            row["request_ms"], row["cost_usd"] = ev.get("request_ms"), ev.get("cost_usd")
            if not row["cache_hit"] and ev.get("cache_stored") is False:
                row["note"] = "not stored: only clean, cited answers are cached"

    ok = (not rows[0]["cache_hit"] and rows[1]["cache_hit"] and not rows[3]["cache_hit"])
    report = {"what": f"Live semantic-cache check against {args.api}, "
                      f"{time.strftime('%Y-%m-%d', time.gmtime())}.",
              "how": "python scripts/cache_live_check.py; wall_s measured by the client, "
                     "request_ms and cost_usd from Cloud Logging when --project is given.",
              "passed": ok, "questions": rows}
    if args.project:
        report["total_cost_usd"] = round(sum(r.get("cost_usd") or 0 for r in rows), 6)
    out = args.out.resolve()
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"{'PASS' if ok else 'FAIL'}; wrote {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
