"""Write the live service's operations report from its logs (sentinel.ops).

    python scripts/ops_report.py --gcloud --project <gcp-project> --days 30
    python scripts/ops_report.py --log-file service.log

Writes data/ops_report.json and a copy for the dashboard (dashboard/data/ops.json; run
`npm run sync-data` in dashboard/ to publish it). Reads logs only: no model calls, no cost. The
report holds counts and timings, never questions or answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sentinel.config import DASHBOARD_DATA_DIR, DATA_DIR
from sentinel.ops import read_cloud_logging, read_log_file, summarise

OUT = DATA_DIR / "ops_report.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gcloud", action="store_true", help="read the Cloud Run service's logs")
    src.add_argument("--log-file", type=Path, help="read JSON log lines from a file")
    ap.add_argument("--project", help="GCP project id (with --gcloud)")
    ap.add_argument("--days", type=int, default=30, help="how far back to read (with --gcloud)")
    args = ap.parse_args()
    if args.gcloud and not args.project:
        ap.error("--gcloud needs --project")

    if args.gcloud:
        events, source = read_cloud_logging(args.project, args.days), f"cloud-logging, {args.days}d"
    else:
        events, source = read_log_file(args.log_file), f"log file {args.log_file.name}"
    report = summarise(events, source)
    text = json.dumps(report, indent=2) + "\n"
    OUT.write_text(text, encoding="utf-8")
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DASHBOARD_DATA_DIR / "ops.json").write_text(text, encoding="utf-8")

    print(f"{report['requests']} requests, {report['window']['from']} to {report['window']['to']}")
    for k, v in report["outcomes"].items():
        print(f"  {k:17} {v['n']:5}  {v['rate']:.1%}")
    print(f"  {'stage':<13} {'n':>5} {'p50':>9} {'p95':>9} {'p99':>9}")
    for st in report["latency"]:
        print(f"  {st['stage']:<13} {st['n']:>5} {st['p50_ms']:>7.0f}ms {st['p95_ms']:>7.0f}ms "
              f"{st['p99_ms']:>7.0f}ms")
    c = report["cost_usd"]
    if c["requests_with_cost"]:
        print(f"  cost ${c['total']:.4f} over {c['requests_with_cost']} requests "
              f"(${c['per_request_mean']:.5f} each)")
    f = report["feedback"]
    print(f"  feedback up {f['up']}  down {f['down']}")
    print(f"wrote {OUT.relative_to(DATA_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
