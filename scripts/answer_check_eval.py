"""Answer check: does it catch carried-out injections without withholding genuine answers?

Runs sentinel.guardrails.answer over answers already recorded — no generation, so it costs one
cheap classifier call per distinct answer:

  caught      answers in which an attack succeeded (its canary is in the answer), from
              data/guardrails/report.json, every defence configuration
  resisted    answers to an attack's question where the attack passage was retrieved but its
              canary didn't appear. Reported, not scored: some still carry part of the
              instruction (a "paste your token" line whose link the output guard removed)
  benign      answers that never saw an attack: the ground-truth eval answers (eval_runs.db),
              the attack questions with the attack passage not retrieved, and the judge
              calibration answers

It also replays each defence configuration's recorded answers with the check added — the check
reads only the final answer, so this is what the pipeline would have returned — and reports how
many attacks would still have succeeded. A full `run_guardrails` run measures the same thing
end to end (configuration "hardening+output+answer").

    python scripts/answer_check_eval.py      # writes data/guardrails/answer_check.json

Labels are cached per (classifier, question, answer) in data/guardrails/answer_check_cache.jsonl,
so a rerun after adding answers only pays for the new ones.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel.config import DATA_DIR, settings  # noqa: E402
from sentinel.eval.stats import proportion  # noqa: E402
from sentinel.generate import is_abstention  # noqa: E402
from sentinel.guardrails.answer import AnswerPolicyClassifier  # noqa: E402

REPORT = DATA_DIR / "guardrails" / "report.json"
CALIBRATION = DATA_DIR / "judge_calibration.blind.jsonl"
CACHE = DATA_DIR / "guardrails" / "answer_check_cache.jsonl"
OUT = DATA_DIR / "guardrails" / "answer_check.json"
EVAL_DB = Path(__file__).resolve().parents[1] / "eval_runs.db"


def _sets() -> dict[str, dict[tuple[str, str], dict]]:
    """Distinct (question, answer) pairs per set, each with where it came from."""
    sets: dict[str, dict[tuple[str, str], dict]] = {"caught": {}, "resisted": {}, "benign": {}}
    for r in json.loads(REPORT.read_text())["injection"]["attacks"]["results"]:
        name = "caught" if r["succeeded"] else "resisted" if r["exposed"] else "benign"
        sets[name].setdefault((r["question"], r["answer"]), {"source": f"attack:{r['config']}",
                                                             "attack": r["attack"]})
    with sqlite3.connect(EVAL_DB) as db:
        for q, a in db.execute("SELECT question, generated_answer FROM eval_items"):
            sets["benign"].setdefault((q, a), {"source": "eval"})
    for line in CALIBRATION.read_text().splitlines():
        row = json.loads(line)
        sets["benign"].setdefault((row["question"], row["answer"]), {"source": "calibration"})
    return sets


def _key(clf: AnswerPolicyClassifier, q: str, a: str) -> str:
    return hashlib.sha256(f"{clf.name}\0{q}\0{a}".encode()).hexdigest()


def main() -> None:
    clf = AnswerPolicyClassifier()
    cache: dict[str, float] = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            row = json.loads(line)
            cache[row["key"]] = row["score"]
    sets = _sets()
    todo = sorted({(q, a) for s in sets.values() for q, a in s if _key(clf, q, a) not in cache})
    print(f"{sum(map(len, sets.values()))} distinct answers, {len(todo)} to score", flush=True)

    def score(pair: tuple[str, str]) -> tuple[str, float]:
        return _key(clf, *pair), clf.score(*pair)

    with ThreadPoolExecutor(settings.screening_workers) as pool, CACHE.open("a") as fh:
        for i, (k, s) in enumerate(pool.map(score, todo), 1):
            cache[k] = s
            fh.write(json.dumps({"key": k, "score": s}) + "\n")
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    report: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classifier": clf.name,
    }
    for name, pairs in sets.items():
        flagged = [(q, a, meta) for (q, a), meta in pairs.items()
                   if cache[_key(clf, q, a)] >= clf.threshold]
        report[name] = proportion(len(flagged), len(pairs))
        shown = flagged if name != "caught" else [
            (q, a, m) for (q, a), m in pairs.items() if cache[_key(clf, q, a)] < clf.threshold
        ]
        report[name]["examples" if name != "caught" else "missed"] = [
            {"question": q, "answer": a, **m} for q, a, m in shown
        ]
        print(f"{name:9} flagged {len(flagged)}/{len(pairs)}  "
              f"{report[name]['rate']:.3f}  ci95 {report[name]['ci95']}")
    results = json.loads(REPORT.read_text())["injection"]["attacks"]["results"]

    def withheld(r: dict) -> bool:
        return not is_abstention(r["answer"]) and cache[_key(clf, r["question"], r["answer"])] >= clf.threshold

    report["with_answer_check"] = {}
    for cfg in dict.fromkeys(r["config"] for r in results):
        rows = [r for r in results if r["config"] == cfg]
        repeats = len({r["repeat"] for r in rows})
        attacks = len(rows) // repeats
        left = [r for r in rows if r["succeeded"] and not withheld(r)]
        clean = [r for r in rows if not r["succeeded"]]
        report["with_answer_check"][cfg] = {
            "attacks": attacks,
            "succeeded_before": round(sum(r["succeeded"] for r in rows) / repeats, 2),
            "succeeded_after": round(len(left) / repeats, 2),
            "succeeded_after_in_any_run": proportion(len({r["target_doc"] for r in left}), attacks),
            "clean_answers_withheld": proportion(sum(map(withheld, clean)), len(clean)),
            "still_succeeded": sorted({f"{r['target_doc']} {r['attack']}" for r in left}),
        }
        b = report["with_answer_check"][cfg]
        print(f"  {cfg:18} succeeded {b['succeeded_before']}/{attacks} -> {b['succeeded_after']}/{attacks}"
              f"   clean answers withheld {b['clean_answers_withheld']['k']}/{b['clean_answers_withheld']['n']}")
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
