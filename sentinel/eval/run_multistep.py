"""Multi-step questions: does query planning find every part of a multi-part question?

data/multistep_questions.jsonl holds 16 questions that each ask about two things, usually from
two different RFCs. Each part has a verbatim evidence quote and its gold chunks: exactly the
retrievable chunks that contain the quote (chunks overlap, so it can be more than one). The
loader checks both against the index, so a rebuilt index that moved the text fails loudly
instead of scoring against the wrong chunks.

Retrieval (the default; paid only for planner calls, about $0.0001 each):
  * multi-part questions: the share of parts with at least one gold chunk in the passages the
    generator would see, and how many questions have every part covered. Three configurations:
    a single search's top 5 (what ships), a single search's top 6 (as many passages as a split
    question gets, so a gain isn't just one more passage), and the planned search.
  * over-splitting: the planner on the 48 single-topic ground-truth questions. A question it
    leaves whole is retrieved exactly as before; for any it splits, hit@K and recall@K are
    compared with the single search, so a regression on ordinary questions shows.
  * latency: the planner call, and retrieval + rerank, planned vs single.

Answers (--judge, paid): generates an answer from the single top-5 and from the planned passages
for each multi-part question, through the same context redaction and output guard as the main
eval, and scores Ragas faithfulness and context recall with the calibrated judge. Context recall
asks whether the passages support every claim in the reference answer, which is exactly what a
missed part loses.

Planner outputs are saved in the report; --reuse-plans reads them back, so the retrieval numbers
can be recomputed without calling the planner again.

    python -m sentinel.eval.run_multistep --max-spend 0.05            # retrieval only
    python -m sentinel.eval.run_multistep --judge --max-spend 0.25    # + judged answers
    python -m sentinel.eval.run_multistep --reuse-plans               # recompute, no API calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from functools import partial

from sentinel.config import DATA_DIR, settings
from sentinel.plan import Plan, plan
from sentinel.retrieve import _load_index, hybrid_retrieve_timed, multi_retrieve_timed, warmup
from sentinel.spend import BudgetExceeded, meter

QUESTIONS_PATH = DATA_DIR / "multistep_questions.jsonl"
REPORT_PATH = DATA_DIR / "multistep_eval.json"


def _norm(s: str) -> str:
    return " ".join(s.split())


def load_items() -> list[dict]:
    """The multi-part questions, with every part's evidence checked against the index."""
    idx = _load_index()
    items = [json.loads(line) for line in QUESTIONS_PATH.open(encoding="utf-8")]
    for n, it in enumerate(items, 1):
        if len(it["parts"]) < 2:
            raise ValueError(f"item {n} has fewer than two parts")
        for part in it["parts"]:
            missing = [c for c in part["relevant"] if c not in idx.id2chunk]
            if missing:
                raise ValueError(f"item {n} ({part['about']}): not in the index: {missing}")
            evidence = _norm(part["evidence"])
            if not any(evidence in _norm(idx.id2chunk[c].text) for c in part["relevant"]):
                raise ValueError(f"item {n} ({part['about']}): evidence not in its gold chunks")
            # Chunks overlap, so the same text can sit in two of them: every retrievable chunk
            # holding the evidence must be gold, or finding the other copy would count as a miss.
            holders = {c.chunk_id for c in idx.chunks
                       if not c.quarantined and evidence in _norm(c.text)}
            if unlisted := sorted(holders - set(part["relevant"])):
                raise ValueError(f"item {n} ({part['about']}): evidence also in {unlisted}; "
                                 "add them to its gold chunks or make the evidence specific")
    return items


def coverage(chunk_ids: list[str], parts: list[dict]) -> list[bool]:
    got = set(chunk_ids)
    return [bool(got & set(p["relevant"])) for p in parts]


def _plans(questions: list[str], reuse: dict[str, dict] | None) -> dict[str, Plan]:
    """Planner output per question: saved (reuse is a dict) or fresh from the planner (None)."""
    if reuse is not None and (missing := [q for q in questions if q not in reuse]):
        raise SystemExit(f"--reuse-plans: {len(missing)} question(s) have no saved plan; "
                         "run once with --max-spend")
    out = {}
    for n, q in enumerate(questions, 1):
        if reuse is not None:
            r = reuse[q]
            out[q] = Plan(q, r["queries"], error=r.get("error"), latency_ms=r.get("latency_ms", 0))
        else:
            out[q] = plan(q)
        print(f"  planned {n}/{len(questions)}", end="\r", flush=True)
    print()
    return out


def _rank_metrics(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    first = next((i for i, c in enumerate(retrieved, 1) if c in relevant), None)
    return {"hit": 1.0 if first else 0.0, "recall": len(relevant & set(retrieved)) / len(relevant)}


def retrieval_stage(items: list[dict], reuse: dict[str, dict] | None) -> dict:
    from sentinel.eval.run_eval import load_ground_truth

    gt = load_ground_truth()
    plans = _plans([it["question"] for it in items] + [g.question for g in gt], reuse)

    rows, lat = [], {"plan": [], "single": [], "planned": []}
    for it in items:
        q, parts, p = it["question"], it["parts"], plans[it["question"]]
        single6, r_ms, k_ms = hybrid_retrieve_timed(q, 6)
        single = single6[:settings.rerank_top_k]     # the top 5 is a prefix of the top 6
        lat["single"].append(r_ms + k_ms)
        planned, r_ms, k_ms = multi_retrieve_timed(p.queries)
        lat["planned"].append(r_ms + k_ms)
        lat["plan"].append(p.latency_ms)
        cov = {name: coverage([c.chunk_id for c in chunks], parts)
               for name, chunks in (("single_top5", single), ("single_top6", single6),
                                    ("planned", planned))}
        rows.append({"question": q, "queries": p.queries, "plan_error": p.error,
                     "coverage": cov, "planned_ids": [c.chunk_id for c in planned],
                     "single_ids": [c.chunk_id for c in single]})

    idx = _load_index()
    text2id = {c.text: cid for cid, c in idx.id2chunk.items()}
    splits = []
    for g in gt:
        p = plans[g.question]
        if not p.multi:
            continue
        relevant = {text2id[t] for t in g.relevant_contexts}
        single, _, _ = hybrid_retrieve_timed(g.question)
        planned, _, _ = multi_retrieve_timed(p.queries)
        splits.append({"question": g.question, "queries": p.queries,
                       "single": _rank_metrics([c.chunk_id for c in single], relevant),
                       "planned": _rank_metrics([c.chunk_id for c in planned], relevant)})

    def summary(name: str) -> dict:
        parts = [c for r in rows for c in r["coverage"][name]]
        return {"parts_covered": f"{sum(parts)}/{len(parts)}",
                "parts_covered_rate": round(sum(parts) / len(parts), 3),
                "all_parts_covered": f"{sum(all(r['coverage'][name]) for r in rows)}/{len(rows)}"}

    def p50(xs: list[float]) -> float:
        return round(statistics.median(xs), 1) if xs else 0.0

    return {
        "multi_part": {
            "questions": len(rows),
            "split_by_planner": sum(len(r["queries"]) > 1 for r in rows),
            "plan_errors": sum(r["plan_error"] is not None for r in rows),
            **{name: summary(name) for name in ("single_top5", "single_top6", "planned")},
            "latency_ms_p50": {"plan": p50(lat["plan"]),
                               "single_retrieve_rerank": p50(lat["single"]),
                               "planned_retrieve_rerank": p50(lat["planned"])},
        },
        "single_topic": {
            "questions": len(gt),
            "split_by_planner": len(splits),
            "plan_errors": sum(plans[g.question].error is not None for g in gt),
            "split_questions": splits,
            "hit_single": sum(s["single"]["hit"] for s in splits),
            "hit_planned": sum(s["planned"]["hit"] for s in splits),
        },
        "per_question": rows,
        "plans": {q: {"queries": p.queries, "error": p.error, "latency_ms": round(p.latency_ms, 1)}
                  for q, p in plans.items()},
    }


async def _judge(items: list[dict], report: dict, only: set[int] | None = None,
                 previous: dict | None = None) -> dict:
    from ragas import SingleTurnSample

    from sentinel.eval.run_eval import (
        _build_metrics,
        _coerce,
        _coerce_faithfulness,
        _score_async,
        _with_backoff,
    )
    from sentinel.generate import collect_answer, is_abstention
    from sentinel.guardrails.output import guard_answer
    from sentinel.guardrails.pii import redact_chunks

    faithfulness, _, context_recall = _build_metrics()
    idx = _load_index()
    by_q = {r["question"]: r for r in report["per_question"]}
    kept = {s["question"]: s for s in (previous or {}).get("per_question", [])}
    scored = []
    for n, it in enumerate(items, 1):
        if only is not None and n not in only and it["question"] in kept:
            scored.append(kept[it["question"]])    # judged earlier and unchanged since
            continue
        row = by_q[it["question"]]
        out = {"question": it["question"]}
        for name, ids in (("single_top5", row["single_ids"]), ("planned", row["planned_ids"])):
            chunks = [_retrieved(idx, cid) for cid in ids]
            if settings.redact_context_pii:
                chunks, _ = redact_chunks(chunks)
            answer, _, _ = await asyncio.to_thread(
                _with_backoff, partial(collect_answer, it["question"], chunks, retry=False))
            if settings.output_guard:
                answer, _ = guard_answer(answer, chunks)
            sample = SingleTurnSample(user_input=it["question"], response=answer,
                                      retrieved_contexts=[c.text for c in chunks],
                                      reference=it["reference_answer"])
            out[name] = {
                "answer": answer,
                "abstained": is_abstention(answer),
                "faithfulness": _coerce_faithfulness(
                    await _score_async(faithfulness, sample), is_abstention(answer)),
                "context_recall": _coerce(await _score_async(context_recall, sample)),
            }
        scored.append(out)
        print(f"  judged {n}/{len(items)}", end="\r", flush=True)
    print()
    means = {name: {m: round(statistics.mean(s[name][m] for s in scored), 3)
                    for m in ("faithfulness", "context_recall")}
             | {"abstained": sum(s[name]["abstained"] for s in scored)}
             for name in ("single_top5", "planned")}
    return {"items": len(scored), "means": means, "per_question": scored}


def _retrieved(idx, chunk_id: str):
    from sentinel.schema import RetrievedChunk
    c = idx.id2chunk[chunk_id]
    return RetrievedChunk(chunk_id=chunk_id, text=c.text, scope=c.scope)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judge", action="store_true", help="also generate and judge answers")
    ap.add_argument("--reuse-plans", action="store_true",
                    help="read planner outputs from the last report instead of calling it")
    ap.add_argument("--rejudge", metavar="N,N", help="with --judge: judge only these items "
                    "(1-based) and keep the last report's verdicts for the rest")
    ap.add_argument("--max-spend", type=float, metavar="USD", help="required unless reusing plans")
    args = ap.parse_args()
    if args.max_spend is None and (args.judge or not args.reuse_plans):
        sys.exit("--max-spend is required for a paid run (the Groq budget is fixed)")
    if args.max_spend is not None:
        settings.spend_limit_usd = args.max_spend

    reuse = None
    if args.reuse_plans:
        saved = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
        reuse = saved.get("retrieval", {}).get("plans", {})
    warmup()
    items = load_items()
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "planner_model": settings.planner_model,
              "multistep_top_k": settings.multistep_top_k}
    previous = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    only = {int(n) for n in args.rejudge.split(",")} if args.rejudge else None
    try:
        report["retrieval"] = retrieval_stage(items, reuse)
        if args.judge:
            report["answers"] = asyncio.run(_judge(items, report["retrieval"], only,
                                                   previous.get("answers")))
        elif "answers" in previous:
            report["answers"] = previous["answers"]    # judged answers don't depend on gold chunks
    except BudgetExceeded as exc:
        report["stopped"] = str(exc)
    report["spend"] = meter.summary()
    REPORT_PATH.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    r = report.get("retrieval", {})
    print(json.dumps({k: r[k] for k in ("multi_part",) if k in r}, indent=1))
    if "single_topic" in r:
        st = r["single_topic"]
        print(f"single-topic: {st['split_by_planner']}/{st['questions']} split; "
              f"hit {st['hit_single']:.0f} -> {st['hit_planned']:.0f} on those")
    if "answers" in report:
        print(json.dumps(report["answers"]["means"], indent=1))
    print(f"spend: {report['spend']}" + (f"; stopped: {report['stopped']}"
                                         if "stopped" in report else ""))
    print(f"wrote {REPORT_PATH.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
