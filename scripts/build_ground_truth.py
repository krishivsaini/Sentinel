"""Ground-truth draft generator (§7, §15 day 8) — Phase 3, Day 8.

For each seed question (data/gt_seed_questions.jsonl), runs the real hybrid retriever and drafts
a GROUNDED reference answer with our generation layer — which answers only from retrieved
context and cites the chunks it used. The cited chunks become the candidate `relevant_contexts`.
Output is written to data/ground_truth.draft.jsonl.

CRITICAL (FR-GT2): these are DRAFTS. Every item must then be hand-corrected before it becomes
data/ground_truth.jsonl. Drafts are grounded in real corpus text (not free-floating synthetic
answers), but a human must still verify each answer + its relevant contexts. Abstentions and
citation-less drafts are flagged needs_review so bad questions surface.

Resumable: re-running skips questions already drafted (append-only), so a free-tier throttle
mid-run never loses progress. Calls are spaced and per-item timeout-guarded.

Run:  python scripts/build_ground_truth.py                 # draft all not-yet-done
      python scripts/build_ground_truth.py --limit 10      # only 10 this run
      python scripts/build_ground_truth.py --spacing 5     # 5s between LLM calls
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from sentinel.config import DATA_DIR, settings

SEED_PATH = DATA_DIR / "gt_seed_questions.jsonl"
DRAFT_PATH = DATA_DIR / "ground_truth.draft.jsonl"
REVIEW_PATH = DATA_DIR / "ground_truth.review.jsonl"
ASSIST_POOL = 6  # candidate chunks surfaced per question for LLM-free hand-authoring


def _load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


# Draft from a slightly larger pool than the service top-5 so a question whose answer spans
# several chunks still gets a grounded draft; the human trims relevant_contexts during review.
DRAFT_POOL = 8


def draft_item(question: str, cluster: str) -> dict:
    """Retrieve gold-candidate chunks and draft a grounded, cited reference answer."""
    from sentinel.generate import collect_answer, is_abstention
    from sentinel.retrieve import hybrid_retrieve

    chunks = hybrid_retrieve(question, top_k=DRAFT_POOL)
    id2text = {c.chunk_id: c.text for c in chunks}
    text, citations, _ = collect_answer(question, chunks, retry=True)
    cited_ids = [c.chunk_id for c in citations]

    if is_abstention(text):
        status, contexts = "abstained", [id2text[c.chunk_id] for c in chunks[:3]]
    elif cited_ids:
        status, contexts = "ok", [id2text[cid] for cid in cited_ids if cid in id2text]
    else:
        status, contexts = "no_citation", [id2text[c.chunk_id] for c in chunks[:3]]

    return {
        "question": question,
        "reference_answer": text,
        "relevant_contexts": contexts,
        "_meta": {
            "cluster": cluster,
            "status": status,                       # ok | abstained | no_citation
            "needs_review": True,                   # FR-GT2: hand-correct before promoting
            "cited_chunk_ids": cited_ids,
            "retrieved_chunk_ids": [c.chunk_id for c in chunks],
            "drafted_by": settings.generation_model,
        },
    }


def assist() -> int:
    """LLM-FREE: for each seed, surface top candidate chunks (local retrieval only) plus any
    existing draft, so the reference answers can be hand-authored from the corpus with zero API
    dependency. Writes data/ground_truth.review.jsonl."""
    from sentinel.retrieve import hybrid_retrieve, warmup

    seeds = _load_jsonl(SEED_PATH)
    drafts = {d["question"]: d for d in _load_jsonl(DRAFT_PATH)}
    warmup()

    with REVIEW_PATH.open("w", encoding="utf-8") as out:
        for i, seed in enumerate(seeds, 1):
            q = seed["question"]
            chunks = hybrid_retrieve(q, top_k=ASSIST_POOL)
            draft = drafts.get(q)
            out.write(json.dumps({
                "question": q,
                "cluster": seed.get("cluster", ""),
                "existing_draft": (
                    {"reference_answer": draft["reference_answer"],
                     "status": draft["_meta"]["status"],
                     "cited_chunk_ids": draft["_meta"]["cited_chunk_ids"]}
                    if draft else None
                ),
                "candidates": [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks],
            }, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(seeds)}] {seed.get('cluster',''):<16} "
                  f"draft={'yes' if draft else 'no ':<3} {q[:50]}")

    print(f"\nwrote {REVIEW_PATH} ({len(seeds)} questions, {ASSIST_POOL} candidates each)")
    print("NEXT: hand-author reference_answer + relevant_contexts -> data/ground_truth.jsonl")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Draft grounded ground-truth triples for review.")
    ap.add_argument("--assist", action="store_true",
                    help="LLM-free: emit candidate chunks per seed for hand-authoring")
    ap.add_argument("--limit", type=int, default=None, help="max items to draft this run")
    ap.add_argument("--spacing", type=float, default=4.0, help="seconds between LLM calls (RPM)")
    ap.add_argument("--timeout", type=float, default=90.0, help="per-item timeout seconds")
    args = ap.parse_args()

    if args.assist:
        return assist()

    seeds = _load_jsonl(SEED_PATH)
    done = {d["question"] for d in _load_jsonl(DRAFT_PATH)}
    todo = [s for s in seeds if s["question"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"seeds={len(seeds)}  already drafted={len(done)}  drafting now={len(todo)}")

    from sentinel.retrieve import warmup

    warmup()

    ok = failed = 0
    with DRAFT_PATH.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=1) as pool:
        for i, seed in enumerate(todo, 1):
            q, cluster = seed["question"], seed.get("cluster", "")
            fut = pool.submit(draft_item, q, cluster)
            try:
                item = fut.result(timeout=args.timeout)
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                out.flush()
                ok += 1
                print(f"  [{i}/{len(todo)}] {item['_meta']['status']:<11} {cluster:<16} {q[:52]}")
            except (FutureTimeout, Exception) as exc:
                failed += 1
                print(f"  [{i}/{len(todo)}] SKIP ({type(exc).__name__}) {q[:52]}")
            time.sleep(args.spacing)

    print(f"\ndrafted={ok}  skipped={failed}  -> {DRAFT_PATH}")
    print("NEXT: hand-correct every item, then promote to data/ground_truth.jsonl")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
