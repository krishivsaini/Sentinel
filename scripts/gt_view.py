"""View candidate chunks from ground_truth.review.jsonl for hand-authoring (LLM-free helper)."""
import json
import sys
from pathlib import Path

REVIEW = Path("data/ground_truth.review.jsonl")
lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
hi = int(sys.argv[2]) if len(sys.argv) > 2 else lo + 12
ncand = int(sys.argv[3]) if len(sys.argv) > 3 else 4

rows = [json.loads(l) for l in REVIEW.open()]
for idx in range(lo, min(hi, len(rows))):
    r = rows[idx]
    print(f"\n{'='*90}\n[{idx}] ({r['cluster']}) {r['question']}")
    d = r.get("existing_draft")
    if d and d["status"] == "ok":
        print(f"  DRAFT[{d['status']}] cites {d['cited_chunk_ids']}: {d['reference_answer'][:300]}")
    for c in r["candidates"][:ncand]:
        print(f"  --- {c['chunk_id']} ---")
        print("  " + c["text"].strip().replace("\n", "\n  ")[:900])
