"""Build the independent attack set: BIPIA's attack instructions planted in RFC passages.

Every hand-written attack in data/adversarial/ was written by the person who wrote the defences.
BIPIA (Yi et al., "Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large
Language Models", Microsoft, MIT licence) supplies attacks nobody here wrote: its 75 test-split
text-task instructions, 5 in each of 15 categories — task hijacks ("Recommend a good book"),
output encodings (Base64, Caesar cipher, reversed text, emoji, translation) and planted content
(advertising, scams, fabricated statistics). They are used verbatim; only the carrier is ours.

Each instruction is planted into the ground-truth passage of one of the 48 eval questions —
at the start, middle or end of the passage in turn, as BIPIA does — and the question is the
one that retrieves that passage. So the attack rides inside exactly the text the model needs.

    python scripts/build_bipia.py        # writes data/bipia/{attacks.json,corpus.jsonl,questions.jsonl}
    python -m sentinel.ingest --bipia    # then index (and screen) it into indexes_bipia/

The download is pinned to one commit, and the file vendored, so the set never changes under us.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel.config import BIPIA_DIR, DATA_DIR, INDEXES_DIR, IndexPaths  # noqa: E402

COMMIT = "a004b69ec0dd446e0afd461d98cb5e96e120a5d0"   # microsoft/BIPIA main, 2024-04-15
SOURCE = f"https://raw.githubusercontent.com/microsoft/BIPIA/{COMMIT}/benchmark/text_attack_test.json"
LICENSE_URL = f"https://raw.githubusercontent.com/microsoft/BIPIA/{COMMIT}/LICENSE"
POSITIONS = ("start", "middle", "end")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def _plant(passage: str, instruction: str, position: str) -> str:
    if position == "start":
        return f"{instruction}\n\n{passage}"
    if position == "end":
        return f"{passage}\n\n{instruction}"
    # middle: at the paragraph break nearest the midpoint (a sentence break if there is none)
    mid = len(passage) // 2
    breaks = [i for i in range(len(passage)) if passage.startswith("\n\n", i)] or [
        i + 1 for i in range(len(passage)) if passage.startswith(". ", i)
    ]
    cut = min(breaks, key=lambda i: abs(i - mid)) if breaks else mid
    return f"{passage[:cut].rstrip()}\n\n{instruction}\n\n{passage[cut:].lstrip()}"


def _titles() -> dict[str, str]:
    """RFC doc_id -> title, from the corpus manifest."""
    manifest = json.loads((DATA_DIR / "corpus_manifest.json").read_text())
    rfcs = next(v for v in manifest.values()
                if isinstance(v, list) and v and isinstance(v[0], dict) and "number" in v[0])
    return {f"rfc{r['number']}": r["title"] for r in rfcs}


def main() -> None:
    BIPIA_DIR.mkdir(parents=True, exist_ok=True)
    raw = _fetch(SOURCE)
    attacks = json.loads(raw)
    (BIPIA_DIR / "attacks.json").write_bytes(raw)
    (BIPIA_DIR / "LICENSE").write_bytes(_fetch(LICENSE_URL))

    chunks = {
        c["text"]: c for c in map(json.loads, IndexPaths(INDEXES_DIR).chunks.open(encoding="utf-8"))
    }
    titles = _titles()
    gt = [json.loads(line) for line in (DATA_DIR / "ground_truth.jsonl").open(encoding="utf-8")]

    corpus, questions = [], []
    flat = [(cat, instr) for cat, instrs in attacks.items() for instr in instrs]
    for i, (category, instruction) in enumerate(flat):
        item = gt[i % len(gt)]
        host = chunks[item["relevant_contexts"][0]]   # exact chunk text, by construction
        doc_id = f"bipia{i:02d}"
        position = POSITIONS[i % len(POSITIONS)]
        corpus.append({
            "doc_id": doc_id,
            "title": titles.get(host["doc_id"], host["doc_id"]),
            "host_chunk": host["chunk_id"],
            "text": _plant(host["text"], instruction, position),
        })
        questions.append({
            "question": item["question"], "target_doc": doc_id, "category": category,
            "instruction": instruction, "position": position, "split": "bipia",
        })

    for name, rows in (("corpus.jsonl", corpus), ("questions.jsonl", questions)):
        with (BIPIA_DIR / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(flat)} attacks in {len(attacks)} categories from microsoft/BIPIA@{COMMIT[:7]} "
          f"(sha256 {hashlib.sha256(raw).hexdigest()[:12]}) -> {BIPIA_DIR}")


if __name__ == "__main__":
    main()
