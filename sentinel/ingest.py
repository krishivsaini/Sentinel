"""Ingestion pipeline (§8) — Phase 1, Day 2.

load source docs -> clean/normalize -> chunk (~512 tok, ~10–15% overlap)
-> embed (local bge model) -> build FAISS dense index
-> build BM25 index over the SAME chunks (shared chunk IDs) -> persist both.

Requirements: idempotent (re-run rebuilds cleanly, FR-I3); dense + sparse indexes share
identical chunk_ids (FR-I2); the chunking strategy is recorded on the index (FR-I4).

Run:  python -m sentinel.ingest
"""

from __future__ import annotations


def main() -> None:
    # TODO(Phase 1, Day 2): implement load -> chunk -> embed -> FAISS + BM25.
    raise NotImplementedError("ingest.py is a Phase 1 (Day 2) stub — not yet implemented.")


if __name__ == "__main__":
    main()
