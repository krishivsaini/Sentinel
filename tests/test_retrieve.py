"""Retrieval tests (§4) — Phase 1, Days 3–4.

Planned coverage:
  - dense and BM25 indexes share identical chunk IDs (guards the silent fusion-corruption bug)
  - seed queries return their expected chunks
  - FR-R7: at least one seed query retrieved by BM25 but missed by pure dense (the concrete
    justification for hybrid) — this becomes a README artifact
  - FR-R6: cross-encoder rerank measurably improves context recall vs fusion-only
"""

import pytest


@pytest.mark.skip(reason="Phase 1 (Days 3–4): retrieval not yet implemented.")
def test_bm25_and_dense_share_chunk_ids() -> None: ...


@pytest.mark.skip(reason="Phase 1 (Days 3–4): retrieval not yet implemented.")
def test_hybrid_recovers_exact_match_query_missed_by_dense() -> None: ...
