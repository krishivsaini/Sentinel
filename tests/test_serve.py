"""API contract tests (§4, §11) — Phase 2, Day 6.

Planned coverage:
  - GET /healthz returns 200
  - unauthenticated POST /query is rejected (FR-S3)
  - rate limit fires under load (FR-S4) — a test must prove it
  - /query returns a well-formed Answer with per-stage latency + citations
"""

import pytest


@pytest.mark.skip(reason="Phase 2 (Day 6): service not yet implemented.")
def test_unauthenticated_query_is_rejected() -> None: ...


@pytest.mark.skip(reason="Phase 2 (Day 6): service not yet implemented.")
def test_rate_limit_fires_under_load() -> None: ...
