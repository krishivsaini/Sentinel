"""Shared test setup."""

from __future__ import annotations

import pytest

from sentinel.config import settings


@pytest.fixture(autouse=True)
def _no_query_planning(monkeypatch):
    """Query planning and the semantic cache are on by default. The planner is a paid model
    call, and a cache would let one test's answer leak into the next, so tests run with both
    off; tests/test_plan.py and tests/test_cache.py turn them on where they need them."""
    monkeypatch.setattr(settings, "query_planning", False)
    monkeypatch.setattr(settings, "semantic_cache", False)
