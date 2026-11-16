"""Shared test setup."""

from __future__ import annotations

import pytest

from sentinel.config import settings


@pytest.fixture(autouse=True)
def _no_query_planning(monkeypatch):
    """Query planning is on by default, and the planner is a paid model call. Tests stay
    hermetic with it off; tests/test_plan.py turns it on where it fakes the planner."""
    monkeypatch.setattr(settings, "query_planning", False)
