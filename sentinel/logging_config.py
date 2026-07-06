"""Structured logging + per-stage latency capture (§11) — Phase 2, Days 6–7.

Every request logs, as structured fields: per-stage latency (retrieve/rerank/generate),
retrieved chunk IDs, and outcome (FR-S5). This is what makes honest per-stage p95 possible
(NFR-3) — latency is measured per stage in isolation, never blended.

Provides a `stage_timer` context manager and a configured structlog logger.
"""

from __future__ import annotations


def configure_logging() -> None:
    # TODO(Phase 2, Day 6): configure structlog (JSON renderer, timestamps, level).
    raise NotImplementedError("logging_config.py is a Phase 2 stub — not yet implemented.")
