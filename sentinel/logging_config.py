"""Structured logging + per-stage latency capture (§11) — Phase 2, Days 6-7.

Every /query logs, as structured fields: per-stage latency (retrieve/rerank/generate),
retrieved chunk IDs, and outcome (FR-S5). This is what makes honest per-stage p95 possible
(NFR-3) — latency is measured per stage in isolation, never blended.

Provides `configure_logging()`, a module `log`, and a `stage_timer` context manager that
records a stage's elapsed milliseconds into a dict.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

import structlog


def _level_to_int(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog to emit one structured event per log call. JSON in production
    (greppable, machine-parseable); a readable console renderer when json_output is False."""
    renderer = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_level_to_int(level)),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger("sentinel")


@contextmanager
def stage_timer(sink: dict[str, float], name: str) -> Iterator[None]:
    """Time a pipeline stage in isolation and store elapsed ms under `name` in `sink`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        sink[name] = (time.perf_counter() - start) * 1000.0
