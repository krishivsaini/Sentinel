"""Which corpus RFCs have been replaced or amended since publication (version awareness).

An RFC is never edited; a later RFC obsoletes or updates it. The corpus is a snapshot, so an
answer can quote a document accurately and still be out of date — RFC 2818 is obsoleted by
RFC 9110, and RFC 8446 (TLS 1.3) by RFC 9846, which isn't in the corpus at all. Faithfulness
checks can't see that. Every citation therefore carries its RFC's standing, taken from the RFC
Editor's metadata (scripts/fetch_rfc_status.py writes data/rfc_status.json), so a client can
mark outdated sources instead of presenting them as current.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import structlog

from sentinel.config import DATA_DIR

STATUS_PATH = DATA_DIR / "rfc_status.json"
log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Standing:
    """An RFC's standing. Related RFCs are doc ids ("rfc9110"), oldest first."""

    status: str = ""                                   # e.g. "PROPOSED STANDARD"
    obsoleted_by: tuple[str, ...] = field(default=())  # replaced outright
    updated_by: tuple[str, ...] = field(default=())    # amended in part

    @property
    def obsolete(self) -> bool:
        return bool(self.obsoleted_by)


@cache
def _load(path: Path = STATUS_PATH) -> dict[str, Standing]:
    try:
        records = json.loads(path.read_text(encoding="utf-8"))["rfcs"]
    except FileNotFoundError:
        # A missing file means "unknown", never "current": callers get an empty Standing and
        # the gap is logged, but answering doesn't depend on it.
        log.warning("rfc_status.missing", path=str(path))
        return {}
    ids = lambda nums: tuple(f"rfc{n}" for n in nums)  # noqa: E731
    return {doc_id: Standing(status=r["status"], obsoleted_by=ids(r["obsoleted_by"]),
                             updated_by=ids(r["updated_by"]))
            for doc_id, r in records.items()}


def standing(doc_id: str) -> Standing:
    """The standing of a corpus RFC; an empty Standing when it isn't known."""
    return _load().get(doc_id, Standing())
