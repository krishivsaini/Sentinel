"""Version awareness: each citation carries its RFC's standing (sentinel/rfc_status.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sentinel.generate import parse_citations
from sentinel.rfc_status import STATUS_PATH, Standing, _load, standing
from sentinel.schema import RetrievedChunk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_corpus import load_manifest  # noqa: E402
from fetch_rfc_status import _numbers, check  # noqa: E402


def test_committed_status_covers_exactly_the_corpus() -> None:
    """Regenerating the corpus without the status file (or vice versa) fails here, not in prod."""
    assert check(json.loads(STATUS_PATH.read_text())["rfcs"], load_manifest()) == []


def test_known_replacements() -> None:
    """The two obsoleted RFCs that motivated the feature, one replaced inside the corpus and one
    replaced by an RFC published after the corpus was fetched."""
    assert standing("rfc2818").obsoleted_by == ("rfc9110",)
    assert standing("rfc8446").obsoleted_by == ("rfc9846",)
    assert "rfc8174" in standing("rfc2119").updated_by
    assert not standing("rfc9110").obsolete


def test_unknown_doc_is_unknown_not_current() -> None:
    assert standing("rfc9999") == Standing()


def test_missing_file_degrades_to_unknown(tmp_path: Path) -> None:
    assert _load(tmp_path / "absent.json") == {}


def test_numbers_drops_non_rfc_references() -> None:
    assert _numbers(["RFC9110", "RFC0793", "BCP0014", "STD0097"]) == [793, 9110]


def test_citations_carry_standing() -> None:
    chunks = [RetrievedChunk(chunk_id=c, text="x") for c in ("rfc2818#0003", "rfc9110#0100")]
    old, new = parse_citations("Use TLS [rfc2818#0003] [rfc9110#0100].", chunks)
    assert old.obsoleted_by == ["rfc9110"]
    assert new.obsoleted_by == []
    assert old.model_dump()["obsoleted_by"] == ["rfc9110"]  # reaches the API response
