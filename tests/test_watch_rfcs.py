"""The corpus watcher reads the RFC Editor's index and reports what concerns the corpus.

A small hand-written index stands in for rfc-index.xml (no network), in the real schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import watch_rfcs as w  # noqa: E402


def _entry(n, title, year, month, wg="", status="PROPOSED STANDARD", **rel):
    def refs(tag, nums):
        if not nums:
            return ""
        return f"<{tag}>" + "".join(f"<doc-id>RFC{m:04d}</doc-id>" for m in nums) + f"</{tag}>"
    return (f"<rfc-entry><doc-id>RFC{n:04d}</doc-id><title>{title}</title>"
            f"<date><month>{month}</month><year>{year}</year></date>"
            + refs("obsoletes", rel.get("obsoletes"))
            + refs("obsoleted-by", rel.get("obsoleted_by"))
            + refs("updates", rel.get("updates"))
            + refs("updated-by", rel.get("updated_by"))
            + f"<current-status>{status}</current-status><wg_acronym>{wg}</wg_acronym>"
            "</rfc-entry>")


INDEX = ("<?xml version='1.0'?><rfc-index xmlns='https://www.rfc-editor.org/rfc-index'>"
         "<bcp-entry><doc-id>BCP14</doc-id></bcp-entry>"
         + _entry(2119, "Key words", 1997, "March", status="BEST CURRENT PRACTICE",
                  updated_by=[8174])
         + _entry(8446, "TLS 1.3", 2018, "August", wg="tls", obsoleted_by=[9846],
                  updated_by=[9999])
         + _entry(9110, "HTTP Semantics", 2022, "June", wg="httpbis")
         + _entry(9111, "HTTP Caching", 2022, "June", wg="httpbis")
         + _entry(5000, "An old TLS thing", 2008, "May", wg="tls")              # old, same WG
         + _entry(7000, "Old replacement", 2013, "May", obsoletes=[2119])       # old, but replaces
         + _entry(9846, "TLS 1.3 again", 2026, "July", wg="tls", obsoletes=[8446])
         + _entry(9999, "A TLS update", 2026, "September", updates=[8446])
         + _entry(10036, "Incremental HTTP", 2026, "August", wg="httpbis")
         + _entry(10050, "Unrelated", 2026, "October", wg="sidrops")
         + "</rfc-index>").encode()

MANIFEST = {"fetched_utc": "2026-07-06T17:59:29+00:00", "rfcs": [
    {"number": 2119, "title": "Key words", "cluster": "keywords"},
    {"number": 8446, "title": "TLS 1.3", "cluster": "tls"},
    {"number": 9110, "title": "HTTP Semantics", "cluster": "http-core"},
    {"number": 9111, "title": "HTTP Caching", "cluster": "http-core"},
]}
STATUS = {
    "rfc2119": {"status": "BEST CURRENT PRACTICE", "obsoleted_by": [], "updated_by": [8174]},
    "rfc8446": {"status": "PROPOSED STANDARD", "obsoleted_by": [], "updated_by": []},
    "rfc9110": {"status": "PROPOSED STANDARD", "obsoleted_by": [], "updated_by": []},
    "rfc9111": {"status": "PROPOSED STANDARD", "obsoleted_by": [], "updated_by": []},
}


def test_the_index_is_parsed_into_relationships() -> None:
    idx = w.parse_index(INDEX)
    assert 14 not in idx                                     # BCP entries are skipped
    assert idx[8446]["obsoleted_by"] == [9846] and idx[8446]["wg"] == "tls"
    assert (idx[9846]["year"], idx[9846]["month"]) == (2026, 7)


def test_standing_changes_are_what_rfc_status_json_does_not_know_yet() -> None:
    report = w.review(w.parse_index(INDEX), MANIFEST, STATUS, {})
    assert report["standing_changes"] == [
        {"rfc": 8446, "title": "TLS 1.3", "obsoleted_by": [9846], "updated_by": [9999]}]


def test_candidates_replacements_any_date_others_only_since_the_snapshot() -> None:
    report = w.review(w.parse_index(INDEX), MANIFEST, STATUS, {})
    got = {c["rfc"]: c for c in report["candidates"]}
    assert set(got) == {7000, 9846, 9999, 10036}
    assert got[7000]["why"] == ["obsoletes RFC 2119"]                 # old, but a replacement
    assert got[9846]["suggested_cluster"] == "tls"
    assert got[9999]["why"] == ["updates RFC 8446"]
    assert got[10036]["suggested_cluster"] == "http-core"             # the WG's cluster
    assert 5000 not in got and 10050 not in got                       # old same-WG; other WG


def test_decided_rfcs_are_not_reported_again() -> None:
    decisions = {"rfc9846": {"decision": "skip"}, "rfc7000": {"decision": "skip"}}
    report = w.review(w.parse_index(INDEX), MANIFEST, STATUS, decisions)
    assert {c["rfc"] for c in report["candidates"]} == {9999, 10036}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST))
    (tmp_path / "status.json").write_text(json.dumps({"rfcs": STATUS}))
    (tmp_path / "index.xml").write_bytes(INDEX)
    monkeypatch.setattr(w, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(w, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(w, "DECISIONS_PATH", tmp_path / "decisions.json")
    monkeypatch.setattr(w, "load_manifest", lambda: json.loads(manifest.read_text()))
    return tmp_path


def test_check_add_and_skip(repo, capsys) -> None:
    index = ["--index-file", str(repo / "index.xml")]
    assert w.main([*index, "check", "--summary", str(repo / "s.md")]) == w.NEEDS_REVIEW
    assert "RFC 9846" in (repo / "s.md").read_text()

    assert w.main([*index, "add", "9846", "--cluster", "tls"]) == 0
    manifest = json.loads((repo / "manifest.json").read_text())
    assert manifest["rfcs"][-1] == {"number": 9846, "title": "TLS 1.3 again", "cluster": "tls"}
    with pytest.raises(SystemExit):
        w.main([*index, "add", "9846", "--cluster", "tls"])           # already in

    for n in (7000, 9999, 10036):
        w.main(["skip", str(n), "--reason", "not needed"])
    decisions = json.loads((repo / "decisions.json").read_text())["decisions"]
    assert decisions["rfc9846"]["decision"] == "added"
    assert decisions["rfc9999"] == {"decision": "skip", "reason": "not needed",
                                    "decided_utc": decisions["rfc9999"]["decided_utc"]}
    # Only the standing change for 8446 is left to act on.
    assert w.main([*index, "check"]) == w.NEEDS_REVIEW
    assert "0 candidate(s)" in capsys.readouterr().out
