"""Watch the RFC Editor for changes that concern the corpus: what to re-ingest, what to review.

The corpus is a curated snapshot of 48 RFCs. RFCs are never edited, they are replaced, so the
snapshot drifts in two ways, and this script reports both from the RFC Editor's index
(https://www.rfc-editor.org/rfc-index.xml, one request):

  * standing changes: a corpus RFC newly obsoleted or updated, or its status changed, since
    data/rfc_status.json was written. Fix: `python scripts/fetch_rfc_status.py`, so citations
    show the new standing.
  * candidates: RFCs outside the corpus that obsolete a corpus RFC (at any date: a replacement
    is always worth a look), or that update one or come from a working group the corpus already
    covers, published since the corpus snapshot.

Adding an RFC stays a person's decision: it changes what the eval measures. `add` puts it in
the manifest; `skip` records why not, so it isn't reported again. Decisions are committed in
data/corpus_watch.json.

    python scripts/watch_rfcs.py                       # report; exit 3 if anything needs review
    python scripts/watch_rfcs.py add 9846 --cluster tls
    python scripts/watch_rfcs.py skip 9999 --reason "operational guidance, not protocol"

After an `add`:  python scripts/fetch_corpus.py && python scripts/fetch_rfc_status.py
                 && python -m sentinel.ingest --incremental
The last step screens only the new chunks for prompt injection (Groq calls, so it costs a
little) and embeds only them; everything else comes from the existing index.

Stdlib only. Reading the index costs nothing; nothing here calls a model.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_corpus import DownloadError, download, load_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "corpus_manifest.json"
STATUS_PATH = REPO_ROOT / "data" / "rfc_status.json"
DECISIONS_PATH = REPO_ROOT / "data" / "corpus_watch.json"
INDEX_URL = "https://www.rfc-editor.org/rfc-index.xml"
NS = "{https://www.rfc-editor.org/rfc-index}"
NO_WG = {"", "NON WORKING GROUP", "IETF - NON WORKING GROUP", "NONE"}
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September",
     "October", "November", "December"], start=1)}
NEEDS_REVIEW = 3


def _rfcs(entry, tag: str) -> list[int]:
    el = entry.find(NS + tag)
    if el is None:
        return []
    ids = (d.text or "" for d in el.findall(NS + "doc-id"))
    return sorted(int(i[3:]) for i in ids if i.startswith("RFC") and i[3:].isdigit())


def parse_index(xml: bytes) -> dict[int, dict]:
    """RFC number -> the fields this script uses, for every RFC in the index."""
    out = {}
    for e in ET.fromstring(xml).iter(NS + "rfc-entry"):
        doc_id = e.findtext(NS + "doc-id", "")
        if not doc_id.startswith("RFC"):
            continue
        out[int(doc_id[3:])] = {
            "title": " ".join(e.findtext(NS + "title", "").split()),
            "year": int(e.findtext(f"{NS}date/{NS}year", "0") or 0),
            "month": MONTHS.get(e.findtext(f"{NS}date/{NS}month", ""), 0),
            "status": e.findtext(NS + "current-status", "").upper(),
            "wg": e.findtext(NS + "wg_acronym", "").strip(),
            **{f: _rfcs(e, tag) for f, tag in [
                ("obsoletes", "obsoletes"), ("obsoleted_by", "obsoleted-by"),
                ("updates", "updates"), ("updated_by", "updated-by")]},
        }
    return out


def review(index: dict[int, dict], manifest: dict, status: dict, decisions: dict) -> dict:
    """What changed for the corpus, and which outside RFCs deserve a look."""
    corpus = {e["number"]: e for e in manifest["rfcs"]}
    snap = datetime.fromisoformat(manifest["fetched_utc"])
    since = (snap.year, snap.month)

    changes = []
    for n in sorted(corpus):
        cur, old = index.get(n), status.get(f"rfc{n}")
        if cur is None or old is None:
            continue
        diff = {f: sorted(set(cur[f]) - set(old[f])) for f in ("obsoleted_by", "updated_by")}
        diff = {f: v for f, v in diff.items() if v}
        if cur["status"] and cur["status"] != old["status"]:
            diff["status"] = f"{old['status']} -> {cur['status']}"
        if diff:
            changes.append({"rfc": n, "title": corpus[n]["title"], **diff})

    # Each working group's most common cluster in the corpus (ties: the earlier RFC's).
    by_wg: dict[str, Counter] = {}
    for n in sorted(corpus):
        wg = index.get(n, {}).get("wg", "")
        if wg.upper() not in NO_WG:
            by_wg.setdefault(wg, Counter())[corpus[n]["cluster"]] += 1
    wg_cluster = {wg: c.most_common(1)[0][0] for wg, c in by_wg.items()}

    candidates = []
    for n, e in sorted(index.items()):
        if n in corpus or f"rfc{n}" in decisions:
            continue
        recent = (e["year"], e["month"]) >= since
        why, cluster = [], None
        if replaced := [m for m in e["obsoletes"] if m in corpus]:
            why.append("obsoletes " + ", ".join(f"RFC {m}" for m in replaced))
            cluster = corpus[replaced[0]]["cluster"]
        if recent and (amended := [m for m in e["updates"] if m in corpus]):
            why.append("updates " + ", ".join(f"RFC {m}" for m in amended))
            cluster = cluster or corpus[amended[0]]["cluster"]
        if recent and e["wg"] in wg_cluster:
            why.append(f"from the {e['wg']} working group")
            cluster = cluster or wg_cluster[e["wg"]]
        if why:
            candidates.append({"rfc": n, "title": e["title"], "published": f"{e['year']}-"
                               f"{e['month']:02d}", "status": e["status"], "why": why,
                               "suggested_cluster": cluster})
    return {"since": f"{since[0]}-{since[1]:02d}", "standing_changes": changes,
            "candidates": candidates}


def _load_decisions() -> dict:
    if DECISIONS_PATH.exists():
        return json.loads(DECISIONS_PATH.read_text())["decisions"]
    return {}


def _save_decision(number: int, decision: str, reason: str) -> None:
    decisions = _load_decisions()
    decisions[f"rfc{number}"] = {"decision": decision, "reason": reason,
                                 "decided_utc": datetime.now(UTC).date().isoformat()}
    DECISIONS_PATH.write_text(json.dumps({"decisions": dict(sorted(decisions.items()))},
                                         indent=1) + "\n")


def _index(args) -> dict[int, dict]:
    xml = args.index_file.read_bytes() if args.index_file else download(INDEX_URL)
    return parse_index(xml)


def _markdown(report: dict) -> str:
    lines = [f"## Corpus watch: {len(report['standing_changes'])} standing change(s), "
             f"{len(report['candidates'])} candidate(s)", ""]
    for c in report["standing_changes"]:
        parts = [f"{k.replace('_', ' ')} {v if isinstance(v, str) else ', '.join(map(str, v))}"
                 for k, v in c.items() if k not in ("rfc", "title")]
        lines.append(f"- **RFC {c['rfc']}** ({c['title']}): {'; '.join(parts)}")
    if report["standing_changes"]:
        lines += ["", "Run `python scripts/fetch_rfc_status.py` and commit data/rfc_status.json.",
                  ""]
    for c in report["candidates"]:
        lines.append(f"- RFC {c['rfc']} *{c['title']}* ({c['published']}, {c['status']}): "
                     f"{'; '.join(c['why'])}; suggested cluster `{c['suggested_cluster']}`")
    if report["candidates"]:
        lines += ["", "Decide each: `python scripts/watch_rfcs.py add N --cluster C` or "
                  "`skip N --reason ...`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index-file", type=Path, help="read a saved rfc-index.xml (no network)")
    sub = ap.add_subparsers(dest="cmd")
    chk = sub.add_parser("check", help="report (the default)")
    chk.add_argument("--summary", type=Path, help="also write the report as Markdown here")
    add = sub.add_parser("add", help="add an RFC to the corpus manifest")
    add.add_argument("number", type=int)
    add.add_argument("--cluster", required=True)
    skip = sub.add_parser("skip", help="record that an RFC stays out")
    skip.add_argument("number", type=int)
    skip.add_argument("--reason", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "skip":
        _save_decision(args.number, "skip", args.reason)
        print(f"RFC {args.number}: skipped ({args.reason})")
        return 0
    try:
        index = _index(args)
    except DownloadError as exc:
        print(f"couldn't read the RFC index: {exc}")
        return 1
    manifest = load_manifest()

    if args.cmd == "add":
        if args.number not in index:
            sys.exit(f"RFC {args.number} isn't in the RFC index")
        if any(e["number"] == args.number for e in manifest["rfcs"]):
            sys.exit(f"RFC {args.number} is already in the corpus")
        entry = {"number": args.number, "title": index[args.number]["title"],
                 "cluster": args.cluster}
        manifest["rfcs"].append(entry)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        _save_decision(args.number, "added", f"cluster {args.cluster}")
        print(f"added RFC {args.number} ({entry['title']}) to the manifest. Next:\n"
              "  python scripts/fetch_corpus.py && python scripts/fetch_rfc_status.py\n"
              "  python -m sentinel.ingest --incremental   # screens + embeds only the new RFC")
        return 0

    status = json.loads(STATUS_PATH.read_text())["rfcs"]
    report = review(index, manifest, status, _load_decisions())
    text = _markdown(report)
    print(text, end="")
    if getattr(args, "summary", None):
        args.summary.write_text(text)
    return NEEDS_REVIEW if report["standing_changes"] or report["candidates"] else 0


if __name__ == "__main__":
    sys.exit(main())
