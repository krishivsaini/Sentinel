"""Training data for the reranker: pure functions, so every rule is unit-tested.

The data is synthetic — a doc2query model writes questions for RFC passages — so most of the
work is keeping bad examples out:

  windows      doc2query reads at most 320 word pieces, and a chunk is ~450. Questions are
               generated from paragraph-packed windows of a chunk instead, so the end of a chunk
               gets questions too; the positive is still the whole chunk.
  boilerplate  tables of contents, reference entries, author addresses, licence text, RFC header
               blocks and ASCII diagrams are dropped paragraph by paragraph before windowing:
               questions about them would teach the reranker noise.
  queries      cleaned, length-bounded, deduplicated; a query written for two chunks is
               ambiguous and dropped.
  leakage      the 48 ground-truth questions are for evaluation only: a synthetic query too
               similar to any of them never enters training (tests/test_reranker_data.py).
  negatives    from the same first-stage pool the reranker sees at inference, minus the positive,
               its neighbouring chunks (they overlap it by ~12%) and near-identical chunks.
  split        by RFC, so near-duplicate questions about one chunk can't sit on both sides of the
               train / validation line.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable, Iterable

WINDOW_TOKENS = 300        # doc2query was trained on inputs truncated to 320 word pieces
STUB_SLACK = 20            # a short window may run this far past WINDOW_TOKENS: still <= 320
MIN_WINDOW_WORDS = 25      # a window shorter than this is mostly a heading
QUERY_WORDS = (3, 30)      # doc2query writes MS MARCO-style questions, a few words to a sentence

_DOT_LEADER = re.compile(r"\.\s?\.\s?\.\s?\.")
_TWO_COLUMN = re.compile(r"\S\s{6,}\S")
_REFERENCE_ENTRY = re.compile(r"^\s*\[[^\]\s]{1,40}\]")
_REFERENCE_SIGNS = ("RFC ", "DOI", "<http", "Work in Progress", "Internet-Draft")
_CONTACT_LINE = re.compile(r"^\s*(E-?mail|EMail|Phone|Fax|URI|Tel):", re.I)
_LICENCE_SIGNS = (
    "BCP 78", "Trust Legal Provisions", "Simplified BSD License", "Revised BSD License",
    "Status of This Memo", "Status of this Memo", "Internet Standards Track document",
    "Information about the current status of this document",
    "This document is a product of the Internet Engineering Task Force",
    "Copyright (c)", "Copyright Notice",
)
_HEADER_SIGNS = ("Request for Comments:", "ISSN:", "Category:", "Obsoletes:", "Updates:")
_SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")
_ABNF_RULE = re.compile(r"^\s*[A-Za-z][\w-]*\s+=/?\s", re.M)   # "token68 = 1*( ALPHA ..."


# --------------------------------------------------------------------------- paragraphs


def paragraphs(text: str) -> list[str]:
    """Blank-line-separated paragraphs, raw (line structure kept for classification)."""
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def boilerplate_kind(paragraph: str) -> str | None:
    """Why this paragraph is boilerplate ('toc', 'reference', 'contact', 'licence', 'header',
    'diagram'), or None if it is content."""
    lines = [ln for ln in paragraph.splitlines() if ln.strip()]
    if not lines:
        return "empty"
    n = len(lines)
    if sum(bool(_DOT_LEADER.search(ln)) for ln in lines) >= max(1, 0.4 * n):
        return "toc"
    if _REFERENCE_ENTRY.match(lines[0]) and any(s in paragraph for s in _REFERENCE_SIGNS):
        return "reference"
    if sum(bool(_CONTACT_LINE.match(ln)) for ln in lines) >= 1 and n <= 12:
        return "contact"
    if any(s in paragraph for s in _LICENCE_SIGNS):
        return "licence"
    if any(s in paragraph for s in _HEADER_SIGNS) and sum(bool(_TWO_COLUMN.search(ln)) for ln in lines) >= 0.5 * n:
        return "header"
    visible = [c for c in paragraph if not c.isspace()]
    symbolic = len(visible) >= 40 and sum(c.isalpha() for c in visible) < 0.35 * len(visible)
    if symbolic and not _ABNF_RULE.search(paragraph):   # grammar rules are content
        return "diagram"
    return None


def flatten(paragraph: str) -> str:
    """One line, single-spaced: how a paragraph is shown to doc2query."""
    return " ".join(paragraph.split())


# --------------------------------------------------------------------------- windows


def make_windows(
    text: str, count_tokens: Callable[[str], int], max_tokens: int = WINDOW_TOKENS
) -> tuple[list[str], dict[str, int]]:
    """Content windows of one chunk for query generation, and the boilerplate dropped.

    Content paragraphs are packed greedily, in order, while they fit in max_tokens; a paragraph
    too long on its own is split at sentence ends (and, failing that, by words). A short window —
    typically a section heading — is carried into the next one (up to STUB_SLACK over the
    limit) rather than standing alone, and a short last window is evened out with the one
    before it, so no content is lost to the minimum size. A chunk whose content is only a stub
    (fewer than MIN_WINDOW_WORDS words in all) yields no window."""
    dropped: dict[str, int] = defaultdict(int)
    pieces: list[str] = []
    for para in paragraphs(text):
        kind = boilerplate_kind(para)
        if kind:
            dropped[kind] += 1
            continue
        flat = flatten(para)
        if count_tokens(flat) <= max_tokens:
            pieces.append(flat)
        else:
            pieces.extend(_split_long(flat, count_tokens, max_tokens))

    windows: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        stub = len(current.split()) < MIN_WINDOW_WORDS
        limit = max_tokens + STUB_SLACK if stub else max_tokens
        if current and count_tokens(candidate) > limit:
            windows.append(current)
            current = piece
        else:
            current = candidate
    if current:
        windows.append(current)
    if len(windows) >= 2 and len(windows[-1].split()) < MIN_WINDOW_WORDS:
        merged = f"{windows[-2]} {windows[-1]}".split()
        windows[-2:] = [" ".join(merged[: len(merged) // 2]), " ".join(merged[len(merged) // 2:])]
    return [w for w in windows if len(w.split()) >= MIN_WINDOW_WORDS], dict(dropped)


def _split_long(flat: str, count_tokens: Callable[[str], int], max_tokens: int) -> list[str]:
    out, current = [], ""
    for sentence in _SENTENCE_END.split(flat):
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > max_tokens:
            out.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        out.append(current)
    final = []
    for piece in out:          # a single "sentence" longer than the window: cut by words
        while count_tokens(piece) > max_tokens:
            words = piece.split()
            lo, hi = 1, len(words)
            while lo < hi:     # the longest word prefix that fits
                mid = (lo + hi + 1) // 2
                if count_tokens(" ".join(words[:mid])) <= max_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            final.append(" ".join(words[:lo]))
            piece = " ".join(words[lo:])
        if piece:
            final.append(piece)
    return final


# --------------------------------------------------------------------------- queries


def normalize_query(q: str) -> str:
    """The form queries are compared in: lower case, letters and digits, single spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", q.lower()).split())


def clean_queries(by_chunk: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Length-bound and deduplicate queries per chunk, then drop any query written for more than
    one chunk: it has no single right answer. Returns the kept queries and the drop counts."""
    stats = {"raw": 0, "bad_length": 0, "duplicate": 0, "ambiguous": 0}
    kept: dict[str, list[str]] = {}
    owners: dict[str, set[str]] = defaultdict(set)
    for cid, queries in by_chunk.items():
        seen, out = set(), []
        for q in queries:
            stats["raw"] += 1
            q = " ".join(q.split())
            norm = normalize_query(q)
            words = len(norm.split())
            if not (QUERY_WORDS[0] <= words <= QUERY_WORDS[1]):
                stats["bad_length"] += 1
                continue
            if norm in seen:
                stats["duplicate"] += 1
                continue
            seen.add(norm)
            out.append(q)
            owners[norm].add(cid)
        kept[cid] = out
    ambiguous = {n for n, cids in owners.items() if len(cids) > 1}
    for cid in kept:
        before = len(kept[cid])
        kept[cid] = [q for q in kept[cid] if normalize_query(q) not in ambiguous]
        stats["ambiguous"] += before - len(kept[cid])
    return {c: qs for c, qs in kept.items() if qs}, stats


def leaked(query: str, max_similarity: float, gt_normalized: set[str], threshold: float) -> bool:
    """A synthetic query is leakage if it repeats a ground-truth question as text, or is
    semantically too close to one (max cosine similarity of their embeddings >= threshold)."""
    return normalize_query(query) in gt_normalized or max_similarity >= threshold


# --------------------------------------------------------------------------- negatives


def chunk_ordinal(chunk_id: str) -> tuple[str, int]:
    doc, _, ordinal = chunk_id.partition("#")
    return doc, int(ordinal.split(".")[0])


def select_negatives(
    positive: str,
    pool: Iterable[str],
    near_duplicate: Callable[[str, str], bool],
    limit: int,
) -> tuple[list[str], dict[str, int]]:
    """Hard negatives for one query, in first-stage order: the pool minus the positive, the
    positive's neighbouring chunks in the same RFC (they share ~12% of its text, so they are
    partly correct answers), and chunks that are near-identical to it."""
    pdoc, pord = chunk_ordinal(positive)
    stats = {"neighbour": 0, "near_duplicate": 0}
    out = []
    for cid in pool:
        if cid == positive:
            continue
        doc, ordinal = chunk_ordinal(cid)
        if doc == pdoc and abs(ordinal - pord) <= 1:
            stats["neighbour"] += 1
            continue
        if near_duplicate(positive, cid):
            stats["near_duplicate"] += 1
            continue
        out.append(cid)
        if len(out) == limit:
            break
    return out, stats


# --------------------------------------------------------------------------- split


def is_validation_doc(doc_id: str, fraction: float, salt: str = "reranker-v1") -> bool:
    """A stable, seed-free assignment of whole RFCs to validation (~fraction of them)."""
    h = int(hashlib.sha256(f"{salt}:{doc_id}".encode()).hexdigest(), 16)
    return (h % 10_000) / 10_000 < fraction
