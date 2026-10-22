"""Output validation (v2-R3) — the last check before an answer reaches the user.

Sanitising (rewrites the answer):
  * markdown images are removed outright — a rendered image fetches its URL, which is how an
    injected answer exfiltrates data without the user clicking anything;
  * links to hosts outside config.answer_link_hosts are replaced. "Only allow URLs that appear
    in the retrieved passages" would not do: a poisoned passage carries its own phishing link;
  * PII is redacted (sentinel.guardrails.pii) as a backstop to context redaction.
Checks (reported, not rewritten):
  * every [chunk_id] citation must resolve to a retrieved chunk;
  * a substantive (non-abstaining) answer must cite at least one passage.

/query streams tokens, so sanitising can't wait for the full answer. StreamSanitizer holds back a
short tail (HOLDBACK chars) and only ever emits up to a whitespace boundary that no detected span
crosses, so a link or an email is never released half-checked. Holding ~100 characters costs
roughly 20 tokens of delay.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from sentinel.config import settings
from sentinel.generate import is_abstention, parse_citations
from sentinel.guardrails import pii
from sentinel.schema import RetrievedChunk

LINK_REMOVED = "[link removed]"
_IMAGE_RE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")
_URL_RE = re.compile(r"\b(?:https?|ftp)://[^\s<>()\"'\[\]{}]+", re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?'\""
_ANY_CITATION_RE = re.compile(r"[\[【]\s*([a-z][a-z0-9]*#\d+)\s*[\]】]")


def link_allowed(url: str, hosts: tuple[str, ...] | None = None) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    allowed = settings.answer_link_hosts if hosts is None else hosts
    return any(host == h or host.endswith("." + h) for h in allowed)


@dataclass(frozen=True)
class _Edit:
    start: int
    end: int
    kind: str          # "image" | "link" | a PII kind
    replacement: str

    @property
    def length(self) -> int:
        return self.end - self.start


def _edits(text: str) -> list[_Edit]:
    """Every rewrite the policy wants in `text`, non-overlapping, in order."""
    edits = [_Edit(m.start(), m.end(), "image", "") for m in _IMAGE_RE.finditer(text)]
    for m in _URL_RE.finditer(text):
        url = m.group().rstrip(_TRAILING_PUNCT)
        if not link_allowed(url):
            edits.append(_Edit(m.start(), m.start() + len(url), "link", LINK_REMOVED))
    edits += [
        _Edit(s.start, s.end, s.kind, pii.REDACTION.format(kind=s.kind)) for s in pii.find_pii(text)
    ]
    edits.sort(key=lambda e: (e.start, -e.length))
    out: list[_Edit] = []
    for e in edits:  # an image swallows the link inside it; otherwise the longer edit wins
        if out and e.start < out[-1].end:
            if e.length > out[-1].length:
                out[-1] = e
            continue
        out.append(e)
    return out


def _apply(text: str, edits: list[_Edit]) -> str:
    parts, pos = [], 0
    for e in edits:
        parts += [text[pos:e.start], e.replacement]
        pos = e.end
    parts.append(text[pos:])
    return "".join(parts)


@dataclass
class OutputReport:
    """What the output guard changed and found."""

    removed: Counter = field(default_factory=Counter)     # kind -> count (image, link, PII kinds)
    removed_links: list[str] = field(default_factory=list)
    unresolved_citations: list[str] = field(default_factory=list)
    uncited: bool = False

    @property
    def flagged(self) -> bool:
        return bool(self.removed or self.unresolved_citations or self.uncited)

    def as_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "removed": dict(self.removed),
            "removed_links": self.removed_links,
            "unresolved_citations": self.unresolved_citations,
            "uncited": self.uncited,
        }


def sanitize(text: str, report: OutputReport | None = None) -> str:
    """Rewrite a complete text under the output policy, recording what changed."""
    edits = _edits(text)
    if report is not None:
        for e in edits:
            report.removed[e.kind] += 1
            if e.kind == "link":
                report.removed_links.append(text[e.start:e.end])
    return _apply(text, edits)


class StreamSanitizer:
    """Incremental sanitize() for a token stream. feed() returns text that is safe to emit now;
    finish() returns the rest. Concatenated, the output equals sanitize(full text)."""

    HOLDBACK = 96

    def __init__(self, report: OutputReport | None = None) -> None:
        self.report = report if report is not None else OutputReport()
        self._buf = ""

    def feed(self, token: str) -> str:
        self._buf += token
        limit = len(self._buf) - self.HOLDBACK
        if limit <= 0:
            return ""
        cut = max(self._buf.rfind(" ", 0, limit + 1), self._buf.rfind("\n", 0, limit + 1))
        if cut <= 0:
            return ""
        for e in _edits(self._buf):  # never cut through something the policy rewrites
            if e.start < cut < e.end:
                cut = e.start
        out, self._buf = self._buf[:cut], self._buf[cut:]
        return sanitize(out, self.report)

    def finish(self) -> str:
        out, self._buf = self._buf, ""
        return sanitize(out, self.report)


def check_citations(text: str, chunks: list[RetrievedChunk], report: OutputReport) -> None:
    """Fill in the citation checks for a final (already sanitised) answer."""
    retrieved = {c.chunk_id for c in chunks}
    cited = list(dict.fromkeys(_ANY_CITATION_RE.findall(text)))
    report.unresolved_citations = [c for c in cited if c not in retrieved]
    report.uncited = not is_abstention(text) and not parse_citations(text, chunks)


def guard_answer(text: str, chunks: list[RetrievedChunk]) -> tuple[str, OutputReport]:
    """Non-streaming convenience: sanitise a complete answer and run every check."""
    report = OutputReport()
    clean = sanitize(text, report)
    check_citations(clean, chunks, report)
    return clean, report
