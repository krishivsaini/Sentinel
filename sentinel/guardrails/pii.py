"""PII detection and redaction (v2-R2).

Applied twice: to retrieved passages before they reach the generator (so the model never sees
personal data it could repeat) and to the answer stream as a backstop (sentinel.guardrails.output),
since a model can also produce PII from its own memory or echo it from the question.

Policy — what counts, decided by surveying the real corpus, not assumed:
  email        personal or organisational mailboxes. Addresses at the domains RFC 2606 / 6761
               reserve for documentation (example.com, *.example, *.test, *.invalid, localhost)
               are not personal data and are kept. Role addresses (iesg@ietf.org) can't be told
               apart from personal ones by pattern, so they are redacted too; the eval measures
               that over-redaction instead of hiding it.
  phone        international (+CC ...) and North-American numbers, phone and fax.
  credit_card  13-19 digit card numbers in a card layout, with a known issuer prefix and a valid
               Luhn checksum — the checksum is what separates them from hex dumps and OIDs.
  secret       credentials in recognisable provider formats (OpenAI, Groq, AWS, GitHub, Slack,
               Google, NVIDIA) and PEM private-key blocks. JWTs are deliberately excluded: the
               JOSE RFCs are full of example tokens, and the answers that explain them need them.
Out of scope, stated rather than half-done:
  IP addresses — of 145 IPv4-shaped strings in the corpus, 114 are section numbers and the rest
  are documentation, private-range or 1980s example hosts; none is personal, and redacting them
  would break correct answers. Names and postal addresses need entity recognition, not patterns.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from sentinel.schema import RetrievedChunk

REDACTION = "[REDACTED {kind}]"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str

    @property
    def length(self) -> int:
        return self.end - self.start


# --------------------------------------------------------------------------- email

_EMAIL_RE = re.compile(
    r"(?<![\w.+%-])"
    r"[A-Za-z0-9][A-Za-z0-9._%+-]*"
    r"@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}"
    r"(?![\w-])"
)
_RESERVED_DOMAIN_RE = re.compile(
    r"(?:^|\.)(?:example\.(?:com|net|org)|example|test|invalid|localhost)$", re.IGNORECASE
)


def _in_url(text: str, start: int) -> bool:
    """True when the match sits inside a URL token — the userinfo of 'http://user@host' is not a
    mailbox."""
    token_start = max(text.rfind(" ", 0, start), text.rfind("\n", 0, start)) + 1
    return "://" in text[token_start:start]


def find_emails(text: str) -> list[Span]:
    spans = []
    for m in _EMAIL_RE.finditer(text):
        domain = m.group().rsplit("@", 1)[1]
        if _RESERVED_DOMAIN_RE.search(domain) or _in_url(text, m.start()):
            continue
        spans.append(Span(m.start(), m.end(), "email"))
    return spans


# --------------------------------------------------------------------------- phone

_PHONE_RE = re.compile(
    r"(?<![\w+])(?:"
    r"\+\d{1,3}(?:[ .-]?\(?\d{1,4}\)?){2,5}"                     # +44 20 8133 3718, +1-617-253-5702
    r"|\(\d{3}\)\s?\d{3}[ .-]\d{4}"                              # (650) 328-8631, (720)562-6785
    r"|\b\d{3}[.-]\d{3}[.-]\d{4}\b"                              # 415-555-0100, 415.555.0100
    r")(?!\w)(?!\.\d)"                                          # a full stop may follow; ".5" may not
)


def find_phones(text: str) -> list[Span]:
    spans = []
    for m in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 8 <= len(digits) <= 15:
            spans.append(Span(m.start(), m.end(), "phone"))
    return spans


# --------------------------------------------------------------------------- credit card

_CARD_RE = re.compile(
    r"(?<![\w-])(?:"
    r"\d{4}([ -]?)\d{4}\1\d{4}\1\d{1,7}"                         # 4-4-4-x, uniform separators
    r"|3[47]\d{2}[ -]?\d{6}[ -]?\d{5}"                           # Amex 4-6-5
    r")(?![\w-])"
)
_ISSUER_RE = re.compile(r"^(?:4|5[1-5]|2[2-7]|3[47]|3[0689]|6(?:011|5)|35)")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def find_cards(text: str) -> list[Span]:
    spans = []
    for m in _CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if (
            13 <= len(digits) <= 19
            and _ISSUER_RE.match(digits)
            and _luhn_ok(digits)
        ):
            spans.append(Span(m.start(), m.end(), "credit_card"))
    return spans


# --------------------------------------------------------------------------- secrets

_SECRET_RES = [
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"),       # OpenAI / Anthropic style
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),                       # Groq
    re.compile(r"\bnvapi-[A-Za-z0-9_-]{20,}"),                   # NVIDIA
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),                # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),               # GitHub
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),               # Slack
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),                    # Google API key
    re.compile(                                                  # PEM private key block
        r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z]+ )?PRIVATE KEY-----"
    ),
]


def find_secrets(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), "secret") for r in _SECRET_RES for m in r.finditer(text)]


# --------------------------------------------------------------------------- combined

DETECTORS = {
    "email": find_emails,
    "phone": find_phones,
    "credit_card": find_cards,
    "secret": find_secrets,
}


def find_pii(text: str) -> list[Span]:
    """Every PII span, non-overlapping, in order. Where detectors overlap (a card number that
    also looks like a phone), the longer span wins."""
    spans = sorted(
        (s for find in DETECTORS.values() for s in find(text)),
        key=lambda s: (s.start, -s.length),
    )
    out: list[Span] = []
    for s in spans:
        if out and s.start < out[-1].end:
            if s.length > out[-1].length:
                out[-1] = s
            continue
        out.append(s)
    return out


def redact(text: str, spans: list[Span] | None = None) -> tuple[str, list[Span]]:
    """Replace each PII span with a typed placeholder. Returns the redacted text and the spans
    (offsets into the original)."""
    spans = find_pii(text) if spans is None else spans
    if not spans:
        return text, []
    parts, pos = [], 0
    for s in spans:
        parts.append(text[pos:s.start])
        parts.append(REDACTION.format(kind=s.kind))
        pos = s.end
    parts.append(text[pos:])
    return "".join(parts), spans


def redact_chunks(chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], Counter]:
    """Context redaction: copies of the retrieved chunks with PII replaced, so the generator never
    sees it. Chunk ids (and so citations) are unchanged. Returns the chunks and a count by kind."""
    counts: Counter = Counter()
    out = []
    for c in chunks:
        text, spans = redact(c.text)
        counts.update(s.kind for s in spans)
        out.append(c.model_copy(update={"text": text}) if spans else c)
    return out, counts
