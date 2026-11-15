"""PII detection and redaction (v2-R2).

Applied twice: to retrieved passages before they reach the generator (so the model never sees
personal data it could repeat) and to the answer stream as a backstop (sentinel.guardrails.output),
since a model can also produce PII from its own memory or echo it from the question.

Policy — what counts, decided by surveying the real corpus, not assumed:
  email        personal or organisational mailboxes. Addresses at the domains RFC 2606 / 6761
               reserve for documentation (example.com, *.example, *.test, *.invalid, localhost)
               are not personal data and are kept. So are role and list mailboxes, but only on
               strong evidence in the address itself: an RFC 2142 role name (hostmaster@...) or a
               group/function word in the local part (ietf-http-wg@, oauth-ext-review@, ipr@).
               Anything else is treated as personal: a missed mailbox leaks, an over-redacted
               one only costs a little text. Lists named after a working group (hybi@ietf.org)
               look like any other name and stay redacted; the eval counts them.
  phone        international (+CC ...) and North-American numbers, phone and fax.
  credit_card  13-19 digit card numbers in a card layout, with a known issuer prefix and a valid
               Luhn checksum — the checksum is what separates them from hex dumps and OIDs.
  secret       credentials in recognisable provider formats (OpenAI, Groq, AWS, GitHub, Slack,
               Google, NVIDIA) and PEM private-key blocks. JWTs are deliberately excluded: the
               JOSE RFCs are full of example tokens, and the answers that explain them need them.
  person       private documents only (any scope but "public"): a spaCy PERSON entity that also
               looks like a person's name — two to four capitalised words or initials, starting
               or ending with a known given name (data/guardrails/given_names.txt, from Wikidata,
               so not only Western names) or starting with an initial ("J. Martinez") — and any
               capitalised word after an honorific ("Dr. Okafor"). Entity recognition alone is
               useless on technical text: over the RFC corpus it tagged JSON, MUST, Huffman and
               "Header Parameter" as people. A lone first or last name is not caught.
  address      private documents only: a street line (number, name, street type) and the unit,
               city, state/ZIP or UK postcode that follow it. Other countries' formats are not
               caught.
Why names and addresses stop at the public scope: the public corpus is published RFCs, whose
names are authors and editors on the public record, and whose addresses are their employers'.
Redacting them would break "who edited RFC 9110?" for no privacy gain. Their mailboxes and phone
numbers are still redacted: those are ways to reach a person. The answer stream (output.py) stays
pattern-only for the same reason — it can't tell which source a name came from.
Out of scope, stated rather than half-done:
  IP addresses — of 145 IPv4-shaped strings in the corpus, 114 are section numbers and the rest
  are documentation, private-range or 1980s example hosts; none is personal, and redacting them
  would break correct answers.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import cache

from sentinel.config import GUARDRAILS_DIR
from sentinel.schema import RetrievedChunk

GIVEN_NAMES_PATH = GUARDRAILS_DIR / "given_names.txt"
PUBLIC_SCOPES = frozenset({"public"})

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


# RFC 2142, "Mailbox Names for Common Services, Roles and Functions".
_RFC2142_ROLES = frozenset({
    "info", "marketing", "sales", "support", "abuse", "noc", "security", "postmaster",
    "hostmaster", "usenet", "news", "webmaster", "www", "uucp", "ftp",
})
# Words that name a group, list or function rather than a person, as tokens of the local part.
# "ietf" alone is deliberately absent: personal addresses use it (jri.ietf@gmail.com).
_GROUP_TOKENS = frozenset({
    "wg", "review", "reviews", "request", "requests", "announce", "discuss", "list", "lists",
    "types", "ipr", "iesg", "iab", "registrar", "registry", "editors",
})


def is_role_address(address: str) -> bool:
    """True when the mailbox is a role or list rather than a person, judged from the address."""
    local = address.rsplit("@", 1)[0].lower()
    return local in _RFC2142_ROLES or not _GROUP_TOKENS.isdisjoint(re.split(r"[._+-]", local))


def find_emails(text: str) -> list[Span]:
    spans = []
    for m in _EMAIL_RE.finditer(text):
        domain = m.group().rsplit("@", 1)[1]
        if _RESERVED_DOMAIN_RE.search(domain) or _in_url(text, m.start()):
            continue
        if is_role_address(m.group()):
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


# --------------------------------------------------------------------------- person names


@cache
def _nlp():
    """spaCy's small English pipeline, loaded on first use: a service whose documents are all
    public never pays for it."""
    import spacy

    return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])


@cache
def _given_names() -> frozenset[str]:
    lines = GIVEN_NAMES_PATH.read_text(encoding="utf-8").splitlines()
    return frozenset(n.casefold() for n in lines if n and not n.startswith("#"))


_HONORIFIC = r"(?:Mr|Mrs|Ms|Mx|Dr|Prof|Sir|Dame)\.?"
_HONORIFIC_NAME_RE = re.compile(rf"\b{_HONORIFIC}\s+([^\W\d_][^\W\d_'’-]+)")
_LEADING_HONORIFIC_RE = re.compile(rf"^(?:{_HONORIFIC}\s+)+")
_INITIAL_RE = re.compile(r"^[^\W\d_]\.$")
_WORD_RE = re.compile(r"^[^\W\d_][^\W\d_]*(?:['’-][^\W\d_]+)*$")


def _is_capitalised_word(tok: str) -> bool:
    # "Lena", "O'Neill", "Al-Sayed" — not "JSON" or "MUST" (all capitals) or "iPhone"
    return bool(_WORD_RE.match(tok)) and tok[0].isupper() and not (len(tok) > 1 and tok.isupper())


def looks_like_person(name: str) -> bool:
    """Whether an entity-recognition PERSON span is shaped like a person's name."""
    toks = name.split()
    if not 2 <= len(toks) <= 4:
        return False
    initials = [bool(_INITIAL_RE.match(t)) and t[0].isupper() for t in toks]
    if not all(i or _is_capitalised_word(t) for i, t in zip(initials, toks, strict=True)):
        return False
    if all(initials):
        return False
    given = _given_names()
    return initials[0] or toks[0].casefold() in given or toks[-1].casefold() in given


def find_names(text: str) -> list[Span]:
    spans = [Span(m.start(1), m.end(1), "person") for m in _HONORIFIC_NAME_RE.finditer(text)]
    for ent in _nlp()(text).ents:
        if ent.label_ != "PERSON":
            continue
        start, name = ent.start_char, re.sub(r"['’]s$", "", ent.text)
        lead = _LEADING_HONORIFIC_RE.match(name)
        if lead:
            start, name = start + lead.end(), name[lead.end():]
        if looks_like_person(name):
            spans.append(Span(start, start + len(name), "person"))
    return spans


# --------------------------------------------------------------------------- postal addresses

_STREET_TYPES = (
    "Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|"
    "Terrace|Parkway|Pkwy|Highway|Hwy|Square|Sq|Close|Crescent|Row|Gardens"
)
_STREET_RE = re.compile(
    rf"\b\d{{1,6}}[A-Z]?\s+(?:[A-Z][\w'’.-]*\s+){{1,4}}(?:{_STREET_TYPES})\b\.?"
)
_ADDRESS_PART_RE = re.compile(
    r",\s*(?:"
    r"(?:Apt|Apartment|Suite|Ste|Unit|Floor|Fl|Room|Rm|Building|Bldg)\.?\s*[\w-]+"   # unit
    r"|[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3}"                                   # city
    r"(?:\s+(?:[A-Z]{2}\s+)?\d{5}(?:-\d{4})?"                                     # ST 12345
    r"|\s+[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}"                                       # UK postcode
    r"|\s+\d{3,6})?"                                                               # other code
    r")"
)


def find_addresses(text: str) -> list[Span]:
    spans = []
    for m in _STREET_RE.finditer(text):
        end = m.end()
        while part := _ADDRESS_PART_RE.match(text, end):
            end = part.end()
        spans.append(Span(m.start(), end, "address"))
    return spans


# --------------------------------------------------------------------------- combined

DETECTORS = {
    "email": find_emails,
    "phone": find_phones,
    "credit_card": find_cards,
    "secret": find_secrets,
}
# Run only on private documents (see the module docstring).
ENTITY_DETECTORS = {
    "person": find_names,
    "address": find_addresses,
}


def find_pii(text: str, *, entities: bool = False) -> list[Span]:
    """Every PII span, non-overlapping, in order. Where detectors overlap (a card number that
    also looks like a phone), the longer span wins. entities=True adds names and addresses."""
    finders = [*DETECTORS.values(), *(ENTITY_DETECTORS.values() if entities else ())]
    spans = sorted(
        (s for find in finders for s in find(text)),
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


def redact(
    text: str, spans: list[Span] | None = None, *, entities: bool = False
) -> tuple[str, list[Span]]:
    """Replace each PII span with a typed placeholder. Returns the redacted text and the spans
    (offsets into the original)."""
    spans = find_pii(text, entities=entities) if spans is None else spans
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
    sees it. Chunk ids (and so citations) are unchanged. Names and addresses are redacted only in
    chunks outside the public scope. Returns the chunks and a count by kind."""
    counts: Counter = Counter()
    out = []
    for c in chunks:
        text, spans = redact(c.text, entities=c.scope not in PUBLIC_SCOPES)
        counts.update(s.kind for s in spans)
        out.append(c.model_copy(update={"text": text}) if spans else c)
    return out, counts
