"""Semantic cache: answer a question again without retrieval or generation when it has, in
effect, been asked before.

A hit needs two things:

  1. Similar meaning: the cosine similarity of the two questions' bge embeddings (the same
     embedding retrieval uses, with its query instruction) is at least config.cache_threshold.
  2. The same exact terms: both questions name the same technical tokens (technical_tokens):
     status codes, RFC and version numbers, header and parameter names, error codes, ALL-CAPS
     keywords and acronyms. Embeddings place "What does 307 mean?" next to "What does 308 mean?";
     the answers differ, so the tokens must match exactly. Questions that differ only in
     ordinary words ("the client's certificate" vs "the server's") rely on the threshold, which
     is set from a measurement of such pairs (scripts/cache_eval.py).

An entry is only ever served to a request in the same context: the caller's scopes, the index
build, the generation model and prompt, and every setting that changes an answer, all hashed
into one key. A different scope never sees another scope's answers, and rebuilding the index or
changing a guardrail starts the cache afresh. Entries expire after config.cache_ttl_seconds.

What's stored is only an answer that went out clean: cited, not abstained, not withheld by the
answer check, nothing removed by the output guard, and no PII in the question. The input guard
still runs on every question before the cache is consulted, so a rephrased attack can't ride on
a cached answer. A thumbs-down on any answer served from an entry evicts it.

The cache is in memory, per instance: no new service, nothing to pay for. A restart or a new
instance starts empty. A shared store (Redis, Postgres) would make hits survive restarts and be
shared between instances, at the price of a standing service.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

# A "word" keeps the punctuation technical names are made of: rfc9110, HTTP/2, max-age,
# invalid_grant, 1.3, Sec-WebSocket-Key.
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./+-]*[A-Za-z0-9]|[A-Za-z0-9]")
_QUOTED = re.compile(r"[\"'`“‘]([^\"'`”’]{1,60})[\"'`”’]")
# Small capitalised words that aren't names, so a capital after "?" or in a title doesn't count.
_COMMON = {"what", "how", "why", "when", "which", "who", "does", "do", "is", "are", "the", "a",
           "an", "in", "of", "and", "or", "to", "for", "by", "default", "using", "during"}


def technical_tokens(question: str) -> frozenset[str]:
    """The exact terms two questions must share for one to reuse the other's answer, lowercased.
    A token is a word with a digit, an underscore or a hyphen; an ALL-CAPS word or one with an
    inner capital (HttpOnly, WebSocket); a capitalised word that doesn't start the sentence
    (Secure, Basic, Path); or anything in quotes. Lower-case prose doesn't count, so rephrasing
    it is free, while changing a name, number or keyword never is."""
    out = {m.group(1).strip().lower() for m in _QUOTED.finditer(question)}
    prev_end = None
    for m in _WORD.finditer(question):
        w = m.group(0)
        gap = "" if prev_end is None else question[prev_end:m.start()]
        sentence_start = prev_end is None or any(ch in ".?!:" for ch in gap)
        prev_end = m.end()
        technical = (any(ch.isdigit() for ch in w) or "_" in w or "-" in w
                     or (len(w) > 1 and w.isupper())
                     or any(ch.isupper() for ch in w[1:])
                     or (w[0].isupper() and not sentence_start and w.lower() not in _COMMON))
        if technical:
            out.add(w.lower())
    return frozenset(out)


@dataclass
class Entry:
    context: str
    question: str
    tokens: frozenset[str]
    vector: np.ndarray
    payload: dict                       # what a hit replays: text, citations, retrieved, ...
    answer_id: str                      # the answer it was first generated as
    created: float = field(default_factory=lambda: time.time())
    hits: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class Hit:
    entry: Entry
    similarity: float

    @property
    def age_s(self) -> float:
        return time.time() - self.entry.created


def context_key(**parts) -> str:
    """One key for everything an answer depends on besides the question."""
    blob = json.dumps(parts, sort_keys=True, default=lambda v: sorted(v) if isinstance(
        v, (set, frozenset)) else str(v))
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class AnswerCache:
    """In-memory semantic cache, safe to share between request threads."""

    def __init__(self, threshold: float, ttl_seconds: float, max_entries: int) -> None:
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, Entry] = OrderedDict()   # oldest first
        self._served: OrderedDict[str, str] = OrderedDict()      # answer_id -> entry id
        self.stats = {"lookups": 0, "hits": 0, "stored": 0, "evicted": 0}

    def __len__(self) -> int:
        return len(self._entries)

    def _expire(self, now: float) -> None:
        while self._entries:
            oldest = next(iter(self._entries.values()))
            if now - oldest.created < self.ttl and len(self._entries) <= self.max_entries:
                break
            self._entries.pop(oldest.id)

    def lookup(self, question: str, vector: np.ndarray, context: str) -> Hit | None:
        """The closest entry in `context` with the same technical tokens, if it is similar
        enough. The answer_id the caller serves it under must then be recorded with `served`."""
        tokens = technical_tokens(question)
        with self._lock:
            self.stats["lookups"] += 1
            self._expire(time.time())
            best: Hit | None = None
            for e in self._entries.values():
                if e.context != context or e.tokens != tokens:
                    continue
                sim = float(np.dot(e.vector, vector))
                if sim >= self.threshold and (best is None or sim > best.similarity):
                    best = Hit(e, sim)
            if best is not None:
                best.entry.hits += 1
                self.stats["hits"] += 1
            return best

    def store(self, question: str, vector: np.ndarray, context: str, payload: dict,
              answer_id: str) -> Entry:
        entry = Entry(context, question, technical_tokens(question),
                      np.asarray(vector, dtype="float32"), payload, answer_id)
        with self._lock:
            self._entries[entry.id] = entry
            self.stats["stored"] += 1
            self._expire(time.time())
            self._remember(answer_id, entry.id)
        return entry

    def served(self, answer_id: str, entry: Entry) -> None:
        """Record that `answer_id` was a replay of `entry`, so feedback on it can evict it."""
        with self._lock:
            self._remember(answer_id, entry.id)

    def _remember(self, answer_id: str, entry_id: str) -> None:
        self._served[answer_id] = entry_id
        while len(self._served) > 10 * self.max_entries:
            self._served.popitem(last=False)

    def evict_answer(self, answer_id: str) -> bool:
        """Drop the entry an answer came from (a reader said it was wrong). True if one was."""
        with self._lock:
            entry_id = self._served.pop(answer_id, None)
            if entry_id is None or self._entries.pop(entry_id, None) is None:
                return False
            self.stats["evicted"] += 1
            return True
