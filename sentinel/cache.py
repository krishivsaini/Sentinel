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

Two backends with the same interface and the same tests (tests/test_cache.py runs every test
against both), chosen by config.cache_backend:

  * "memory" (AnswerCache): per instance, nothing to run. A restart starts empty, and on Cloud
    Run, which scales to zero when idle, so does every cold start.
  * "redis" (RedisAnswerCache): entries live in Redis, so they survive restarts and cold starts
    and are shared by every instance. Fails open: if Redis is unreachable a lookup is a miss and
    a store is skipped, so the cache can never fail a request.
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
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import numpy as np

from sentinel.logging_config import log

if TYPE_CHECKING:
    import redis

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

    backend = "memory"

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

    def ping(self) -> bool:
        return True

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


# --------------------------------------------------------------------------- Redis


def redacted(url: str) -> str:
    """The URL without its password, for logs."""
    parts = urlsplit(url)
    return url.replace(f":{parts.password}@", ":***@", 1) if parts.password else url


class RedisAnswerCache:
    """The same cache in Redis, shared by every instance and kept across restarts.

    Keys, under `prefix`:

      e:<id>                    hash    one entry: context, question, tokens, vector (float32
                                        bytes), payload (JSON), answer_id, created, hits
      b:<context>:<tokens>      set     the ids a lookup compares against: the same context and
                                        exactly the same technical tokens
      order                     zset    every id by creation time, for expiry and the size cap
      s:<answer_id>             string  the entry an answer was served from (thumbs-down)
      stats                     hash    lookups, hits, stored, evicted, errors
      seq                       counter the insertion order, the prefix of every id

    The token check is an exact match, so it's a key rather than a search: a lookup reads one
    bucket (usually zero to a few entries) and compares those vectors with numpy. No vector
    index or Redis module is needed, so any Redis works: open-source Redis, Valkey, Redis Cloud,
    Upstash. Every key has a TTL, so Redis reclaims memory on its own; expiry is also checked
    against each entry's `created` time, so the answer doesn't depend on when Redis gets to it.
    """

    backend = "redis"

    def __init__(self, client: redis.Redis, threshold: float, ttl_seconds: float,
                 max_entries: int, prefix: str = "sentinel:cache:") -> None:
        import redis as redis_lib
        self._errors: tuple[type[BaseException], ...] = (redis_lib.RedisError, OSError)
        self._r = client
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.prefix = prefix
        self._ttl_ms = max(1, int(ttl_seconds * 1000))

    @classmethod
    def from_url(cls, url: str, threshold: float, ttl_seconds: float, max_entries: int,
                 prefix: str = "sentinel:cache:") -> RedisAnswerCache:
        """Short timeouts: an unreachable Redis costs a request at most about half a second
        once, then the connection error makes later calls fail fast."""
        import redis as redis_lib
        client = redis_lib.Redis.from_url(url, socket_connect_timeout=0.5, socket_timeout=0.5,
                                          health_check_interval=30)
        return cls(client, threshold, ttl_seconds, max_entries, prefix)

    # -- keys
    def _k(self, *parts: str) -> str:
        return self.prefix + ":".join(parts)

    def _bucket(self, context: str, tokens: frozenset[str]) -> str:
        digest = hashlib.sha256(json.dumps(sorted(tokens)).encode()).hexdigest()[:16]
        return self._k("b", context, digest)

    def _failed(self, op: str, exc: BaseException) -> None:
        log.warning("cache.redis_error", op=op, error=type(exc).__name__, detail=str(exc)[:200])
        try:
            self._r.hincrby(self._k("stats"), "errors", 1)
        except self._errors:
            pass

    # -- interface shared with AnswerCache
    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except self._errors:
            return False

    def __len__(self) -> int:
        try:
            return int(self._r.zcount(self._k("order"), time.time() - self.ttl, "+inf"))
        except self._errors as exc:
            self._failed("len", exc)
            return 0

    @property
    def stats(self) -> dict[str, int]:
        try:
            raw = self._r.hgetall(self._k("stats"))
        except self._errors:
            return {}
        return {k.decode(): int(v) for k, v in raw.items()}

    def lookup(self, question: str, vector: np.ndarray, context: str) -> Hit | None:
        tokens = technical_tokens(question)
        bucket = self._bucket(context, tokens)
        now = time.time()
        try:
            pipe = self._r.pipeline(transaction=False)
            pipe.hincrby(self._k("stats"), "lookups", 1)
            pipe.smembers(bucket)
            ids = sorted(i.decode() for i in pipe.execute()[1])
            if not ids:
                return None
            pipe = self._r.pipeline(transaction=False)
            for i in ids:
                pipe.hgetall(self._k("e", i))
            rows = pipe.execute()
            best: tuple[float, str, dict] | None = None
            gone: list[str] = []
            for i, row in zip(ids, rows, strict=True):
                if not row or now - float(row[b"created"]) >= self.ttl:
                    gone.append(i)      # expired, evicted or trimmed: tidy the bucket
                    continue
                if row[b"context"].decode() != context:   # a bucket-key collision; never
                    continue
                vec = np.frombuffer(row[b"vector"], dtype="float32")
                sim = float(np.dot(vec, vector))
                if sim >= self.threshold and (best is None or sim > best[0]):
                    best = (sim, i, row)
            pipe = self._r.pipeline(transaction=False)
            if gone:
                pipe.srem(bucket, *gone)
                pipe.delete(*(self._k("e", i) for i in gone))
                pipe.zrem(self._k("order"), *gone)
            if best is not None:
                pipe.hincrby(self._k("e", best[1]), "hits", 1)
                pipe.hincrby(self._k("stats"), "hits", 1)
            if gone or best is not None:
                pipe.execute()
        except self._errors as exc:
            self._failed("lookup", exc)
            return None
        if best is None:
            return None
        sim, i, row = best
        entry = Entry(context=context, question=row[b"question"].decode(), tokens=tokens,
                      vector=np.frombuffer(row[b"vector"], dtype="float32").copy(),
                      payload=json.loads(row[b"payload"]), answer_id=row[b"answer_id"].decode(),
                      created=float(row[b"created"]), hits=int(row.get(b"hits", 0)) + 1, id=i)
        return Hit(entry, sim)

    def store(self, question: str, vector: np.ndarray, context: str, payload: dict,
              answer_id: str) -> Entry:
        entry = Entry(context, question, technical_tokens(question),
                      np.asarray(vector, dtype="float32"), payload, answer_id)
        try:
            # Ids start with a shared insertion counter: entries created in the same instant
            # (score ties in `order`) then sort oldest first, as they do in memory.
            entry.id = f"{self._r.incr(self._k('seq')):012d}-{entry.id[:12]}"
            key = self._k("e", entry.id)
            bucket = self._bucket(context, entry.tokens)
            pipe = self._r.pipeline(transaction=True)
            pipe.hset(key, mapping={
                "context": context, "question": question,
                "tokens": json.dumps(sorted(entry.tokens)),
                "vector": entry.vector.tobytes(), "payload": json.dumps(payload),
                "answer_id": answer_id, "created": repr(entry.created), "hits": 0})
            pipe.pexpire(key, self._ttl_ms)
            pipe.sadd(bucket, entry.id)
            pipe.pexpire(bucket, self._ttl_ms)
            pipe.zadd(self._k("order"), {entry.id: entry.created})
            pipe.set(self._k("s", answer_id), entry.id, px=self._ttl_ms)
            pipe.hincrby(self._k("stats"), "stored", 1)
            pipe.execute()
            self._trim(time.time())
        except self._errors as exc:
            self._failed("store", exc)
        return entry

    def _trim(self, now: float) -> None:
        """Drop expired entries, then the oldest beyond max_entries. Their bucket memberships
        are tidied by the next lookup that meets them."""
        order = self._k("order")
        pipe = self._r.pipeline(transaction=False)
        pipe.zrangebyscore(order, "-inf", f"({now - self.ttl}")
        pipe.zcard(order)
        expired, total = pipe.execute()
        over = max(0, total - len(expired) - self.max_entries)
        oldest = self._r.zrange(order, len(expired), len(expired) + over - 1) if over else []
        drop = [i.decode() for i in (*expired, *oldest)]
        if drop:
            pipe = self._r.pipeline(transaction=False)
            pipe.delete(*(self._k("e", i) for i in drop))
            pipe.zrem(order, *drop)
            pipe.execute()

    def served(self, answer_id: str, entry: Entry) -> None:
        try:
            self._r.set(self._k("s", answer_id), entry.id, px=self._ttl_ms)
        except self._errors as exc:
            self._failed("served", exc)

    def evict_answer(self, answer_id: str) -> bool:
        try:
            pipe = self._r.pipeline(transaction=True)
            pipe.get(self._k("s", answer_id))
            pipe.delete(self._k("s", answer_id))
            entry_id = pipe.execute()[0]
            if entry_id is None:
                return False
            entry_id = entry_id.decode()
            pipe = self._r.pipeline(transaction=True)
            pipe.delete(self._k("e", entry_id))
            pipe.zrem(self._k("order"), entry_id)
            if not pipe.execute()[0]:
                return False
            self._r.hincrby(self._k("stats"), "evicted", 1)
            return True
        except self._errors as exc:
            self._failed("evict", exc)
            return False

    def clear(self) -> int:
        """Delete every key under the prefix (tests, or after a bad deploy). Returns how many."""
        n = 0
        for key in self._r.scan_iter(match=self.prefix + "*", count=500):
            n += self._r.delete(key)
        return n
