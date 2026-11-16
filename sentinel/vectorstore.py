"""Dense search in Postgres with pgvector, as an alternative to the in-memory FAISS index.

With SENTINEL_VECTOR_STORE=pgvector, retrieval's dense stage runs as one SQL query: nearest
neighbours by inner product (the vectors are unit length, so that is cosine, the same score
FAISS gives), with the caller's scope and quarantine filters in the WHERE clause. The filters
stay inside the search, as they do with FAISS, so a chunk the caller may not see never takes a
candidate slot. BM25 and the chunk text stay in process: only the vectors move.

The index directory stays the build artifact and the source of truth. `sync` copies it into
Postgres, and later syncs write only what changed: a row is rewritten when its text or vector
changed, updated in place when only its position, scope or quarantine flag did, and deleted when
its chunk is gone. A fingerprint of the index files is stored with each collection; the service
refuses to start on a collection that doesn't match its index, so it can never search vectors
from a different build than the chunk text it hands the reranker.

    docker compose -f deploy/pgvector/compose.yaml up -d
    python -m sentinel.vectorstore sync                          # indexes/
    python -m sentinel.vectorstore sync --index-dir indexes_adversarial
    python -m sentinel.vectorstore status

Layout: one table per embedding dimension (sentinel_chunks_d384 for bge-small), keyed by
(collection, chunk_id), where a collection is an index directory's name. An HNSW index serves the
nearest-neighbour query; pgvector's iterative scan (0.8+) keeps reading past ef_search when the
filters reject candidates, so a filtered query still returns its full top-N.

Needs the optional dependencies: `uv sync --extra pgvector`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from sentinel.config import INDEXES_DIR, IndexPaths, settings
from sentinel.schema import Chunk

COLLECTIONS = "sentinel_collections"
MIN_PGVECTOR = (0, 8, 0)   # iterative index scans


class VectorStoreError(RuntimeError):
    """Postgres is unreachable, misconfigured, or out of step with the index directory."""


def collection_name(root: Path) -> str:
    return root.resolve().name


def _table(dim: int):
    from psycopg import sql
    return sql.Identifier(f"sentinel_chunks_d{int(dim)}")


def redacted(url: str) -> str:
    """The URL without its password, for messages."""
    parts = urlsplit(url)
    if parts.password:
        return url.replace(f":{parts.password}@", ":***@", 1)
    return url


def fingerprint(paths: IndexPaths) -> str:
    """Identifies one build: the chunk records (text, scope, quarantine, order) and the vectors."""
    h = hashlib.sha256()
    for f in (paths.chunks, paths.faiss):
        h.update(f.read_bytes())
    return h.hexdigest()


def _row_sha(text: str, vector: np.ndarray) -> str:
    return hashlib.sha256(text.encode() + b"\0" + vector.astype("<f4").tobytes()).hexdigest()


# --------------------------------------------------------------------------- connections


def connect(url: str | None = None, *, create: bool = False):
    """A new autocommit connection with the vector type registered and the HNSW search settings
    applied. `create` installs the extension first (sync does; searching never does)."""
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError:
        raise VectorStoreError(
            "pgvector support isn't installed: uv sync --extra pgvector") from None

    url = url or settings.database_url
    try:
        conn = psycopg.connect(url, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError as exc:
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise VectorStoreError(f"can't reach Postgres at {redacted(url)}: {reason}") from None
    try:
        if create:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        if row is None:
            raise VectorStoreError("the pgvector extension isn't installed in this database; "
                                   "run `python -m sentinel.vectorstore sync`")
        version = tuple(int(p) for p in row[0].split(".")[:3])
        if version < MIN_PGVECTOR:
            raise VectorStoreError(f"pgvector {row[0]} is too old; need 0.8 or later")
        register_vector(conn)
        conn.execute("SELECT set_config('hnsw.ef_search', %s, false), "
                     "set_config('hnsw.iterative_scan', 'strict_order', false)",
                     (str(settings.pgvector_ef_search),))
    except BaseException:
        conn.close()
        raise
    return conn


_shared_lock = threading.Lock()
_shared = None


def _connection():
    """The process's search connection, reopened if it dropped. psycopg connections are safe to
    share between threads (their operations take turns); one query is a few milliseconds."""
    global _shared
    with _shared_lock:
        if _shared is None or _shared.closed or _shared.broken:
            _shared = connect()
        return _shared


def _fetch(query, params) -> list[tuple]:
    import psycopg
    for attempt in (1, 2):
        try:
            return _connection().execute(query, params).fetchall()
        except psycopg.OperationalError:
            if attempt == 2:
                raise
    return []


def close() -> None:
    global _shared
    with _shared_lock:
        if _shared is not None:
            _shared.close()
        _shared = None
        _dims.clear()


# --------------------------------------------------------------------------- sync


def _ensure_schema(conn, dim: int) -> None:
    from psycopg import sql
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {COLLECTIONS} (
            name text PRIMARY KEY,
            fingerprint text NOT NULL,
            dim integer NOT NULL,
            embedding_model text NOT NULL,
            row_count integer NOT NULL,
            synced_utc timestamptz NOT NULL DEFAULT now())""")
    table = _table(dim)
    conn.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {t} (
            collection text NOT NULL,
            chunk_id text NOT NULL,
            row_num integer NOT NULL,
            doc_id text NOT NULL,
            scope text NOT NULL,
            quarantined boolean NOT NULL,
            row_sha text NOT NULL,
            text text NOT NULL,
            embedding vector({d}) NOT NULL,
            PRIMARY KEY (collection, chunk_id))""").format(t=table, d=sql.Literal(int(dim))))
    conn.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} USING hnsw "
                         "(embedding vector_ip_ops)").format(
        i=sql.Identifier(f"sentinel_chunks_d{int(dim)}_hnsw"), t=table))


def _read_index(root: Path) -> tuple[list[Chunk], np.ndarray, str]:
    import faiss
    paths = IndexPaths(root)
    if not paths.chunks.exists() or not paths.faiss.exists():
        raise VectorStoreError(f"no index at {root}; build it with python -m sentinel.ingest")
    chunks = [Chunk.model_validate_json(line) for line in paths.chunks.open(encoding="utf-8")]
    index = faiss.read_index(str(paths.faiss))
    vectors = index.reconstruct_n(0, index.ntotal)
    if len(vectors) != len(chunks):
        raise VectorStoreError(f"{root}: {len(chunks)} chunks but {len(vectors)} vectors")
    return chunks, vectors, fingerprint(paths)


def sync(index_dir: Path = INDEXES_DIR, *, url: str | None = None) -> dict:
    """Make the collection for `index_dir` match it, writing only what changed. One transaction:
    a search never sees a half-synced collection."""
    from psycopg import sql

    t0 = time.perf_counter()
    root = index_dir.resolve()
    chunks, vectors, fp = _read_index(root)
    dim, name = int(vectors.shape[1]), collection_name(root)
    conn = connect(url, create=True)
    try:
        with conn.transaction():
            _ensure_schema(conn, dim)
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (name,))
            prev = conn.execute(f"SELECT fingerprint, dim FROM {COLLECTIONS} WHERE name = %s",
                                (name,)).fetchone()
            if prev == (fp, dim):
                return {"mode": "unchanged", "collection": name, "rows": len(chunks)}
            table = _table(dim)
            if prev and prev[1] != dim:   # a new embedding model: the old rows are in another table
                conn.execute(sql.SQL("DELETE FROM {t} WHERE collection = %s").format(
                    t=_table(prev[1])), (name,))
            existing = {r[0]: r[1:] for r in conn.execute(sql.SQL(
                "SELECT chunk_id, row_sha, row_num, doc_id, scope, quarantined FROM {t} "
                "WHERE collection = %s").format(t=table), (name,))}

            writes, moves = [], []
            for i, (c, v) in enumerate(zip(chunks, vectors, strict=True)):
                sha, old = _row_sha(c.text, v), existing.pop(c.chunk_id, None)
                if old is None or old[0] != sha:
                    writes.append((name, c.chunk_id, i, c.doc_id, c.scope, c.quarantined, sha,
                                   c.text, v))
                elif old[1:] != (i, c.doc_id, c.scope, c.quarantined):
                    moves.append((i, c.doc_id, c.scope, c.quarantined, name, c.chunk_id))
            with conn.cursor() as cur:
                if existing:
                    cur.execute(sql.SQL("DELETE FROM {t} WHERE collection = %s AND chunk_id = "
                                        "ANY(%s)").format(t=table), (name, list(existing)))
                if moves:
                    cur.executemany(sql.SQL(
                        "UPDATE {t} SET row_num = %s, doc_id = %s, scope = %s, quarantined = %s "
                        "WHERE collection = %s AND chunk_id = %s").format(t=table), moves)
                if writes:
                    cur.executemany(sql.SQL("""
                        INSERT INTO {t} (collection, chunk_id, row_num, doc_id, scope,
                                         quarantined, row_sha, text, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (collection, chunk_id) DO UPDATE SET
                            row_num = EXCLUDED.row_num, doc_id = EXCLUDED.doc_id,
                            scope = EXCLUDED.scope, quarantined = EXCLUDED.quarantined,
                            row_sha = EXCLUDED.row_sha, text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding""").format(t=table), writes)
            conn.execute(f"""
                INSERT INTO {COLLECTIONS} (name, fingerprint, dim, embedding_model, row_count)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET fingerprint = EXCLUDED.fingerprint,
                    dim = EXCLUDED.dim, embedding_model = EXCLUDED.embedding_model,
                    row_count = EXCLUDED.row_count, synced_utc = now()""",
                         (name, fp, dim, settings.embedding_model, len(chunks)))
    finally:
        conn.close()
    _dims.pop(name, None)
    return {"mode": "incremental" if prev else "full", "collection": name, "rows": len(chunks),
            "written": len(writes), "updated": len(moves), "removed": len(existing),
            "seconds": round(time.perf_counter() - t0, 2)}


# --------------------------------------------------------------------------- search

_dims: dict[str, int] = {}


def check_synced(root: Path) -> None:
    """Raise unless the collection for `root` holds exactly this build of the index."""
    name = collection_name(root)
    rows = _fetch(f"SELECT fingerprint, dim FROM {COLLECTIONS} WHERE name = %s", (name,)) \
        if _has_collections_table() else []
    if not rows:
        raise VectorStoreError(f"no pgvector collection {name!r}; run "
                               f"`python -m sentinel.vectorstore sync --index-dir {root}`")
    if rows[0][0] != fingerprint(IndexPaths(root)):
        raise VectorStoreError(f"pgvector collection {name!r} is from a different build than "
                               f"{root}; run `python -m sentinel.vectorstore sync "
                               f"--index-dir {root}`")
    _dims[name] = rows[0][1]


def _has_collections_table() -> bool:
    return bool(_fetch("SELECT to_regclass(%s) IS NOT NULL", (COLLECTIONS,))[0][0])


def dense_search(
    root: Path,
    query_vector: np.ndarray,
    top_n: int,
    scopes: frozenset[str],
    include_quarantined: bool = False,
) -> list[tuple[str, float]]:
    """(chunk_id, inner product) best-first, among the chunks in `scopes` and, unless asked for,
    not quarantined: the FAISS search's contract."""
    from psycopg import sql
    name = collection_name(root)
    if name not in _dims:
        check_synced(root)
    quarantine = sql.SQL("") if include_quarantined else sql.SQL("AND NOT quarantined")
    query = sql.SQL(
        "SELECT chunk_id, -(embedding <#> %(q)s) FROM {t} "
        "WHERE collection = %(c)s AND scope = ANY(%(s)s) {quarantine} "
        "ORDER BY embedding <#> %(q)s LIMIT %(n)s").format(t=_table(_dims[name]),
                                                            quarantine=quarantine)
    vec = np.asarray(query_vector, dtype="float32").reshape(-1)
    rows = _fetch(query, {"q": vec, "c": name, "s": sorted(scopes), "n": int(top_n)})
    return [(cid, float(score)) for cid, score in rows]


def status() -> list[dict]:
    if not _has_collections_table():
        return []
    rows = _fetch(f"SELECT name, dim, row_count, embedding_model, synced_utc FROM {COLLECTIONS} "
                  "ORDER BY name", ())
    return [{"collection": n, "dim": d, "rows": r, "embedding_model": m,
             "synced_utc": s.isoformat(timespec="seconds")} for n, d, r, m, s in rows]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync", help="copy an index directory's vectors into Postgres")
    s.add_argument("--index-dir", type=Path, default=INDEXES_DIR)
    sub.add_parser("status", help="list the collections in Postgres")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "sync":
            print(sync(args.index_dir))
        else:
            for row in status() or [{"collections": "none"}]:
                print(row)
    except VectorStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
