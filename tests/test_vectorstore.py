"""pgvector dense search gives FAISS's results under the same filters, and sync writes only what
changed.

Needs a Postgres with pgvector (deploy/pgvector/compose.yaml locally; a service container in CI)
at SENTINEL_TEST_DATABASE_URL, default the local container. Skipped when it isn't reachable or
the optional dependencies aren't installed. Each test works in its own schema, dropped after.
"""

from __future__ import annotations

import os
import pickle
import uuid
from urllib.parse import quote

import numpy as np
import pytest

from sentinel import retrieve
from sentinel import vectorstore as vs
from sentinel.config import INDEXES_DIR, IndexPaths, settings
from sentinel.schema import Chunk

TEST_URL = os.environ.get("SENTINEL_TEST_DATABASE_URL",
                          "postgresql://sentinel:sentinel@127.0.0.1:5433/sentinel")
SCOPES = ("public", "internal", "adversarial")


@pytest.fixture
def pg(monkeypatch):
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    try:
        admin = vs.connect(TEST_URL, create=True)
    except vs.VectorStoreError as exc:
        pytest.skip(f"no Postgres for the pgvector tests: {exc}")
    schema = f"t_{uuid.uuid4().hex[:12]}"
    admin.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in TEST_URL else "?"
    url = f"{TEST_URL}{sep}options={quote(f'-csearch_path={schema},public')}"
    monkeypatch.setattr(settings, "database_url", url)
    vs.close()
    retrieve._load_index_at.cache_clear()
    yield url
    vs.close()
    retrieve._load_index_at.cache_clear()
    admin.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.close()


def _write_index(root, chunks: list[Chunk], vectors: np.ndarray) -> None:
    import faiss
    from rank_bm25 import BM25Okapi
    paths = IndexPaths(root)
    paths.faiss.parent.mkdir(parents=True, exist_ok=True)
    paths.chunks.write_text("".join(c.model_dump_json() + "\n" for c in chunks))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(paths.faiss))
    with paths.bm25.open("wb") as fh:
        pickle.dump({"bm25": BM25Okapi([c.text.split() for c in chunks]),
                     "chunk_ids": [c.chunk_id for c in chunks]}, fh)


def _unit(rng, n, dim=16):
    v = rng.standard_normal((n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _synthetic(root, n=300, seed=0):
    rng = np.random.default_rng(seed)
    chunks = [Chunk(chunk_id=f"d{i // 10}#{i % 10}", doc_id=f"d{i // 10}", text=f"chunk {i}",
                    token_count=2, ordinal=i % 10, scope=SCOPES[rng.integers(3)],
                    quarantined=bool(rng.random() < 0.1)) for i in range(n)]
    vectors = _unit(rng, n)
    _write_index(root, chunks, vectors)
    return chunks, vectors


def _faiss_ids(chunks, vectors, q, top_n, scopes, include_quarantined):
    ok = [i for i, c in enumerate(chunks)
          if c.scope in scopes and (include_quarantined or not c.quarantined)]
    sims = vectors[ok] @ q
    return [chunks[ok[j]].chunk_id for j in np.argsort(-sims, kind="stable")[:top_n]]


def test_filtered_searches_match_exact_search(pg, tmp_path) -> None:
    root = tmp_path / "idx"
    chunks, vectors = _synthetic(root)
    assert vs.sync(root)["written"] == len(chunks)
    allowed = {c.chunk_id: c for c in chunks}
    overlaps = []
    for q in _unit(np.random.default_rng(1), 20):
        for scopes, iq in [({"public"}, False), ({"public", "internal"}, False),
                           (set(SCOPES), True)]:
            got = vs.dense_search(root, q, 20, frozenset(scopes), iq)
            want = _faiss_ids(chunks, vectors, q, 20, scopes, iq)
            assert len(got) == len(want)                       # filters never shorten the list
            assert all(allowed[c].scope in scopes and (iq or not allowed[c].quarantined)
                       for c, _ in got)                        # and never let a chunk through
            assert [s for _, s in got] == sorted((s for _, s in got), reverse=True)
            overlaps.append(len({c for c, _ in got} & set(want)) / len(want))
    assert np.mean(overlaps) >= 0.98                           # HNSW is approximate, barely


def test_sync_writes_only_what_changed(pg, tmp_path) -> None:
    root = tmp_path / "idx"
    chunks, vectors = _synthetic(root)
    vs.sync(root)
    # One chunk's text and vector change, one flips to quarantined, one is removed.
    chunks[5] = chunks[5].model_copy(update={"text": "revised"})
    vectors[5] = _unit(np.random.default_rng(9), 1)[0]
    flip = next(i for i, c in enumerate(chunks) if i > 250 and not c.quarantined)
    chunks[flip] = chunks[flip].model_copy(update={"quarantined": True})
    flip_id, flip_vec = chunks[flip].chunk_id, vectors[flip].copy()
    del chunks[200]
    vectors = np.delete(vectors, 200, axis=0)
    _write_index(root, chunks, vectors)

    report = vs.sync(root)
    # Rows after the removed one moved up a place: metadata updates, no vectors rewritten.
    assert report["mode"] == "incremental"
    assert (report["written"], report["removed"]) == (1, 1)
    assert report["updated"] == len(chunks) - 200
    assert vs.sync(root)["mode"] == "unchanged"

    # The newly quarantined chunk is its own nearest neighbour, and is no longer returned.
    assert flip_id not in {c for c, _ in vs.dense_search(
        root, flip_vec, 10, frozenset(SCOPES), False)}
    assert vs.dense_search(root, flip_vec, 1, frozenset(SCOPES), True)[0][0] == flip_id
    assert vs.dense_search(root, vectors[5], 1, frozenset(SCOPES), True)[0][0] \
        == chunks[5].chunk_id


def test_a_collection_from_another_build_is_refused(pg, tmp_path, monkeypatch) -> None:
    root = tmp_path / "idx"
    chunks, vectors = _synthetic(root)
    monkeypatch.setattr(settings, "vector_store", "pgvector")
    with pytest.raises(vs.VectorStoreError, match="no pgvector collection"):
        retrieve._load_index(root)
    vs.sync(root)
    assert retrieve._load_index(root).faiss is None            # FAISS isn't loaded at all
    chunks[0] = chunks[0].model_copy(update={"quarantined": not chunks[0].quarantined})
    _write_index(root, chunks, vectors)                        # rebuilt, not re-synced
    retrieve._load_index_at.cache_clear()
    vs.close()
    with pytest.raises(vs.VectorStoreError, match="different build"):
        retrieve._load_index(root)


@pytest.mark.skipif(not IndexPaths(INDEXES_DIR).faiss.exists(), reason="needs the real index")
def test_the_pipeline_returns_the_same_answers_from_pgvector(pg, monkeypatch) -> None:
    question = "What does the HTTP 429 status code mean?"
    faiss_top = [r.chunk_id for r in retrieve.hybrid_retrieve(question)]
    assert vs.sync(INDEXES_DIR)["rows"] > 0
    monkeypatch.setattr(settings, "vector_store", "pgvector")
    retrieve._load_index_at.cache_clear()
    dense_pg = retrieve.dense_search(question)
    assert len(dense_pg) == settings.dense_top_n
    assert [r.chunk_id for r in retrieve.hybrid_retrieve(question)] == faiss_top


def test_an_unknown_vector_store_is_an_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "vector_store", "chroma")
    retrieve._load_index_at.cache_clear()
    with pytest.raises(ValueError, match="unknown vector_store"):
        retrieve._load_index(tmp_path)


def test_messages_never_show_the_password() -> None:
    assert vs.redacted("postgresql://u:s3cret@db:5432/x") == "postgresql://u:***@db:5432/x"
    assert vs.redacted("postgresql://db/x") == "postgresql://db/x"
