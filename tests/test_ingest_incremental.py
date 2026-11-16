"""Incremental ingestion: updating an index for a changed corpus gives the index a full rebuild
would, and embeds only what changed.

Each test builds tiny indexes in tmp_path from four small real RFCs, with a stand-in embedder
(a fixed vector per text, so equal text means an equal vector and every call is counted).
Screening is replaced by a deterministic rule, so nothing calls a model or the network.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil

import numpy as np
import pytest

from sentinel import ingest
from sentinel.config import CORPUS_DIR, IndexPaths, settings

RFCS = [2119, 8174, 6585, 2818]   # small ones from the real corpus
EMBEDDED: list[int] = []


def _fake_embed(chunks):
    EMBEDDED.append(len(chunks))
    rows = [np.frombuffer(hashlib.sha256(c.text.encode()).digest()[:32], dtype=np.uint8)
            .astype("float32") for c in chunks]
    m = np.vstack(rows) if rows else np.zeros((0, 32), "float32")
    return m / np.linalg.norm(m, axis=1, keepdims=True).clip(min=1e-9)


def _fake_screen(chunks, label):
    # Quarantine anything mentioning "MUST NOT", so the flags have to be recomputed correctly.
    return [c.model_copy(update={"quarantined": "MUST NOT" in c.text}) for c in chunks]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    src = tmp_path / "corpus"
    src.mkdir()
    for n in RFCS:
        shutil.copy(CORPUS_DIR / f"rfc{n}.txt", src / f"rfc{n}.txt")
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(ingest, "CORPUS_DIR", src)
    monkeypatch.setattr(ingest, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(ingest, "_embed", _fake_embed)
    monkeypatch.setattr(ingest, "_screen", _fake_screen)
    EMBEDDED.clear()

    def use(numbers):
        manifest.write_text(json.dumps({
            "source_base_url": "https://www.rfc-editor.org/rfc/",
            "rfcs": [{"number": n, "title": f"RFC {n}", "cluster": "x"} for n in numbers]}))
    use(RFCS[:3])
    return tmp_path, src, use


def _snapshot(index_dir):
    import faiss
    paths = IndexPaths(index_dir)
    index = faiss.read_index(str(paths.faiss))
    with paths.bm25.open("rb") as fh:
        bm25 = pickle.load(fh)
    meta = json.loads(paths.meta.read_text())
    return {
        "chunks": paths.chunks.read_text(),
        "vectors": index.reconstruct_n(0, index.ntotal),
        "bm25_ids": bm25["chunk_ids"],
        "bm25_idf": bm25["bm25"].idf,
        "meta": {k: v for k, v in meta.items() if k != "built_utc"},
    }


def _assert_same(a, b):
    assert a["chunks"] == b["chunks"]
    np.testing.assert_array_equal(a["vectors"], b["vectors"])
    assert a["bm25_ids"] == b["bm25_ids"] and a["bm25_idf"] == b["bm25_idf"]
    assert a["meta"] == b["meta"]


def _full(tmp_path, name):
    ingest.ingest(index_dir=tmp_path / name, screen=True)
    return _snapshot(tmp_path / name)


def test_an_added_rfc_is_the_only_one_embedded_and_matches_a_full_build(corpus) -> None:
    tmp, _, use = corpus
    ingest.ingest(index_dir=tmp / "inc", screen=True)
    use(RFCS)
    EMBEDDED.clear()
    report = ingest.ingest_incremental(index_dir=tmp / "inc", screen=True)
    added_chunks = sum(c.doc_id == "rfc2818" for c in ingest.chunk_documents(
        ingest.load_source_docs(ingest.load_manifest())))
    assert report["mode"] == "incremental" and report["added"] == ["rfc2818"]
    assert EMBEDDED == [added_chunks] and report["embedded"] == added_chunks
    _assert_same(_snapshot(tmp / "inc"), _full(tmp, "full"))
    assert not (tmp / "inc.building").exists() and not (tmp / "inc.old").exists()


def test_a_changed_rfc_is_re_embedded_and_a_removed_one_dropped(corpus) -> None:
    tmp, src, use = corpus
    ingest.ingest(index_dir=tmp / "inc", screen=True)
    # A revised text for one RFC (an erratum, say), and another taken out of the manifest.
    path = src / "rfc8174.txt"
    path.write_text(path.read_text() + "\n\nThe keywords MUST NOT be read in lowercase.\n")
    use([2119, 8174])
    EMBEDDED.clear()
    report = ingest.ingest_incremental(index_dir=tmp / "inc", screen=True)
    assert report["updated"] == ["rfc8174"] and report["removed"] == ["rfc6585"]
    n_8174 = sum(c.doc_id == "rfc8174" for c in ingest.chunk_documents(
        ingest.load_source_docs(ingest.load_manifest())))
    assert EMBEDDED == [n_8174]
    inc = _snapshot(tmp / "inc")
    assert "rfc6585#" not in inc["chunks"]
    _assert_same(inc, _full(tmp, "full"))


def test_an_unchanged_corpus_is_left_alone(corpus) -> None:
    tmp, _, _ = corpus
    ingest.ingest(index_dir=tmp / "inc", screen=True)
    before = IndexPaths(tmp / "inc").faiss.stat().st_mtime_ns
    EMBEDDED.clear()
    assert ingest.ingest_incremental(index_dir=tmp / "inc", screen=True)["mode"] == "unchanged"
    assert EMBEDDED == [] and IndexPaths(tmp / "inc").faiss.stat().st_mtime_ns == before


def test_a_change_in_chunking_forces_a_full_rebuild(corpus, monkeypatch) -> None:
    tmp, _, _ = corpus
    ingest.ingest(index_dir=tmp / "inc", screen=True)
    monkeypatch.setattr(settings, "chunk_tokens", settings.chunk_tokens // 2)
    report = ingest.ingest_incremental(index_dir=tmp / "inc", screen=True)
    assert report["mode"] == "full" and "chunk_tokens" in report["reason"]
    _assert_same(_snapshot(tmp / "inc"), _full(tmp, "full"))


def test_an_index_without_document_hashes_is_rebuilt_in_full(corpus) -> None:
    tmp, _, _ = corpus
    ingest.ingest(index_dir=tmp / "inc", screen=True)
    meta_path = IndexPaths(tmp / "inc").meta
    meta = json.loads(meta_path.read_text())
    del meta["doc_sha256"]                       # an index built before incremental ingestion
    meta_path.write_text(json.dumps(meta))
    assert ingest.ingest_incremental(index_dir=tmp / "inc", screen=True)["mode"] == "full"
