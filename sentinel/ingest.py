"""Ingestion pipeline (§8) — Phase 1, Day 2.

load source docs -> clean/normalize -> chunk (~512 tok, ~12% overlap)
-> embed (local bge model) -> build FAISS dense index
-> build BM25 index over the SAME chunks (shared chunk IDs) -> persist both + metadata.

Requirements: idempotent (re-run rebuilds cleanly, FR-I3); dense + sparse indexes share
identical chunk_ids (FR-I2 — both are built over one canonical, ordered chunk list, so row i
of FAISS and doc i of BM25 both map to chunks[i]); the chunking strategy + models are recorded
on the index (FR-I4) in indexes/meta.json.

Persisted under an index directory (gitignored; layout in config.IndexPaths):
  chunks.jsonl   canonical ordered Chunk store — the single source of truth for chunk IDs
  faiss/index.faiss   IndexFlatIP over L2-normalized bge embeddings (row order == chunks.jsonl)
  bm25.pkl       pickled BM25Okapi + chunk_ids (same order) for the sparse index
  meta.json      strategy, params, models, counts, corpus provenance, build timestamp

Two indexes are built, and they never share a directory:
  indexes/              the real RFC corpus — what the service and the normal eval read
  indexes_adversarial/  the same RFC chunks plus the labelled adversarial corpus, for the
                        guardrail eval only. Built FROM indexes/: the RFC chunks and their
                        vectors are copied, not re-embedded, so they are identical in both and
                        only the adversarial passages are embedded.

Run:  python -m sentinel.ingest                  # full rebuild of indexes/
      python -m sentinel.ingest --incremental    # only new/changed RFCs (same result as full)
      python -m sentinel.ingest --limit 3        # first 3 RFCs (fast smoke test)
      python -m sentinel.ingest --dry-run        # clean + chunk only, no embed/index
      python -m sentinel.ingest --adversarial    # build indexes_adversarial/ from indexes/
      python -m sentinel.ingest --no-screen      # skip prompt-injection screening

Every chunk is screened for prompt injection before indexing (sentinel.guardrails.injection);
flagged chunks are written with quarantined=true and never retrieved. Verdicts come from the
committed cache when the text is unchanged, so an unchanged corpus rebuilds without API calls.

Incremental (`--incremental`): meta.json records a hash of every document. A rebuild compares
them with the corpus, then chunks, screens and embeds only new or changed documents; the other
chunks and their vectors are carried over, removed documents dropped, and BM25 rebuilt (its
statistics are global). The result is the index a full rebuild would produce, written beside
the old one and swapped in only when complete. Any change to how chunks are made or embedded
(the chunking settings, the embedding model, CHUNKER_VERSION) forces a full rebuild.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import time
from collections import Counter
from hashlib import sha256
from pathlib import Path

from sentinel.config import (
    ADVERSARIAL_DIR,
    ADVERSARIAL_INDEX_DIR,
    BIPIA_DIR,
    BIPIA_INDEX_DIR,
    CORPUS_DIR,
    DATA_DIR,
    INDEXES_DIR,
    IndexPaths,
    settings,
)
from sentinel.schema import Chunk, SourceDoc

MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"
ADVERSARIAL_CORPUS_PATH = ADVERSARIAL_DIR / "corpus.jsonl"
BIPIA_CORPUS_PATH = BIPIA_DIR / "corpus.jsonl"
TIKTOKEN_ENCODING = "cl100k_base"  # only used to *measure* chunk length in tokens
MIN_CHUNK_TOKENS = 5               # drop degenerate fragments (stray letters, page artifacts)
# Bump when cleaning or chunking code changes in a way the settings don't capture: an index
# built by another version is then rebuilt in full rather than updated incrementally.
CHUNKER_VERSION = 1

# --- RFC boilerplate patterns (two formats: modern unpaginated + legacy paginated) ---
# Legacy page footer, e.g. "Mockapetris                              [Page 1]"
_FOOTER_RE = re.compile(r"^.*\[Page \d+\]\s*$")
# Legacy running header, e.g. "RFC 7519        JSON Web Token (JWT)        May 2015"
_HEADER_RE = re.compile(r"^RFC \d{3,5}\s{2,}.*\b(?:19|20)\d{2}\s*$")
# 3+ blank lines (left behind by stripped page breaks) -> collapse to a paragraph break.
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
# BM25 token: keep hyphenated/underscored identifiers whole (e.g. "Retry-After",
# "SETTINGS_MAX_CONCURRENT_STREAMS", "429") — that exact-match power is why BM25 earns
# its place (FR-R7). Query-time tokenization MUST reuse this function.
_BM25_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def bm25_tokenize(text: str) -> list[str]:
    """Tokenizer shared by ingest (index build) and retrieve (query) so terms align."""
    return _BM25_TOKEN_RE.findall(text.lower())


def clean_rfc_text(text: str) -> str:
    """Strip page furniture (form feeds, running headers, [Page N] footers) from both the
    modern unpaginated and legacy paginated RFC text formats. Content is otherwise verbatim."""
    text = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "")
    kept = [
        line for line in text.split("\n")
        if not _FOOTER_RE.match(line) and not _HEADER_RE.match(line)
    ]
    cleaned = "\n".join(line.rstrip() for line in kept)
    return _MULTI_BLANK_RE.sub("\n\n", cleaned).strip() + "\n"


def load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_source_docs(manifest: dict, limit: int | None = None) -> list[SourceDoc]:
    base = manifest["source_base_url"]
    entries = manifest["rfcs"][:limit] if limit else manifest["rfcs"]
    docs: list[SourceDoc] = []
    for entry in entries:
        number = entry["number"]
        path = CORPUS_DIR / f"rfc{number}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run scripts/fetch_corpus.py first")
        docs.append(
            SourceDoc(
                doc_id=f"rfc{number}",
                title=entry["title"],
                source_uri=f"{base}rfc{number}.txt",
                text=clean_rfc_text(path.read_text(encoding="utf-8", errors="replace")),
            )
        )
    return docs


def _build_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    overlap = int(settings.chunk_tokens * settings.chunk_overlap_pct)
    if settings.chunk_strategy != "recursive":
        raise NotImplementedError(
            f"chunk_strategy={settings.chunk_strategy!r} not implemented; use 'recursive' (v1)."
        )
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TIKTOKEN_ENCODING,
        chunk_size=settings.chunk_tokens,
        chunk_overlap=overlap,
    )


def chunk_documents(docs: list[SourceDoc]) -> list[Chunk]:
    """Split every doc into ~512-token chunks with deterministic, globally-unique chunk_ids
    of the form '<doc_id>#<ordinal>' (e.g. 'rfc9110#0007')."""
    import tiktoken

    splitter = _build_splitter()
    enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    chunks: list[Chunk] = []
    for doc in docs:
        for ordinal, piece in enumerate(splitter.split_text(doc.text)):
            piece = piece.strip()
            n_tok = len(enc.encode(piece)) if piece else 0
            if n_tok < MIN_CHUNK_TOKENS:  # drop empty/degenerate fragments; ordinals may gap
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{ordinal:04d}",
                    doc_id=doc.doc_id,
                    text=piece,
                    token_count=n_tok,
                    ordinal=ordinal,
                    scope=doc.scope,
                )
            )
    return chunks


def _write_chunks(chunks: list[Chunk], paths: IndexPaths) -> None:
    with paths.chunks.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")


def _embed(chunks: list[Chunk]):
    """L2-normalized bge embeddings, one row per chunk, in chunk order (float32 ndarray)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model, device=settings.torch_device)
    # bge-v1.5 wants no instruction prefix on the passage side (only queries get one).
    return model.encode(
        [c.text for c in chunks],
        batch_size=64,
        normalize_embeddings=True,   # unit vectors -> inner product == cosine
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")


def _write_faiss(embeddings, paths: IndexPaths) -> int:
    """Persist a cosine (IP) FAISS index over `embeddings`; returns the dimension. Row order is
    the caller's chunk order."""
    import faiss

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    paths.faiss.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(paths.faiss))
    return dim


def _build_bm25(chunks: list[Chunk], paths: IndexPaths) -> None:
    """Build a BM25 index over the SAME ordered chunks; persist it with its chunk_id order."""
    from rank_bm25 import BM25Okapi

    tokenized = [bm25_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with paths.bm25.open("wb") as fh:
        pickle.dump({"bm25": bm25, "chunk_ids": [c.chunk_id for c in chunks]}, fh)


def _write_meta(
    docs: list[SourceDoc], chunks: list[Chunk], dim: int | None, paths: IndexPaths
) -> None:
    manifest_sha = sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    meta = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **_build_settings(),
        "embedding_dim": dim,
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "doc_ids": [d.doc_id for d in docs],
        "corpus_manifest_sha256": manifest_sha,
        "scopes": dict(Counter(c.scope for c in chunks)),
        "quarantined": sum(c.quarantined for c in chunks),
        "doc_sha256": {d.doc_id: _doc_sha(d) for d in docs},
    }
    paths.meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _build_settings() -> dict:
    """Everything that decides a document's chunks and vectors. An index built with different
    values can't be updated incrementally."""
    return {
        "chunker_version": CHUNKER_VERSION,
        "chunk_strategy": settings.chunk_strategy,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap_pct": settings.chunk_overlap_pct,
        "embedding_model": settings.embedding_model,
        "bm25_tokenizer": "hyphen_underscore_lower",
    }


def _doc_sha(doc: SourceDoc) -> str:
    """Identifies a document's chunks: its cleaned text and scope (chunk ids derive from doc_id)."""
    return sha256(f"{doc.scope}\0{doc.text}".encode()).hexdigest()


def _reset_dir(root: Path) -> IndexPaths:
    """Idempotent (FR-I3): wipe and recreate one index directory, never anything outside it."""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return IndexPaths(root)


def _screen(chunks: list[Chunk], label: str) -> list[Chunk]:
    """Injection screening (v2-R1): quarantine every chunk a detector flags. Cached verdicts
    are reused; without GROQ_API_KEY, uncached chunks are an error rather than silently
    unscreened (pass --no-screen to build without screening)."""
    from sentinel.guardrails.injection import screen_chunks

    chunks, screenings = screen_chunks(chunks, offline=not settings.groq_api_key)
    released = sum(s.released for s in screenings)
    print(f"[screen] {label}: {sum(c.quarantined for c in chunks)} of {len(chunks)} quarantined"
          + (f" ({released} flagged but released by a reviewed exception)" if released else ""))
    return chunks


def ingest(
    limit: int | None = None,
    dry_run: bool = False,
    index_dir: Path = INDEXES_DIR,
    screen: bool | None = None,
) -> None:
    t0 = time.perf_counter()
    screen = settings.injection_screening if screen is None else screen

    manifest = load_manifest()
    docs = load_source_docs(manifest, limit=limit)
    print(f"[load]  {len(docs)} source docs (cleaned)")

    chunks = chunk_documents(docs)
    toks = [c.token_count for c in chunks]
    avg = sum(toks) / len(toks) if toks else 0
    print(f"[chunk] {len(chunks)} chunks | avg {avg:.0f} tok | "
          f"min {min(toks, default=0)} max {max(toks, default=0)}")

    if dry_run:
        print("[dry-run] skipping embed + index build")
        return

    if screen:
        chunks = _screen(chunks, "corpus")
    paths = _reset_dir(index_dir)
    _write_chunks(chunks, paths)
    print(f"[write] {paths.chunks} ({len(chunks)} chunks)")

    dim = _write_faiss(_embed(chunks), paths)
    print(f"[faiss] IndexFlatIP dim={dim} vectors={len(chunks)}")

    _build_bm25(chunks, paths)
    print(f"[bm25]  BM25Okapi over {len(chunks)} chunks")

    _write_meta(docs, chunks, dim, paths)
    print(f"[meta]  {paths.meta}")
    print(f"[done]  {time.perf_counter() - t0:.1f}s — {len(docs)} docs, {len(chunks)} chunks")


def ingest_incremental(index_dir: Path = INDEXES_DIR, screen: bool | None = None) -> dict:
    """Update index_dir for what changed in the corpus since it was built; see the module
    docstring. Returns what was done: {"mode": "full" | "incremental" | "unchanged", ...}."""
    import faiss
    import numpy as np

    t0 = time.perf_counter()
    screen = settings.injection_screening if screen is None else screen
    paths = IndexPaths(index_dir)
    meta = json.loads(paths.meta.read_text()) if paths.meta.exists() else {}
    stale = [k for k, v in _build_settings().items() if meta.get(k) != v]
    if not (paths.chunks.exists() and paths.faiss.exists()) or "doc_sha256" not in meta or stale:
        why = f"built with different {stale}" if meta and stale else "no incremental metadata"
        print(f"[incremental] {index_dir.name}/ {why}: full rebuild")
        ingest(index_dir=index_dir, screen=screen)
        return {"mode": "full", "reason": why}

    docs = load_source_docs(load_manifest())
    old_sha = meta["doc_sha256"]
    changed = [d for d in docs if old_sha.get(d.doc_id) != _doc_sha(d)]
    added = [d.doc_id for d in changed if d.doc_id not in old_sha]
    updated = [d.doc_id for d in changed if d.doc_id in old_sha]
    removed = sorted(set(old_sha) - {d.doc_id for d in docs})
    if not changed and not removed:
        print(f"[incremental] {index_dir.name}/ is up to date ({len(docs)} docs)")
        return {"mode": "unchanged", "added": [], "updated": [], "removed": [], "embedded": 0}

    old_chunks = [Chunk.model_validate_json(line) for line in paths.chunks.open(encoding="utf-8")]
    old_index = faiss.read_index(str(paths.faiss))
    old_vectors = old_index.reconstruct_n(0, old_index.ntotal)
    if len(old_vectors) != len(old_chunks):
        raise RuntimeError(f"{index_dir} is inconsistent — rebuild it with `sentinel.ingest`")
    kept: dict[str, list[tuple[Chunk, object]]] = {}
    for chunk, vec in zip(old_chunks, old_vectors, strict=True):
        kept.setdefault(chunk.doc_id, []).append((chunk, vec))

    fresh = chunk_documents(changed)
    fresh_vectors = _embed(fresh) if fresh else np.zeros((0, old_vectors.shape[1]), "float32")
    for chunk, vec in zip(fresh, fresh_vectors, strict=True):
        kept.setdefault(f"new:{chunk.doc_id}", []).append((chunk, vec))
    changed_ids = {d.doc_id for d in changed}
    # Corpus order, as a full rebuild has it: each changed doc's fresh chunks, others' old ones.
    rows = [row for d in docs
            for row in kept.get(f"new:{d.doc_id}" if d.doc_id in changed_ids else d.doc_id, [])]
    print(f"[incremental] +{len(added)} new, {len(updated)} changed, -{len(removed)} removed docs;"
          f" embedded {len(fresh)} chunks, kept {len(rows) - len(fresh)}")
    chunks = [c.model_copy(update={"quarantined": False}) for c, _ in rows]
    vectors = np.vstack([v for _, v in rows]).astype("float32")
    if screen:   # re-screens everything, but only new text misses the verdict cache
        chunks = _screen(chunks, "corpus")

    tmp_dir = index_dir.with_name(index_dir.name + ".building")
    tmp = _reset_dir(tmp_dir)
    _write_chunks(chunks, tmp)
    dim = _write_faiss(vectors, tmp)
    _build_bm25(chunks, tmp)
    _write_meta(docs, chunks, dim, tmp)
    old_dir = index_dir.with_name(index_dir.name + ".old")
    if old_dir.exists():
        shutil.rmtree(old_dir)
    index_dir.rename(old_dir)
    tmp_dir.rename(index_dir)
    shutil.rmtree(old_dir)
    print(f"[done]  {time.perf_counter() - t0:.1f}s — {len(docs)} docs, {len(chunks)} chunks")
    return {"mode": "incremental", "added": added, "updated": updated, "removed": removed,
            "embedded": len(fresh)}


def load_adversarial_docs() -> list[SourceDoc]:
    """The labelled adversarial corpus (data/adversarial/corpus.jsonl). Every doc is forced into
    the "adversarial" scope here, whatever its record says, so a labelling slip can't leak a
    poisoned passage into the public scope."""
    docs = []
    for line in ADVERSARIAL_CORPUS_PATH.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            docs.append(
                SourceDoc(
                    doc_id=rec["doc_id"],
                    title=rec["title"],
                    source_uri=f"adversarial://{rec['doc_id']}",
                    text=rec["text"],
                    scope="adversarial",
                )
            )
    return docs


def load_bipia_docs() -> list[SourceDoc]:
    """BIPIA's attack instructions planted in RFC passages (data/bipia/corpus.jsonl, built by
    scripts/build_bipia.py). Each doc gets a scope of its own, "bipia:<doc_id>", so a run can
    retrieve exactly one attack at a time; like the adversarial corpus, never public."""
    return [
        SourceDoc(
            doc_id=rec["doc_id"],
            title=rec["title"],
            source_uri=f"bipia://{rec['doc_id']}",
            text=rec["text"],
            scope=f"bipia:{rec['doc_id']}",
        )
        for rec in map(json.loads, filter(str.strip, BIPIA_CORPUS_PATH.open(encoding="utf-8")))
    ]


def _holds_exactly(index_dir: Path, docs: list[SourceDoc]) -> bool:
    paths = IndexPaths(index_dir)
    if not paths.chunks.exists():
        return False
    chunks = (Chunk.model_validate_json(line) for line in paths.chunks.open(encoding="utf-8"))
    indexed = {c.doc_id for c in chunks if c.scope != "public"}
    return indexed == {d.doc_id for d in docs}


def adversarial_index_is_current(index_dir: Path = ADVERSARIAL_INDEX_DIR) -> bool:
    """True when index_dir exists and holds exactly the documents of the adversarial corpus file —
    so an eval or test never runs attacks against a stale index."""
    return _holds_exactly(index_dir, load_adversarial_docs())


def bipia_index_is_current(index_dir: Path = BIPIA_INDEX_DIR) -> bool:
    return _holds_exactly(index_dir, load_bipia_docs())


def ingest_adversarial(
    base_dir: Path = INDEXES_DIR,
    out_dir: Path = ADVERSARIAL_INDEX_DIR,
    screen: bool | None = None,
    docs: list[SourceDoc] | None = None,
) -> None:
    """Build a guardrail-eval index: every chunk of the real index (copied verbatim, vectors
    and quarantine verdicts included) followed by the chunked, screened, embedded attack corpus —
    the adversarial one by default, or `docs`."""
    import faiss
    import numpy as np

    if out_dir.resolve() == base_dir.resolve():
        raise ValueError("the adversarial index must not overwrite the base index")
    t0 = time.perf_counter()
    base = IndexPaths(base_dir)
    base_chunks = [Chunk.model_validate_json(line) for line in base.chunks.open(encoding="utf-8")]
    base_index = faiss.read_index(str(base.faiss))
    base_vectors = base_index.reconstruct_n(0, base_index.ntotal)
    if len(base_vectors) != len(base_chunks):
        raise RuntimeError(f"{base_dir} is inconsistent — rebuild it with `sentinel.ingest`")

    adv_docs = load_adversarial_docs() if docs is None else docs
    adv_chunks = chunk_documents(adv_docs)
    clash = {c.chunk_id for c in adv_chunks} & {c.chunk_id for c in base_chunks}
    if clash:
        raise ValueError(f"adversarial chunk ids collide with the real corpus: {sorted(clash)}")
    print(f"[load]  {len(base_chunks)} chunks from {base_dir.name}/ + "
          f"{len(adv_docs)} adversarial docs -> {len(adv_chunks)} chunks")
    if settings.injection_screening if screen is None else screen:
        adv_chunks = _screen(adv_chunks, "adversarial")

    chunks = base_chunks + adv_chunks
    paths = _reset_dir(out_dir)
    _write_chunks(chunks, paths)
    dim = _write_faiss(np.vstack([base_vectors, _embed(adv_chunks)]), paths)
    _build_bm25(chunks, paths)
    _write_meta(load_source_docs(load_manifest()) + adv_docs, chunks, dim, paths)
    print(f"[done]  {time.perf_counter() - t0:.1f}s — {out_dir.name}/: {len(chunks)} chunks "
          f"({len(adv_chunks)} adversarial)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the FAISS + BM25 indexes from the corpus.")
    ap.add_argument("--limit", type=int, default=None, help="ingest only the first N RFCs")
    ap.add_argument("--dry-run", action="store_true", help="clean + chunk only; no embed/index")
    ap.add_argument("--adversarial", action="store_true",
                    help="build indexes_adversarial/ (real index + adversarial corpus)")
    ap.add_argument("--bipia", action="store_true",
                    help="build indexes_bipia/ (real index + the BIPIA attack set)")
    ap.add_argument("--incremental", action="store_true",
                    help="update indexes/ for new or changed RFCs only (= a full build)")
    ap.add_argument("--no-screen", action="store_true",
                    help="skip prompt-injection screening (nothing is quarantined)")
    args = ap.parse_args()
    screen = False if args.no_screen else None
    if args.adversarial:
        ingest_adversarial(screen=screen)
    elif args.bipia:
        ingest_adversarial(out_dir=BIPIA_INDEX_DIR, screen=screen, docs=load_bipia_docs())
    elif args.incremental:
        ingest_incremental(screen=screen)
    else:
        ingest(limit=args.limit, dry_run=args.dry_run, screen=screen)


if __name__ == "__main__":
    main()
