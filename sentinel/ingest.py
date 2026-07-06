"""Ingestion pipeline (§8) — Phase 1, Day 2.

load source docs -> clean/normalize -> chunk (~512 tok, ~12% overlap)
-> embed (local bge model) -> build FAISS dense index
-> build BM25 index over the SAME chunks (shared chunk IDs) -> persist both + metadata.

Requirements: idempotent (re-run rebuilds cleanly, FR-I3); dense + sparse indexes share
identical chunk_ids (FR-I2 — both are built over one canonical, ordered chunk list, so row i
of FAISS and doc i of BM25 both map to chunks[i]); the chunking strategy + models are recorded
on the index (FR-I4) in indexes/meta.json.

Persisted under indexes/ (gitignored):
  chunks.jsonl   canonical ordered Chunk store — the single source of truth for chunk IDs
  faiss/index.faiss   IndexFlatIP over L2-normalized bge embeddings (row order == chunks.jsonl)
  bm25.pkl       pickled BM25Okapi + chunk_ids (same order) for the sparse index
  meta.json      strategy, params, models, counts, corpus provenance, build timestamp

Run:  python -m sentinel.ingest              # full rebuild
      python -m sentinel.ingest --limit 3    # first 3 RFCs (fast smoke test)
      python -m sentinel.ingest --dry-run    # clean + chunk only, no embed/index
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import time
from hashlib import sha256

from sentinel.config import (
    BM25_PATH,
    CHUNKS_PATH,
    CORPUS_DIR,
    DATA_DIR,
    FAISS_DIR,
    INDEXES_DIR,
    settings,
)
from sentinel.schema import Chunk, SourceDoc

MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"
META_PATH = INDEXES_DIR / "meta.json"
TIKTOKEN_ENCODING = "cl100k_base"  # only used to *measure* chunk length in tokens
MIN_CHUNK_TOKENS = 5               # drop degenerate fragments (stray letters, page artifacts)

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
                )
            )
    return chunks


def _write_chunks(chunks: list[Chunk]) -> None:
    with CHUNKS_PATH.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")


def _build_faiss(chunks: list[Chunk]) -> int:
    """Embed chunk texts with the local bge model and build a cosine (IP) FAISS index.
    Returns the embedding dimension. Row order matches `chunks` exactly."""
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)
    # bge-v1.5 wants no instruction prefix on the passage side (only queries get one).
    embeddings = model.encode(
        [c.text for c in chunks],
        batch_size=64,
        normalize_embeddings=True,   # unit vectors -> inner product == cosine
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_DIR / "index.faiss"))
    return dim


def _build_bm25(chunks: list[Chunk]) -> None:
    """Build a BM25 index over the SAME ordered chunks; persist it with its chunk_id order."""
    from rank_bm25 import BM25Okapi

    tokenized = [bm25_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with BM25_PATH.open("wb") as fh:
        pickle.dump({"bm25": bm25, "chunk_ids": [c.chunk_id for c in chunks]}, fh)


def _write_meta(docs: list[SourceDoc], chunks: list[Chunk], dim: int | None) -> None:
    manifest_sha = sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    meta = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunk_strategy": settings.chunk_strategy,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap_pct": settings.chunk_overlap_pct,
        "embedding_model": settings.embedding_model,
        "embedding_dim": dim,
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "doc_ids": [d.doc_id for d in docs],
        "corpus_manifest_sha256": manifest_sha,
        "bm25_tokenizer": "hyphen_underscore_lower",
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def ingest(limit: int | None = None, dry_run: bool = False) -> None:
    t0 = time.perf_counter()

    # Idempotent (FR-I3): wipe and rebuild the whole indexes/ tree from scratch.
    if INDEXES_DIR.exists():
        shutil.rmtree(INDEXES_DIR)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

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

    _write_chunks(chunks)
    print(f"[write] {CHUNKS_PATH.relative_to(INDEXES_DIR.parent)} ({len(chunks)} chunks)")

    dim = _build_faiss(chunks)
    print(f"[faiss] IndexFlatIP dim={dim} vectors={len(chunks)}")

    _build_bm25(chunks)
    print(f"[bm25]  BM25Okapi over {len(chunks)} chunks")

    _write_meta(docs, chunks, dim)
    print(f"[meta]  {META_PATH.relative_to(INDEXES_DIR.parent)}")
    print(f"[done]  {time.perf_counter() - t0:.1f}s — {len(docs)} docs, {len(chunks)} chunks")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the FAISS + BM25 indexes from the corpus.")
    ap.add_argument("--limit", type=int, default=None, help="ingest only the first N RFCs")
    ap.add_argument("--dry-run", action="store_true", help="clean + chunk only; no embed/index")
    args = ap.parse_args()
    ingest(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
