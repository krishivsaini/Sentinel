"""Sentinel — production hybrid-retrieval RAG service with a Ragas eval gate in CI.

Package layout (see docs/architecture.md):
    config.py           single source of truth for knobs, model IDs, thresholds, paths
    schema.py           Pydantic data contracts for every stage
    ingest.py           load -> chunk -> embed -> FAISS + BM25 (shared chunk IDs)
    retrieve.py         dense + sparse -> RRF fusion -> cross-encoder rerank
    generate.py         grounded generation with inline citations
    serve.py            FastAPI: /query (SSE), /healthz, JWT, rate limit
    guardrails/         injection screening, PII redaction, output validation
    eval/               Ragas pipeline, failure attribution, SQLite + JSON store
    logging_config.py   structured logging + per-stage latency capture
"""

import os
import sys

# On macOS the faiss-cpu and torch wheels each bundle their own libomp, which fails two ways:
#   1. whichever runtime initializes second aborts the process ("OMP: Error #15 ... already
#      initialized") — KMP_DUPLICATE_LIB_OK lets both coexist. Verified against exact numpy
#      search: 400 filtered + unfiltered queries interleaved with torch work, zero mismatches.
#   2. if faiss is *loaded* before torch, torch's OpenMP calls bind to faiss's runtime and the
#      embedder/reranker segfault (exit 139). Importing torch here, before any submodule can load
#      faiss, fixes the order for every process that imports sentinel first. (A script that
#      imports faiss before sentinel can't be protected from in here.)
# Linux wheels share one libgomp, so none of this runs there.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch  # noqa: F401,E402

__version__ = "0.1.0"
