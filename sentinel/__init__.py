"""Sentinel — production hybrid-retrieval RAG service with a Ragas eval gate in CI.

Package layout (see docs/architecture.md):
    config.py           single source of truth for knobs, model IDs, thresholds, paths
    schema.py           Pydantic data contracts for every stage
    ingest.py           load -> chunk -> embed -> FAISS + BM25 (shared chunk IDs)
    retrieve.py         dense + sparse -> RRF fusion -> cross-encoder rerank
    generate.py         grounded generation with inline citations
    serve.py            FastAPI: /query (SSE), /healthz, JWT, rate limit
    eval/               Ragas pipeline, failure attribution, SQLite + JSON store
    logging_config.py   structured logging + per-stage latency capture
"""

__version__ = "0.1.0"
