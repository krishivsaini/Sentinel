# The Sentinel API as a container (deploy/ runs it on Cloud Run).
#
# Two stages. The builder installs the locked dependencies (CPU-only PyTorch on Linux, see
# pyproject.toml), builds the service index from the committed corpus and downloads the two local
# models. Index building makes no API calls: the corpus's injection-screening verdicts are
# committed. The runtime stage copies only what serving needs — the virtualenv, the package, the
# index and the model weights — and runs offline as an unprivileged user, so a cold start loads
# from disk instead of downloading.
#
#   docker build -t sentinel-api .
#   docker run -p 8080:8080 -e PORT=8080 -e GROQ_API_KEY=... -e SENTINEL_JWT_SECRET=... sentinel-api

FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/hf
WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY sentinel/ sentinel/
RUN uv sync --frozen --no-dev

# The service index (FAISS + BM25, screened) and the embedding + rerank models.
COPY data/corpus/ data/corpus/
COPY data/corpus_manifest.json data/
COPY data/guardrails/screening_cache.jsonl data/guardrails/
RUN .venv/bin/python -m sentinel.ingest \
 && .venv/bin/python -c "from sentinel.retrieve import warmup; warmup()"


FROM python:3.11-slim
ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/app/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app
RUN useradd --system --uid 10001 --home-dir /app sentinel
COPY --from=builder --chown=sentinel /app/.venv .venv
COPY --from=builder --chown=sentinel /app/hf hf
COPY --from=builder --chown=sentinel /app/indexes indexes
COPY --chown=sentinel sentinel/ sentinel/
USER sentinel
EXPOSE 8080
CMD ["python", "-m", "sentinel.serve"]
