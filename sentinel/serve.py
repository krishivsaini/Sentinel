"""FastAPI service (§11) — Phase 2, Day 6.

Endpoints:
    POST /query   -> Answer, SSE-streamed tokens; JWT-protected; rate-limited.
    POST /token   -> issue a JWT for the single demo user.
    GET  /healthz -> liveness.

Production hygiene: JWT auth (FR-S3), slowapi per-IP rate limit (FR-S4, test must prove it
fires), structured per-stage-latency logging (FR-S5), Pydantic validation everywhere (FR-S6).

Run:  python -m sentinel.serve   (or: uvicorn sentinel.serve:app)
"""

from __future__ import annotations


def main() -> None:
    # TODO(Phase 2, Day 6): build the FastAPI app (SSE /query, JWT, rate limit) and run uvicorn.
    raise NotImplementedError("serve.py is a Phase 2 (Day 6) stub — not yet implemented.")


if __name__ == "__main__":
    main()
