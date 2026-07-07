"""Honest per-stage latency benchmark (§11, NFR-3) — Phase 2, Day 7.

Measures retrieve / rerank / generate latency PER STAGE, never blended. The two local stages
(retrieve, rerank) are free and fast, so they get a few hundred samples; generation hits the
Gemini free tier (~10 RPM), so it gets a smaller, honestly-reported sample. Models are warmed
first so no one-time load/first-inference cost leaks into the timings.

Writes data/latency_report.json (committable artifact for the README) and prints a table.

Run:  python scripts/bench_latency.py                       # 200 local, 25 generate
      python scripts/bench_latency.py --local-iters 120 --gen-samples 15
      python scripts/bench_latency.py --no-generate         # local stages only (no API)
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone

from sentinel.config import settings

# Diverse questions spanning the corpus clusters so the latency distribution is realistic
# (chunk sizes, rerank pool contents, and answer lengths vary by topic).
QUERIES = [
    "Which HTTP status code indicates the client has sent too many requests?",
    "What header field tells a client how long to wait before retrying?",
    "How does HTTP caching decide whether a stored response is fresh?",
    "What is the purpose of the HTTP PATCH method?",
    "How does HPACK compress HTTP/2 header fields?",
    "What is the QPACK encoder stream used for in HTTP/3?",
    "What does the 308 Permanent Redirect status code mean?",
    "How is a URI's authority component structured?",
    "What characters are allowed in a percent-encoded URI?",
    "How does a URI Template expand a simple string variable?",
    "What is the SameSite attribute of a cookie?",
    "How does the WebSocket opening handshake work?",
    "Which TLS 1.3 handshake message carries the server certificate?",
    "What is the purpose of the TLS ALPN extension?",
    "How does Server Name Indication work in TLS?",
    "What is a QUIC stateless reset token?",
    "How does QUIC provide loss detection?",
    "What is the TCP maximum segment lifetime?",
    "How does the TCP three-way handshake establish a connection?",
    "What is the OAuth 2.0 authorization code grant flow?",
    "What does the invalid_grant error mean in OAuth 2.0?",
    "How does PKCE protect the authorization code flow?",
    "What is the OAuth 2.0 device authorization grant?",
    "How is a Bearer token presented in an HTTP request?",
    "What are the registered claims in a JSON Web Token?",
    "How is a JWS signature computed and encoded?",
    "What is a JSON Web Key and what fields does it contain?",
    "How does base64url encoding differ from standard base64?",
    "What is the format of an Internet timestamp?",
    "How does a DNS resolver look up an A record?",
    "What is DNS over HTTPS and how does it frame queries?",
    "What does HSTS protect against?",
]


def _summary(name: str, samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)

    def pct(p: float) -> float:
        if not s:
            return 0.0
        k = (len(s) - 1) * p / 100.0
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return {
        "stage": name,
        "n": len(s),
        "mean_ms": round(statistics.fmean(s), 1) if s else 0.0,
        "p50_ms": round(pct(50), 1),
        "p95_ms": round(pct(95), 1),
        "p99_ms": round(pct(99), 1),
        "min_ms": round(s[0], 1) if s else 0.0,
        "max_ms": round(s[-1], 1) if s else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-stage latency benchmark.")
    ap.add_argument("--local-iters", type=int, default=200, help="retrieve+rerank samples")
    ap.add_argument("--gen-samples", type=int, default=20, help="generation samples (API-bound)")
    ap.add_argument("--gen-spacing", type=float, default=3.0, help="seconds between gen calls (RPM)")
    ap.add_argument("--gen-timeout", type=float, default=25.0, help="per-call timeout; skip if over")
    ap.add_argument("--no-generate", action="store_true", help="skip the API-bound generate stage")
    ap.add_argument("--out", default="data/latency_report.json")
    args = ap.parse_args()

    from sentinel.generate import collect_answer
    from sentinel.retrieve import _embedder, hybrid_retrieve_timed, retrieve, warmup

    print("warming up (load + first inference)...")
    warmup()
    device = str(getattr(_embedder(), "device", "unknown"))

    # --- local stages: retrieve + rerank (free, a few hundred samples) ---
    retrieve_ms: list[float] = []
    rerank_ms: list[float] = []
    print(f"local stages: {args.local_iters} iterations...")
    t0 = time.perf_counter()
    for i in range(args.local_iters):
        _, r_ms, rr_ms = hybrid_retrieve_timed(QUERIES[i % len(QUERIES)])
        retrieve_ms.append(r_ms)
        rerank_ms.append(rr_ms)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.local_iters}  ({time.perf_counter() - t0:.0f}s)")

    from pathlib import Path

    out_path = Path(args.out)

    def write_report(stages: list[dict], gen_failed: int) -> None:
        report = {
            "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "device": device,
            "generation_model": settings.generation_model,
            "config": {
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "dense_top_n": settings.dense_top_n,
                "sparse_top_n": settings.sparse_top_n,
                "rrf_k": settings.rrf_k,
                "rerank_top_k": settings.rerank_top_k,
            },
            "generate_throttled_skipped": gen_failed,
            "note": "Per-stage, never blended (NFR-3). retrieve/rerank are local; generate is "
                    "Gemini free-tier measured with retries OFF (throttled calls are skipped, "
                    "not backoff-inflated). Measured on the platform/device above.",
            "stages": stages,
        }
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def print_table(stages: list[dict]) -> None:
        print(f"\n{'stage':<10} {'n':>4} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8} {'max':>9}")
        for st in stages:
            print(f"{st['stage']:<10} {st['n']:>4} {st['p50_ms']:>7.0f}m {st['p95_ms']:>7.0f}m "
                  f"{st['p99_ms']:>7.0f}m {st['mean_ms']:>7.0f}m {st['max_ms']:>8.0f}m")

    # Persist the local stages immediately so an API hiccup can't lose them.
    stages = [_summary("retrieve", retrieve_ms), _summary("rerank", rerank_ms)]
    write_report(stages, gen_failed=0)

    # --- generate stage: API-bound; retries OFF, spaced out, per-sample fault-tolerant ---
    gen_failed = 0
    if not args.no_generate:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

        generate_ms: list[float] = []
        print(f"generate stage: up to {args.gen_samples} samples (Gemini free tier)...")
        with ThreadPoolExecutor(max_workers=1) as pool:
            for i in range(args.gen_samples):
                q = QUERIES[i % len(QUERIES)]
                # Hard timeout so a throttled call (SDK transport retries for minutes) can't
                # hang or pollute the sample — it's skipped instead.
                fut = pool.submit(lambda qq=q: collect_answer(qq, retrieve(qq), retry=False))
                try:
                    _, _, g_ms = fut.result(timeout=args.gen_timeout)
                    generate_ms.append(g_ms)
                    print(f"  {i + 1}/{args.gen_samples}  {g_ms:.0f}ms")
                except (FutureTimeout, Exception) as exc:  # throttle/disconnect/timeout -> skip
                    gen_failed += 1
                    print(f"  {i + 1}/{args.gen_samples}  SKIP ({type(exc).__name__})")
                time.sleep(args.gen_spacing)  # stay under ~10 RPM
        if generate_ms:
            stages.append(_summary("generate", generate_ms))
        write_report(stages, gen_failed)

    print_table(stages)
    print(f"\ngenerate samples skipped (throttled): {gen_failed}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
