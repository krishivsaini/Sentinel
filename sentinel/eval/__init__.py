"""Evaluation subpackage (§12) — engineering-signal #2, the project's headline.

    run_eval.py    Ragas pipeline over ground_truth.jsonl (faithfulness/relevance/recall)
    attribution.py retrieval-failure vs generation-failure classifier
    store.py       persist EvalResult to SQLite (git-SHA-keyed) + export JSON for dashboard

--- ragas <-> langchain-community 0.4.x compatibility shim ---
ragas 0.4.x hard-imports `ChatVertexAI` from `langchain_community.chat_models.vertexai` at
module load (an isinstance target in its completion-count optimization). langchain-community
0.4.x — the version that matches our langchain-v1 retrieval stack — dropped that deprecated
path (ChatVertexAI now lives in langchain-google-vertexai). We use Gemini via
langchain-google-genai and never touch Vertex, so we register a stub module here BEFORE any
ragas import. This is a targeted, reproducible fix (lives in our code, not site-packages) that
only satisfies an unused import; it changes no evaluation behaviour. Importing anything from
`sentinel.eval` runs this first, so every submodule that imports ragas is safe.
"""

from __future__ import annotations

import sys
import types

_VERTEX_SHIM = "langchain_community.chat_models.vertexai"
if _VERTEX_SHIM not in sys.modules:
    try:  # if a real one is importable (older lc-community), prefer it and don't shim.
        __import__(_VERTEX_SHIM)
    except ModuleNotFoundError:
        _stub = types.ModuleType(_VERTEX_SHIM)

        class ChatVertexAI:  # noqa: D401 - unused placeholder; never instantiated on our path
            """Stub for ragas' isinstance check; we run Gemini, never Vertex."""

        _stub.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
        sys.modules[_VERTEX_SHIM] = _stub
