"""Security guardrails (v2) — each one measured by sentinel.eval.run_guardrails, not just present.

    injection.py   screen documents for prompt injection at index build; quarantine what's flagged
    pii.py         detect + redact PII in retrieved context (before generation) and in answers
    output.py      validate an answer before it's returned: citations, links, leaked canaries

Retrieval-scope enforcement lives in sentinel.retrieve (Access), because it has to act inside the
first-stage searches rather than after them.
"""
