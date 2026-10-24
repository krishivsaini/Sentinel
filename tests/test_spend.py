"""Spend meter tests — hermetic: fake models, no API calls."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from sentinel.config import settings
from sentinel.retry import is_transient
from sentinel.spend import BudgetExceeded, SpendCallback, meter


@pytest.fixture(autouse=True)
def fresh_meter(monkeypatch):
    monkeypatch.setattr(settings, "spend_limit_usd", None)   # whatever the environment sets
    meter.reset()
    yield
    meter.reset()


def _fake(model: str) -> GenericFakeChatModel:
    reply = AIMessage(content="ok", usage_metadata={
        "input_tokens": 2_000, "output_tokens": 500, "total_tokens": 2_500})
    return GenericFakeChatModel(messages=iter([reply] * 3), callbacks=[SpendCallback(model)])


def test_calls_are_priced_per_model() -> None:
    meter.record("openai/gpt-oss-20b", 1_000_000, 0)
    meter.record("openai/gpt-oss-120b", 0, 1_000_000)
    assert meter.summary()["usd"] == pytest.approx(0.075 + 0.60)


def test_unknown_models_are_priced_at_the_dearest_rate() -> None:
    meter.record("some/new-model", 1_000_000, 1_000_000)
    assert meter.usd == pytest.approx(0.15 + 0.60)


def test_langchain_calls_are_metered() -> None:
    _fake("openai/gpt-oss-20b").invoke("hi")
    assert meter.usd == pytest.approx((2_000 * 0.075 + 500 * 0.30) / 1e6)
    assert meter.summary()["by_model"]["openai/gpt-oss-20b"]["calls"] == 1


def test_no_call_starts_past_the_limit(monkeypatch) -> None:
    monkeypatch.setattr(settings, "spend_limit_usd", 0.0001)
    model = _fake("openai/gpt-oss-120b")
    model.invoke("first")                      # under the limit: sent, and it crosses it
    with pytest.raises(BudgetExceeded):
        model.invoke("second")
    assert meter.summary()["by_model"]["openai/gpt-oss-120b"]["calls"] == 1


def test_guardrail_classifier_calls_are_refused_past_the_limit(monkeypatch) -> None:
    from sentinel.guardrails import _groq

    monkeypatch.setattr(settings, "spend_limit_usd", 0.0)
    monkeypatch.setattr(_groq, "_client", lambda: pytest.fail("a request was sent"))
    with pytest.raises(BudgetExceeded):
        _groq.chat("openai/gpt-oss-safeguard-20b", [{"role": "user", "content": "x"}])


def test_budget_errors_are_never_retried() -> None:
    assert not is_transient(BudgetExceeded("spend limit reached: $0.2500 of $0.25 spent"))


def test_streamed_calls_are_metered_from_the_final_chunk() -> None:
    """Groq reports a stream's usage on its last chunk; generation streams."""
    from collections.abc import Iterator

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    class Streaming(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "streaming"

        def _generate(self, *args, **kwargs):
            raise NotImplementedError

        def _stream(self, *args, **kwargs) -> Iterator[ChatGenerationChunk]:
            for text in ("The 429 ", "status code"):
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))
            yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata={
                "input_tokens": 1_000, "output_tokens": 100, "total_tokens": 1_100}))

    model = Streaming(callbacks=[SpendCallback("openai/gpt-oss-20b")])
    assert "".join(c.content for c in model.stream("q")) == "The 429 status code"
    assert meter.usd == pytest.approx((1_000 * 0.075 + 100 * 0.30) / 1e6)
