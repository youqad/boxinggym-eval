"""Tests for AsyncLiteLLM Responses API parameter handling."""

import asyncio

import litellm
from boxing_gym.agents.async_litellm_wrapper import AsyncLiteLLM


class DummyResponses:
    """Minimal Responses-style object for extract_text()."""

    output = [{"type": "message", "content": [{"text": "ok"}]}]
    usage = {"input_tokens": 1, "output_tokens": 2}


def _run(coro):
    """Run an async coroutine without pytest-asyncio."""
    return asyncio.run(coro)


def test_responses_api_honors_max_output_tokens(monkeypatch):
    monkeypatch.setenv("BOXINGGYM_FAKE_LLM", "0")
    captured = {}

    async def fake_aresponses(**kwargs):
        captured.update(kwargs)
        return DummyResponses()

    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

    llm = AsyncLiteLLM(model_name="gpt-5.1", max_tokens=2048)
    _run(llm([{"role": "user", "content": "hi"}], max_output_tokens=77))

    assert captured["max_output_tokens"] == 77


def test_responses_api_falls_back_to_max_tokens(monkeypatch):
    monkeypatch.setenv("BOXINGGYM_FAKE_LLM", "0")
    captured = {}

    async def fake_aresponses(**kwargs):
        captured.update(kwargs)
        return DummyResponses()

    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

    llm = AsyncLiteLLM(model_name="gpt-5.1", max_tokens=2048)
    _run(llm([{"role": "user", "content": "hi"}], max_tokens=123))

    assert captured["max_output_tokens"] == 123


def test_responses_api_defaults_to_instance_max_tokens(monkeypatch):
    monkeypatch.setenv("BOXINGGYM_FAKE_LLM", "0")
    captured = {}

    async def fake_aresponses(**kwargs):
        captured.update(kwargs)
        return DummyResponses()

    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

    llm = AsyncLiteLLM(model_name="gpt-5.1", max_tokens=4096)
    _run(llm([{"role": "user", "content": "hi"}]))

    assert captured["max_output_tokens"] == llm.max_tokens
