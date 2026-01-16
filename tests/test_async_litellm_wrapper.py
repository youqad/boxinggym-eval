"""Tests for AsyncLiteLLM Responses API parameter handling.

Phase 0 safety net: tests for call recording and fake mode behavior.
"""

import asyncio
import os
import tempfile

import litellm
import pytest
from boxing_gym.agents.async_litellm_wrapper import AsyncLiteLLM
from boxing_gym.agents.call_recorder import CallRecorder


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


class TestCallRecording:
    """Tests for LLM call recording with finally block."""

    def test_records_even_on_exception(self, monkeypatch):
        """Failed calls still get recorded."""
        monkeypatch.setenv("BOXINGGYM_FAKE_LLM", "0")

        async def failing_acompletion(**kwargs):
            raise RuntimeError("API connection failed")

        monkeypatch.setattr(litellm, "acompletion", failing_acompletion)

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = CallRecorder(run_id="test-error-recording", output_dir=tmpdir)
            llm = AsyncLiteLLM(model_name="gpt-4o", max_tokens=100)
            llm.set_call_recorder(recorder, agent_name="test_agent")

            with pytest.raises(RuntimeError, match="API connection failed"):
                _run(llm([{"role": "user", "content": "test"}]))

            calls = recorder.get_all_calls()
            assert len(calls) == 1
            assert calls[0]["error"] == "API connection failed"
            assert calls[0]["agent"] == "test_agent"
            assert calls[0]["model"] == "gpt-4o"

    def test_records_successful_call(self, monkeypatch):
        """Successful calls recorded with response."""
        monkeypatch.setenv("BOXINGGYM_FAKE_LLM", "0")

        class FakeMsg:
            def __init__(self):
                self.content = "test response"
                self.reasoning_content = None

        class FakeChoice:
            def __init__(self):
                self.message = FakeMsg()

        class FakeResp:
            def __init__(self):
                self.model = "gpt-4o"
                self.choices = [FakeChoice()]
                self.usage = {"prompt_tokens": 10, "completion_tokens": 5}

        async def fake_acompletion(**kwargs):
            return FakeResp()

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = CallRecorder(run_id="test-success-recording", output_dir=tmpdir)
            llm = AsyncLiteLLM(model_name="gpt-4o", max_tokens=100)
            llm.set_call_recorder(recorder, agent_name="scientist")

            _run(llm([{"role": "user", "content": "hello"}]))

            calls = recorder.get_all_calls()
            assert len(calls) == 1
            assert calls[0]["error"] is None
            assert calls[0]["response"] == "test response"
            assert calls[0]["prompt_tokens"] == 10
            assert calls[0]["completion_tokens"] == 5


class TestFakeMode:
    """Tests for BOXINGGYM_FAKE_LLM mode."""

    def test_fake_mode_returns_mock_response(self, monkeypatch):
        """Returns mock response, no API call."""
        monkeypatch.setenv("BOXINGGYM_FAKE_LLM", "1")

        llm = AsyncLiteLLM(model_name="gpt-4o", max_tokens=100)
        response = _run(llm([{"role": "user", "content": "test"}]))

        assert response.choices[0].message.content == "mock response"
        assert response.model == "gpt-4o"

    def test_fake_mode_consistent_across_calls(self, monkeypatch):
        """Consistent output across calls."""
        monkeypatch.setenv("BOXINGGYM_FAKE_LLM", "true")

        llm = AsyncLiteLLM(model_name="deepseek-chat", max_tokens=100)

        r1 = _run(llm([{"role": "user", "content": "first"}]))
        r2 = _run(llm([{"role": "user", "content": "second"}]))

        assert r1.choices[0].message.content == "mock response"
        assert r2.choices[0].message.content == "mock response"

    def test_fake_mode_records_calls(self, monkeypatch):
        """Calls still recorded in fake mode."""
        monkeypatch.setenv("BOXINGGYM_FAKE_LLM", "yes")

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = CallRecorder(run_id="test-fake-recording", output_dir=tmpdir)
            llm = AsyncLiteLLM(model_name="gpt-4o", max_tokens=100)
            llm.set_call_recorder(recorder, agent_name="fake_test")

            _run(llm([{"role": "user", "content": "fake test"}]))

            calls = recorder.get_all_calls()
            assert len(calls) == 1
            assert calls[0]["call_type"] == "fake"
            assert calls[0]["response"] == "mock response"
            assert calls[0]["latency_ms"] == 0.0

    def test_fake_mode_disabled_with_zero(self, monkeypatch):
        """BOXINGGYM_FAKE_LLM=0 disables fake mode."""
        monkeypatch.setenv("BOXINGGYM_FAKE_LLM", "0")
        captured = {}

        async def fake_acompletion(**kwargs):
            captured["called"] = True
            return DummyResponses()

        class FakeMsg:
            content = "real api response"
            reasoning_content = None

        class FakeChoice:
            message = FakeMsg()

        class FakeResp:
            model = "gpt-4o"
            choices = [FakeChoice()]
            usage = {"prompt_tokens": 1, "completion_tokens": 2}

        async def fake_acompletion_real(**kwargs):
            captured["called"] = True
            return FakeResp()

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion_real)

        llm = AsyncLiteLLM(model_name="gpt-4o", max_tokens=100)
        response = _run(llm([{"role": "user", "content": "test"}]))

        assert captured.get("called") is True
        assert response.choices[0].message.content == "real api response"

    def test_fake_mode_disabled_with_false(self, monkeypatch):
        """BOXINGGYM_FAKE_LLM=false disables fake mode."""
        monkeypatch.setenv("BOXINGGYM_FAKE_LLM", "false")
        captured = {}

        class FakeMsg:
            content = "real response"
            reasoning_content = None

        class FakeChoice:
            message = FakeMsg()

        class FakeResp:
            model = "gpt-4o"
            choices = [FakeChoice()]
            usage = {"prompt_tokens": 1, "completion_tokens": 2}

        async def fake_acompletion(**kwargs):
            captured["called"] = True
            return FakeResp()

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        llm = AsyncLiteLLM(model_name="gpt-4o", max_tokens=100)
        _run(llm([{"role": "user", "content": "test"}]))

        assert captured.get("called") is True
