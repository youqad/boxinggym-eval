"""Tests for LMExperimenter agent."""

import os
import pytest
from unittest.mock import patch, MagicMock

from boxing_gym.agents.agent import (
    LMExperimenter,
    _model_profile,
    _compute_effective_max_tokens,
    ModelProfile,
    _build_weave_payload,
)


class TestModelProfile:
    """Tests for model profile detection and token limits."""

    def test_deepseek_reasoner_is_reasoning_model(self):
        profile = _model_profile("deepseek/deepseek-reasoner")
        assert profile.is_reasoning is True
        assert profile.api_max_tokens == 32768

    def test_deepseek_chat_is_not_reasoning_model(self):
        profile = _model_profile("deepseek/deepseek-chat")
        assert profile.is_reasoning is False
        assert profile.api_max_tokens == 4096

    def test_o1_series_is_reasoning_model(self):
        for model in ["o1", "o1-mini", "openai/o1", "o3-mini"]:
            profile = _model_profile(model)
            assert profile.is_reasoning is True, f"Expected {model} to be reasoning"

    def test_gpt5_is_reasoning_model(self):
        profile = _model_profile("gpt-5")
        assert profile.is_reasoning is True
        assert profile.api_max_tokens == 131072

    def test_kimi_thinking_is_reasoning_model(self):
        profile = _model_profile("moonshot/kimi-k2-thinking")
        assert profile.is_reasoning is True

    def test_standard_model_fallback(self):
        profile = _model_profile("gpt-4o")
        assert profile.is_reasoning is False
        assert profile.api_max_tokens == 16384  # gpt-4o has higher output limit


class TestComputeEffectiveMaxTokens:
    """Tests for max_tokens computation with reasoning model handling."""

    def test_none_returns_api_default(self):
        tokens, overridden = _compute_effective_max_tokens("gpt-4", None)
        profile = _model_profile("gpt-4")
        assert tokens == profile.api_max_tokens
        assert overridden is False

    def test_non_reasoning_respects_user_value(self):
        tokens, overridden = _compute_effective_max_tokens("gpt-4o", 1000)
        assert tokens == 1000
        assert overridden is False

    def test_reasoning_model_low_tokens_gets_overridden(self):
        # o1 reasoning model with low max_tokens should be overridden
        tokens, overridden = _compute_effective_max_tokens("o1", 500)
        profile = _model_profile("o1")
        assert tokens == profile.api_max_tokens
        assert overridden is True

    def test_reasoning_model_allows_truncation_when_requested(self):
        tokens, overridden = _compute_effective_max_tokens("o1", 500, allow_truncation=True)
        assert tokens == 500
        assert overridden is False

    def test_reasoning_model_high_tokens_not_overridden(self):
        # o1 has api_max_tokens=100000, so need to request >= 100000 to not be overridden
        tokens, overridden = _compute_effective_max_tokens("o1", 100000)
        assert tokens == 100000
        assert overridden is False

    def test_invalid_max_tokens_returns_default(self):
        tokens, overridden = _compute_effective_max_tokens("gpt-4", -1)
        profile = _model_profile("gpt-4")
        assert tokens == profile.api_max_tokens
        assert overridden is True


class TestLMExperimenterFakeMode:
    """Tests for fake LLM mode (used in testing without real API calls)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        original = os.environ.get("BOXINGGYM_FAKE_LLM")
        os.environ["BOXINGGYM_FAKE_LLM"] = "1"
        try:
            yield
        finally:
            if original is not None:
                os.environ["BOXINGGYM_FAKE_LLM"] = original
            else:
                os.environ.pop("BOXINGGYM_FAKE_LLM", None)

    def test_fake_mode_returns_parseable_observation(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)
        response, retries = agent.prompt_llm_and_parse(
            "What observation would you like to make?",
            is_observation=True
        )
        assert response is not None
        assert response.strip(), "Response should not be empty"

    def test_fake_mode_returns_parseable_answer(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)
        response, retries = agent.prompt_llm_and_parse(
            "What is your answer?",
            is_observation=False
        )
        assert response is not None
        assert response.strip(), "Response should not be empty"

    def test_fake_mode_dugongs_returns_valid_age(self):
        os.environ["BOXINGGYM_FAKE_LLM"] = "dugongs"
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)
        response, retries = agent.prompt_llm_and_parse(
            "What is the dugong age?",
            is_observation=False
        )
        assert response == "2.0"

    def test_fake_mode_lotka_returns_valid_params(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)
        response, retries = agent.prompt_llm_and_parse(
            "What are the lotka-volterra parameters?",
            is_observation=False
        )
        assert response is not None
        assert response.strip(), "Response should not be empty"


class TestLMExperimenterParseResponse:
    """Tests for response parsing methods."""

    def test_parse_response_extracts_observe_tag(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)
        response = agent.parse_response("<observe>42</observe>", is_observation=True)
        assert response == "42"

    def test_parse_response_extracts_answer_tag(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)
        response = agent.parse_response("<answer>3.14</answer>", is_observation=False)
        assert response == "3.14"

    def test_parse_response_returns_none_for_missing_tag(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)
        response = agent.parse_response("No tags here", is_observation=True)
        assert response is None

    def test_parse_response_v2_handles_whitespace(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)
        response = agent.parse_response_v2("< observe >  42  < / observe >", is_observation=True)
        assert response is not None
        assert "42" in response


class TestLMExperimenterRetryLogic:
    """Tests for retry logic in prompt_llm_and_parse."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # save original value (if any) and disable fake mode for mocking
        original = os.environ.get("BOXINGGYM_FAKE_LLM")
        os.environ["BOXINGGYM_FAKE_LLM"] = "0"
        try:
            yield
        finally:
            # always restore original state
            if original is not None:
                os.environ["BOXINGGYM_FAKE_LLM"] = original
            else:
                os.environ.pop("BOXINGGYM_FAKE_LLM", None)

    def test_retries_on_parse_failure(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)

        call_count = 0
        def mock_prompt_llm(prompt, _weave_context=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return "Invalid response without tags"
            return "<answer>42</answer>"

        with patch.object(agent, 'prompt_llm', side_effect=mock_prompt_llm):
            response, retries = agent.prompt_llm_and_parse("test", is_observation=False)

        assert response == "42"
        assert retries == 2  # 2 failed attempts before success
        assert call_count == 3

    def test_raises_after_max_retries(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)

        def mock_prompt_llm(prompt, _weave_context=None):
            return "Always invalid - no tags"

        with patch.object(agent, 'prompt_llm', side_effect=mock_prompt_llm):
            with pytest.raises(ValueError, match="Failed to get valid response"):
                agent.prompt_llm_and_parse("test", is_observation=False, max_tries=4)

    def test_rejects_response_without_numbers(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)

        call_count = 0
        def mock_prompt_llm(prompt, _weave_context=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return "<answer>no numbers here</answer>"
            return "<answer>42</answer>"

        with patch.object(agent, 'prompt_llm', side_effect=mock_prompt_llm):
            response, retries = agent.prompt_llm_and_parse("test", is_observation=False)

        assert response == "42"
        assert retries == 1

    def test_accepts_word_numbers(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)

        def mock_prompt_llm(prompt, _weave_context=None):
            return "<answer>one half</answer>"

        with patch.object(agent, 'prompt_llm', side_effect=mock_prompt_llm):
            response, retries = agent.prompt_llm_and_parse("test", is_observation=False)

        # word numbers should be accepted
        assert response is not None


class TestLMExperimenterUsageStats:
    """Tests for usage statistics tracking."""

    def test_usage_stats_initialized_empty(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)
        stats = agent._usage_stats
        assert stats["prompt_tokens"] == 0
        assert stats["completion_tokens"] == 0
        assert stats["retry_count"] == 0
        assert stats["error_count"] == 0

    def test_retry_count_accumulates(self):
        original = os.environ.get("BOXINGGYM_FAKE_LLM")
        try:
            os.environ["BOXINGGYM_FAKE_LLM"] = "0"
            agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)

            call_count = 0
            def mock_prompt_llm(prompt, _weave_context=None):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    return "Invalid"
                return "<answer>42</answer>"

            with patch.object(agent, 'prompt_llm', side_effect=mock_prompt_llm):
                agent.prompt_llm_and_parse("test", is_observation=False)

            assert agent._usage_stats["retry_count"] == 2
        finally:
            if original is not None:
                os.environ["BOXINGGYM_FAKE_LLM"] = original
            else:
                os.environ.pop("BOXINGGYM_FAKE_LLM", None)


class TestLMExperimenterInitialization:
    """Tests for LMExperimenter initialization."""

    def test_default_initialization(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.5)
        assert agent.model_name == "gpt-4o"
        assert agent.temperature == 0.5

    def test_with_api_base(self):
        agent = LMExperimenter(
            model_name="deepseek/deepseek-chat",
            temperature=0.0,
            api_base="https://api.deepseek.com/v1"
        )
        assert agent.api_base == "https://api.deepseek.com/v1"

    def test_messages_initialized_empty(self):
        agent = LMExperimenter(model_name="gpt-4o", temperature=0.0)
        assert agent.messages == []


class TestWeavePayload:
    """Tests for Weave payload construction safeguards."""

    def test_payload_defaults_model_and_usage(self):
        payload = _build_weave_payload(
            weave_context=None,
            args=(),
            kwargs={},
            latency_ms=12.3,
            result=None,
        )
        assert payload["model"] == "unknown"
        assert isinstance(payload["usage"], dict)
        assert payload["usage"]["prompt_tokens"] == 0
        assert payload["usage"]["completion_tokens"] == 0
        assert payload["usage"]["reasoning_tokens"] == 0

    def test_payload_preserves_model_from_context(self):
        payload = _build_weave_payload(
            weave_context={"model": "gpt-4o"},
            args=(),
            kwargs={},
            latency_ms=1.0,
            result=None,
        )
        assert payload["model"] == "gpt-4o"
