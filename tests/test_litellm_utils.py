"""Tests for boxing_gym.agents.litellm_utils module.

These tests lock current behavior for paper reproducibility.
"""
import pytest
from unittest.mock import Mock, MagicMock
from types import SimpleNamespace

from boxing_gym.agents.litellm_utils import (
    is_responses_api_model,
    messages_to_input_text,
    extract_text,
)


class TestIsResponsesApiModel:
    """Tests for is_responses_api_model function."""

    # GPT-5 detection
    def test_identifies_gpt5(self):
        assert is_responses_api_model("gpt-5") is True

    def test_identifies_gpt5_turbo(self):
        assert is_responses_api_model("gpt-5-turbo") is True

    def test_identifies_gpt5_with_version(self):
        assert is_responses_api_model("gpt-5.1") is True
        assert is_responses_api_model("gpt-5.1-mini") is True

    def test_identifies_gpt5_uppercase(self):
        assert is_responses_api_model("GPT-5") is True
        assert is_responses_api_model("GPT-5-TURBO") is True

    def test_identifies_gpt5_with_prefix(self):
        assert is_responses_api_model("openai/gpt-5") is True

    # o-series (reasoning models) detection
    def test_identifies_o1(self):
        assert is_responses_api_model("o1") is True
        assert is_responses_api_model("o1-preview") is True
        assert is_responses_api_model("o1-mini") is True

    def test_identifies_o3(self):
        assert is_responses_api_model("o3") is True
        assert is_responses_api_model("o3-mini") is True

    def test_identifies_o4(self):
        assert is_responses_api_model("o4-mini") is True

    def test_identifies_o_series_with_prefix(self):
        assert is_responses_api_model("openai/o1") is True
        assert is_responses_api_model("openai/o3-mini") is True

    def test_identifies_o_series_uppercase(self):
        assert is_responses_api_model("O1") is True
        assert is_responses_api_model("O3-MINI") is True

    # GPT-4.1 detection
    def test_identifies_gpt41(self):
        assert is_responses_api_model("gpt-4.1") is True
        assert is_responses_api_model("gpt-4.1-mini") is True
        assert is_responses_api_model("gpt-4.1-nano") is True

    # Negative cases
    def test_rejects_gpt4(self):
        assert is_responses_api_model("gpt-4") is False
        assert is_responses_api_model("gpt-4-turbo") is False

    def test_rejects_gpt4o(self):
        assert is_responses_api_model("gpt-4o") is False
        assert is_responses_api_model("gpt-4o-mini") is False

    def test_rejects_claude(self):
        assert is_responses_api_model("claude-3-opus") is False
        assert is_responses_api_model("claude-3-sonnet") is False

    def test_rejects_deepseek(self):
        assert is_responses_api_model("deepseek-chat") is False
        assert is_responses_api_model("deepseek-v3") is False

    def test_handles_none(self):
        assert is_responses_api_model(None) is False

    def test_handles_empty_string(self):
        assert is_responses_api_model("") is False

    # edge case: "o" in model name but not reasoning model
    def test_rejects_non_reasoning_o(self):
        assert is_responses_api_model("llama-70b") is False
        assert is_responses_api_model("mistral-7b-instruct") is False


class TestMessagesToInputText:
    """Tests for messages_to_input_text function."""

    def test_simple_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = messages_to_input_text(messages)
        assert result == "user: Hello"

    def test_system_and_user_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there"},
        ]
        result = messages_to_input_text(messages)
        assert result == "system: You are helpful.\nuser: Hi there"

    def test_multi_turn_conversation(self):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        result = messages_to_input_text(messages)
        lines = result.split("\n")
        assert len(lines) == 4
        assert lines[0] == "system: Be concise."
        assert lines[1] == "user: What is 2+2?"
        assert lines[2] == "assistant: 4"
        assert lines[3] == "user: And 3+3?"

    def test_content_as_list_with_text(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello world"}]}
        ]
        result = messages_to_input_text(messages)
        assert result == "user: Hello world"

    def test_content_as_list_with_value(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "value": "Hello value"}]}
        ]
        result = messages_to_input_text(messages)
        assert result == "user: Hello value"

    def test_content_as_list_multiple_parts(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ]}
        ]
        result = messages_to_input_text(messages)
        assert result == "user: Part 1 Part 2"

    def test_content_as_list_with_string_items(self):
        messages = [
            {"role": "user", "content": ["Hello", "World"]}
        ]
        result = messages_to_input_text(messages)
        assert result == "user: Hello World"

    def test_empty_messages(self):
        assert messages_to_input_text([]) == ""
        assert messages_to_input_text(None) == ""

    def test_missing_role_defaults_to_user(self):
        messages = [{"content": "No role specified"}]
        result = messages_to_input_text(messages)
        assert result == "user: No role specified"

    def test_missing_content_defaults_to_empty(self):
        messages = [{"role": "user"}]
        result = messages_to_input_text(messages)
        assert result == "user: "


class TestExtractText:
    """Tests for extract_text function."""

    # ChatCompletions-style responses
    def test_chat_completions_with_object(self):
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hello from ChatCompletions")
                )
            ]
        )
        assert extract_text(resp) == "Hello from ChatCompletions"

    def test_chat_completions_with_dict(self):
        resp = {
            "choices": [
                {"message": {"content": "Dict response"}}
            ]
        }
        assert extract_text(resp) == "Dict response"

    def test_chat_completions_empty_choices(self):
        resp = SimpleNamespace(choices=[])
        assert extract_text(resp) == ""

    def test_chat_completions_none_content(self):
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=None))
            ]
        )
        assert extract_text(resp) == ""

    # Responses-style responses (output array)
    def test_responses_style_with_objects(self):
        resp = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="text", text="Responses API output")
                    ]
                )
            ]
        )
        assert extract_text(resp) == "Responses API output"

    def test_responses_style_with_dicts(self):
        resp = SimpleNamespace(
            output=[
                {
                    "type": "message",
                    "content": [
                        {"type": "text", "text": "Dict Responses output"}
                    ]
                }
            ]
        )
        assert extract_text(resp) == "Dict Responses output"

    def test_responses_style_multiple_outputs(self):
        # should return first message's text
        resp = SimpleNamespace(
            output=[
                SimpleNamespace(type="thinking", content=[]),  # skip
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(text="The answer")]
                )
            ]
        )
        assert extract_text(resp) == "The answer"

    def test_responses_style_empty_output(self):
        resp = SimpleNamespace(output=[])
        assert extract_text(resp) == ""

    def test_responses_style_no_text(self):
        resp = SimpleNamespace(
            output=[
                SimpleNamespace(type="message", content=[])
            ]
        )
        assert extract_text(resp) == ""

    # Edge cases
    def test_none_response(self):
        assert extract_text(None) == ""

    def test_empty_dict(self):
        assert extract_text({}) == ""

    def test_dict_without_choices_or_output(self):
        assert extract_text({"other": "data"}) == ""

    def test_dict_with_empty_choices(self):
        assert extract_text({"choices": []}) == ""

    def test_fallback_when_first_method_fails(self):
        # object with choices that raise exception, but has output
        class FailingChoices:
            @property
            def choices(self):
                raise RuntimeError("Simulated failure")

        resp = FailingChoices()
        resp.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(text="Fallback worked")]
            )
        ]
        assert extract_text(resp) == "Fallback worked"
