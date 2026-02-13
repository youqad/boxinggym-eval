"""Tests for the unified pricing module."""

import pytest

from boxing_gym.agents.pricing import (
    MODEL_COST_PER_INPUT,
    MODEL_COST_PER_OUTPUT,
    MODEL_REGISTRY,
    calculate_cost,
    get_api_config,
    get_input_cost,
    get_litellm_pricing_dict,
    get_max_tokens,
    get_model_pricing,
    get_output_cost,
)


class TestModelRegistry:
    """Tests for MODEL_REGISTRY structure and contents."""

    def test_registry_has_substantial_model_coverage(self):
        count = len(MODEL_REGISTRY)
        assert count >= 50, f"Registry should have substantial coverage (got {count})"
        # check critical models exist rather than arbitrary upper bound
        critical_models = ["gpt-4o", "claude-sonnet-4", "deepseek-chat", "o1-mini"]
        for model in critical_models:
            assert model in MODEL_REGISTRY, f"Critical model {model} missing from registry"

    def test_all_entries_have_required_fields(self):
        required = {
            "input_cost_per_token",
            "output_cost_per_token",
            "max_tokens",
            "litellm_provider",
            "mode",
        }
        for model, config in MODEL_REGISTRY.items():
            missing = required - set(config.keys())
            assert not missing, f"{model} missing fields: {missing}"

    def test_costs_are_non_negative(self):
        for model, config in MODEL_REGISTRY.items():
            assert config["input_cost_per_token"] >= 0, f"{model} has negative input cost"
            assert config["output_cost_per_token"] >= 0, f"{model} has negative output cost"

    def test_max_tokens_are_positive(self):
        for model, config in MODEL_REGISTRY.items():
            assert config["max_tokens"] > 0, f"{model} has non-positive max_tokens"

    def test_known_models_exist(self):
        known = ["gpt-4o", "claude-sonnet-4", "deepseek-chat", "gemini-2.0-flash", "qwen-max"]
        for model in known:
            assert model in MODEL_REGISTRY, f"{model} not in registry"


class TestBackwardCompatDicts:
    """Tests for MODEL_COST_PER_INPUT and MODEL_COST_PER_OUTPUT dicts."""

    def test_dicts_have_same_length_as_registry(self):
        assert len(MODEL_COST_PER_INPUT) == len(MODEL_REGISTRY)
        assert len(MODEL_COST_PER_OUTPUT) == len(MODEL_REGISTRY)

    def test_dicts_match_registry_values(self):
        for model in MODEL_REGISTRY:
            assert MODEL_COST_PER_INPUT[model] == MODEL_REGISTRY[model]["input_cost_per_token"]
            assert MODEL_COST_PER_OUTPUT[model] == MODEL_REGISTRY[model]["output_cost_per_token"]


class TestGetModelPricing:
    """Tests for get_model_pricing function."""

    def test_returns_dict_for_known_model(self):
        result = get_model_pricing("gpt-4o")
        assert isinstance(result, dict)
        assert "input_cost_per_token" in result

    def test_returns_none_for_unknown_model(self):
        result = get_model_pricing("nonexistent-model-xyz")
        assert result is None


class TestGetInputCost:
    """Tests for get_input_cost function."""

    def test_returns_cost_for_known_model(self):
        cost = get_input_cost("gpt-4o")
        assert cost > 0

    def test_returns_default_for_unknown_model(self):
        cost = get_input_cost("nonexistent", default=0.123)
        assert cost == 0.123

    def test_default_is_zero(self):
        cost = get_input_cost("nonexistent")
        assert cost == 0.0


class TestGetOutputCost:
    """Tests for get_output_cost function."""

    def test_returns_cost_for_known_model(self):
        cost = get_output_cost("gpt-4o")
        assert cost > 0

    def test_output_cost_typically_higher_than_input(self):
        # most models charge more for output tokens
        for model in ["gpt-4o", "claude-sonnet-4", "gemini-2.0-flash"]:
            input_cost = get_input_cost(model)
            output_cost = get_output_cost(model)
            assert output_cost >= input_cost, f"{model}: output should be >= input cost"


class TestGetApiConfig:
    """Tests for get_api_config function."""

    def test_returns_dict_for_known_model(self):
        config = get_api_config("gpt-4o")
        assert isinstance(config, dict)
        assert config["litellm_provider"] == "openai"

    def test_returns_dict_with_none_provider_for_unknown_model(self):
        config = get_api_config("nonexistent")
        assert isinstance(config, dict)
        assert config["litellm_provider"] is None

    def test_known_model_has_provider(self):
        config = get_api_config("gpt-4o")
        assert config["litellm_provider"] == "openai"

    def test_anthropic_model_has_anthropic_provider(self):
        config = get_api_config("claude-sonnet-4")
        assert config["litellm_provider"] == "anthropic"


class TestGetMaxTokens:
    """Tests for get_max_tokens function."""

    def test_returns_value_for_known_model(self):
        tokens = get_max_tokens("gpt-4o")
        assert tokens > 0

    def test_returns_default_for_unknown_model(self):
        tokens = get_max_tokens("nonexistent", default=4096)
        assert tokens == 4096


class TestCalculateCost:
    """Tests for calculate_cost function."""

    def test_correct_calculation_for_gpt4o(self):
        # gpt-4o: $2.5/1M input, $10/1M output
        cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        expected = 1000 * 2.5e-6 + 500 * 10e-6
        assert abs(cost - expected) < 1e-10

    def test_zero_tokens_returns_zero(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=0, completion_tokens=0)
        assert cost == 0.0

    def test_unknown_model_returns_zero(self):
        cost = calculate_cost("nonexistent", prompt_tokens=1000, completion_tokens=500)
        assert cost == 0.0

    def test_free_models_return_zero(self):
        # ollama models are free
        cost = calculate_cost("ollama/gpt-oss:20b", prompt_tokens=10000, completion_tokens=5000)
        assert cost == 0.0


class TestGetLitellmPricingDict:
    """Tests for get_litellm_pricing_dict function."""

    def test_returns_dict_with_correct_length(self):
        result = get_litellm_pricing_dict()
        assert len(result) == len(MODEL_REGISTRY)

    def test_values_contain_required_litellm_fields(self):
        result = get_litellm_pricing_dict()
        required = {
            "input_cost_per_token",
            "output_cost_per_token",
            "max_tokens",
            "litellm_provider",
            "mode",
        }
        for model, config in result.items():
            missing = required - set(config.keys())
            assert not missing, f"{model} missing litellm fields: {missing}"


class TestPricingAccuracy:
    """Spot-check pricing against known values."""

    @pytest.mark.parametrize(
        "model,input_per_million,output_per_million",
        [
            ("gpt-4o", 2.50, 10.00),
            ("claude-sonnet-4", 3.00, 15.00),
            ("deepseek-chat", 0.28, 0.42),
            ("gemini-2.0-flash", 0.10, 0.40),
        ],
    )
    def test_known_pricing_values(self, model, input_per_million, output_per_million):
        input_cost = get_input_cost(model) * 1_000_000
        output_cost = get_output_cost(model) * 1_000_000
        assert abs(input_cost - input_per_million) < 0.01, f"{model} input cost mismatch"
        assert abs(output_cost - output_per_million) < 0.01, f"{model} output cost mismatch"
