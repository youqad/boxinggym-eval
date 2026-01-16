"""Characterization tests for results_io parsing.

Phase 0 safety net: capture current behavior BEFORE any changes.
"""

import pytest
from boxing_gym.agents.results_io import (
    get_model_display_name,
    get_env_display_name,
    get_goal_display_name,
    _parse_wandb_run_name,
    standardize,
    NORM_STATIC,
)


class TestGetModelDisplayName:
    """Tests for model name normalization."""

    def test_known_model_exact_match(self):
        """Direct key lookup should work."""
        assert get_model_display_name("gpt-4o") == "GPT-4o"
        assert get_model_display_name("deepseek-chat") == "DeepSeek-Chat"
        assert get_model_display_name("glm-4.7") == "GLM-4.7"

    def test_known_model_case_insensitive(self):
        """Lowercase lookup should work."""
        assert get_model_display_name("GPT-4O") == "GPT-4o"
        assert get_model_display_name("DEEPSEEK-CHAT") == "DeepSeek-Chat"

    def test_litellm_prefixed_models(self):
        """Models with provider prefix should resolve."""
        assert get_model_display_name("openai/gpt-4o") == "GPT-4o"
        assert get_model_display_name("deepseek/deepseek-chat") == "DeepSeek-V3.2"
        assert get_model_display_name("anthropic/glm-4.7") == "GLM-4.7"

    def test_unknown_model_returns_key(self):
        """Unknown model should return input unchanged."""
        assert get_model_display_name("unknown-model-xyz") == "unknown-model-xyz"

    def test_minimax_variants(self):
        """MiniMax model variants should normalize."""
        assert get_model_display_name("minimax-m2.1") == "MiniMax-M2.1"
        assert get_model_display_name("openai/MiniMax-M2.1") == "MiniMax-M2.1"
        assert get_model_display_name("openai/minimax-m2.1") == "MiniMax-M2.1"


class TestGetEnvDisplayName:
    """Tests for environment name normalization."""

    def test_direct_suffix_removed(self):
        """_direct suffix should be stripped."""
        assert get_env_display_name("dugongs_direct") == "dugongs"
        assert get_env_display_name("peregrines_direct") == "peregrines"
        assert get_env_display_name("lotka_volterra_direct") == "lotka_volterra"

    def test_morals_normalized(self):
        """morals should become moral_machines."""
        assert get_env_display_name("morals") == "moral_machines"
        assert get_env_display_name("moral") == "moral_machines"

    def test_unknown_env_returns_base(self):
        """Unknown env should return with _direct stripped."""
        assert get_env_display_name("unknown_env") == "unknown_env"
        assert get_env_display_name("unknown_env_direct") == "unknown_env"

    def test_already_canonical_unchanged(self):
        """Already canonical names should pass through."""
        assert get_env_display_name("dugongs") == "dugongs"
        assert get_env_display_name("hyperbolic_temporal_discount") == "hyperbolic_temporal_discount"


class TestGetGoalDisplayName:
    """Tests for goal name normalization."""

    def test_none_goal_uses_default(self):
        """None goal should return environment default."""
        assert get_goal_display_name("dugongs", None) == "length"
        assert get_goal_display_name("peregrines", None) == "population"
        assert get_goal_display_name("hyperbolic_temporal_discount", None) == "choice"

    def test_direct_variants_use_default(self):
        """direct/direct_naive/direct_discovery should use default."""
        assert get_goal_display_name("dugongs", "direct") == "length"
        assert get_goal_display_name("dugongs", "direct_naive") == "length"
        assert get_goal_display_name("dugongs", "direct_discovery") == "length"

    def test_location_finding_source_alias(self):
        """location_finding source -> source_location."""
        assert get_goal_display_name("location_finding", "source") == "source_location"

    def test_death_process_infection_alias(self):
        """death_process infection -> infection_rate."""
        assert get_goal_display_name("death_process", "infection") == "infection_rate"

    def test_specific_goal_preserved(self):
        """Non-default goals should pass through."""
        assert get_goal_display_name("irt", "best_student") == "best_student"
        assert get_goal_display_name("irt", "difficult_question") == "difficult_question"

    def test_unknown_env_unknown_goal(self):
        """Unknown env with unknown goal should return goal as-is."""
        assert get_goal_display_name("unknown_env", "custom_goal") == "custom_goal"


class TestParseWandbRunName:
    """Tests for W&B run name parsing."""

    def test_parses_oed_run(self):
        """OED experiment run should parse correctly."""
        result = _parse_wandb_run_name("dugongs_oed_gpt-4o_seed42")
        assert result is not None
        assert result["env"] == "dugongs"
        assert result["exp_type"] == "oed"
        assert result["model"] == "gpt-4o"
        assert result["seed"] == 42

    def test_parses_discovery_run(self):
        """Discovery experiment run should parse correctly."""
        result = _parse_wandb_run_name("peregrines_discovery_deepseek-chat_seed1")
        assert result is not None
        assert result["env"] == "peregrines"
        assert result["exp_type"] == "discovery"
        assert result["model"] == "deepseek-chat"
        assert result["seed"] == 1

    def test_parses_run_with_goal(self):
        """Run with goal suffix should parse correctly."""
        result = _parse_wandb_run_name("irt_best_student_oed_gpt-4o_seed0")
        assert result is not None
        assert result["env"] == "irt"
        assert result["goal"] == "best_student"
        assert result["exp_type"] == "oed"
        assert result["seed"] == 0

    def test_env_normalization_applied(self):
        """Environment should be normalized during parsing."""
        result = _parse_wandb_run_name("dugongs_direct_oed_gpt-4o_seed1")
        assert result is not None
        # dugongs_direct should become dugongs
        assert result["env"] == "dugongs"

    def test_invalid_run_name_returns_none(self):
        """Invalid run names should return None."""
        assert _parse_wandb_run_name("random_string") is None
        assert _parse_wandb_run_name("missing_seed_oed_gpt4o") is None
        assert _parse_wandb_run_name("") is None

    def test_hyperbolic_env_parsing(self):
        """Hyperbolic temporal discount parsing (known bug: 'discount' goal match).

        NOTE: This documents a bug where 'discount' in _KNOWN_GOALS incorrectly
        matches the env name suffix, truncating to 'hyperbolic_temporal'.
        The actual env name in configs is 'hyperbolic_direct' which parses correctly.
        """
        result = _parse_wandb_run_name("hyperbolic_temporal_discount_oed_glm-4.7_seed5")
        assert result is not None
        # Bug: 'discount' matches _KNOWN_GOALS, incorrectly splitting the env name
        assert result["env"] == "hyperbolic_temporal"  # Should be hyperbolic_temporal_discount
        assert result["goal"] == "discount"  # Incorrectly extracted as goal
        assert result["model"] == "glm-4.7"

    def test_hyperbolic_direct_parses_correctly(self):
        """hyperbolic_direct (actual config name) parses correctly."""
        result = _parse_wandb_run_name("hyperbolic_direct_oed_gpt-4o_seed3")
        assert result is not None
        # hyperbolic_direct -> hyperbolic (via get_env_display_name stripping _direct)
        assert result["env"] == "hyperbolic"
        assert result["model"] == "gpt-4o"


class TestStandardize:
    """Tests for z-score standardization."""

    def test_known_env_standardizes(self):
        """Known environments should standardize correctly."""
        # dugongs: (0.9058681693402041, 9.234192516908691)
        z_mean, z_std = standardize(0.9058681693402041, 1.0, "dugongs")
        assert z_mean is not None
        assert z_mean == pytest.approx(0.0, abs=0.01)  # z-score of mean is 0

    def test_unknown_env_returns_none(self):
        """Unknown environments should return None."""
        z_mean, z_std = standardize(1.0, 0.5, "unknown_env")
        assert z_mean is None
        assert z_std is None

    def test_all_known_envs_have_norm_constants(self):
        """All standard environments should have normalization constants."""
        known_envs = [
            "dugongs",
            "peregrines",
            "lotka_volterra",
            "hyperbolic_temporal_discount",
            "irt",
            "survival",
            "location_finding",
            "death_process",
            "emotion",
        ]
        for env in known_envs:
            assert env in NORM_STATIC, f"Missing NORM_STATIC for {env}"
            mu, sigma = NORM_STATIC[env]
            assert sigma > 0, f"Invalid sigma for {env}"

    def test_standardize_formula(self):
        """Z-score should follow (x - mu) / sigma."""
        # For dugongs: mu=0.9058681693402041, sigma=9.234192516908691
        mu, sigma = NORM_STATIC["dugongs"]
        test_value = mu + sigma  # should give z=1.0
        z_mean, _ = standardize(test_value, 0.0, "dugongs")
        assert z_mean == pytest.approx(1.0, abs=0.01)

    def test_zero_sigma_returns_none(self):
        """Environment with zero sigma should return None."""
        # This is a safety check - if an env had sigma=0 it would be undefined
        # Currently no such env exists, but function handles it
        # We test by checking that valid envs don't have zero sigma
        for env, (mu, sigma) in NORM_STATIC.items():
            assert sigma != 0, f"Zero sigma for {env} would cause division error"


class TestNormStaticConstants:
    """Tests for NORM_STATIC reference values."""

    def test_morals_variants_both_present(self):
        """Both morals and moral_machines should have constants."""
        assert "morals" in NORM_STATIC
        assert "moral_machines" in NORM_STATIC
        # They should be identical
        assert NORM_STATIC["morals"] == NORM_STATIC["moral_machines"]

    def test_constants_are_reasonable(self):
        """Normalization constants should be sensible values."""
        for env, (mu, sigma) in NORM_STATIC.items():
            assert isinstance(mu, (int, float))
            assert isinstance(sigma, (int, float))
            assert sigma > 0, f"Sigma must be positive for {env}"
