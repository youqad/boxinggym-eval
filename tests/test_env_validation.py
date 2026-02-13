"""Parametrized environment validation tests.

Tests that all registered environments can be instantiated and used correctly.
"""

from typing import Any

import pytest

from boxing_gym.envs.registry import get_environment_registry

# get all registered environments and goals
ENVS, GOALS = get_environment_registry()

# list of (env_name, goal_type) tuples for parametrized tests
DIRECT_GOALS = [
    (env_name, goal_type)
    for (env_name, goal_type) in GOALS.keys()
    if "direct" in goal_type and "naive" not in goal_type
]

ALL_GOALS = list(GOALS.keys())


class TestEnvironmentRegistry:
    """Tests for environment registration."""

    def test_registry_not_empty(self):
        """Verify environments are registered."""
        assert len(ENVS) > 0
        assert len(GOALS) > 0

    def test_all_envs_have_direct_goal(self):
        """Every environment should have at least a direct goal."""
        for env_name in ENVS:
            has_direct = any(e == env_name and "direct" in g for (e, g) in GOALS.keys())
            assert has_direct, f"Environment '{env_name}' missing direct goal"


class TestEnvironmentInstantiation:
    """Test that all environments can be instantiated."""

    @pytest.mark.parametrize("env_name", list(ENVS.keys()))
    def test_env_can_be_instantiated(self, env_name: str):
        """Test that environment classes can be instantiated."""
        EnvClass = ENVS[env_name]
        env = EnvClass()
        assert env is not None

    @pytest.mark.parametrize("env_name", list(ENVS.keys()))
    def test_env_has_required_methods(self, env_name: str):
        """Test that environments have required interface methods."""
        EnvClass = ENVS[env_name]
        env = EnvClass()

        # all envs should have these methods
        assert hasattr(env, "validate_input")
        assert callable(env.validate_input)
        assert hasattr(env, "run_experiment")
        assert callable(env.run_experiment)


class TestGoalInstantiation:
    """Test that all goals can be instantiated with their environments."""

    @pytest.mark.parametrize("env_goal", ALL_GOALS)
    def test_goal_can_be_instantiated(self, env_goal: tuple[str, str]):
        """Test that goal classes can be instantiated."""
        env_name, goal_type = env_goal
        EnvClass = ENVS[env_name]
        GoalClass = GOALS[(env_name, goal_type)]

        env = EnvClass()
        goal = GoalClass(env)
        assert goal is not None

    @pytest.mark.parametrize("env_goal", ALL_GOALS)
    def test_goal_has_required_methods(self, env_goal: tuple[str, str]):
        """Test that goals have required interface methods."""
        env_name, goal_type = env_goal
        EnvClass = ENVS[env_name]
        GoalClass = GOALS[(env_name, goal_type)]

        env = EnvClass()
        goal = GoalClass(env)

        # all goals should have these methods
        required_methods = [
            "get_goal_eval_question",
            "evaluate_predictions",
        ]
        for method in required_methods:
            assert hasattr(goal, method), f"Goal missing method: {method}"
            assert callable(getattr(goal, method))


class TestDirectGoalEvaluation:
    """Test that direct goals can generate and evaluate predictions."""

    @pytest.mark.parametrize("env_goal", DIRECT_GOALS)
    def test_goal_generates_eval_question(self, env_goal: tuple[str, str]):
        """Test that goals can generate evaluation questions."""
        env_name, goal_type = env_goal
        EnvClass = ENVS[env_name]
        GoalClass = GOALS[(env_name, goal_type)]

        env = EnvClass()
        goal = GoalClass(env)

        # generate a question
        question, gt = goal.get_goal_eval_question(include_prior=False)

        assert isinstance(question, str)
        assert len(question) > 0
        # gt can be various types (float, int, str, list, etc.)

    @pytest.mark.parametrize("env_goal", DIRECT_GOALS)
    def test_goal_evaluates_predictions(self, env_goal: tuple[str, str]):
        """Test that goals can evaluate predictions."""
        env_name, goal_type = env_goal
        EnvClass = ENVS[env_name]
        GoalClass = GOALS[(env_name, goal_type)]

        env = EnvClass()
        goal = GoalClass(env)

        # environment-specific prediction formats
        prediction_format = {
            "lotka_volterra": "[0.5, 0.4]",  # expects list of 2 floats
            "moral_machines": "1",  # expects 0 or 1
            "morals": "1",  # expects 0 or 1
            "emotion": "happy",  # expects emotion string
        }.get(env_name, "0.5")  # default to simple float string

        # generate questions and dummy predictions
        questions, gts, predictions = [], [], []
        for _ in range(3):
            q, gt = goal.get_goal_eval_question(include_prior=False)
            questions.append(q)
            gts.append(gt)
            predictions.append(prediction_format)

        # evaluate
        result = goal.evaluate_predictions(predictions, gts)

        # result should be a dict or something meaningful
        assert result is not None


class TestEnvironmentValidation:
    """Test input validation for environments."""

    def _is_invalid_result(self, result: Any) -> bool:
        """Check if validation result indicates invalid input.

        Environments can indicate invalid input by:
        - returning False or None
        - returning an error string
        - raising an exception (handled by caller)
        """
        if result is False or result is None:
            return True
        if isinstance(result, str) and any(
            word in result.lower() for word in ["invalid", "must", "error", "expected"]
        ):
            return True
        return False

    @pytest.mark.parametrize("env_name", list(ENVS.keys()))
    def test_validate_input_rejects_empty(self, env_name: str):
        """Test that validate_input rejects empty input."""
        EnvClass = ENVS[env_name]
        env = EnvClass()
        # some envs need include_prior set
        if hasattr(env, "include_prior") or env_name == "emotion":
            env.include_prior = False

        # empty string should be invalid
        try:
            result = env.validate_input("")
            assert self._is_invalid_result(result), (
                f"Expected invalid result for empty input, got: {result}"
            )
        except (ValueError, TypeError, AttributeError):
            pass  # raising is also acceptable

    @pytest.mark.parametrize("env_name", list(ENVS.keys()))
    def test_validate_input_rejects_none(self, env_name: str):
        """Test that validate_input handles None gracefully."""
        EnvClass = ENVS[env_name]
        env = EnvClass()
        if hasattr(env, "include_prior") or env_name == "emotion":
            env.include_prior = False

        try:
            result = env.validate_input(None)
            assert self._is_invalid_result(result), (
                f"Expected invalid result for None input, got: {result}"
            )
        except (TypeError, AttributeError, ValueError):
            pass  # raising is acceptable for None input

    @pytest.mark.parametrize("env_name", list(ENVS.keys()))
    def test_validate_input_rejects_invalid_format(self, env_name: str):
        """Test that validate_input rejects clearly invalid format."""
        EnvClass = ENVS[env_name]
        env = EnvClass()
        if hasattr(env, "include_prior") or env_name == "emotion":
            env.include_prior = False

        # gibberish should be invalid
        try:
            result = env.validate_input("not a valid observation format xyz123")
            assert self._is_invalid_result(result), (
                f"Expected invalid result for gibberish input, got: {result}"
            )
        except (ValueError, TypeError, AttributeError):
            pass  # raising is also acceptable


class TestEnvironmentDescription:
    """Test that environments provide descriptions."""

    @pytest.mark.parametrize("env_name", list(ENVS.keys()))
    def test_env_has_description(self, env_name: str):
        """Test that environments have a description or docstring."""
        EnvClass = ENVS[env_name]
        env = EnvClass()

        # should have either get_description() or __doc__
        has_description = hasattr(env, "get_description") or (
            EnvClass.__doc__ and len(EnvClass.__doc__) > 10
        )
        assert has_description, f"Environment '{env_name}' lacks description"


class TestGoalSystemMessage:
    """Test that goals can generate system messages."""

    @pytest.mark.parametrize("env_goal", DIRECT_GOALS)
    def test_goal_has_system_message(self, env_goal: tuple[str, str]):
        """Test that goals can generate system messages for agents."""
        env_name, goal_type = env_goal
        EnvClass = ENVS[env_name]
        GoalClass = GOALS[(env_name, goal_type)]

        env = EnvClass()
        goal = GoalClass(env)

        # goals should have a way to get system message
        if hasattr(goal, "get_system_message"):
            msg = goal.get_system_message(include_prior=False)
            assert isinstance(msg, str)
            assert len(msg) > 0
        elif hasattr(goal, "system_message"):
            msg = goal.system_message
            assert isinstance(msg, str)
