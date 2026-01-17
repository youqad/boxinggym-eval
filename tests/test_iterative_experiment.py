"""Integration tests for the experiment loop."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from boxing_gym.experiment.loop import (
    iterative_experiment,
    _normalize_budgets,
    _compute_z_stats,
)


class TestNormalizeBudgets:
    """Test budget normalization helper."""

    def test_none_returns_empty(self):
        assert _normalize_budgets(None) == []

    def test_single_int(self):
        assert _normalize_budgets(5) == [5]

    def test_list_sorted_unique(self):
        assert _normalize_budgets([10, 5, 10, 15]) == [5, 10, 15]

    def test_tuple_works(self):
        assert _normalize_budgets((3, 1, 2)) == [1, 2, 3]

    def test_invalid_values_skipped(self):
        assert _normalize_budgets([5, "bad", 10]) == [5, 10]

    def test_string_int_fails(self):
        # string that's not convertible
        assert _normalize_budgets("not_a_number") == []


class TestComputeZStats:
    """Test z-score computation helper."""

    def test_none_inputs_return_none(self):
        z_mean, z_std = _compute_z_stats(None, 0.0, 1.0)
        assert z_mean is None
        assert z_std is None

    def test_zero_sigma_returns_none(self):
        z_mean, z_std = _compute_z_stats({"mse": 0.5}, 0.0, 0.0)
        assert z_mean is None
        assert z_std is None

    def test_dict_with_mse(self):
        z_mean, z_std = _compute_z_stats({"mse": 1.5}, 0.5, 1.0)
        assert z_mean == pytest.approx(1.0)
        assert z_std == 0.0  # no std provided

    def test_dict_with_std_mse(self):
        z_mean, z_std = _compute_z_stats({"mse": 1.5, "std_mse": 0.5}, 0.5, 1.0)
        assert z_mean == pytest.approx(1.0)
        assert z_std == pytest.approx(0.5)

    def test_list_input(self):
        z_mean, z_std = _compute_z_stats([2.0, 0.2], 1.0, 0.5)
        assert z_mean == pytest.approx(2.0)
        assert z_std == pytest.approx(0.4)


class TestIterativeExperimentMocked:
    """Integration tests with mocked goal and scientist."""

    @pytest.fixture
    def mock_goal(self):
        """Create a mock goal with required interface."""
        goal = Mock()
        goal.get_system_message.return_value = "You are a scientist."
        goal.get_goal_description.return_value = "Predict the outcome."
        goal.get_simulation_description.return_value = "Run simulation."
        goal.get_queries.return_value = []
        goal.evaluate_predictions.return_value = {"mse": 0.5, "std_mse": 0.1}
        goal.get_norm_params.return_value = (0.0, 1.0)
        # mock the environment
        goal.env = Mock()
        goal.env.run_experiment.return_value = ("observation result", True)
        return goal

    @pytest.fixture
    def mock_scientist(self):
        """Create a mock scientist with required interface."""
        scientist = Mock()
        scientist.generate_actions.return_value = "observe(x=1)"
        scientist.generate_predictions.return_value = "0.5"
        scientist.update_history = Mock()
        scientist.get_latencies_ms.return_value = []
        return scientist

    def test_empty_budget_runs(self, mock_goal, mock_scientist):
        """Test with empty budget list."""
        # returns: (results, queries, observations, successes, explanations, eigs, programs)
        result_tuple = iterative_experiment(
            goal=mock_goal,
            scientist=mock_scientist,
            num_experiments=[],
            num_evals=1,
            include_prior=False,
        )
        results, queries, observations, successes, explanations, eigs, programs = result_tuple
        assert results == []
        assert queries == []

    def test_single_budget_calls_scientist(self, mock_goal, mock_scientist):
        """Test scientist is called for each experiment step."""
        with patch("boxing_gym.experiment.loop.evaluate") as mock_eval:
            mock_eval.return_value = ({"mse": 0.5}, [], [])

            results = iterative_experiment(
                goal=mock_goal,
                scientist=mock_scientist,
                num_experiments=[3],
                num_evals=1,
                include_prior=False,
            )

            # scientist should generate actions 3 times (budget=3)
            assert mock_scientist.generate_actions.call_count == 3

    def test_prior_only_mode(self, mock_goal, mock_scientist):
        """Test budget=0 triggers prior-only evaluation."""
        with patch("boxing_gym.experiment.loop.evaluate") as mock_eval:
            mock_eval.return_value = ({"mse": 0.5}, [], [])

            results = iterative_experiment(
                goal=mock_goal,
                scientist=mock_scientist,
                num_experiments=[0],
                num_evals=1,
                include_prior=True,
            )

            # no actions generated for prior-only
            assert mock_scientist.generate_actions.call_count == 0

    def test_multiple_budgets_incremental(self, mock_goal, mock_scientist):
        """Test multiple budgets run incrementally."""
        with patch("boxing_gym.experiment.loop.evaluate") as mock_eval:
            mock_eval.return_value = ({"mse": 0.5}, [], [])

            results = iterative_experiment(
                goal=mock_goal,
                scientist=mock_scientist,
                num_experiments=[2, 5],  # budgets 2 and 5
                num_evals=1,
                include_prior=False,
            )

            # should call 5 times total (incremental: 2 + 3 more = 5)
            assert mock_scientist.generate_actions.call_count == 5

    def test_result_structure(self, mock_goal, mock_scientist):
        """Test returned result has expected structure."""
        with patch("boxing_gym.experiment.loop.evaluate") as mock_eval:
            mock_eval.return_value = ({"mse": 0.5, "std_mse": 0.1}, [], [])

            result_tuple = iterative_experiment(
                goal=mock_goal,
                scientist=mock_scientist,
                num_experiments=[2],
                num_evals=1,
                include_prior=False,
                norm_mu=0.0,
                norm_sigma=1.0,
            )

            # returns: (results, queries, observations, successes, explanations, eigs, programs)
            assert len(result_tuple) == 7
            results, queries, observations, successes, explanations, eigs, programs = result_tuple
            assert isinstance(results, list)
            assert isinstance(queries, list)
            assert isinstance(observations, list)
            assert len(results) == 1  # one budget checkpoint

    def test_step_logger_called(self, mock_goal, mock_scientist):
        """Test step logger receives updates."""
        mock_logger = Mock()
        mock_logger.log_step = Mock()
        mock_logger.log_evaluation = Mock()

        with patch("boxing_gym.experiment.loop.evaluate") as mock_eval:
            mock_eval.return_value = ({"mse": 0.5}, [], [])

            iterative_experiment(
                goal=mock_goal,
                scientist=mock_scientist,
                num_experiments=[2],
                num_evals=1,
                include_prior=False,
                step_logger=mock_logger,
            )

            # step logger should be called for each step
            assert mock_logger.log_step.call_count >= 1


class TestIterativeExperimentEdgeCases:
    """Edge case tests for experiment loop."""

    @pytest.fixture
    def minimal_goal(self):
        goal = Mock()
        goal.get_system_message.return_value = ""
        goal.get_goal_description.return_value = ""
        goal.get_simulation_description.return_value = ""
        goal.get_queries.return_value = []
        goal.evaluate_predictions.return_value = {"mse": 0.0}
        goal.env = Mock()
        goal.env.run_experiment.return_value = ("obs", True)
        return goal

    @pytest.fixture
    def minimal_scientist(self):
        scientist = Mock()
        scientist.generate_actions.return_value = ""
        scientist.generate_predictions.return_value = "0"
        scientist.update_history = Mock()
        scientist.get_latencies_ms.return_value = []
        return scientist

    def test_scientist_exception_propagates(self, minimal_goal, minimal_scientist):
        """Test that scientist exceptions propagate (not swallowed)."""
        minimal_scientist.generate_actions.side_effect = Exception("LLM failed")

        with patch("boxing_gym.experiment.loop.evaluate") as mock_eval:
            mock_eval.return_value = ({"mse": 1.0}, [], [])

            # scientist exceptions in generate_actions should propagate
            with pytest.raises(Exception, match="LLM failed"):
                iterative_experiment(
                    goal=minimal_goal,
                    scientist=minimal_scientist,
                    num_experiments=[1],
                    num_evals=1,
                    include_prior=False,
                )

    def test_handles_goal_exception(self, minimal_goal, minimal_scientist):
        """Test graceful handling when goal evaluation raises."""
        with patch("boxing_gym.experiment.loop.evaluate") as mock_eval:
            mock_eval.side_effect = Exception("Evaluation failed")

            result_tuple = iterative_experiment(
                goal=minimal_goal,
                scientist=minimal_scientist,
                num_experiments=[1],
                num_evals=1,
                include_prior=False,
            )

            # should still return 7-tuple structure
            assert len(result_tuple) == 7
            results, queries, observations, successes, _, _, _ = result_tuple
            # results should contain fallback error result
            assert len(results) == 1
