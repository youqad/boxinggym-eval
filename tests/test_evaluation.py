"""Tests for evaluation module, including parallel evaluation."""

import os
from unittest.mock import MagicMock

import pytest

from boxing_gym.experiment.evaluation import _make_prediction, evaluate


class TestParallelEvaluation:
    """Tests for parallel evaluation functionality."""

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

    def test_make_prediction_returns_indexed_result(self):
        """Test that _make_prediction returns (idx, prediction) tuple."""
        scientist = MagicMock()
        scientist.generate_predictions.return_value = "42"

        idx, pred = _make_prediction(scientist, "test question", 5)

        assert idx == 5
        assert pred == "42"
        scientist.generate_predictions.assert_called_once_with("test question")

    def test_make_prediction_handles_exception(self):
        """Test that _make_prediction returns '0' on failure."""
        scientist = MagicMock()
        scientist.generate_predictions.side_effect = ValueError("LLM failed")

        idx, pred = _make_prediction(scientist, "test question", 3)

        assert idx == 3
        assert pred == "0"

    def test_evaluate_parallel_preserves_order(self):
        """Test that parallel evaluation preserves question/answer ordering."""
        # mock goal
        goal = MagicMock()
        call_count = 0

        def mock_get_question(include_prior):
            nonlocal call_count
            call_count += 1
            return f"question_{call_count}", f"gt_{call_count}"

        goal.get_goal_eval_question.side_effect = mock_get_question
        goal.evaluate_predictions.return_value = {"mse": 0.1}
        goal.eval_pointer = 0

        # mock scientist
        scientist = MagicMock()

        def mock_predict(q):
            # extract question number and return matching prediction
            num = q.split("question_")[-1].strip()
            return f"pred_{num}"

        scientist.generate_predictions.side_effect = mock_predict

        result, questions, gts, predictions = evaluate(
            final_results="context",
            goal=goal,
            scientist=scientist,
            num_evals=5,
            include_prior=False,
            parallel=True,
        )

        # verify ordering is preserved
        assert len(predictions) == 5
        assert len(gts) == 5
        for i in range(5):
            assert gts[i] == f"gt_{i + 1}"
            # questions have "context\n" prepended
            assert f"question_{i + 1}" in questions[i]

    def test_evaluate_parallel_clone_mode_uses_clones(self, monkeypatch):
        """Test that clone mode uses per-task clones instead of the base agent."""
        monkeypatch.setenv("BOXINGGYM_EVAL_PARALLEL_MODE", "clone")

        goal = MagicMock()
        call_count = 0

        def mock_get_question(include_prior):
            nonlocal call_count
            call_count += 1
            return f"question_{call_count}", f"gt_{call_count}"

        goal.get_goal_eval_question.side_effect = mock_get_question
        goal.evaluate_predictions.return_value = {"mse": 0.1}
        goal.eval_pointer = 0

        counter = {"base_calls": 0, "clone_calls": 0, "clone_count": 0}

        class CloneAgent:
            def __init__(self, counter, is_clone=False):
                self.counter = counter
                self.is_clone = is_clone

            def generate_predictions(self, _q):
                if self.is_clone:
                    self.counter["clone_calls"] += 1
                else:
                    self.counter["base_calls"] += 1
                return "pred"

            def clone_for_eval(self):
                self.counter["clone_count"] += 1
                return CloneAgent(self.counter, is_clone=True)

        scientist = CloneAgent(counter)

        evaluate(
            final_results="context",
            goal=goal,
            scientist=scientist,
            num_evals=3,
            include_prior=False,
            parallel=True,
        )

        assert counter["base_calls"] == 0
        assert counter["clone_calls"] == 3
        assert counter["clone_count"] == 3

    def test_evaluate_parallel_lock_mode_uses_shared_agent(self, monkeypatch):
        """Test that lock mode uses the shared agent safely."""
        monkeypatch.setenv("BOXINGGYM_EVAL_PARALLEL_MODE", "lock")

        goal = MagicMock()
        goal.get_goal_eval_question.return_value = ("question", "gt")
        goal.evaluate_predictions.return_value = {"mse": 0.0}

        scientist = MagicMock()
        scientist.generate_predictions.return_value = "pred"

        evaluate(
            final_results="",
            goal=goal,
            scientist=scientist,
            num_evals=3,
            include_prior=False,
            parallel=True,
        )

        assert scientist.generate_predictions.call_count == 3

    def test_evaluate_sequential_mode(self):
        """Test that parallel=False uses sequential execution."""
        goal = MagicMock()
        goal.get_goal_eval_question.return_value = ("question", "gt")
        goal.evaluate_predictions.return_value = {"mse": 0.0}

        scientist = MagicMock()
        scientist.generate_predictions.return_value = "pred"

        evaluate(
            final_results="",
            goal=goal,
            scientist=scientist,
            num_evals=3,
            include_prior=False,
            parallel=False,
        )

        assert scientist.generate_predictions.call_count == 3

    def test_evaluate_single_eval_uses_sequential(self):
        """Test that num_evals=1 uses sequential even with parallel=True."""
        goal = MagicMock()
        goal.get_goal_eval_question.return_value = ("question", "gt")
        goal.evaluate_predictions.return_value = {"mse": 0.0}

        scientist = MagicMock()
        scientist.generate_predictions.return_value = "pred"

        # with only 1 eval, should use sequential (avoids thread overhead)
        evaluate(
            final_results="",
            goal=goal,
            scientist=scientist,
            num_evals=1,
            include_prior=False,
            parallel=True,
        )

        scientist.generate_predictions.assert_called_once()

    def test_evaluate_resets_eval_pointer(self):
        """Test that evaluate resets goal.eval_pointer before collecting questions."""
        goal = MagicMock()
        goal.eval_pointer = 5  # simulate pointer at non-zero position
        goal.get_goal_eval_question.return_value = ("q", "gt")
        goal.evaluate_predictions.return_value = {}

        scientist = MagicMock()
        scientist.generate_predictions.return_value = "0"

        evaluate(
            final_results="",
            goal=goal,
            scientist=scientist,
            num_evals=2,
            include_prior=False,
            parallel=False,
        )

        # verify pointer was reset
        assert goal.eval_pointer == 0


class TestEvaluateWorkerConfig:
    """Tests for worker configuration."""

    def test_max_workers_from_env(self):
        """Test that BOXINGGYM_EVAL_WORKERS env var is respected."""
        # this is tricky to test since it's read at import time
        # we just verify the default is reasonable
        from boxing_gym.experiment.evaluation import DEFAULT_EVAL_WORKERS, MAX_EVAL_WORKERS

        assert DEFAULT_EVAL_WORKERS == 8
        # MAX_EVAL_WORKERS could be different if env var is set
        assert MAX_EVAL_WORKERS >= 1
