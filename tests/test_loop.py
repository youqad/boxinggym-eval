"""Tests for boxing_gym.experiment.loop module.

These tests lock current behavior for paper reproducibility.
"""

from unittest.mock import Mock

import pytest

from boxing_gym.experiment.loop import (
    _compute_z_stats,
    _get_latency_count,
    _get_step_latency_ms,
    _normalize_budgets,
)


class TestNormalizeBudgets:
    """Tests for _normalize_budgets function."""

    def test_none_returns_empty_list(self):
        assert _normalize_budgets(None) == []

    def test_single_int_returns_single_element_list(self):
        assert _normalize_budgets(5) == [5]
        assert _normalize_budgets(10) == [10]

    def test_single_string_int_returns_single_element_list(self):
        assert _normalize_budgets("5") == [5]

    def test_list_returns_sorted_unique(self):
        assert _normalize_budgets([3, 1, 2]) == [1, 2, 3]
        assert _normalize_budgets([5, 5, 3, 3, 1]) == [1, 3, 5]

    def test_tuple_returns_sorted_unique(self):
        assert _normalize_budgets((3, 1, 2)) == [1, 2, 3]

    def test_set_returns_sorted_list(self):
        result = _normalize_budgets({3, 1, 2})
        assert result == [1, 2, 3]

    def test_empty_list_returns_empty_list(self):
        assert _normalize_budgets([]) == []

    def test_mixed_types_in_list_converts_to_int(self):
        assert _normalize_budgets([1, "2", 3.0]) == [1, 2, 3]

    def test_invalid_values_in_list_are_skipped(self):
        assert _normalize_budgets([1, "invalid", 3]) == [1, 3]

    def test_invalid_single_value_returns_empty(self):
        assert _normalize_budgets("invalid") == []


class TestGetStepLatencyMs:
    """Tests for _get_step_latency_ms function."""

    def test_returns_sum_of_new_latencies(self):
        scientist = Mock()
        scientist.get_latencies_ms.return_value = [100.0, 200.0, 150.0]

        # prev_latency_count=1 means skip first, sum remaining
        result = _get_step_latency_ms(scientist, prev_latency_count=1)
        assert result == 350.0  # 200 + 150

    def test_returns_none_when_no_new_latencies(self):
        scientist = Mock()
        scientist.get_latencies_ms.return_value = [100.0, 200.0]

        result = _get_step_latency_ms(scientist, prev_latency_count=2)
        assert result is None

    def test_returns_none_when_no_latency_method(self):
        scientist = Mock(spec=[])  # no get_latencies_ms
        result = _get_step_latency_ms(scientist, prev_latency_count=0)
        assert result is None

    def test_handles_exception_gracefully(self):
        scientist = Mock()
        scientist.get_latencies_ms.side_effect = RuntimeError("test error")

        result = _get_step_latency_ms(scientist, prev_latency_count=0)
        assert result is None


class TestGetLatencyCount:
    """Tests for _get_latency_count function."""

    def test_returns_length_of_latencies(self):
        scientist = Mock()
        scientist.get_latencies_ms.return_value = [100.0, 200.0, 300.0]

        assert _get_latency_count(scientist) == 3

    def test_returns_zero_when_no_latency_method(self):
        scientist = Mock(spec=[])
        assert _get_latency_count(scientist) == 0

    def test_returns_zero_on_exception(self):
        scientist = Mock()
        scientist.get_latencies_ms.side_effect = RuntimeError("test error")

        assert _get_latency_count(scientist) == 0


class TestComputeZStats:
    """Tests for _compute_z_stats function."""

    def test_returns_none_when_eval_score_is_none(self):
        z_mean, z_std = _compute_z_stats(None, norm_mu=0.5, norm_sigma=0.1)
        assert z_mean is None
        assert z_std is None

    def test_returns_none_when_norm_mu_is_none(self):
        z_mean, z_std = _compute_z_stats({"mse": 0.5}, norm_mu=None, norm_sigma=0.1)
        assert z_mean is None
        assert z_std is None

    def test_returns_none_when_norm_sigma_is_none(self):
        z_mean, z_std = _compute_z_stats({"mse": 0.5}, norm_mu=0.5, norm_sigma=None)
        assert z_mean is None
        assert z_std is None

    def test_returns_none_when_norm_sigma_is_zero(self):
        z_mean, z_std = _compute_z_stats({"mse": 0.5}, norm_mu=0.5, norm_sigma=0)
        assert z_mean is None
        assert z_std is None

    def test_returns_none_when_norm_sigma_is_negative(self):
        z_mean, z_std = _compute_z_stats({"mse": 0.5}, norm_mu=0.5, norm_sigma=-0.1)
        assert z_mean is None
        assert z_std is None

    def test_dict_with_mse_key(self):
        z_mean, z_std = _compute_z_stats({"mse": 0.7, "std_mse": 0.05}, norm_mu=0.5, norm_sigma=0.1)
        assert z_mean == pytest.approx(2.0)  # (0.7 - 0.5) / 0.1
        assert z_std == pytest.approx(0.5)  # 0.05 / 0.1

    def test_dict_with_accuracy_key(self):
        z_mean, z_std = _compute_z_stats(
            {"accuracy": 0.8, "std_accuracy": 0.02}, norm_mu=0.7, norm_sigma=0.05
        )
        assert z_mean == pytest.approx(2.0)  # (0.8 - 0.7) / 0.05
        assert z_std == pytest.approx(0.4)  # 0.02 / 0.05

    def test_dict_with_score_key(self):
        z_mean, z_std = _compute_z_stats({"score": 1.0}, norm_mu=0.5, norm_sigma=0.25)
        assert z_mean == pytest.approx(2.0)  # (1.0 - 0.5) / 0.25
        assert z_std == 0.0  # no std provided, defaults to 0

    def test_tuple_with_mean_only(self):
        z_mean, z_std = _compute_z_stats((0.6,), norm_mu=0.5, norm_sigma=0.1)
        assert z_mean == pytest.approx(1.0)
        assert z_std == 0.0  # defaults to 0 when no std

    def test_tuple_with_mean_and_std(self):
        z_mean, z_std = _compute_z_stats((0.6, 0.02), norm_mu=0.5, norm_sigma=0.1)
        assert z_mean == pytest.approx(1.0)
        assert z_std == pytest.approx(0.2)

    def test_list_with_mean_and_std(self):
        z_mean, z_std = _compute_z_stats([0.6, 0.02], norm_mu=0.5, norm_sigma=0.1)
        assert z_mean == pytest.approx(1.0)
        assert z_std == pytest.approx(0.2)

    def test_empty_dict_returns_none(self):
        z_mean, z_std = _compute_z_stats({}, norm_mu=0.5, norm_sigma=0.1)
        assert z_mean is None
        assert z_std is None

    def test_empty_list_returns_none(self):
        z_mean, z_std = _compute_z_stats([], norm_mu=0.5, norm_sigma=0.1)
        assert z_mean is None
        assert z_std is None
