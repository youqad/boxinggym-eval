"""Tests for statistical analysis functions.

Phase 0 safety net: capture current behavior BEFORE any changes.
"""

import numpy as np
import pytest
from boxing_gym.analysis.stats import (
    welch_ttest,
    paired_ttest,
    bootstrap_ci,
    cohens_d,
    benjamini_hochberg,
    bonferroni_correct,
    sem,
    TTestResult,
    BootstrapCIResult,
    EffectSizeResult,
)


class TestWelchTTest:
    """Tests for Welch's t-test."""

    def test_detects_significant_difference(self):
        """Clearly different distributions should be significant."""
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([10, 11, 12, 13, 14])
        result = welch_ttest(a, b)

        assert isinstance(result, TTestResult)
        assert result.p_value < 0.01
        assert result.significant
        assert "significantly better" in result.interpretation

    def test_no_difference_same_distribution(self):
        """Similar distributions should not be significant."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100)
        b = rng.normal(0, 1, 100)
        result = welch_ttest(a, b)

        assert result.p_value > 0.05
        assert not result.significant
        assert "no significant difference" in result.interpretation

    def test_handles_unequal_variances(self):
        """Welch's test should handle unequal variances."""
        a = np.array([1, 2, 3])  # small variance
        b = np.array([0, 5, 10, 15, 20])  # large variance
        result = welch_ttest(a, b)

        # Should complete without error
        assert isinstance(result.t_statistic, float)
        assert isinstance(result.p_value, float)

    def test_handles_nan_values(self):
        """NaN values should be dropped."""
        a = np.array([1, 2, np.nan, 4, 5])
        b = np.array([10, np.nan, 12, 13, 14])
        result = welch_ttest(a, b)

        # Should still detect significant difference
        assert result.significant

    def test_insufficient_data_returns_not_significant(self):
        """Single element arrays should return not significant."""
        a = np.array([1])
        b = np.array([2])
        result = welch_ttest(a, b)

        assert not result.significant
        assert result.p_value == 1.0
        assert "insufficient data" in result.interpretation

    def test_custom_alpha(self):
        """Custom alpha should affect significance threshold."""
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([3, 4, 5, 6, 7])  # overlapping but different

        # With alpha=0.05, might be significant
        result_05 = welch_ttest(a, b, alpha=0.05)
        # With alpha=0.001, likely not significant
        result_001 = welch_ttest(a, b, alpha=0.001)

        # p-value should be the same, significance should differ
        assert result_05.p_value == result_001.p_value


class TestPairedTTest:
    """Tests for paired t-test."""

    def test_detects_paired_difference(self):
        """Paired differences should be detected."""
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 3, 4, 5, 6])  # consistently +1
        result = paired_ttest(a, b)

        assert result.significant

    def test_no_difference_same_values(self):
        """Identical values should show no difference."""
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1, 2, 3, 4, 5])
        result = paired_ttest(a, b)

        # When all differences are 0, t-statistic is NaN (0/0)
        assert np.isnan(result.t_statistic)
        assert not result.significant

    def test_requires_equal_length(self):
        """Should raise error for unequal lengths."""
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3, 4])

        with pytest.raises(ValueError, match="equal length"):
            paired_ttest(a, b)

    def test_handles_nan_pairs(self):
        """NaN pairs should be dropped together."""
        a = np.array([1, 2, np.nan, 4, 5])
        b = np.array([2, np.nan, 4, 5, 6])
        result = paired_ttest(a, b)

        # Should complete without error (3 valid pairs remain)
        assert isinstance(result, TTestResult)


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_ci_contains_true_mean(self):
        """CI should contain the population mean with high probability."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, 100)
        result = bootstrap_ci(data, n_bootstrap=1000)

        assert isinstance(result, BootstrapCIResult)
        # The true mean (5.0) should be within CI most of the time
        # (this specific seed should work)
        assert result.ci_low < 5.0 < result.ci_high

    def test_reproducible_default_seed(self):
        """Bootstrap should be reproducible with default seed=42."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r1 = bootstrap_ci(data, n_bootstrap=1000)
        r2 = bootstrap_ci(data, n_bootstrap=1000)

        # Should be identical due to default seed=42
        assert r1.ci_low == r2.ci_low
        assert r1.ci_high == r2.ci_high

    def test_seed_none_produces_variation(self):
        """seed=None should produce different results each call."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        results = [bootstrap_ci(data, n_bootstrap=100, seed=None) for _ in range(5)]

        # At least some CI bounds should differ (not all identical)
        ci_lows = [r.ci_low for r in results]
        assert len(set(ci_lows)) > 1, "seed=None should produce variation"

    def test_median_statistic(self):
        """Median statistic should compute median."""
        data = np.array([1, 2, 3, 100])  # outlier
        result = bootstrap_ci(data, statistic="median")

        # Mean would be ~26.5, median is 2.5
        assert result.mean == np.median(data)

    def test_mean_statistic(self):
        """Mean statistic should compute mean."""
        data = np.array([1, 2, 3, 4, 5])
        result = bootstrap_ci(data, statistic="mean")

        assert result.mean == np.mean(data)

    def test_single_element(self):
        """Single element should return that value as CI bounds."""
        data = np.array([5.0])
        result = bootstrap_ci(data, n_bootstrap=100)

        assert result.mean == 5.0
        assert result.ci_low == 5.0
        assert result.ci_high == 5.0

    def test_empty_after_nan_removal(self):
        """All NaN array should return NaN."""
        data = np.array([np.nan, np.nan])
        result = bootstrap_ci(data)

        assert np.isnan(result.mean)

    def test_handles_nan_values(self):
        """NaN values should be dropped."""
        data = np.array([1, 2, np.nan, 4, 5])
        result = bootstrap_ci(data)

        # Should compute based on [1, 2, 4, 5]
        assert result.mean == np.mean([1, 2, 4, 5])

    def test_invalid_statistic_raises(self):
        """Invalid statistic should raise ValueError."""
        data = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="unknown statistic"):
            bootstrap_ci(data, statistic="invalid")

    def test_confidence_level(self):
        """Different confidence levels should affect CI width."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        ci_95 = bootstrap_ci(data, confidence=0.95)
        ci_99 = bootstrap_ci(data, confidence=0.99)

        # 99% CI should be wider than 95% CI
        width_95 = ci_95.ci_high - ci_95.ci_low
        width_99 = ci_99.ci_high - ci_99.ci_low
        assert width_99 >= width_95


class TestCohensD:
    """Tests for Cohen's d effect size."""

    def test_large_effect(self):
        """Clearly different groups should have large effect."""
        a = np.array([1, 2, 3])
        b = np.array([10, 11, 12])
        result = cohens_d(a, b)

        assert isinstance(result, EffectSizeResult)
        assert abs(result.d) > 0.8
        assert result.interpretation == "large"

    def test_negligible_effect(self):
        """Very similar groups should have negligible effect."""
        # Groups with tiny difference relative to variance
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # +0.1 shift vs std ~1.58
        result = cohens_d(a, b)

        assert abs(result.d) < 0.2
        assert result.interpretation == "negligible"

    def test_small_effect(self):
        """Small but noticeable difference."""
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = cohens_d(a, b)

        assert 0.2 <= abs(result.d) < 0.5
        assert result.interpretation == "small"

    def test_medium_effect(self):
        """Medium difference."""
        # For medium effect (0.5 <= |d| < 0.8), shift by ~1.0 vs std ~1.58
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 3, 4, 5, 6])  # +1 shift
        result = cohens_d(a, b)

        assert 0.5 <= abs(result.d) < 0.8
        assert result.interpretation == "medium"

    def test_identical_distributions(self):
        """Identical distributions should have d=0."""
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1, 2, 3, 4, 5])
        result = cohens_d(a, b)

        assert result.d == 0.0
        assert result.interpretation == "negligible"

    def test_handles_nan(self):
        """NaN values should be dropped."""
        a = np.array([1, 2, np.nan, 4, 5])
        b = np.array([10, 11, 12, np.nan, 14])
        result = cohens_d(a, b)

        # Should still detect large effect
        assert result.interpretation == "large"

    def test_insufficient_data(self):
        """Single element should return NaN d."""
        a = np.array([1])
        b = np.array([2])
        result = cohens_d(a, b)

        assert np.isnan(result.d)
        assert result.interpretation == "negligible"


class TestBenjaminiHochberg:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_controls_fdr(self):
        """Adjusted p-values should maintain monotonicity."""
        p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        adjusted, significant = benjamini_hochberg(p_values)

        # Adjusted p-values should be monotonic (in sorted order)
        sorted_idx = np.argsort(p_values)
        adjusted_sorted = adjusted[sorted_idx]
        for i in range(len(adjusted_sorted) - 1):
            assert adjusted_sorted[i] <= adjusted_sorted[i + 1]

        # Most significant should remain significant at alpha=0.05
        assert significant[0]

    def test_all_significant(self):
        """All very small p-values should remain significant."""
        p_values = np.array([0.001, 0.002, 0.003])
        adjusted, significant = benjamini_hochberg(p_values)

        assert all(significant)
        assert all(adjusted < 0.05)

    def test_none_significant(self):
        """All large p-values should remain non-significant."""
        p_values = np.array([0.5, 0.6, 0.7])
        adjusted, significant = benjamini_hochberg(p_values)

        assert not any(significant)

    def test_empty_array(self):
        """Empty array should return empty results."""
        p_values = np.array([])
        adjusted, significant = benjamini_hochberg(p_values)

        assert len(adjusted) == 0
        assert len(significant) == 0

    def test_single_value(self):
        """Single p-value should be unchanged."""
        p_values = np.array([0.03])
        adjusted, significant = benjamini_hochberg(p_values)

        assert len(adjusted) == 1
        assert adjusted[0] == 0.03
        assert significant[0]

    def test_adjusted_capped_at_one(self):
        """Adjusted p-values should not exceed 1.0."""
        p_values = np.array([0.9, 0.95, 0.99])
        adjusted, _ = benjamini_hochberg(p_values)

        assert all(adjusted <= 1.0)

    def test_custom_alpha(self):
        """Custom alpha should affect significance."""
        p_values = np.array([0.01, 0.02, 0.03])

        _, sig_05 = benjamini_hochberg(p_values, alpha=0.05)
        _, sig_001 = benjamini_hochberg(p_values, alpha=0.001)

        # More significant with alpha=0.05 than alpha=0.001
        assert sum(sig_05) >= sum(sig_001)


class TestBonferroniCorrect:
    """Tests for Bonferroni correction."""

    def test_multiplies_by_n(self):
        """Bonferroni should multiply p-values by n."""
        p_values = np.array([0.01, 0.02, 0.03])
        adjusted, _ = bonferroni_correct(p_values)

        # Should multiply each by 3
        np.testing.assert_array_almost_equal(adjusted, [0.03, 0.06, 0.09])

    def test_capped_at_one(self):
        """Adjusted p-values should not exceed 1.0."""
        p_values = np.array([0.5, 0.6, 0.7])
        adjusted, _ = bonferroni_correct(p_values)

        assert all(adjusted <= 1.0)

    def test_empty_array(self):
        """Empty array should return empty results."""
        p_values = np.array([])
        adjusted, significant = bonferroni_correct(p_values)

        assert len(adjusted) == 0
        assert len(significant) == 0

    def test_more_conservative_than_bh(self):
        """Bonferroni should be more conservative than BH."""
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        _, sig_bonf = bonferroni_correct(p_values)
        _, sig_bh = benjamini_hochberg(p_values)

        # BH should find more significant results
        assert sum(sig_bh) >= sum(sig_bonf)


class TestSEM:
    """Tests for standard error of the mean."""

    def test_computes_sem(self):
        """SEM should be std / sqrt(n)."""
        data = np.array([1, 2, 3, 4, 5])
        result = sem(data)

        expected = np.std(data, ddof=1) / np.sqrt(len(data))
        assert result == pytest.approx(expected)

    def test_handles_nan(self):
        """NaN values should be dropped."""
        data = np.array([1, 2, np.nan, 4, 5])
        result = sem(data)

        clean_data = np.array([1, 2, 4, 5])
        expected = np.std(clean_data, ddof=1) / np.sqrt(len(clean_data))
        assert result == pytest.approx(expected)

    def test_single_element_returns_nan(self):
        """Single element should return NaN."""
        data = np.array([5.0])
        result = sem(data)

        assert np.isnan(result)

    def test_empty_returns_nan(self):
        """Empty array should return NaN."""
        data = np.array([])
        result = sem(data)

        assert np.isnan(result)


class TestBootstrapCIPerformance:
    """Performance characterization tests for bootstrap CI.

    These tests document current performance to catch regressions
    or verify improvements after optimization.
    """

    def test_performance_baseline(self):
        """Document baseline performance for comparison after optimization."""
        import time

        data = np.random.randn(1000)

        start = time.time()
        bootstrap_ci(data, n_bootstrap=1000)
        elapsed = time.time() - start

        # Current implementation takes ~0.5-2s for 1000 bootstrap samples
        # After optimization, this should be <0.1s
        # For now, just verify it completes in reasonable time
        assert elapsed < 10.0  # generous timeout for CI environments
