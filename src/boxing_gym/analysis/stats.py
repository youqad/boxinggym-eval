"""Statistical tests for benchmark analysis with multiple comparison corrections."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class TTestResult:
    """Result of a t-test."""

    t_statistic: float
    p_value: float
    significant: bool  # at alpha=0.05
    interpretation: str


@dataclass
class BootstrapCIResult:
    """Result of bootstrap confidence interval."""

    mean: float
    ci_low: float
    ci_high: float
    confidence: float


@dataclass
class EffectSizeResult:
    """Result of effect size calculation."""

    d: float
    interpretation: Literal["negligible", "small", "medium", "large"]


def welch_ttest(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> TTestResult:
    """Welch's t-test for unequal variances.

    Use this for comparing independent samples (e.g., different models).
    """
    from scipy import stats

    a = np.asarray(a)
    b = np.asarray(b)

    # drop NaN values (align with Pandas behavior)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) < 2 or len(b) < 2:
        return TTestResult(
            t_statistic=float("nan"),
            p_value=1.0,
            significant=False,
            interpretation="insufficient data",
        )

    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    significant = p_val < alpha

    # interpret direction
    if significant:
        if np.mean(a) < np.mean(b):
            interp = "A significantly better (lower z)"
        else:
            interp = "B significantly better (lower z)"
    else:
        interp = "no significant difference"

    return TTestResult(
        t_statistic=float(t_stat),
        p_value=float(p_val),
        significant=significant,
        interpretation=interp,
    )


def paired_ttest(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> TTestResult:
    """Paired t-test for matched samples.

    Use this when comparing matched runs (same seed/prompt/environment).
    """
    from scipy import stats

    a = np.asarray(a)
    b = np.asarray(b)

    if len(a) != len(b):
        raise ValueError("paired t-test requires equal length arrays")

    # drop pairs where either value is NaN (maintain pairing)
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        return TTestResult(
            t_statistic=float("nan"),
            p_value=1.0,
            significant=False,
            interpretation="insufficient data",
        )

    t_stat, p_val = stats.ttest_rel(a, b)
    significant = p_val < alpha

    diff_mean = np.mean(a - b)
    if significant:
        if diff_mean < 0:
            interp = "A significantly better (lower z)"
        else:
            interp = "B significantly better (lower z)"
    else:
        interp = "no significant difference"

    return TTestResult(
        t_statistic=float(t_stat),
        p_value=float(p_val),
        significant=significant,
        interpretation=interp,
    )


def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = "mean",
    seed: int | None = 42,
) -> BootstrapCIResult:
    """Bootstrap confidence interval via resampling. NaN values dropped.

    Default seed=42 for reproducibility. Pass None for fresh randomness.
    Uses vectorized sampling when memory-safe (~100x faster); RNG draw order
    differs from loop-based sampling so results may differ for the same seed.
    """
    data = np.asarray(data)

    # drop NaN values (align with Pandas behavior)
    data = data[~np.isnan(data)]

    if len(data) < 2:
        mean_val = float(data[0]) if len(data) == 1 else float("nan")
        return BootstrapCIResult(
            mean=mean_val, ci_low=mean_val, ci_high=mean_val, confidence=confidence
        )

    if statistic not in ("mean", "median"):
        raise ValueError(f"unknown statistic: {statistic}")

    rng = np.random.default_rng(seed)

    # memory guard: vectorize only if safe
    # threshold: ~100M elements (~800MB for float64)
    MEMORY_THRESHOLD = 100_000_000

    if len(data) * n_bootstrap < MEMORY_THRESHOLD:
        # vectorized path (fast, higher memory)
        samples = rng.choice(data, size=(n_bootstrap, len(data)), replace=True)
        if statistic == "mean":
            bootstrap_stats = np.mean(samples, axis=1)
        else:  # median
            bootstrap_stats = np.median(samples, axis=1)
    else:
        # fallback to loop for large datasets (slower, safe memory)
        bootstrap_stats = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            if statistic == "mean":
                bootstrap_stats[i] = np.mean(sample)
            else:  # median
                bootstrap_stats[i] = np.median(sample)

    alpha = 1 - confidence
    ci_low = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    if statistic == "mean":
        center = float(np.mean(data))
    else:
        center = float(np.median(data))

    return BootstrapCIResult(mean=center, ci_low=ci_low, ci_high=ci_high, confidence=confidence)


def cohens_d(a: np.ndarray, b: np.ndarray) -> EffectSizeResult:
    """Cohen's d effect size.

    Interpretation thresholds (Cohen 1988):
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # drop NaN values (align with Pandas behavior)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) < 2 or len(b) < 2:
        return EffectSizeResult(d=float("nan"), interpretation="negligible")

    # pooled standard deviation
    n1, n2 = len(a), len(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        # if means differ but variance is 0, effect is infinite
        mean_diff = np.mean(a) - np.mean(b)
        if mean_diff == 0:
            d = 0.0  # identical distributions
        else:
            d = float("inf") if mean_diff > 0 else float("-inf")
    else:
        d = (np.mean(a) - np.mean(b)) / pooled_std

    # interpret
    abs_d = abs(d)
    if np.isinf(abs_d):
        interp = "large"  # infinite = perfect separation
    elif abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return EffectSizeResult(d=float(d), interpretation=interp)


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Returns:
        adjusted_p: FDR-adjusted p-values
        significant: boolean array of significant tests
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    # sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH adjustment
    adjusted = np.zeros(n)
    for i, rank in enumerate(range(1, n + 1)):
        adjusted[sorted_idx[i]] = sorted_p[i] * n / rank

    # ensure monotonicity (adjusted p-values should be non-decreasing in original order)
    # work backwards through sorted order
    adjusted_sorted = adjusted[sorted_idx]
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    # put back in original order
    for i, idx in enumerate(sorted_idx):
        adjusted[idx] = min(adjusted_sorted[i], 1.0)  # cap at 1.0

    significant = adjusted < alpha

    return adjusted, significant


def bonferroni_correct(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Bonferroni correction for multiple comparisons.

    More conservative than Benjamini-Hochberg.

    Returns:
        adjusted_p: Bonferroni-adjusted p-values
        significant: boolean array of significant tests
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    adjusted = np.minimum(p_values * n, 1.0)
    significant = adjusted < alpha

    return adjusted, significant


def sem(data: np.ndarray) -> float:
    """Standard error of the mean."""
    data = np.asarray(data)
    # drop NaN values (align with Pandas behavior)
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return float("nan")
    return float(np.std(data, ddof=1) / np.sqrt(len(data)))


def compare_models(
    df, model_col: str = "model", score_col: str = "z_mean", reference_model: str | None = None
) -> dict:
    """Compare all models against a reference (or best model).

    Returns dict with pairwise comparisons and FDR-corrected p-values.
    """
    models = df[model_col].unique()

    if reference_model is None:
        # use best model as reference
        means = df.groupby(model_col)[score_col].mean()
        reference_model = means.idxmin()

    ref_scores = df[df[model_col] == reference_model][score_col].values

    comparisons = []
    p_values = []

    for model in models:
        if model == reference_model:
            continue

        model_scores = df[df[model_col] == model][score_col].values
        result = welch_ttest(ref_scores, model_scores)

        comparisons.append(
            {
                "model": model,
                "reference": reference_model,
                "t_statistic": result.t_statistic,
                "p_value": result.p_value,
                "interpretation": result.interpretation,
            }
        )
        p_values.append(result.p_value)

    # FDR correction
    if p_values:
        adjusted_p, significant = benjamini_hochberg(np.array(p_values))
        for i, comp in enumerate(comparisons):
            comp["p_adjusted"] = adjusted_p[i]
            comp["significant_fdr"] = significant[i]

    return {
        "reference": reference_model,
        "comparisons": comparisons,
    }
