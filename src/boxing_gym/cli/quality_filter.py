"""Quality filtering for analysis views."""

from __future__ import annotations

import numpy as np
import pandas as pd

from boxing_gym.data_quality.config import Z_OUTLIER_THRESHOLD


def apply_quality_filters(
    df: pd.DataFrame,
    z_col: str = "z_mean",
    z_threshold: float = Z_OUTLIER_THRESHOLD,
    min_budget: int = 0,
    budget_col: str = "config/budget",
    drop_nan: bool = True,
    drop_inf: bool = True,
) -> pd.DataFrame:
    """Filter DataFrame by z-score validity and optional budget threshold."""
    if z_col not in df.columns:
        return df.copy()

    z = pd.to_numeric(df[z_col], errors="coerce")

    mask = pd.Series(True, index=df.index)

    if drop_nan:
        mask &= z.notna()

    if drop_inf:
        mask &= ~np.isinf(z)

    # threshold only applies to finite values
    mask &= (z.abs() <= z_threshold) | ~np.isfinite(z)

    if min_budget > 0 and budget_col in df.columns:
        mask &= df[budget_col] >= min_budget

    return df[mask].copy()
