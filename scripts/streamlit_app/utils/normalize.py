"""Helpers to normalize common column names across data sources."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first column that exists in the DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_env_column(df: pd.DataFrame) -> str | None:
    return first_existing_column(df, ["config/envs", "env", "envs"])


def get_model_column(df: pd.DataFrame) -> str | None:
    return first_existing_column(df, ["config/llms", "model", "llms"])


def get_metric_column(df: pd.DataFrame, target_metric: str = "metric/eval/z_mean") -> str | None:
    return first_existing_column(df, [target_metric, "metric/eval/z_mean", "z_mean"])


def get_budget_column(df: pd.DataFrame) -> str | None:
    return first_existing_column(
        df, ["config/budget", "config/exp.budget", "metric/budget", "budget"]
    )


def normalize_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return numeric series for a column, coercing errors to NaN."""
    if column not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")
