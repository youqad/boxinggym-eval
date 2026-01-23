"""Cached data loading utilities for BoxingGym dashboard.

Wraps results_io.py functions with Streamlit caching.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Add src to path for imports
_src = Path(__file__).resolve().parent.parent.parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from boxing_gym.agents.results_io import (
    DEFAULT_ENTITY,
    DEFAULT_GOALS,
    DEFAULT_PROJECT,
    DataSource,
    PAPER_RESULTS,
    RunResult,
    aggregate_results,
    generate_paper_reference_rows,
    get_paper_value_4d,
    load_results_from_json_dir,
    load_results_from_local_wandb,
    load_results_from_wandb,
)


@st.cache_data(ttl=300, show_spinner="Loading from W&B...")
def load_wandb_results(
    sweep_id: str,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> List[RunResult]:
    """Load results from W&B API with 5-minute cache."""
    return load_results_from_wandb(sweep_id, entity, project)


@st.cache_data(ttl=60, show_spinner="Loading local results...")
def load_local_results(root: str) -> List[RunResult]:
    """Load results from local JSON files with 1-minute cache."""
    return load_results_from_json_dir(root)


@st.cache_data(ttl=60, show_spinner="Loading local W&B runs...")
def load_local_wandb_results(wandb_dir: str = "wandb-dir") -> List[RunResult]:
    """Load results from local W&B run directories with 1-minute cache.

    Much faster than API calls when data is already synced locally.
    Scans wandb-dir/run-*/files/wandb-summary.json for completed runs.
    """
    return load_results_from_local_wandb(wandb_dir)


@st.cache_data(ttl=300, show_spinner="Loading cached results...")
def load_parquet_results(parquet_path: str | None = None) -> pd.DataFrame:
    """Load results from parquet cache. Search order:
    1. Explicit path argument
    2. .boxing-gym-cache/runs.parquet (local dev, from `box sync`)
    3. demo_data/demo_runs.parquet (bundled for HF Space)

    Returns empty DataFrame if nothing found.
    """
    search_paths = []

    if parquet_path:
        search_paths.append(Path(parquet_path))

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    search_paths.append(project_root / ".boxing-gym-cache" / "runs.parquet")
    search_paths.append(Path(__file__).resolve().parent.parent / "demo_data" / "demo_runs.parquet")

    for p in search_paths:
        if p.exists():
            return pd.read_parquet(p)

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_paper_baselines() -> List[RunResult]:
    """Load paper reference rows (static, cache forever)."""
    gpt_rows = generate_paper_reference_rows("gpt-4o")
    box_rows = generate_paper_reference_rows("box")
    return gpt_rows + box_rows


@st.cache_data(show_spinner=False)
def compute_aggregated_results(
    results_hash: str,  # Hash of results for cache key
    results: List[Dict],
    group_by: Tuple[str, ...],
) -> List[Dict]:
    """Compute aggregated results with caching."""
    # Convert dicts back to RunResult objects
    run_results = [RunResult(**r) for r in results]
    agg = aggregate_results(run_results, group_by=group_by)
    return [r.__dict__ for r in agg]


def result_to_row_dict(r: RunResult) -> Dict:
    """Convert a RunResult to a row dict for display."""
    env_name = r.env
    goal_name = r.goal or ""
    include_prior = r.include_prior if r.include_prior is not None else False

    # Get paper values for comparison
    paper_gpt4o_b0 = get_paper_value_4d(env_name, goal_name, include_prior, 0, "gpt-4o")
    paper_gpt4o_b10 = get_paper_value_4d(env_name, goal_name, include_prior, 10, "gpt-4o")
    paper_box_b0 = get_paper_value_4d(env_name, goal_name, include_prior, 0, "box")
    paper_box_b10 = get_paper_value_4d(env_name, goal_name, include_prior, 10, "box")

    return {
        "path": r.path or "(W&B)" if r.source == DataSource.WANDB_API else r.path or "",
        "env": env_name,
        "goal": goal_name,
        "experiment_type": r.experiment_type or "",
        "include_prior": include_prior,
        "use_ppl": r.use_ppl if r.use_ppl is not None else False,
        "model": r.model,
        "seed": r.seed,
        "budget": r.budget,
        "raw_mean": r.raw_mean,
        "raw_std": r.raw_std,
        "z_mean": r.z_mean,
        "z_std": r.z_std,
        "paper_gpt4o_b0": paper_gpt4o_b0,
        "paper_gpt4o_b10": paper_gpt4o_b10,
        "paper_box_b0": paper_box_b0,
        "paper_box_b10": paper_box_b10,
        "source": r.source.value if r.source else "unknown",
    }


def results_to_dataframe(results: List[RunResult]) -> pd.DataFrame:
    """Convert list of RunResult to DataFrame."""
    rows = [result_to_row_dict(r) for r in results]
    return pd.DataFrame(rows)


def get_sweep_ids_from_env() -> List[str]:
    """Get sweep IDs from environment variable (set by CLI)."""
    sweep_ids_str = os.environ.get("SWEEP_IDS", "")
    if sweep_ids_str:
        return [s.strip() for s in sweep_ids_str.split(",") if s.strip()]
    return []


def get_entity_from_env() -> str:
    """Get W&B entity from environment variable."""
    return os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY)


def get_project_from_env() -> str:
    """Get W&B project from environment variable."""
    return os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT)


def get_target_metric_from_env() -> str:
    """Get target metric from environment variable (set by CLI)."""
    return os.environ.get("TARGET_METRIC", "metric/eval/z_mean")


def get_wandb_dir_from_env() -> str:
    """Get W&B directory from environment variable."""
    return os.environ.get("WANDB_DIR", "wandb-dir")


@st.cache_data(show_spinner=False)
def get_paper_data_as_df(default_goals_only: bool = False) -> pd.DataFrame:
    """Convert PAPER_RESULTS to DataFrame for comparison charts.

    Args:
        default_goals_only: If True, filter to only default goal per environment

    Returns DataFrame with columns: env, goal, include_prior, budget, z_mean, source
    """
    rows = []
    for source_key in ["gpt-4o", "box"]:
        source_label = "Paper (GPT-4o)" if source_key == "gpt-4o" else "Paper (BOX)"
        for (env, goal, prior, budget), z_mean in PAPER_RESULTS.get(source_key, {}).items():
            # Filter to default goal if requested
            if default_goals_only and goal != DEFAULT_GOALS.get(env):
                continue
            rows.append({
                "env": env,
                "goal": goal,
                "include_prior": prior,
                "budget": budget,
                "z_mean": z_mean,
                "source": source_label,
                "model": source_label,
            })
    return pd.DataFrame(rows)


def filter_to_default_goals(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include default goals per environment."""
    if df.empty:
        return df
    mask = df.apply(lambda row: row["goal"] == DEFAULT_GOALS.get(row["env"]), axis=1)
    return df[mask].copy()
