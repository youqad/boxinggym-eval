"""Cached data loading utilities for BoxingGym dashboard.

Wraps results_io.py functions with Streamlit caching.
"""

import json
import os
import sys
from pathlib import Path

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
    PAPER_RESULTS,
    DataSource,
    RunResult,
    aggregate_results,
    generate_paper_reference_rows,
    get_paper_value_4d,
    load_results_from_json_dir,
    load_results_from_local_wandb,
    load_results_from_wandb,
)

CANONICAL_PARQUET = "canonical_runs.parquet"
CANONICAL_METADATA = "canonical_metadata.json"
DEMO_PARQUET = "demo_runs.parquet"


def is_demo_mode() -> bool:
    """Whether dashboard is running in pinned HF demo mode."""
    return os.environ.get("BOXING_GYM_DEMO_MODE", "0") == "1"


def _default_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    demo_data_dir = Path(__file__).resolve().parent.parent / "demo_data"
    return project_root, demo_data_dir


def resolve_results_path(
    parquet_path: str | None = None,
    *,
    project_root: Path | None = None,
    demo_data_dir: Path | None = None,
    demo_mode: bool | None = None,
) -> tuple[Path | None, str]:
    """Resolve the parquet source with deterministic ordering.

    Returns:
        (path, source_tag) where source_tag is one of:
        - explicit_path
        - canonical_snapshot
        - local_cache
        - bundled_demo_fallback
        - none
    """
    if parquet_path:
        explicit = Path(parquet_path)
        if explicit.exists():
            return explicit, "explicit_path"
        return None, "none"

    if project_root is None or demo_data_dir is None:
        default_project_root, default_demo_data_dir = _default_paths()
        project_root = project_root or default_project_root
        demo_data_dir = demo_data_dir or default_demo_data_dir

    if demo_mode is None:
        demo_mode = is_demo_mode()

    cache_path = project_root / ".boxing-gym-cache" / "runs.parquet"
    canonical_path = demo_data_dir / CANONICAL_PARQUET
    bundled_demo_path = demo_data_dir / DEMO_PARQUET

    if demo_mode:
        ordered = [
            (canonical_path, "canonical_snapshot"),
            (bundled_demo_path, "bundled_demo_fallback"),
            (cache_path, "local_cache"),
        ]
    else:
        ordered = [
            (cache_path, "local_cache"),
            (canonical_path, "canonical_snapshot"),
            (bundled_demo_path, "bundled_demo_fallback"),
        ]

    for path, source in ordered:
        if path.exists():
            return path, source
    return None, "none"


def get_active_results_source(parquet_path: str | None = None) -> str:
    """Return active source tag for the parquet loader."""
    _, source = resolve_results_path(parquet_path=parquet_path)
    return source


def get_canonical_metadata_path(demo_data_dir: Path | None = None) -> Path:
    """Canonical metadata file path."""
    if demo_data_dir is None:
        _, demo_data_dir = _default_paths()
    return demo_data_dir / CANONICAL_METADATA


@st.cache_data(show_spinner=False)
def load_canonical_metadata() -> dict:
    """Load canonical snapshot metadata JSON if present."""
    metadata_path = get_canonical_metadata_path()
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner="Loading from W&B...")
def load_wandb_results(
    sweep_id: str,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> list[RunResult]:
    """Load results from W&B API with 5-minute cache."""
    return load_results_from_wandb(sweep_id, entity, project)


@st.cache_data(ttl=60, show_spinner="Loading local results...")
def load_local_results(root: str) -> list[RunResult]:
    """Load results from local JSON files with 1-minute cache."""
    return load_results_from_json_dir(root)


@st.cache_data(ttl=60, show_spinner="Loading local W&B runs...")
def load_local_wandb_results(wandb_dir: str = "wandb-dir") -> list[RunResult]:
    """Load results from local W&B run directories with 1-minute cache.

    Much faster than API calls when data is already synced locally.
    Scans wandb-dir/run-*/files/wandb-summary.json for completed runs.
    """
    return load_results_from_local_wandb(wandb_dir)


@st.cache_data(ttl=300, show_spinner="Loading cached results...")
def load_parquet_results(parquet_path: str | None = None) -> pd.DataFrame:
    """Load results from parquet cache, honoring demo mode resolution."""
    path, _ = resolve_results_path(parquet_path=parquet_path)
    if path is not None:
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_paper_baselines() -> list[RunResult]:
    """Load paper reference rows (static, cache forever)."""
    gpt_rows = generate_paper_reference_rows("gpt-4o")
    box_rows = generate_paper_reference_rows("box")
    return gpt_rows + box_rows


@st.cache_data(show_spinner=False)
def compute_aggregated_results(
    results_hash: str,  # Hash of results for cache key
    results: list[dict],
    group_by: tuple[str, ...],
) -> list[dict]:
    """Compute aggregated results with caching."""
    # Convert dicts back to RunResult objects
    run_results = [RunResult(**r) for r in results]
    agg = aggregate_results(run_results, group_by=group_by)
    return [r.__dict__ for r in agg]


def result_to_row_dict(r: RunResult) -> dict:
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


def results_to_dataframe(results: list[RunResult]) -> pd.DataFrame:
    """Convert list of RunResult to DataFrame."""
    rows = [result_to_row_dict(r) for r in results]
    return pd.DataFrame(rows)


def get_sweep_ids_from_env() -> list[str]:
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
            rows.append(
                {
                    "env": env,
                    "goal": goal,
                    "include_prior": prior,
                    "budget": budget,
                    "z_mean": z_mean,
                    "source": source_label,
                    "model": source_label,
                }
            )
    return pd.DataFrame(rows)


def filter_to_default_goals(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include default goals per environment."""
    if df.empty:
        return df
    mask = df.apply(lambda row: row["goal"] == DEFAULT_GOALS.get(row["env"]), axis=1)
    return df[mask].copy()
