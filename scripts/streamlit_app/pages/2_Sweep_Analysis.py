"""Sweep Analysis - W&B sweep analysis with interactive charts.

Analyze sweep results with parameter importance, model rankings, and heatmaps.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Page config
st.set_page_config(
    page_title="Sweep Analysis - BoxingGym",
    page_icon="📈",
    layout="wide",
)

# Add parent to path for imports
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from components.charts import (
    build_heatmap_chart,
    build_importance_chart,
    build_model_ranking_chart,
    build_parameter_effect_chart,
)
from utils.data_loader import (
    get_entity_from_env,
    get_project_from_env,
    get_sweep_ids_from_env,
    get_target_metric_from_env,
    load_wandb_results,
)
from utils.theme import inject_custom_css

# Add src to path for analysis functions
_src = Path(__file__).resolve().parent.parent.parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Inject custom CSS
inject_custom_css()

# Title
st.title("📈 Sweep Analysis")
st.markdown("Parameter importance, model rankings, and heatmaps from W&B sweeps.")


# Analysis functions (from analyze_sweep_results.py)
@st.cache_data(show_spinner=False)
def compute_parameter_importance(
    df: pd.DataFrame,
    target_metric: str = "metric/eval/z_mean",
) -> pd.DataFrame:
    """Compute parameter importance using Random Forest."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        st.warning("sklearn not available. Install with: uv pip install scikit-learn")
        return pd.DataFrame()

    if target_metric not in df.columns:
        return pd.DataFrame()

    # Filter to meaningful hyperparameters only
    noise_patterns = ["results", "ppl/", "ppl.", "hydra", "filename", "system_prompt", "wandb", "_wandb", "_runtime", "_step", "_timestamp"]
    config_cols = [
        c for c in df.columns
        if c.startswith("config/")
        and not any(noise in c.lower() for noise in noise_patterns)
    ]

    if not config_cols:
        return pd.DataFrame()

    X_data = df[config_cols].copy()
    y = df[target_metric].values

    valid_mask = ~np.isnan(y)
    X_data = X_data[valid_mask]
    y = y[valid_mask]

    if len(y) < 10:
        return pd.DataFrame()

    # Encode categorical variables
    X_encoded = pd.DataFrame()
    for col in X_data.columns:
        if X_data[col].dtype == object or X_data[col].dtype.name == "category":
            le = LabelEncoder()
            values = X_data[col].fillna("__NULL__").astype(str)
            X_encoded[col] = le.fit_transform(values)
        else:
            X_encoded[col] = X_data[col].fillna(0)

    # Fit Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_encoded, y)

    # Get importance
    importance = pd.DataFrame({
        "parameter": config_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Add correlations
    correlations = []
    for col in config_cols:
        try:
            if X_encoded[col].nunique() > 1 and len(y) > 1:
                corr = np.corrcoef(X_encoded[col].values.astype(float), y.astype(float))[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
        except Exception:
            corr = 0.0
        correlations.append(corr)

    importance["correlation"] = correlations
    return importance


def aggregate_by_parameter(
    df: pd.DataFrame,
    parameter: str,
    target_metric: str = "metric/eval/z_mean",
) -> pd.DataFrame:
    """Aggregate metric by a single parameter."""
    if parameter not in df.columns or target_metric not in df.columns:
        return pd.DataFrame()

    agg = df.groupby(parameter)[target_metric].agg(["mean", "std", "count"])
    agg = agg.sort_values("mean")
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    return agg.reset_index()


def find_best_configurations(
    df: pd.DataFrame,
    target_metric: str = "metric/eval/z_mean",
    n: int = 15,
) -> pd.DataFrame:
    """Find the best N configurations."""
    if target_metric not in df.columns:
        return pd.DataFrame()

    key_configs = [
        "config/llms",
        "config/envs",
        "config/include_prior",
        "config/use_ppl",
        "config/seed",
    ]
    key_configs = [c for c in key_configs if c in df.columns]

    best = df.nsmallest(n, target_metric)

    result_cols = key_configs + [target_metric, "run_id"]
    result_cols = [c for c in result_cols if c in best.columns]

    return best[result_cols].reset_index(drop=True)


# Helper to fetch available sweeps
@st.cache_data(ttl=60, show_spinner=False)
def fetch_available_sweeps(entity: str, project: str):
    """Fetch all sweeps from the project."""
    try:
        import wandb
        import requests
        # Use longer timeout for slow WandB API
        api = wandb.Api(timeout=60)
        # Need entity/project format for API
        full_path = f"{entity}/{project}"
        sweeps = api.project(full_path).sweeps()
        # Don't fetch run counts - too slow (makes API call ~10x slower)
        result = []
        for s in sweeps:
            result.append((s.id, s.name or s.id, s.state))
        return result, None
    except requests.exceptions.HTTPError as e:
        if "502" in str(e):
            return [], "WandB API temporarily unavailable (HTTP 502). Try again in a moment."
        return [], f"WandB API error: {str(e)}"
    except requests.exceptions.Timeout:
        return [], "WandB API timeout after 60s. The service may be slow or unavailable."
    except Exception as e:
        return [], str(e)


# Sidebar - Sweep selection
with st.sidebar:
    st.header("🎯 Sweep Selection")

    # Data source selector
    data_source = st.radio(
        "Data Source",
        options=["Local JSON Files", "WandB API"],
        index=0,  # Default to local for speed
        help="Local files are much faster than WandB API"
    )

    entity = st.text_input("Entity", value=get_entity_from_env())
    project = st.text_input("Project", value=get_project_from_env())

    # Try to fetch available sweeps
    available_sweeps = []
    api_error = None
    if entity and project:
        with st.spinner("Discovering sweeps..."):
            available_sweeps, api_error = fetch_available_sweeps(entity, project)

        if api_error:
            st.warning(f"⚠️ WandB API error: {api_error[:200]}{'...' if len(api_error) > 200 else ''}")
            st.info("Using manual sweep ID entry below.")

    # Show dropdown if sweeps found, otherwise text input
    if available_sweeps:
        st.markdown(f"**{len(available_sweeps)} sweeps found**")

        # Format options: "sweep_id (name) - state"
        sweep_options = {
            f"{sid} ({name[:20]}) - {state}": sid
            for sid, name, state in available_sweeps
        }

        # Check for CLI-provided sweep IDs
        cli_sweep_ids = get_sweep_ids_from_env()
        default_indices = []
        if cli_sweep_ids:
            default_indices = [opt for opt in sweep_options.keys() if any(sid in opt for sid in cli_sweep_ids)]

        selected = st.multiselect(
            "Select Sweep(s)",
            options=list(sweep_options.keys()),
            default=default_indices if default_indices else list(sweep_options.keys())[:1],  # Start with 1 sweep for faster loading
            help="Select one or more sweeps to analyze (start with 1 for faster loading)"
        )
        sweep_ids = [sweep_options[s] for s in selected]
    else:
        # Fallback to text input
        cli_sweep_ids = get_sweep_ids_from_env()
        default_sweep = ",".join(cli_sweep_ids) if cli_sweep_ids else ""

        # Hint for users without default sweeps
        if not default_sweep:
            st.caption("💡 Enter your WandB sweep ID(s) above")

        sweep_input = st.text_input(
            "Sweep ID(s)",
            value=default_sweep,
            placeholder="e.g., abc123,def456",
        )
        sweep_ids = [s.strip() for s in sweep_input.split(",") if s.strip()]

    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Load data
if data_source == "Local JSON Files":
    # Load from local JSON files (much faster!)
    from utils.data_loader import load_local_results
    import os

    results_dir = os.path.join(os.path.dirname(__file__), "../../../results")

    with st.spinner("Loading local JSON files..."):
        try:
            results = load_local_results(results_dir)

            # Convert to run dicts
            all_runs = []
            for r in results:
                run_dict = {
                    "run_id": r.path or "local",
                    "config/llms": r.model,
                    "config/envs": r.env,
                    "config/seed": r.seed,
                    "config/include_prior": r.include_prior,
                    "config/use_ppl": r.use_ppl,
                    "metric/eval/z_mean": r.z_mean,
                    "metric/eval/z_std": r.z_std,
                    "sweep_id": "local",
                }
                all_runs.append(run_dict)

            if not all_runs:
                st.warning("No results found in local JSON files.")
                st.info(f"Searched in: {results_dir}")
                st.stop()

            st.info(f"Loaded {len(all_runs)} runs from local JSON files")

        except Exception as e:
            st.error(f"Failed to load local results: {e}")
            st.stop()

else:  # WandB API
    if not sweep_ids:
        if entity and project and not available_sweeps:
            st.warning("No sweeps found in this project. Check your entity/project names.")
        else:
            st.info("Select sweep(s) in the sidebar to begin analysis.")
        st.stop()

    if not entity or not project:
        st.error("Please set **Entity** and **Project** in the sidebar.")
        st.stop()

    # Fetch sweep data
    all_runs = []
    with st.spinner(f"Loading {len(sweep_ids)} sweep(s) from W&B..."):
        for sweep_id in sweep_ids:
            try:
                runs = load_wandb_results(sweep_id, entity, project)
                # Convert to dicts for DataFrame
                for r in runs:
                    run_dict = {
                        "run_id": getattr(r, "path", None) or sweep_id,
                        "config/llms": r.model,
                        "config/envs": r.env,
                        "config/seed": r.seed,
                        "config/include_prior": r.include_prior,
                        "config/use_ppl": r.use_ppl,
                        "metric/eval/z_mean": r.z_mean,
                        "metric/eval/z_std": r.z_std,
                        "sweep_id": sweep_id,
                    }
                    all_runs.append(run_dict)
            except Exception as e:
                st.warning(f"Failed to load sweep {sweep_id}: {e}")

if not all_runs:
    st.error("No runs loaded. Check the sweep ID and try again.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(all_runs)
target_metric = get_target_metric_from_env()

# Filter extreme outliers (numerical errors produce values like 10^58)
if target_metric in df.columns:
    outlier_threshold = 10  # |z_mean| > 10 std devs is essentially impossible
    n_before = len(df)
    df = df[df[target_metric].abs() <= outlier_threshold]
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        st.info(f"Filtered {n_filtered} runs with extreme values (|z_mean| > {outlier_threshold})")

# Summary stats
st.subheader("Summary")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Runs", len(df))

with c2:
    n_models = df["config/llms"].nunique()
    model_list = ", ".join(str(m)[:15] for m in sorted(df["config/llms"].unique())[:3])
    if n_models > 3:
        model_list += f" +{n_models - 3}"
    st.metric("Models", n_models)
    st.caption(model_list)

with c3:
    n_envs = df["config/envs"].nunique()
    env_list = ", ".join(sorted(df["config/envs"].unique())[:3])
    if n_envs > 3:
        env_list += f" +{n_envs - 3}"
    st.metric("Environments", n_envs)
    st.caption(env_list)

with c4:
    if target_metric in df.columns:
        best_idx = df[target_metric].idxmin()
        best_row = df.loc[best_idx]
        best_val = best_row[target_metric]
        best_env = str(best_row.get("config/envs", "?"))
        best_model = str(best_row.get("config/llms", "?"))[:15]
        st.metric("Best z_mean", f"{best_val:.3f}")
        st.caption(f"{best_env} / {best_model}")

st.divider()

# Charts in columns
col1, col2 = st.columns(2)

# Parameter Importance
with col1:
    with st.spinner("Computing parameter importance..."):
        importance = compute_parameter_importance(df, target_metric)

    if not importance.empty:
        fig = build_importance_chart(importance.to_dict("records"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Could not compute parameter importance (not enough data or sklearn missing)")

# Model Rankings
with col2:
    try:
        from boxing_gym.analysis.stats import bootstrap_ci, compare_models
        from components.charts import build_leaderboard_chart

        model_col = "config/llms"
        if model_col not in df.columns:
            model_col = "model" if "model" in df.columns else None

        if model_col and target_metric in df.columns:
            with st.spinner("Computing bootstrap CIs..."):
                rankings = []
                for model in df[model_col].dropna().unique():
                    scores = df[df[model_col] == model][target_metric].dropna().values
                    if len(scores) < 2:
                        continue
                    ci = bootstrap_ci(scores, confidence=0.95, n_bootstrap=10000)
                    rankings.append({
                        "model": model,
                        "mean": ci.mean,
                        "ci_low": ci.ci_low,
                        "ci_high": ci.ci_high,
                        "n": len(scores),
                    })

                if rankings:
                    best = min(rankings, key=lambda r: r["mean"])["model"]
                    cmp_df = df[[model_col, target_metric]].dropna().rename(
                        columns={model_col: "model", target_metric: "z_mean"}
                    )
                    comparison = compare_models(
                        cmp_df, model_col="model", score_col="z_mean",
                        reference_model=best,
                    )
                    sig_models = {
                        c["model"] for c in comparison["comparisons"]
                        if c.get("significant_fdr")
                    }
                    for r in rankings:
                        r["significant"] = r["model"] in sig_models

                    fig = build_leaderboard_chart(rankings)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data for model rankings")
        else:
            st.info("Model column not found in data")
    except (ImportError, ValueError, KeyError):
        # fall back to existing SEM-based chart
        agg_model = aggregate_by_parameter(df, "config/llms", target_metric)
        if not agg_model.empty:
            fig = build_model_ranking_chart(agg_model.to_dict("records"))
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# Heatmap (full width)
st.subheader("Environment × Model Heatmap")
fig = build_heatmap_chart(all_runs, target_metric)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Parameter effects
st.subheader("Parameter Effects")

param_cols = st.columns(2)

with param_cols[0]:
    agg_prior = aggregate_by_parameter(df, "config/include_prior", target_metric)
    if not agg_prior.empty:
        fig = build_parameter_effect_chart(agg_prior.to_dict("records"), "include_prior")
        st.plotly_chart(fig, use_container_width=True)

with param_cols[1]:
    agg_ppl = aggregate_by_parameter(df, "config/use_ppl", target_metric)
    if not agg_ppl.empty:
        fig = build_parameter_effect_chart(agg_ppl.to_dict("records"), "use_ppl")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# Best configurations table
st.subheader("Top Configurations")

best_configs = find_best_configurations(df, target_metric, n=15)

if not best_configs.empty:
    # Rename columns for display
    display_df = best_configs.copy()
    display_df.columns = [
        c.replace("config/", "").replace("metric/eval/", "")
        for c in display_df.columns
    ]

    # Style z_mean column
    def style_z(val):
        if pd.isna(val):
            return ""
        if val < -0.5:
            return "color: #34d399"
        elif val > 0.5:
            return "color: #fb7185"
        return "color: #f59e0b"

    styled = display_df.style.map(style_z, subset=["z_mean"]).format({
        "z_mean": "{:+.3f}",
        "seed": "{:.0f}",
    }, na_rep="—")

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )

# Footer
st.divider()
sweep_display = "local" if data_source == "Local JSON Files" else ', '.join(sweep_ids)
st.caption(f"Analyzing sweeps: {sweep_display} | {len(df)} total runs")
