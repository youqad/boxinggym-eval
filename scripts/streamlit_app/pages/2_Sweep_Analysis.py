"""Sweep Analysis - parameter importance, model rankings, and heatmaps."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sweep Analysis - BoxingGym",
    page_icon="📈",
    layout="wide",
)

_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from components.charts import (
    build_budget_progression_sweep_chart,
    build_heatmap_chart,
    build_importance_chart,
    build_model_ranking_chart,
    build_parameter_effect_chart,
)
from utils.data_loader import get_target_metric_from_env
from utils.data_selector import load_runs_from_selection, render_data_source_selector
from utils.normalize import get_budget_column
from utils.theme import inject_custom_css

_src = Path(__file__).resolve().parent.parent.parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

inject_custom_css()

st.title("📈 Sweep Analysis")
st.markdown("Parameter importance, model rankings, and heatmaps from W&B sweeps.")


@st.cache_data(show_spinner=False)
def compute_parameter_importance(
    df: pd.DataFrame,
    target_metric: str = "metric/eval/z_mean",
) -> pd.DataFrame:
    """Random Forest feature importance for sweep parameters."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        st.warning("sklearn not available. Install with: uv pip install scikit-learn")
        return pd.DataFrame()

    if target_metric not in df.columns:
        return pd.DataFrame()

    noise_patterns = [
        "results",
        "ppl/",
        "ppl.",
        "hydra",
        "filename",
        "system_prompt",
        "wandb",
        "_wandb",
        "_runtime",
        "_step",
        "_timestamp",
        "seed",
    ]
    config_cols = [
        c
        for c in df.columns
        if c.startswith("config/") and not any(noise in c.lower() for noise in noise_patterns)
    ]

    if not config_cols:
        return pd.DataFrame()

    X_data = df[config_cols].copy()
    y = pd.to_numeric(df[target_metric], errors="coerce").values

    valid_mask = ~np.isnan(y)
    X_data = X_data[valid_mask]
    y = y[valid_mask]

    if len(y) < 10:
        return pd.DataFrame()

    X_encoded = pd.DataFrame()
    for col in X_data.columns:
        if X_data[col].dtype == object or X_data[col].dtype.name == "category":
            le = LabelEncoder()
            values = X_data[col].fillna("__NULL__").astype(str)
            X_encoded[col] = le.fit_transform(values)
        else:
            X_encoded[col] = pd.to_numeric(X_data[col], errors="coerce").fillna(0)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_encoded, y)

    importance = pd.DataFrame(
        {
            "parameter": config_cols,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

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


with st.sidebar:
    selection = render_data_source_selector(st.sidebar, key_prefix="sweep")

if selection.data_source == "Local JSON Files":
    with st.spinner("Loading local JSON files..."):
        try:
            all_runs = load_runs_from_selection(selection)
            if not all_runs:
                st.warning("No results found in local JSON files.")
                st.info(f"Searched in: {selection.results_dir}")
                st.stop()
            st.info(f"Loaded {len(all_runs)} runs from local JSON files")
        except Exception as e:
            st.error(f"Failed to load local results: {e}")
            st.stop()
else:
    if not selection.sweep_ids:
        if selection.entity and selection.project and not selection.available_sweeps:
            st.warning("No sweeps found in this project. Check your entity/project names.")
        else:
            st.info("Select sweep(s) in the sidebar to begin analysis.")
        st.stop()

    if not selection.entity or not selection.project:
        st.error("Please set **Entity** and **Project** in the sidebar.")
        st.stop()

    with st.spinner(f"Loading {len(selection.sweep_ids)} sweep(s) from W&B..."):
        all_runs = load_runs_from_selection(selection)

if not all_runs:
    st.error("No runs loaded. Check the sweep ID and try again.")
    st.stop()

df = pd.DataFrame(all_runs)
target_metric = get_target_metric_from_env()

if target_metric in df.columns:
    metric_series = pd.to_numeric(df[target_metric], errors="coerce")
    df[target_metric] = metric_series
    outlier_threshold = 10
    n_before = len(df)
    mask = metric_series.abs() <= outlier_threshold
    df = df[mask]
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        st.info(f"Filtered {n_filtered} runs with extreme values (|z_mean| > {outlier_threshold})")

st.subheader("Summary")

c1, c2, c3, c4 = st.columns(4)
model_col = "config/llms" if "config/llms" in df.columns else None
env_col = "config/envs" if "config/envs" in df.columns else None
metric_series = (
    pd.to_numeric(df[target_metric], errors="coerce")
    if target_metric in df.columns
    else pd.Series(dtype=float)
)
with c1:
    st.metric("Total Runs", len(df))

with c2:
    if model_col:
        n_models = df[model_col].nunique()
        model_list = ", ".join(str(m)[:15] for m in sorted(df[model_col].dropna().unique())[:3])
        if n_models > 3:
            model_list += f" +{n_models - 3}"
        st.metric("Models", n_models)
        st.caption(model_list)
    else:
        st.metric("Models", "—")
        st.caption("No model column")

with c3:
    if env_col:
        n_envs = df[env_col].nunique()
        env_list = ", ".join(sorted(df[env_col].dropna().unique())[:3])
        if n_envs > 3:
            env_list += f" +{n_envs - 3}"
        st.metric("Environments", n_envs)
        st.caption(env_list)
    else:
        st.metric("Environments", "—")
        st.caption("No environment column")

with c4:
    if not metric_series.empty and metric_series.notna().any():
        best_idx = metric_series.idxmin()
        best_row = df.loc[best_idx]
        best_val = float(metric_series.loc[best_idx])
        best_env = str(best_row.get(env_col, "?")) if env_col else "?"
        best_model = str(best_row.get(model_col, "?"))[:15] if model_col else "?"
        st.metric("Best z_mean", f"{best_val:.3f}")
        st.caption(f"{best_env} / {best_model}")
    else:
        st.metric("Best z_mean", "—")
        st.caption("No valid metric values")

st.divider()

col1, col2 = st.columns(2)

with col1:
    with st.spinner("Computing parameter importance..."):
        importance = compute_parameter_importance(df, target_metric)

    if not importance.empty:
        fig = build_importance_chart(importance.to_dict("records"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Could not compute parameter importance (not enough data or sklearn missing)")

with col2:
    try:
        from components.charts import build_leaderboard_chart

        from boxing_gym.analysis.stats import bootstrap_ci, compare_models

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
                    rankings.append(
                        {
                            "model": model,
                            "mean": ci.mean,
                            "ci_low": ci.ci_low,
                            "ci_high": ci.ci_high,
                            "n": len(scores),
                        }
                    )

                if rankings:
                    best = min(rankings, key=lambda r: r["mean"])["model"]
                    cmp_df = (
                        df[[model_col, target_metric]]
                        .dropna()
                        .rename(columns={model_col: "model", target_metric: "z_mean"})
                    )
                    comparison = compare_models(
                        cmp_df,
                        model_col="model",
                        score_col="z_mean",
                        reference_model=best,
                    )
                    sig_models = {
                        c["model"] for c in comparison["comparisons"] if c.get("significant_fdr")
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

st.subheader("Environment × Model Heatmap")
fig = build_heatmap_chart(all_runs, target_metric)
st.plotly_chart(fig, use_container_width=True)

st.divider()

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

st.subheader("Budget Progression")

budget_col = get_budget_column(df)
if budget_col and target_metric in df.columns and "config/llms" in df.columns:
    envs = sorted(df["config/envs"].dropna().unique()) if "config/envs" in df.columns else []
    env_filter = st.selectbox("Environment", ["All"] + envs, index=0)

    plot_df = df.copy()
    if env_filter != "All" and "config/envs" in plot_df.columns:
        plot_df = plot_df[plot_df["config/envs"] == env_filter]

    budget_num = pd.to_numeric(plot_df[budget_col], errors="coerce")
    if budget_num.notna().sum() >= 2:
        plot_df["_budget_key"] = budget_num
        budget_key = "_budget_key"
    else:
        plot_df["_budget_key"] = plot_df[budget_col].astype(str)
        budget_key = "_budget_key"

    fig = build_budget_progression_sweep_chart(
        plot_df,
        metric_col=target_metric,
        budget_col=budget_key,
        model_col="config/llms",
        max_models=8,
    )
    st.plotly_chart(fig, use_container_width=True)

    if "config/envs" in plot_df.columns:
        df_local = plot_df.copy()
        df_local["env_model"] = (
            df_local["config/envs"].astype(str) + " | " + df_local["config/llms"].astype(str)
        )
        pivot = df_local.pivot_table(
            values=target_metric, index="env_model", columns=budget_key, aggfunc="mean"
        )
        budget_vals = pd.to_numeric(plot_df[budget_key], errors="coerce").dropna().unique()
        budgets = sorted(budget_vals.tolist())
        if len(budgets) >= 2:
            first_b, last_b = budgets[0], budgets[-1]
            if first_b in pivot.columns and last_b in pivot.columns:
                pivot["improvement"] = pivot[first_b] - pivot[last_b]
                top_improved = (
                    pivot.dropna(subset=["improvement"])
                    .sort_values("improvement", ascending=False)
                    .head(10)
                )
                if not top_improved.empty:
                    first_label = (
                        int(first_b) if float(first_b).is_integer() else round(float(first_b), 2)
                    )
                    last_label = (
                        int(last_b) if float(last_b).is_integer() else round(float(last_b), 2)
                    )
                    st.caption(f"Top improvements (B={first_label} → B={last_label})")
                    table_df = top_improved[["improvement"]].reset_index()
                    table_df["improvement"] = table_df["improvement"].map(lambda v: f"{v:+.3f}")
                    st.dataframe(table_df, use_container_width=True, hide_index=True)
else:
    st.info("No budget column found for progression chart.")

st.divider()

st.subheader("Seed Stability Diagnostics")

stderr_col = (
    "summary/z_stderr"
    if "summary/z_stderr" in df.columns
    else "z_stderr"
    if "z_stderr" in df.columns
    else None
)
has_seed_col = "config/seed" in df.columns

if stderr_col and target_metric in df.columns:
    st.caption("Multi-seed format detected. Showing z_stderr as seed variance.")
    cols = ["config/envs", "config/llms", target_metric, stderr_col]
    cols = [c for c in cols if c in df.columns]
    seed_df = df.dropna(subset=[stderr_col]).sort_values(stderr_col, ascending=False).head(10)
    if not seed_df.empty:
        display_df = seed_df[cols].copy()
        display_df = display_df.rename(
            columns={
                "config/envs": "env",
                "config/llms": "model",
                target_metric: "z_mean",
                stderr_col: "z_stderr",
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No seed variance data found.")

if has_seed_col and target_metric in df.columns:
    seed_stats = (
        df.groupby("config/seed")[target_metric].agg(["mean", "std", "count"]).reset_index()
    )
    if not seed_stats.empty:
        seed_stats = seed_stats.sort_values("mean")
        seed_stats_display = seed_stats.rename(columns={"config/seed": "seed"})
        seed_stats_display["mean"] = seed_stats_display["mean"].map(lambda v: f"{v:+.3f}")
        seed_stats_display["std"] = seed_stats_display["std"].map(lambda v: f"{v:.3f}")
        st.caption("Per-seed performance (lower z_mean is better)")
        st.dataframe(seed_stats_display, use_container_width=True, hide_index=True)

        noise_patterns = [
            "results",
            "ppl/",
            "ppl.",
            "hydra",
            "filename",
            "system_prompt",
            "_wandb",
            "_runtime",
            "seed",
        ]
        config_cols = [
            c
            for c in df.columns
            if c.startswith("config/") and not any(noise in c.lower() for noise in noise_patterns)
        ]
        if config_cols:
            variance_rows = []
            grouped = df.groupby(config_cols)
            for config_combo, group in grouped:
                if len(group) < 2:
                    continue
                z_vals = group[target_metric].dropna()
                if len(z_vals) < 2:
                    continue
                variance_rows.append(
                    {
                        "config": dict(
                            zip(
                                config_cols,
                                config_combo if isinstance(config_combo, tuple) else [config_combo],
                            )
                        ),
                        "variance": float(z_vals.var()),
                        "mean": float(z_vals.mean()),
                        "count": int(len(z_vals)),
                    }
                )
            if variance_rows:
                var_df = (
                    pd.DataFrame(variance_rows).sort_values("variance", ascending=False).head(10)
                )
                var_df["config"] = var_df["config"].map(
                    lambda c: ", ".join(
                        [f"{k.replace('config/', '')}={str(v)[:12]}" for k, v in c.items()]
                    )
                )
                st.caption("Most unstable configs across seeds")
                st.dataframe(var_df, use_container_width=True, hide_index=True)
    else:
        st.info("No seed data available for per-seed diagnostics.")
else:
    st.info("No seed column found in data.")

st.divider()

st.subheader("Top Configurations")

best_configs = find_best_configurations(df, target_metric, n=15)

if not best_configs.empty:
    display_df = best_configs.copy()
    display_df.columns = [
        c.replace("config/", "").replace("metric/eval/", "") for c in display_df.columns
    ]

    def style_z(val):
        if pd.isna(val):
            return ""
        if val < -0.5:
            return "color: #34d399"
        elif val > 0.5:
            return "color: #fb7185"
        return "color: #f59e0b"

    styled = display_df.style.map(style_z, subset=["z_mean"]).format(
        {
            "z_mean": "{:+.3f}",
            "seed": "{:.0f}",
        },
        na_rep="—",
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )

st.divider()
sweep_display = (
    "local" if selection.data_source == "Local JSON Files" else ", ".join(selection.sweep_ids)
)
st.caption(f"Analyzing sweeps: {sweep_display} | {len(df)} total runs")
