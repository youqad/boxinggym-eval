"""Benchmark Dashboard - Main results viewer.

View and compare benchmark results with paper baselines.
"""

import sys
from pathlib import Path

import streamlit as st

# Page config
st.set_page_config(
    page_title="Benchmark Dashboard - BoxingGym",
    page_icon="📊",
    layout="wide",
)

# Add parent to path for imports
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from components.filters import apply_filters, render_sidebar_filters
from components.tables import render_results_table, render_summary_stats
from utils.data_loader import (
    get_entity_from_env,
    get_paper_baselines,
    get_project_from_env,
    get_sweep_ids_from_env,
    load_local_results,
    load_wandb_results,
    results_to_dataframe,
)
from utils.theme import inject_custom_css

# Inject custom CSS
inject_custom_css()

# Title
st.title("📊 Benchmark Dashboard")
st.markdown(
    "Compare against paper baselines (GPT-4o and BOX). Lower z_mean = better. Negative delta = better than baseline."
)

# Data source selector
st.subheader("Data Source")

# Check for CLI-provided sweep IDs
cli_sweep_ids = get_sweep_ids_from_env()

col1, col2 = st.columns([1, 3])

with col1:
    source_type = st.radio(
        "Source",
        ["W&B Sweep", "Local JSON"],
        index=0 if cli_sweep_ids else 0,
        horizontal=True,
    )

with col2:
    if source_type == "W&B Sweep":
        # Sweep ID input
        default_sweep = ",".join(cli_sweep_ids) if cli_sweep_ids else ""
        sweep_input = st.text_input(
            "Sweep ID(s)",
            value=default_sweep,
            placeholder="Enter sweep ID(s), comma-separated",
            help="Enter one or more W&B sweep IDs",
        )
        sweep_ids = [s.strip() for s in sweep_input.split(",") if s.strip()]
    else:
        root_dir = st.text_input(
            "Results Directory",
            value="results",
            placeholder="Path to results directory",
        )

st.divider()

# Load data
results = []
error_message = None

try:
    if source_type == "W&B Sweep" and sweep_ids:
        entity = get_entity_from_env()
        project = get_project_from_env()

        with st.spinner(f"Loading {len(sweep_ids)} sweep(s) from W&B..."):
            for sweep_id in sweep_ids:
                try:
                    sweep_results = load_wandb_results(sweep_id, entity, project)
                    results.extend(sweep_results)
                except Exception as e:
                    st.warning(f"Failed to load sweep {sweep_id}: {e}")

    elif source_type == "Local JSON" and root_dir:
        if Path(root_dir).exists():
            with st.spinner("Loading local results..."):
                results = load_local_results(root_dir)
        else:
            error_message = f"Directory not found: {root_dir}"

except Exception as e:
    error_message = f"Error loading data: {e}"

# Add paper baselines
paper_results = get_paper_baselines()
results.extend(paper_results)

if error_message:
    st.error(error_message)

if not results:
    st.info("No results loaded. Enter a sweep ID or results directory above.")
    st.stop()

# Convert to DataFrame
df = results_to_dataframe(results)

# Render filters
filters = render_sidebar_filters(df)

# Apply filters
filtered_df = apply_filters(df, filters)

# Summary stats
render_summary_stats(filtered_df)

st.divider()

# detect if data uses new multi-seed format (has z_stderr from built-in aggregation)
has_builtin_aggregation = (
    "z_stderr" in filtered_df.columns and filtered_df["z_stderr"].notna().any()
)

# Aggregate toggle (only needed for old per-seed format)
col_toggle, col_spacer = st.columns([1, 3])
with col_toggle:
    if has_builtin_aggregation:
        st.info("✓ Data already aggregated (multi-seed format)")
        aggregate = False  # no post-hoc aggregation needed
    else:
        aggregate = st.toggle(
            "Aggregate Seeds",
            value=True,
            help="Group by (env, model, budget) and show Mean ± SEM",
        )

# Aggregation logic (only for old per-seed format)
display_df = filtered_df
if aggregate and not filtered_df.empty and not has_builtin_aggregation:
    # Columns to group by (exclude seed, timestamps, run IDs)
    group_cols = ["env", "model", "budget", "include_prior", "use_ppl", "goal", "experiment_type"]
    group_cols = [c for c in group_cols if c in filtered_df.columns]

    # Columns to preserve (take first) - paper refs are constant per group
    meta_cols = ["paper_gpt4o_b0", "paper_gpt4o_b10", "paper_box_b0", "paper_box_b10"]
    meta_cols = [c for c in meta_cols if c in filtered_df.columns]

    # Perform aggregation
    agg_dict = {"z_mean": ["mean", "sem", "count"]}
    for c in meta_cols:
        agg_dict[c] = "first"

    agg_df = filtered_df.groupby(group_cols).agg(agg_dict).reset_index()

    # Flatten hierarchical columns
    new_cols = []
    for col in agg_df.columns:
        if isinstance(col, tuple):
            if col[1]:
                new_cols.append(f"{col[0]}_{col[1]}")
            else:
                new_cols.append(col[0])
        else:
            new_cols.append(col)
    agg_df.columns = new_cols

    # Rename for compatibility with renderer
    rename_map = {
        "z_mean_mean": "z_mean",
        "z_mean_sem": "sem",
        "z_mean_count": "count",
    }
    # Also rename paper cols back (remove _first)
    for c in meta_cols:
        rename_map[f"{c}_first"] = c

    agg_df = agg_df.rename(columns=rename_map)
    display_df = agg_df
elif has_builtin_aggregation and not filtered_df.empty:
    # new multi-seed format: use z_stderr as sem column for display
    display_df = filtered_df.copy()
    if "sem" not in display_df.columns:
        display_df["sem"] = display_df["z_stderr"]

# Results table
render_results_table(
    display_df,
    baseline=filters["baseline"],
    ref_budget=filters["ref_budget"],
)

# Footer
st.divider()
st.caption(
    f"Data source: {source_type} | "
    f"Baseline: {'GPT-4o' if filters['baseline'] == 'gpt' else 'BOX'} | "
    f"Ref Budget: {filters['ref_budget'] or 'Any'}"
)
