"""Paper Comparison - Compare sweep results with paper baselines.

Replicates key figures from the BoxingGym paper with user sweep data overlay.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Page config
st.set_page_config(
    page_title="Paper Comparison - BoxingGym",
    page_icon="📄",
    layout="wide",
)

# Add parent to path for imports
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from components.charts import (
    build_budget_progression_chart,
    build_model_comparison_bars,
)
from utils.data_loader import (
    filter_to_default_goals,
    get_entity_from_env,
    get_paper_data_as_df,
    get_project_from_env,
    get_sweep_ids_from_env,
    load_wandb_results,
)
from utils.theme import inject_custom_css

# Inject custom CSS
inject_custom_css()

# Title
st.title("📄 Paper Comparison")
st.markdown("Overlay your sweep results on the paper's GPT-4o and BOX baselines.")

# Sidebar controls
with st.sidebar:
    st.header("🎯 Configuration")

    # Sweep ID input
    cli_sweep_ids = get_sweep_ids_from_env()
    default_sweep = ",".join(cli_sweep_ids) if cli_sweep_ids else ""

    sweep_input = st.text_input(
        "Sweep ID(s)",
        value=default_sweep,
        placeholder="Enter sweep ID(s), comma-separated",
        help="Leave empty to view paper baselines only",
    )
    sweep_ids = [s.strip() for s in sweep_input.split(",") if s.strip()]

    entity = st.text_input("Entity", value=get_entity_from_env())
    project = st.text_input("Project", value=get_project_from_env())

    st.divider()

    # Chart filters
    st.subheader("Chart Filters")
    include_prior = st.toggle("Include Prior", value=True, help="Filter to runs with/without prior")
    budget_select = st.selectbox("Budget for Bar Chart", options=[0, 10], index=1)
    default_goals_only = st.toggle(
        "Default Goals Only",
        value=True,
        help="Show only primary goal per environment (e.g., 'choice' for hyperbolic, 'signal' for location_finding)",
    )

    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Load paper baselines (always available)
paper_df_full = get_paper_data_as_df()
paper_df = get_paper_data_as_df(default_goals_only=default_goals_only)

# Load our sweep data
sweep_runs = []
if sweep_ids:
    # Validate entity/project are set
    if not entity or not project:
        st.error("Please set **Entity** and **Project** in the sidebar to load W&B sweeps.")
    else:
        with st.spinner(f"Loading {len(sweep_ids)} sweep(s) from W&B..."):
            for sweep_id in sweep_ids:
                try:
                    runs = load_wandb_results(sweep_id, entity, project)
                    for r in runs:
                        sweep_runs.append({
                            "env": r.env,
                            "goal": r.goal,
                            "include_prior": r.include_prior,
                            "use_ppl": r.use_ppl if r.use_ppl is not None else False,
                            "budget": r.budget,
                            "z_mean": r.z_mean,
                            "model": r.model,
                            "seed": r.seed,
                            "source": "Sweep",
                        })
                except Exception as e:
                    st.warning(f"Failed to load sweep {sweep_id}: {e}")

sweep_df_full = pd.DataFrame(sweep_runs) if sweep_runs else pd.DataFrame()

# Filter extreme outliers from user data
if len(sweep_df_full) > 0 and "z_mean" in sweep_df_full.columns:
    outlier_threshold = 10
    n_before = len(sweep_df_full)
    sweep_df_full = sweep_df_full[sweep_df_full["z_mean"].abs() <= outlier_threshold]
    n_filtered = n_before - len(sweep_df_full)
    if n_filtered > 0:
        st.info(f"Filtered {n_filtered} runs with extreme values (|z_mean| > {outlier_threshold})")

# Filter to default goals if requested (for charts)
sweep_df = filter_to_default_goals(sweep_df_full) if default_goals_only else sweep_df_full

# Summary section
st.subheader("Summary")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Sweep Runs", len(sweep_df) if len(sweep_df) > 0 else "—")

with c2:
    if len(sweep_df) > 0:
        st.metric("Sweep Models", sweep_df["model"].nunique())
    else:
        st.metric("Sweep Models", "—")

with c3:
    # Count envs where user beats GPT-4o
    if len(sweep_df) > 0:
        beat_gpt = 0
        paper_gpt = paper_df[paper_df["source"] == "Paper (GPT-4o)"]
        for _, row in sweep_df.groupby(["env", "goal", "include_prior", "budget"])["z_mean"].mean().reset_index().iterrows():
            paper_val = paper_gpt[
                (paper_gpt["env"] == row["env"]) &
                (paper_gpt["goal"] == row["goal"]) &
                (paper_gpt["include_prior"] == row["include_prior"]) &
                (paper_gpt["budget"] == row["budget"])
            ]["z_mean"]
            if len(paper_val) > 0 and row["z_mean"] < paper_val.values[0]:
                beat_gpt += 1
        st.metric("Beat GPT-4o", f"{beat_gpt}")
    else:
        st.metric("Beat GPT-4o", "—")

with c4:
    # Count envs where user beats BOX
    if len(sweep_df) > 0:
        beat_box = 0
        paper_box = paper_df[paper_df["source"] == "Paper (BOX)"]
        for _, row in sweep_df.groupby(["env", "goal", "include_prior", "budget"])["z_mean"].mean().reset_index().iterrows():
            paper_val = paper_box[
                (paper_box["env"] == row["env"]) &
                (paper_box["goal"] == row["goal"]) &
                (paper_box["include_prior"] == row["include_prior"]) &
                (paper_box["budget"] == row["budget"])
            ]["z_mean"]
            if len(paper_val) > 0 and row["z_mean"] < paper_val.values[0]:
                beat_box += 1
        st.metric("Beat BOX", f"{beat_box}")
    else:
        st.metric("Beat BOX", "—")

st.divider()

# Budget Progression Chart (Figure 4 style)
st.subheader("Budget Progression")
st.caption("Shows z_mean vs experiment budget (in-context learning, not gradient-based). Paper baselines are dashed.")

# Get available environments
all_envs = sorted(paper_df["env"].unique())

# Environment selector
selected_envs = st.multiselect(
    "Select Environments",
    options=all_envs,
    default=all_envs[:6],
    help="Choose which environments to display",
)

if selected_envs:
    fig = build_budget_progression_chart(
        paper_df=paper_df,
        sweep_df=sweep_df,
        selected_envs=selected_envs,
        include_prior=include_prior,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one environment to display budget progression.")

st.divider()

# Model Comparison Bars (Figure 3 style)
st.subheader("Model Comparison")
st.caption(f"Grouped bar chart comparing z_mean at Budget={budget_select}. Lower is better.")

fig = build_model_comparison_bars(
    paper_df=paper_df,
    sweep_df=sweep_df,
    budget=budget_select,
    include_prior=include_prior,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Detailed Comparison Table
st.subheader("Detailed Comparison Table")
st.caption("Side-by-side comparison of z_mean values. Green = model beats GPT-4o baseline.")

# Build comparison table with individual model columns
table_rows = []

# Get unique models from sweep data
sweep_models = sorted(sweep_df["model"].unique()) if len(sweep_df) > 0 and "model" in sweep_df.columns else []

# Get unique (env, goal, prior, budget) combinations from paper data
paper_keys = paper_df.groupby(["env", "goal", "include_prior", "budget"]).first().reset_index()

# Get unique use_ppl values from sweep data (if available)
sweep_has_ppl = len(sweep_df) > 0 and "use_ppl" in sweep_df.columns

for _, row in paper_keys.iterrows():
    env, goal, prior, budget = row["env"], row["goal"], row["include_prior"], row["budget"]

    # Get paper values
    gpt_val = paper_df[
        (paper_df["env"] == env) &
        (paper_df["goal"] == goal) &
        (paper_df["include_prior"] == prior) &
        (paper_df["budget"] == budget) &
        (paper_df["source"] == "Paper (GPT-4o)")
    ]["z_mean"]
    gpt_val = gpt_val.values[0] if len(gpt_val) > 0 else None

    box_val = paper_df[
        (paper_df["env"] == env) &
        (paper_df["goal"] == goal) &
        (paper_df["include_prior"] == prior) &
        (paper_df["budget"] == budget) &
        (paper_df["source"] == "Paper (BOX)")
    ]["z_mean"]
    box_val = box_val.values[0] if len(box_val) > 0 else None

    # For each unique use_ppl value in sweep data (or just one row if no sweep data)
    ppl_values = [False]  # default
    if sweep_has_ppl:
        sweep_match_all = sweep_df[
            (sweep_df["env"] == env) &
            (sweep_df["goal"] == goal) &
            (sweep_df["include_prior"] == prior) &
            (sweep_df["budget"] == budget)
        ]
        if len(sweep_match_all) > 0:
            ppl_values = sorted(sweep_match_all["use_ppl"].unique())
        else:
            ppl_values = [None]  # no sweep data for this config

    for use_ppl in ppl_values:
        row_data = {
            "Environment": env,
            "Goal": goal,
            "Prior": "✓" if prior else "—",
            "PPL": "✓" if use_ppl else "—",
            "Budget": budget,
            "GPT-4o": gpt_val,
            "BOX": box_val,
        }

        # Add column for each model
        for model in sweep_models:
            model_col = str(model)[:20]  # truncate for display (increased from 15)
            if len(sweep_df) > 0 and use_ppl is not None and sweep_has_ppl:
                # Filter including use_ppl
                sweep_match = sweep_df[
                    (sweep_df["env"] == env) &
                    (sweep_df["goal"] == goal) &
                    (sweep_df["include_prior"] == prior) &
                    (sweep_df["budget"] == budget) &
                    (sweep_df["model"] == model) &
                    (sweep_df["use_ppl"] == use_ppl)
                ]["z_mean"]
                row_data[model_col] = sweep_match.mean() if len(sweep_match) > 0 else None
            elif len(sweep_df) > 0 and not sweep_has_ppl:
                # No use_ppl column - filter without it
                sweep_match = sweep_df[
                    (sweep_df["env"] == env) &
                    (sweep_df["goal"] == goal) &
                    (sweep_df["include_prior"] == prior) &
                    (sweep_df["budget"] == budget) &
                    (sweep_df["model"] == model)
                ]["z_mean"]
                row_data[model_col] = sweep_match.mean() if len(sweep_match) > 0 else None
            else:
                row_data[model_col] = None

        table_rows.append(row_data)

comparison_df = pd.DataFrame(table_rows)

# Apply styling - highlight model columns that beat GPT-4o
def style_vs_gpt(row):
    """Style model values based on comparison with GPT-4o."""
    gpt_val = row.get("GPT-4o")
    styles = [""] * len(row)

    for i, (col, val) in enumerate(row.items()):
        if col in ["Environment", "Goal", "Prior", "PPL", "Budget", "GPT-4o", "BOX"]:
            continue
        if pd.isna(val) or pd.isna(gpt_val):
            continue
        diff = val - gpt_val
        if diff < -0.05:  # model is better (lower z_mean)
            styles[i] = "background-color: rgba(52, 211, 153, 0.3); color: #34d399"
        elif diff > 0.05:  # model is worse
            styles[i] = "background-color: rgba(251, 113, 133, 0.3); color: #fb7185"
        else:
            styles[i] = "background-color: rgba(251, 191, 36, 0.15); color: #fbbf24"

    return styles


# Build format dict dynamically
format_dict = {
    "GPT-4o": "{:.2f}",
    "BOX": "{:.2f}",
    "Budget": "{:.0f}",
}
for model in sweep_models:
    model_col = str(model)[:20]
    format_dict[model_col] = "{:.2f}"

styled = comparison_df.style.apply(
    style_vs_gpt,
    axis=1,
).format(format_dict, na_rep="—")

st.dataframe(
    styled,
    use_container_width=True,
    height=500,
    hide_index=True,
)

# Footer
st.divider()
if sweep_ids:
    st.caption(f"Comparing sweeps: {', '.join(sweep_ids)} | {len(sweep_df)} runs | {len(paper_df)} paper baselines")
else:
    st.caption(f"Viewing paper baselines only ({len(paper_df)} entries). Add sweep ID to compare.")
