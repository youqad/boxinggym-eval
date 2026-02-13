"""PPL Examples - View generated PyMC models from Box's Loop.

Shows the best probabilistic programs generated during use_ppl=True runs,
with detailed visualizations, diagnostics, and LLM trajectory data.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page config
st.set_page_config(
    page_title="PPL Examples - BoxingGym",
    page_icon="🧪",
    layout="wide",
)

# Add parent to path for imports
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from utils.data_loader import (
    get_entity_from_env,
    get_project_from_env,
)
from utils.theme import inject_custom_css

# Inject custom CSS
inject_custom_css()

# Title
st.title("🧪 PPL Examples")
st.markdown(
    "PyMC models from `use_ppl=True` runs. Check diagnostics: Rhat < 1.01, ESS > 400, divergences = 0."
)

# Sidebar controls
with st.sidebar:
    st.header("🎯 Configuration")

    entity = st.text_input("Entity", value=get_entity_from_env() or "")
    project = st.text_input("Project", value=get_project_from_env() or "")

    st.divider()

    # Filters
    st.subheader("Filters")
    budget_filter = st.selectbox("Budget", options=["All", 10, 5, 0], index=0)
    env_filter = st.text_input("Environment (contains)", placeholder="e.g., dugongs")
    model_filter = st.text_input("Model (contains)", placeholder="e.g., gpt-4o")

    st.divider()

    max_runs = st.slider("Max runs to load", 10, 200, 50)
    fetch_details = st.toggle(
        "Fetch detailed data", value=True, help="Load tables/artifacts (slower)"
    )

    st.divider()

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


def safe_get_config_value(
    config: dict, key: str, subkey: str, fallback_key: str, default="unknown"
):
    """Safely extract config values handling both dict and string formats."""
    val = config.get(key, {})
    if isinstance(val, dict):
        return val.get(subkey, config.get(fallback_key, default))
    return str(val) if val else config.get(fallback_key, default)


@st.cache_data(ttl=300, show_spinner="Loading PPL runs from W&B...")
def load_ppl_runs(entity: str, project: str, max_runs: int = 50, fetch_details: bool = True):
    """Load runs with use_ppl=True and extract comprehensive PPL data."""
    import wandb

    api = wandb.Api(timeout=120)

    # Query runs with use_ppl=True
    runs = api.runs(
        f"{entity}/{project}",
        filters={"config.use_ppl": True, "state": "finished"},
        order="-created_at",
    )

    results = []
    count = 0
    for run in runs:
        if count >= max_runs:
            break

        config = run.config
        summary = run.summary

        # Extract PPL code from config or summary
        ppl_code = config.get("ppl/best_program_code", "") or summary.get(
            "ppl/best_program_code", ""
        )
        if not ppl_code:
            continue

        # Extract environment and model names safely
        env_name = safe_get_config_value(config, "envs", "name", "env_name")
        model_name = safe_get_config_value(config, "llms", "model", "model_name")

        # Core metrics
        best_loo = summary.get("ppl/best_loo") or config.get("ppl/best_loo")
        best_waic = summary.get("ppl/best_waic") or config.get("ppl/best_waic")
        mean_loo = summary.get("ppl/mean_loo")
        mean_waic = summary.get("ppl/mean_waic")

        # Diagnostic metrics
        num_programs = summary.get("ppl/num_programs", 1)
        num_rounds = summary.get("ppl/num_rounds", 1)
        total_divergences = summary.get("ppl/total_divergences", 0)
        max_rhat = summary.get("ppl/max_rhat_any")
        min_ess = summary.get("ppl/min_ess_any")
        best_divergences = summary.get("ppl/best_n_divergences", 0)
        best_max_rhat = summary.get("ppl/best_max_rhat")
        best_min_ess = summary.get("ppl/best_min_ess_bulk")
        has_ppc = summary.get("ppl/has_ppc", False)

        # Build result
        result = {
            "run_id": run.id,
            "run_name": run.name,
            "env": env_name,
            "model": model_name,
            "budget": config.get("budget", summary.get("eval/budget", 0)),
            "seed": config.get("seed", 0),
            "include_prior": config.get("include_prior", True),
            "z_mean": summary.get("eval/z_mean"),
            "best_loo": best_loo,
            "best_waic": best_waic,
            "mean_loo": mean_loo,
            "mean_waic": mean_waic,
            "ppl_code": ppl_code,
            "num_programs": num_programs,
            "num_rounds": num_rounds,
            "total_divergences": total_divergences,
            "max_rhat": max_rhat,
            "min_ess": min_ess,
            "best_divergences": best_divergences,
            "best_max_rhat": best_max_rhat,
            "best_min_ess": best_min_ess,
            "has_ppc": has_ppc,
            "url": run.url,
            "created_at": run.created_at,
        }

        # Fetch detailed tables if requested
        if fetch_details:
            try:
                # Try to get programs table
                programs_table = []
                round_summary = []
                llm_responses = []

                for key in ["ppl/programs", "ppl/round_summary", "ppl/llm_responses"]:
                    if key in summary:
                        try:
                            table_data = summary[key]
                            if hasattr(table_data, "get_dataframe"):
                                df = table_data.get_dataframe()
                                if key == "ppl/programs":
                                    programs_table = df.to_dict("records")
                                elif key == "ppl/round_summary":
                                    round_summary = df.to_dict("records")
                                elif key == "ppl/llm_responses":
                                    llm_responses = df.to_dict("records")
                        except Exception:
                            pass

                result["programs_table"] = programs_table
                result["round_summary"] = round_summary
                result["llm_responses"] = llm_responses

            except Exception:
                result["programs_table"] = []
                result["round_summary"] = []
                result["llm_responses"] = []

        results.append(result)
        count += 1

    return results


def build_loo_progression_chart(df: pd.DataFrame):
    """Build LOO progression chart across environments and models."""
    if "best_loo" not in df.columns or df["best_loo"].isna().all():
        return None

    # Group by environment and model
    grouped = df.groupby(["env", "model"])["best_loo"].min().reset_index()

    fig = px.bar(
        grouped,
        x="env",
        y="best_loo",
        color="model",
        barmode="group",
        title="Best LOO Score by Environment & Model",
        labels={"best_loo": "LOO (lower is better)", "env": "Environment"},
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def build_diagnostic_chart(df: pd.DataFrame):
    """Build diagnostic metrics overview chart."""
    if len(df) == 0:
        return None

    # Create summary stats
    diag_df = df[["env", "model", "best_divergences", "best_max_rhat", "best_min_ess"]].copy()
    diag_df = diag_df.dropna(subset=["best_max_rhat", "best_min_ess"])

    if len(diag_df) == 0:
        return None

    fig = go.Figure()

    # Add scatter for rhat vs ess colored by divergences
    fig.add_trace(
        go.Scatter(
            x=diag_df["best_max_rhat"],
            y=diag_df["best_min_ess"],
            mode="markers",
            marker=dict(
                size=10,
                color=diag_df["best_divergences"],
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Divergences"),
            ),
            text=diag_df.apply(lambda r: f"{r['env']} ({r['model']})", axis=1),
            hovertemplate="<b>%{text}</b><br>Rhat: %{x:.3f}<br>ESS: %{y:.0f}<extra></extra>",
        )
    )

    # Add reference lines
    fig.add_hline(y=400, line_dash="dash", line_color="green", annotation_text="ESS=400 (good)")
    fig.add_vline(x=1.01, line_dash="dash", line_color="green", annotation_text="Rhat=1.01 (good)")

    fig.update_layout(
        title="MCMC Diagnostics: Rhat vs ESS",
        xaxis_title="Max Rhat (should be < 1.01)",
        yaxis_title="Min ESS (should be > 400)",
        height=400,
    )

    return fig


def build_model_performance_chart(df: pd.DataFrame):
    """Build model performance comparison chart."""
    if len(df) == 0:
        return None

    # Aggregate by model
    model_stats = (
        df.groupby("model")
        .agg(
            {
                "z_mean": "mean",
                "best_loo": "mean",
                "num_programs": "sum",
                "run_id": "count",
            }
        )
        .reset_index()
    )
    model_stats.columns = ["model", "avg_z_mean", "avg_loo", "total_programs", "num_runs"]

    fig = px.scatter(
        model_stats,
        x="avg_z_mean",
        y="avg_loo",
        size="total_programs",
        color="model",
        text="model",
        title="Model Performance: Z-Mean vs LOO",
        labels={
            "avg_z_mean": "Avg Z-Mean (lower is better)",
            "avg_loo": "Avg LOO (lower is better)",
        },
    )

    fig.update_traces(textposition="top center")
    fig.update_layout(height=400, showlegend=False)

    return fig


# Load data
if not entity or not project:
    st.error("Please set **Entity** and **Project** in the sidebar.")
else:
    with st.spinner("Loading PPL runs..."):
        ppl_runs = load_ppl_runs(entity, project, max_runs, fetch_details)

    if not ppl_runs:
        st.warning("No PPL runs found. Make sure you have runs with `use_ppl=True`.")
        st.info("Try running: `uv run python run_experiment.py use_ppl=true envs=dugongs_direct`")
    else:
        # Convert to DataFrame
        df = pd.DataFrame(ppl_runs)

        # Apply filters
        if budget_filter != "All":
            df = df[df["budget"] == int(budget_filter)]
        if env_filter:
            df = df[df["env"].str.contains(env_filter, case=False, na=False)]
        if model_filter:
            df = df[df["model"].str.contains(model_filter, case=False, na=False)]

        # Summary metrics
        st.subheader("Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("PPL Runs", len(df))
        with c2:
            st.metric("Environments", df["env"].nunique() if len(df) > 0 else 0)
        with c3:
            st.metric("Models", df["model"].nunique() if len(df) > 0 else 0)
        with c4:
            if len(df) > 0:
                best_loo = df["best_loo"].dropna().min()
                st.metric("Best LOO", f"{best_loo:.2f}" if pd.notna(best_loo) else "—")
            else:
                st.metric("Best LOO", "—")
        with c5:
            if len(df) > 0:
                total_progs = df["num_programs"].sum()
                st.metric("Programs Generated", f"{int(total_progs)}")
            else:
                st.metric("Programs Generated", "—")

        st.divider()

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Best Models",
                "📈 Visualizations",
                "🔬 Diagnostics",
                "🔍 Detailed View",
            ]
        )

        with tab1:
            if len(df) > 0:
                st.subheader("Best PPL Models by Environment")
                st.caption(
                    "Showing the best-performing PyMC model for each environment (lowest z_mean)."
                )

                # Find best (lowest z_mean) for each env
                valid_df = df.dropna(subset=["z_mean"])
                if len(valid_df) > 0:
                    best_per_env = valid_df.loc[valid_df.groupby("env")["z_mean"].idxmin()]

                    for _, row in best_per_env.iterrows():
                        loo_str = (
                            f"{row['best_loo']:.2f}" if pd.notna(row.get("best_loo")) else "N/A"
                        )
                        with st.expander(
                            f"**{row['env']}** — {row['model']} (z={row['z_mean']:.2f}, LOO={loo_str})",
                            expanded=False,
                        ):
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.markdown("#### Generated PyMC Model")
                                st.code(row["ppl_code"], language="python")

                            with col2:
                                st.markdown("#### Run Info")
                                st.markdown(f"**Environment:** `{row['env']}`")
                                st.markdown(f"**Model:** `{row['model']}`")
                                st.markdown(f"**Budget:** {row['budget']}")
                                st.markdown(f"**Seed:** {row['seed']}")
                                st.markdown(
                                    f"**Include Prior:** {'✓' if row['include_prior'] else '—'}"
                                )

                                st.markdown("---")
                                st.markdown("#### Performance")
                                st.markdown(f"**z_mean:** {row['z_mean']:.3f}")
                                if pd.notna(row.get("best_loo")):
                                    st.markdown(f"**Best LOO:** {row['best_loo']:.2f}")
                                if pd.notna(row.get("best_waic")):
                                    st.markdown(f"**Best WAIC:** {row['best_waic']:.2f}")

                                st.markdown("---")
                                st.markdown("#### Generation Stats")
                                st.markdown(f"**Programs Generated:** {row.get('num_programs', 1)}")
                                st.markdown(f"**Rounds:** {row.get('num_rounds', 1)}")

                                st.markdown("---")
                                st.markdown("#### MCMC Diagnostics")
                                if pd.notna(row.get("best_divergences")):
                                    div_color = (
                                        "🟢"
                                        if row["best_divergences"] == 0
                                        else "🟡"
                                        if row["best_divergences"] < 10
                                        else "🔴"
                                    )
                                    st.markdown(
                                        f"{div_color} **Divergences:** {int(row['best_divergences'])}"
                                    )
                                if pd.notna(row.get("best_max_rhat")):
                                    rhat_color = (
                                        "🟢"
                                        if row["best_max_rhat"] < 1.01
                                        else "🟡"
                                        if row["best_max_rhat"] < 1.1
                                        else "🔴"
                                    )
                                    st.markdown(
                                        f"{rhat_color} **Max Rhat:** {row['best_max_rhat']:.3f}"
                                    )
                                if pd.notna(row.get("best_min_ess")):
                                    ess_color = (
                                        "🟢"
                                        if row["best_min_ess"] > 400
                                        else "🟡"
                                        if row["best_min_ess"] > 100
                                        else "🔴"
                                    )
                                    st.markdown(
                                        f"{ess_color} **Min ESS:** {row['best_min_ess']:.0f}"
                                    )

                                st.markdown("---")
                                st.markdown(f"[View Run on W&B]({row['url']})")
                else:
                    st.info("No runs with valid z_mean scores found.")

        with tab2:
            st.subheader("Performance Visualizations")

            if len(df) > 0:
                col1, col2 = st.columns(2)

                with col1:
                    loo_fig = build_loo_progression_chart(df)
                    if loo_fig:
                        st.plotly_chart(loo_fig, use_container_width=True)
                    else:
                        st.info("No LOO data available for visualization.")

                with col2:
                    perf_fig = build_model_performance_chart(df)
                    if perf_fig:
                        st.plotly_chart(perf_fig, use_container_width=True)
                    else:
                        st.info("No performance data available.")

                # Environment x Model heatmap
                st.markdown("#### LOO Heatmap: Environment × Model")
                pivot = df.pivot_table(
                    values="best_loo", index="env", columns="model", aggfunc="min"
                )
                if not pivot.empty:
                    fig = px.imshow(
                        pivot,
                        aspect="auto",
                        color_continuous_scale="RdYlGn_r",
                        title="Best LOO by Environment and Model (lower is better)",
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data for heatmap.")
            else:
                st.info("No data to visualize.")

        with tab3:
            st.subheader("MCMC Diagnostics Overview")

            if len(df) > 0:
                # Diagnostics chart
                diag_fig = build_diagnostic_chart(df)
                if diag_fig:
                    st.plotly_chart(diag_fig, use_container_width=True)

                    st.markdown("""
                    **Interpretation:**
                    - 🟢 **Rhat < 1.01**: Chains have converged
                    - 🟢 **ESS > 400**: Sufficient effective samples
                    - 🔴 **Divergences > 0**: May indicate model misspecification
                    """)
                else:
                    st.info("No diagnostic data available.")

                # Diagnostics table
                st.markdown("#### Diagnostics Summary Table")
                diag_cols = [
                    "env",
                    "model",
                    "best_divergences",
                    "best_max_rhat",
                    "best_min_ess",
                    "num_programs",
                ]
                diag_df = df[diag_cols].copy()
                diag_df = diag_df.dropna(subset=["best_max_rhat", "best_min_ess"])

                if len(diag_df) > 0:
                    st.dataframe(
                        diag_df.style.format(
                            {
                                "best_max_rhat": "{:.3f}",
                                "best_min_ess": "{:.0f}",
                                "best_divergences": "{:.0f}",
                            },
                            na_rep="—",
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No diagnostic data available.")
            else:
                st.info("No data to analyze.")

        with tab4:
            st.subheader("Detailed Run Exploration")
            st.caption("Select a run to view full details including LLM trajectory.")

            if len(df) > 0:
                # Run selector
                run_options = df.apply(
                    lambda r: f"{r['run_name']} ({r['env']}, {r['model']}, b={r['budget']})", axis=1
                ).tolist()

                selected_run_label = st.selectbox("Select Run", run_options)
                selected_idx = run_options.index(selected_run_label)
                selected_run = df.iloc[selected_idx]

                # Full code view
                st.markdown("#### Full PyMC Model Code")
                st.code(selected_run["ppl_code"], language="python")

                # Metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Performance Metrics**")
                    st.json(
                        {
                            "z_mean": round(selected_run["z_mean"], 4)
                            if pd.notna(selected_run["z_mean"])
                            else None,
                            "best_loo": round(selected_run["best_loo"], 2)
                            if pd.notna(selected_run.get("best_loo"))
                            else None,
                            "best_waic": round(selected_run["best_waic"], 2)
                            if pd.notna(selected_run.get("best_waic"))
                            else None,
                            "mean_loo": round(selected_run["mean_loo"], 2)
                            if pd.notna(selected_run.get("mean_loo"))
                            else None,
                        }
                    )

                with col2:
                    st.markdown("**Generation Stats**")
                    st.json(
                        {
                            "num_programs": int(selected_run.get("num_programs", 1)),
                            "num_rounds": int(selected_run.get("num_rounds", 1)),
                            "has_posterior_predictive": selected_run.get("has_ppc", False),
                        }
                    )

                with col3:
                    st.markdown("**MCMC Diagnostics**")
                    st.json(
                        {
                            "divergences": int(selected_run.get("best_divergences", 0))
                            if pd.notna(selected_run.get("best_divergences"))
                            else None,
                            "max_rhat": round(selected_run.get("best_max_rhat", 0), 4)
                            if pd.notna(selected_run.get("best_max_rhat"))
                            else None,
                            "min_ess": round(selected_run.get("best_min_ess", 0), 0)
                            if pd.notna(selected_run.get("best_min_ess"))
                            else None,
                        }
                    )

                # LLM Responses (if available)
                if (
                    fetch_details
                    and "llm_responses" in selected_run
                    and selected_run["llm_responses"]
                ):
                    st.markdown("---")
                    st.markdown("#### LLM Generation Trajectory")
                    st.caption("The prompts and responses that led to this PyMC model.")

                    for i, resp in enumerate(selected_run["llm_responses"]):
                        with st.expander(
                            f"Round {resp.get('round', i + 1)}, Program {resp.get('program_idx', i + 1)}"
                        ):
                            if "llm_response" in resp:
                                st.markdown("**LLM Response:**")
                                st.code(resp["llm_response"][:5000], language="text")
                            if "program_code" in resp:
                                st.markdown("**Extracted Code:**")
                                st.code(resp["program_code"][:3000], language="python")

                # Programs table (if available)
                if (
                    fetch_details
                    and "programs_table" in selected_run
                    and selected_run["programs_table"]
                ):
                    st.markdown("---")
                    st.markdown("#### All Programs Generated")
                    programs_df = pd.DataFrame(selected_run["programs_table"])
                    if len(programs_df) > 0:
                        display_cols = [
                            c
                            for c in [
                                "round",
                                "program_idx",
                                "loo_score",
                                "waic_score",
                                "n_divergences",
                                "max_rhat",
                                "min_ess_bulk",
                            ]
                            if c in programs_df.columns
                        ]
                        if display_cols:
                            st.dataframe(
                                programs_df[display_cols], use_container_width=True, hide_index=True
                            )

                # Link to W&B
                st.markdown("---")
                st.markdown(f"🔗 [View full run on W&B]({selected_run['url']})")
            else:
                st.info("No runs to explore.")

        # Full table at bottom
        st.divider()
        st.subheader("All PPL Runs")

        if len(df) > 0:
            display_cols = [
                "env",
                "model",
                "budget",
                "seed",
                "z_mean",
                "best_loo",
                "best_waic",
                "num_programs",
                "run_name",
            ]
            display_df = df[[c for c in display_cols if c in df.columns]].copy()
            display_df = display_df.sort_values(["env", "z_mean"])

            st.dataframe(
                display_df.style.format(
                    {
                        "z_mean": "{:.3f}",
                        "best_loo": "{:.2f}",
                        "best_waic": "{:.2f}",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )

        # Code comparison
        st.divider()
        st.subheader("Compare PPL Models")
        st.caption("Select two runs to compare their generated PyMC code side by side.")

        if len(df) >= 2:
            available_runs = df.apply(
                lambda r: f"{r['run_name']} ({r['env']}, {r['model']}, b={r['budget']})", axis=1
            ).tolist()

            col1, col2 = st.columns(2)
            with col1:
                run1_label = st.selectbox("Run 1", available_runs, index=0, key="compare_run1")
            with col2:
                run2_label = st.selectbox(
                    "Run 2",
                    available_runs,
                    index=min(1, len(available_runs) - 1),
                    key="compare_run2",
                )

            run1_idx = available_runs.index(run1_label)
            run2_idx = available_runs.index(run2_label)

            code1 = df.iloc[run1_idx]["ppl_code"]
            code2 = df.iloc[run2_idx]["ppl_code"]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{run1_label}**")
                st.code(code1, language="python")
            with col2:
                st.markdown(f"**{run2_label}**")
                st.code(code2, language="python")
        else:
            st.info("Need at least 2 runs to compare.")

        # Footer
        st.divider()
        st.caption(f"Loaded {len(ppl_runs)} PPL runs from {entity}/{project}")
