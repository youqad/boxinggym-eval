"""Leaderboard - Model rankings with bootstrap CIs and significance testing."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Leaderboard - BoxingGym",
    page_icon="🏆",
    layout="wide",
)

_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

_src = Path(__file__).resolve().parent.parent.parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from components.charts import build_env_champions_chart, build_leaderboard_chart
from utils.data_loader import (
    get_active_results_source,
    is_demo_mode,
    load_canonical_metadata,
    load_parquet_results,
)
from utils.data_selector import load_runs_from_selection, render_data_source_selector
from utils.normalize import get_budget_column
from utils.theme import inject_custom_css

from boxing_gym.agents.results_io import get_model_display_name

inject_custom_css()

st.title("🏆 Leaderboard")

df = load_parquet_results()
active_source = get_active_results_source()
canonical_metadata = load_canonical_metadata()

if df.empty:
    st.error("No results found. Run `uv run box sync --local results/` to cache data.")
    st.stop()

source_labels = {
    "canonical_snapshot": "Canonical Snapshot",
    "local_cache": "Local Cache",
    "bundled_demo_fallback": "Bundled Demo Fallback",
    "explicit_path": "Explicit Path",
    "none": "Not Found",
}

st.caption(f"Data source: **{source_labels.get(active_source, active_source)}** (`{active_source}`)")
if active_source == "canonical_snapshot" and canonical_metadata:
    st.caption(
        "Snapshot generated: "
        f"**{canonical_metadata.get('generated_at_utc', 'unknown')}** | "
        f"source `{canonical_metadata.get('source_path', 'unknown')}` | "
        f"filters `{canonical_metadata.get('filters', {})}`"
    )

canonical_demo_mode = is_demo_mode() and active_source == "canonical_snapshot"

with st.sidebar:
    st.header("Filters")
    min_budget = st.slider("Min budget", 0, 20, 10, help="Minimum experiment budget to include")
    envs_available = sorted(df["env"].dropna().unique())
    env_filter = st.selectbox("Environment", ["All"] + envs_available)
    if canonical_demo_mode:
        st.caption("Outlier policy is fixed in canonical demo mode.")
        show_outliers = False
    else:
        show_outliers = st.checkbox("Include outliers", value=False)

    expander = st.expander("Local Summary Data", expanded=False)
    with expander:
        local_selection = render_data_source_selector(
            expander,
            key_prefix="local_summary",
            title="Local Summary Data",
        )

if not show_outliers and "is_outlier" in df.columns:
    df = df[~df["is_outlier"].fillna(False)]
df = df[df["budget"] >= min_budget]
if env_filter != "All":
    df = df[df["env"] == env_filter]

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
valid_scores_df = df.dropna(subset=["model", "z_mean"]).copy()
models = valid_scores_df["model"].unique()
model_means = valid_scores_df.groupby("model")["z_mean"].mean().dropna()
if model_means.empty:
    st.warning("No valid model scores in filtered data.")
    st.stop()
best_model_name = model_means.idxmin()

best_model_display = get_model_display_name(str(best_model_name))
best_z = model_means.min()

with c1:
    st.metric("Total Runs", f"{len(df):,}")
with c2:
    st.metric("Models", len(models))
with c3:
    st.metric("Best Model", str(best_model_display)[:18])
with c4:
    st.metric("Best z-score", f"{best_z:+.3f}")

st.divider()

from boxing_gym.analysis.stats import bootstrap_ci, compare_models

with st.spinner("Computing bootstrap CIs..."):
    rankings = []
    for model in models:
        scores = valid_scores_df[valid_scores_df["model"] == model]["z_mean"].values
        if len(scores) < 2:
            continue
        ci = bootstrap_ci(scores, confidence=0.95, n_bootstrap=10000)
        rankings.append(
            {
                "model": get_model_display_name(str(model)),
                "model_raw": model,
                "mean": ci.mean,
                "ci_low": ci.ci_low,
                "ci_high": ci.ci_high,
                "n": len(scores),
            }
        )

    if len(rankings) >= 2:
        comparison = compare_models(
            valid_scores_df,
            model_col="model",
            score_col="z_mean",
            reference_model=best_model_name,
        )
        sig_models = {c["model"] for c in comparison["comparisons"] if c.get("significant_fdr")}
        for r in rankings:
            r["significant"] = r["model_raw"] in sig_models
    else:
        comparison = {"comparisons": []}
        for r in rankings:
            r["significant"] = False

rankings.sort(key=lambda x: x["mean"])

if not rankings:
    st.warning("Not enough data for rankings (need ≥2 scores per model).")
    st.stop()

st.subheader("Model Rankings")
fig = build_leaderboard_chart(rankings)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Rankings table", expanded=True):
    table_rows = []
    for i, r in enumerate(rankings, 1):
        p_val = None
        for c in comparison["comparisons"]:
            if c["model"] == r["model_raw"]:
                p_val = c.get("p_adjusted")
                break
        table_rows.append(
            {
                "Rank": i,
                "Model": r["model"],
                "Mean z": f"{r['mean']:+.3f}",
                "95% CI": f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
                "vs #1 (p)": f"{p_val:.3f}" if p_val is not None else "—",
                "n": r["n"],
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

st.divider()

st.subheader("Per-Environment Champions (Best Model Only)")
st.caption("Definition: best model mean z-score within each environment (ignores `use_ppl`).")

champions = []
for env in valid_scores_df["env"].dropna().unique():
    env_df = valid_scores_df[valid_scores_df["env"] == env]
    means = env_df.groupby("model")["z_mean"].mean().dropna()
    if means.empty:
        continue
    best = means.idxmin()
    champions.append(
        {
            "env": env,
            "model": best,
            "z_mean": means[best],
        }
    )

champions.sort(key=lambda x: x["z_mean"])

if not champions:
    st.info("No environment champions available for the current filters.")
else:
    chart_rows = []
    for ch in champions:
        chart_rows.append(
            {
                "env": ch["env"],
                "model": get_model_display_name(str(ch["model"])),
                "z_mean": ch["z_mean"],
            }
        )
    fig = build_env_champions_chart(chart_rows)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Champions table"):
    if not champions:
        st.info("No champions to show.")
    else:
        champ_rows = []
        for ch in champions:
            status = "✓ Beats baseline" if ch["z_mean"] < 0 else "Above baseline"
            champ_rows.append(
                {
                    "Environment": ch["env"].replace("_", " ").title(),
                    "Best Model": get_model_display_name(str(ch["model"])),
                    "Mean z": f"{ch['z_mean']:+.3f}",
                    "Status": status,
                }
            )
        st.dataframe(pd.DataFrame(champ_rows), use_container_width=True, hide_index=True)

st.divider()

st.subheader("Per-Environment Champions (Best Model + PPL)")
st.caption("Definition: best `(model, use_ppl)` mean z-score within each environment.")

champions_with_ppl = []
for env in valid_scores_df["env"].dropna().unique():
    env_df = valid_scores_df[valid_scores_df["env"] == env]
    agg = env_df.groupby(["model", "use_ppl"])["z_mean"].mean().reset_index()
    if agg.empty:
        continue
    best_idx = agg["z_mean"].idxmin()
    row = agg.loc[best_idx]
    champions_with_ppl.append(
        {
            "env": env,
            "model": str(row["model"]),
            "model_display": get_model_display_name(str(row["model"])),
            "use_ppl": bool(row["use_ppl"]),
            "z_mean": float(row["z_mean"]),
        }
    )

champions_with_ppl.sort(key=lambda x: x["z_mean"])

if champions_with_ppl:
    chart_rows = []
    for row in champions_with_ppl:
        ppl_suffix = "PPL" if row["use_ppl"] else "No PPL"
        chart_rows.append(
            {
                "env": row["env"],
                "model": f"{row['model_display']} ({ppl_suffix})",
                "z_mean": row["z_mean"],
            }
        )
    st.plotly_chart(build_env_champions_chart(chart_rows), use_container_width=True)
else:
    st.info("No environment champions available for model+PPL definition.")

with st.expander("Champions table (model + PPL)"):
    if champions_with_ppl:
        rows = []
        for row in champions_with_ppl:
            rows.append(
                {
                    "Environment": row["env"].replace("_", " ").title(),
                    "Best Model": row["model_display"],
                    "use_ppl": "true" if row["use_ppl"] else "false",
                    "Mean z": f"{row['z_mean']:+.3f}",
                    "Status": "✓ Beats baseline" if row["z_mean"] < 0 else "Above baseline",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No champions to show.")

st.divider()

st.subheader("Key Findings")
n_beating = sum(1 for ch in champions if ch["z_mean"] < 0)

findings = []

# 1. overall leader
leader = rankings[0]
runner_up = rankings[1] if len(rankings) > 1 else None
leader_line = f"**{leader['model']}** leads overall (z\u2009=\u2009{leader['mean']:+.3f})"
if runner_up and abs(leader["mean"] - runner_up["mean"]) < 0.05:
    leader_line += f", statistically tied with **{runner_up['model']}**"
findings.append(leader_line)

# 2. baseline performance
findings.append(
    f"**{n_beating}/{len(champions)}** environments beaten (z\u2009<\u20090)"
)

# 3. PPL impact (if column present)
if "use_ppl" in valid_scores_df.columns:
    ppl_groups = valid_scores_df.groupby("use_ppl")["z_mean"].mean()
    if True in ppl_groups.index and False in ppl_groups.index:
        z_ppl = ppl_groups[True]
        z_no_ppl = ppl_groups[False]
        better_label = "without PPL" if z_no_ppl < z_ppl else "with PPL"
        better_z = min(z_ppl, z_no_ppl)
        worse_z = max(z_ppl, z_no_ppl)
        findings.append(
            f"**PPL impact**: runs {better_label} average z\u2009=\u2009{better_z:+.3f} "
            f"vs {worse_z:+.3f}"
        )

# 4. environment difficulty
if len(champions) >= 2:
    easiest = champions[0]  # already sorted ascending by z_mean
    hardest = champions[-1]
    findings.append(
        f"**Easiest env**: {easiest['env'].replace('_', ' ')} "
        f"(z\u2009=\u2009{easiest['z_mean']:+.3f}) · "
        f"**Hardest**: {hardest['env'].replace('_', ' ')} "
        f"(z\u2009=\u2009{hardest['z_mean']:+.3f})"
    )

# 5. model specialization — count env wins per model
if len(champions) >= 3:
    from collections import Counter
    win_counts = Counter(
        get_model_display_name(str(ch["model"])) for ch in champions
    )
    top_winners = win_counts.most_common(2)
    if top_winners[0][1] > 1:
        parts = [f"**{name}** ({count} envs)" for name, count in top_winners if count > 1]
        if parts:
            findings.append(f"**Multi-env winners**: {', '.join(parts)}")

# 6. significance note
n_sig = sum(1 for r in rankings if r.get("significant"))
if n_sig > 0:
    findings.append(
        f"{n_sig} model(s) significantly different from #1 (FDR-corrected p\u2009<\u20090.05)"
    )

st.markdown("\n".join(f"- {f}" for f in findings))

st.divider()

st.header("Local Summary")

with st.expander("Load local summary", expanded=False):
    enable_local = st.checkbox("Enable local summary", value=False, key="local_summary_enable")

    if enable_local:
        if local_selection.data_source == "Local JSON Files":
            with st.spinner("Loading local JSON files..."):
                all_runs = load_runs_from_selection(local_selection)
        else:
            if not local_selection.sweep_ids:
                st.info("Select sweep(s) in the sidebar to load local summary data.")
                all_runs = []
            else:
                with st.spinner(f"Loading {len(local_selection.sweep_ids)} sweep(s) from W&B..."):
                    all_runs = load_runs_from_selection(local_selection)

        if not all_runs:
            st.info("No runs loaded for local summary.")
        else:
            summary_df = pd.DataFrame(all_runs)
            if summary_df.empty:
                st.info("No data available for local summary.")
            else:
                metric_col = "metric/eval/z_mean"
                budget_col = get_budget_column(summary_df)
                if metric_col not in summary_df.columns:
                    st.info("Missing z_mean metric in selected data.")
                else:
                    if budget_col and "run_id" in summary_df.columns:
                        budget_num = pd.to_numeric(summary_df[budget_col], errors="coerce")
                        if budget_num.notna().any():
                            summary_df["_budget_key"] = budget_num
                            idx = summary_df.groupby("run_id")["_budget_key"].idxmax()
                            final_df = (
                                summary_df.loc[idx]
                                .drop(columns=["_budget_key"], errors="ignore")
                                .copy()
                            )
                        else:
                            final_df = summary_df.copy()
                    else:
                        final_df = summary_df.copy()

                    if "summary/n_seeds" not in final_df.columns:
                        final_df["summary/n_seeds"] = 1
                    final_df["summary/n_seeds"] = (
                        pd.to_numeric(final_df["summary/n_seeds"], errors="coerce")
                        .fillna(1)
                        .clip(lower=1)
                    )

                    if metric_col in final_df.columns and "config/llms" in final_df.columns:

                        def weighted_agg(group):
                            weights = group["summary/n_seeds"]
                            z_vals = group[metric_col]
                            mask = z_vals.notna()
                            if mask.sum() == 0:
                                return pd.Series(
                                    {
                                        "mean": float("nan"),
                                        "abs_mean": float("nan"),
                                        "run_count": len(group),
                                        "effective_seeds": 0,
                                        "envs": group["config/envs"].nunique()
                                        if "config/envs" in group.columns
                                        else 0,
                                    }
                                )
                            w_mean = (z_vals[mask] * weights[mask]).sum() / weights[mask].sum()
                            return pd.Series(
                                {
                                    "mean": w_mean,
                                    "abs_mean": abs(w_mean),
                                    "run_count": len(group),
                                    "effective_seeds": int(weights[mask].sum()),
                                    "envs": group["config/envs"].nunique()
                                    if "config/envs" in group.columns
                                    else 0,
                                }
                            )

                        model_rank = final_df.groupby("config/llms").apply(
                            weighted_agg, include_groups=False
                        )
                        model_rank = model_rank.dropna(subset=["mean"])
                        model_rank = model_rank.sort_values("abs_mean")
                        model_rank = model_rank.reset_index().rename(
                            columns={"config/llms": "model"}
                        )
                        model_rank["mean"] = model_rank["mean"].map(lambda v: f"{v:+.3f}")

                        st.subheader("Model Rankings (weighted |z|)")
                        st.dataframe(model_rank, use_container_width=True, hide_index=True)

                    if "config/envs" in final_df.columns and "config/llms" in final_df.columns:
                        st.subheader("Per-Environment Top 3")
                        rows = []
                        for env in sorted(final_df["config/envs"].dropna().unique()):
                            env_df = final_df[final_df["config/envs"] == env]

                            def weighted_mean(group):
                                weights = group["summary/n_seeds"]
                                z_vals = group[metric_col]
                                mask = z_vals.notna()
                                if mask.sum() == 0:
                                    return pd.Series({"mean": float("nan"), "effective_seeds": 0})
                                w_mean = (z_vals[mask] * weights[mask]).sum() / weights[mask].sum()
                                return pd.Series(
                                    {"mean": w_mean, "effective_seeds": int(weights[mask].sum())}
                                )

                            agg = env_df.groupby("config/llms").apply(
                                weighted_mean, include_groups=False
                            )
                            agg = agg.dropna(subset=["mean"])
                            agg["abs_mean"] = agg["mean"].abs()
                            agg = agg.sort_values("abs_mean").head(3)
                            for rank, (model, row) in enumerate(agg.iterrows(), 1):
                                rows.append(
                                    {
                                        "environment": env,
                                        "rank": rank,
                                        "model": model,
                                        "z_mean": f"{row['mean']:+.3f}",
                                        "effective_seeds": row["effective_seeds"],
                                    }
                                )
                        if rows:
                            st.dataframe(
                                pd.DataFrame(rows), use_container_width=True, hide_index=True
                            )

                        st.subheader("Coverage (env × model)")
                        envs = sorted(final_df["config/envs"].dropna().unique())
                        model_coverage = final_df.groupby("config/llms")["config/envs"].nunique()
                        top_models = (
                            model_coverage.sort_values(ascending=False).head(8).index.tolist()
                        )
                        coverage_rows = []
                        for env in envs:
                            row = {"environment": env}
                            for model in top_models:
                                has_data = (
                                    (final_df["config/envs"] == env)
                                    & (final_df["config/llms"] == model)
                                ).any()
                                row[str(model)] = "✓" if has_data else "—"
                            coverage_rows.append(row)
                        if coverage_rows:
                            st.dataframe(
                                pd.DataFrame(coverage_rows),
                                use_container_width=True,
                                hide_index=True,
                            )
    else:
        st.info("Enable this section to load local summary diagnostics.")
