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
from utils.data_loader import load_parquet_results
from utils.theme import inject_custom_css

inject_custom_css()

st.title("🏆 Leaderboard")

df = load_parquet_results()

if df.empty:
    st.error("No results found. Run `uv run box sync --local results/` to cache data.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    min_budget = st.slider("Min budget", 0, 20, 10, help="Minimum experiment budget to include")
    envs_available = sorted(df["env"].dropna().unique())
    env_filter = st.selectbox("Environment", ["All"] + envs_available)
    show_outliers = st.checkbox("Include outliers", value=False)

if not show_outliers and "is_outlier" in df.columns:
    df = df[~df["is_outlier"].fillna(False)]
df = df[df["budget"] >= min_budget]
if env_filter != "All":
    df = df[df["env"] == env_filter]

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
models = df["model"].dropna().unique()
model_means = df.groupby("model")["z_mean"].mean()
if model_means.empty:
    st.warning("No valid model scores in filtered data.")
    st.stop()
best_model_name = model_means.idxmin()
best_z = model_means.min()

with c1:
    st.metric("Total Runs", f"{len(df):,}")
with c2:
    st.metric("Models", len(models))
with c3:
    st.metric("Best Model", str(best_model_name)[:18])
with c4:
    st.metric("Best z-score", f"{best_z:+.3f}")

st.divider()

from boxing_gym.analysis.stats import bootstrap_ci, compare_models

with st.spinner("Computing bootstrap CIs..."):
    rankings = []
    for model in models:
        scores = df[df["model"] == model]["z_mean"].dropna().values
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

    comparison = compare_models(
        df.dropna(subset=["z_mean"]), model_col="model",
        score_col="z_mean", reference_model=best_model_name,
    )
    sig_models = {
        c["model"] for c in comparison["comparisons"] if c.get("significant_fdr")
    }
    for r in rankings:
        r["significant"] = r["model"] in sig_models

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
            if c["model"] == r["model"]:
                p_val = c.get("p_adjusted")
                break
        table_rows.append({
            "Rank": i,
            "Model": r["model"],
            "Mean z": f"{r['mean']:+.3f}",
            "95% CI": f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
            "vs #1 (p)": f"{p_val:.3f}" if p_val is not None else "—",
            "n": r["n"],
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

st.divider()

st.subheader("Per-Environment Champions")

champions = []
for env in df["env"].dropna().unique():
    env_df = df[df["env"] == env]
    means = env_df.groupby("model")["z_mean"].mean()
    if means.empty:
        continue
    best = means.idxmin()
    champions.append({
        "env": env,
        "model": best,
        "z_mean": means[best],
    })

champions.sort(key=lambda x: x["z_mean"])

fig = build_env_champions_chart(champions)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Champions table"):
    champ_rows = []
    for ch in champions:
        status = "✓ Beats baseline" if ch["z_mean"] < 0 else "Above baseline"
        champ_rows.append({
            "Environment": ch["env"].replace("_", " ").title(),
            "Best Model": ch["model"],
            "Mean z": f"{ch['z_mean']:+.3f}",
            "Status": status,
        })
    st.dataframe(pd.DataFrame(champ_rows), use_container_width=True, hide_index=True)

st.divider()

st.subheader("Key Findings")
n_beating = sum(1 for ch in champions if ch["z_mean"] < 0)
st.markdown(f"""
- **{rankings[0]['model']}** leads overall with z = {rankings[0]['mean']:+.3f}
- **{n_beating}/{len(champions)}** environments have models beating baseline (z < 0)
- Models with `*` are significantly different from #1 (FDR-corrected p < 0.05)
""")
