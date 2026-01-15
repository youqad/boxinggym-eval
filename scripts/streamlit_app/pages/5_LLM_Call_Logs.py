"""LLM Call Logs - View and analyze JSONL call recordings.

Browse LLM call logs from experiments with filtering, cost analysis, and latency charts.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

# Page config
st.set_page_config(
    page_title="LLM Call Logs - BoxingGym",
    page_icon="📞",
    layout="wide",
)

# Add parent to path for imports
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from utils.theme import inject_custom_css

# Inject custom CSS
inject_custom_css()


def find_jsonl_files(base_paths: List[Path]) -> List[Path]:
    """Find all JSONL call log files across multiple directories."""
    files = []
    for base in base_paths:
        if base.exists():
            files.extend(base.rglob("llm_calls_*.jsonl"))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False, ttl=60)
def load_jsonl_file(path: str) -> pd.DataFrame:
    """Load JSONL file to DataFrame with caching."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not entries:
        return pd.DataFrame()

    df = pd.DataFrame(entries)

    # ensure numeric columns
    for col in ["latency_ms", "cost_usd", "prompt_tokens", "completion_tokens", "reasoning_tokens"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def format_cost(cost: float) -> str:
    """Format cost with appropriate precision."""
    if cost < 0.0001:
        return f"${cost:.6f}"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def truncate_text(text: str, max_len: int = 100) -> str:
    """Truncate text with ellipsis."""
    if not isinstance(text, str):
        return str(text)[:max_len]
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# Title
st.title("📞 LLM Call Logs")
st.markdown("API call costs, latencies, and token counts.")

# Find JSONL files
project_root = Path(__file__).resolve().parent.parent.parent.parent
search_paths = [
    project_root / "outputs",
    project_root / "results",
    Path.cwd() / "outputs",
    Path.cwd() / "results",
]

jsonl_files = find_jsonl_files(search_paths)

if not jsonl_files:
    st.warning("No JSONL call log files found.")
    st.info(
        """
        **To generate call logs:**
        1. Run an experiment with call recording enabled
        2. Look for `llm_calls_*.jsonl` files in the `outputs/` directory

        ```bash
        uv run python run_experiment.py seeds=null seed=1 exp=oed envs=hyperbolic_direct
        ```
        """
    )
    st.stop()

# File selection
st.sidebar.header("📁 File Selection")
file_options = {str(f): f for f in jsonl_files}
file_labels = [f.name for f in jsonl_files]

selected_file_str = st.sidebar.selectbox(
    "Select JSONL file",
    options=list(file_options.keys()),
    format_func=lambda x: Path(x).name,
)

if not selected_file_str:
    st.stop()

selected_file = file_options[selected_file_str]
st.sidebar.caption(f"Path: `{selected_file}`")

# Load data
with st.spinner("Loading call logs..."):
    df = load_jsonl_file(str(selected_file))

if df.empty:
    st.error("No valid entries in the selected file.")
    st.stop()

# Summary metrics
st.header("📊 Summary")

col1, col2, col3, col4 = st.columns(4)

total_calls = len(df)
total_cost = df["cost_usd"].sum() if "cost_usd" in df.columns else 0
avg_latency = df["latency_ms"].mean() if "latency_ms" in df.columns else 0
error_count = df["error"].notna().sum() if "error" in df.columns else 0

col1.metric("Total Calls", f"{total_calls:,}")
col2.metric("Total Cost", format_cost(total_cost))
col3.metric("Avg Latency", f"{avg_latency:,.0f} ms")
col4.metric("Errors", f"{error_count:,}", delta=None if error_count == 0 else f"-{error_count}", delta_color="inverse")

# Token summary
if all(col in df.columns for col in ["prompt_tokens", "completion_tokens"]):
    total_prompt = df["prompt_tokens"].sum()
    total_completion = df["completion_tokens"].sum()
    total_reasoning = df["reasoning_tokens"].sum() if "reasoning_tokens" in df.columns else 0

    st.caption(
        f"**Tokens:** {total_prompt:,} prompt + {total_completion:,} completion"
        + (f" + {total_reasoning:,} reasoning" if total_reasoning > 0 else "")
    )

st.divider()

# Filters
st.sidebar.header("🔍 Filters")

# Agent filter
agents = ["All"] + sorted(df["agent"].unique().tolist()) if "agent" in df.columns else ["All"]
selected_agent = st.sidebar.selectbox("Agent", agents)

# Model filter
models = ["All"] + sorted(df["model"].unique().tolist()) if "model" in df.columns else ["All"]
selected_model = st.sidebar.selectbox("Model", models)

# Call type filter
call_types = ["All"] + sorted(df["call_type"].unique().tolist()) if "call_type" in df.columns else ["All"]
selected_call_type = st.sidebar.selectbox("Call Type", call_types)

# Errors only toggle
show_errors_only = st.sidebar.checkbox("Show errors only", value=False)

# Apply filters
filtered_df = df.copy()

if selected_agent != "All" and "agent" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["agent"] == selected_agent]

if selected_model != "All" and "model" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["model"] == selected_model]

if selected_call_type != "All" and "call_type" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["call_type"] == selected_call_type]

if show_errors_only and "error" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["error"].notna()]

st.sidebar.caption(f"Showing {len(filtered_df):,} of {len(df):,} calls")

# By-agent breakdown
st.header("👥 By Agent")

if "agent" in df.columns:
    agent_stats = (
        df.groupby("agent")
        .agg(
            calls=("agent", "count"),
            cost=("cost_usd", "sum"),
            avg_latency=("latency_ms", "mean"),
            errors=("error", lambda x: x.notna().sum()),
        )
        .reset_index()
        .sort_values("calls", ascending=False)
    )

    # format for display
    agent_stats["cost"] = agent_stats["cost"].apply(format_cost)
    agent_stats["avg_latency"] = agent_stats["avg_latency"].apply(lambda x: f"{x:,.0f} ms")

    st.dataframe(
        agent_stats,
        column_config={
            "agent": st.column_config.TextColumn("Agent", width="medium"),
            "calls": st.column_config.NumberColumn("Calls", format="%d"),
            "cost": st.column_config.TextColumn("Cost"),
            "avg_latency": st.column_config.TextColumn("Avg Latency"),
            "errors": st.column_config.NumberColumn("Errors", format="%d"),
        },
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# Charts
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("💰 Cost by Model")
    if "model" in df.columns and "cost_usd" in df.columns:
        cost_by_model = df.groupby("model")["cost_usd"].sum().sort_values(ascending=True)
        st.bar_chart(cost_by_model, horizontal=True)
    else:
        st.info("Cost data not available")

with col_right:
    st.subheader("⏱️ Latency Distribution")
    if "latency_ms" in df.columns:
        st.bar_chart(df["latency_ms"].value_counts(bins=20).sort_index())
    else:
        st.info("Latency data not available")

st.divider()

# Call logs table
st.header("📋 Call Logs")

# Prepare display dataframe
display_cols = ["agent", "model", "call_type", "latency_ms", "cost_usd", "step_idx", "timestamp"]
display_cols = [c for c in display_cols if c in filtered_df.columns]

if display_cols:
    display_df = filtered_df[display_cols].copy()

    # Format columns
    if "cost_usd" in display_df.columns:
        display_df["cost_usd"] = display_df["cost_usd"].apply(format_cost)
    if "latency_ms" in display_df.columns:
        display_df["latency_ms"] = display_df["latency_ms"].apply(lambda x: f"{x:,.0f}")

    st.dataframe(
        display_df,
        column_config={
            "agent": st.column_config.TextColumn("Agent", width="small"),
            "model": st.column_config.TextColumn("Model", width="medium"),
            "call_type": st.column_config.TextColumn("Type", width="small"),
            "latency_ms": st.column_config.TextColumn("Latency", width="small"),
            "cost_usd": st.column_config.TextColumn("Cost", width="small"),
            "step_idx": st.column_config.NumberColumn("Step", width="small"),
            "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
        },
        hide_index=True,
        use_container_width=True,
        height=400,
    )

# Expandable call details
st.subheader("🔎 Call Details")

if len(filtered_df) > 0:
    call_idx = st.number_input(
        "Select call index",
        min_value=0,
        max_value=len(filtered_df) - 1,
        value=0,
        step=1,
    )

    selected_call = filtered_df.iloc[call_idx]

    col_meta, col_tokens = st.columns(2)

    with col_meta:
        st.markdown("**Metadata**")
        st.json(
            {
                "agent": selected_call.get("agent", ""),
                "model": selected_call.get("model", ""),
                "call_type": selected_call.get("call_type", ""),
                "step_idx": selected_call.get("step_idx"),
                "latency_ms": selected_call.get("latency_ms", 0),
                "cost_usd": selected_call.get("cost_usd", 0),
                "timestamp": selected_call.get("timestamp", ""),
                "error": selected_call.get("error"),
            }
        )

    with col_tokens:
        st.markdown("**Tokens**")
        st.json(
            {
                "prompt_tokens": selected_call.get("prompt_tokens", 0),
                "completion_tokens": selected_call.get("completion_tokens", 0),
                "reasoning_tokens": selected_call.get("reasoning_tokens", 0),
                "has_reasoning": selected_call.get("has_reasoning", False),
            }
        )

    st.markdown("**Prompt**")
    prompt_text = selected_call.get("prompt", "")
    if len(prompt_text) > 5000:
        st.text_area("Prompt (truncated)", prompt_text[:5000] + "...", height=200, disabled=True)
    else:
        st.text_area("Prompt", prompt_text, height=200, disabled=True)

    st.markdown("**Response**")
    response_text = selected_call.get("response", "")
    if len(response_text) > 5000:
        st.text_area("Response (truncated)", response_text[:5000] + "...", height=200, disabled=True)
    else:
        st.text_area("Response", response_text, height=200, disabled=True)

    if selected_call.get("error"):
        st.error(f"**Error:** {selected_call['error']}")

st.divider()

# Download
st.header("📥 Export")

col_csv, col_json = st.columns(2)

with col_csv:
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        "Download filtered data (CSV)",
        data=csv_data,
        file_name=f"call_logs_{selected_file.stem}.csv",
        mime="text/csv",
    )

with col_json:
    json_data = filtered_df.to_json(orient="records", indent=2)
    st.download_button(
        "Download filtered data (JSON)",
        data=json_data,
        file_name=f"call_logs_{selected_file.stem}.json",
        mime="application/json",
    )
