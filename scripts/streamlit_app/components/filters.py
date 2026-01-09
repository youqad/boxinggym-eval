"""Sidebar filter components for BoxingGym dashboard."""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def render_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render sidebar filters and return filter state.

    Args:
        df: DataFrame with results to filter

    Returns:
        Dictionary with filter values
    """
    with st.sidebar:
        st.header("🔍 Filters")

        # Data source filters
        st.subheader("Data")

        # Environment filter
        envs = ["All"] + sorted(df["env"].dropna().unique().tolist())
        env = st.selectbox("Environment", envs, key="filter_env")

        # Model filter
        models = ["All"] + sorted(df["model"].dropna().unique().tolist())
        model = st.selectbox("Model", models, key="filter_model")

        # Budget filter
        budgets = ["All"] + [str(b) for b in sorted(df["budget"].dropna().unique())]
        budget = st.selectbox("Budget", budgets, key="filter_budget")

        st.divider()

        # Comparison filters
        st.subheader("Paper Comparison")

        baseline = st.selectbox(
            "Baseline",
            ["GPT-4o Paper", "BOX Paper"],
            key="filter_baseline",
        )

        ref_budget = st.selectbox(
            "Reference Budget",
            ["Any", "0", "10"],
            key="filter_ref_budget",
        )

        prior = st.selectbox(
            "Prior Included",
            ["All", "True", "False"],
            key="filter_prior",
        )

        st.divider()

        # Additional filters
        st.subheader("Advanced")

        search = st.text_input(
            "Search",
            placeholder="Filter by path, goal...",
            key="filter_search",
        )

        matches_only = st.checkbox(
            "Paper matches only",
            value=False,
            key="filter_matches",
            help="Show only results that have paper baseline values",
        )

        st.divider()

        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return {
        "env": env if env != "All" else None,
        "model": model if model != "All" else None,
        "budget": int(budget) if budget != "All" else None,
        "baseline": "gpt" if "GPT" in baseline else "box",
        "ref_budget": ref_budget.lower() if ref_budget != "Any" else None,
        "prior": prior == "True" if prior != "All" else None,
        "search": search.lower().strip() if search else None,
        "matches_only": matches_only,
    }


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to DataFrame.

    Args:
        df: DataFrame to filter
        filters: Filter values from render_sidebar_filters

    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()

    # Environment filter
    if filters["env"]:
        filtered = filtered[filtered["env"] == filters["env"]]

    # Model filter
    if filters["model"]:
        filtered = filtered[filtered["model"] == filters["model"]]

    # Budget filter
    if filters["budget"] is not None:
        filtered = filtered[filtered["budget"] == filters["budget"]]

    # Prior filter
    if filters["prior"] is not None:
        filtered = filtered[filtered["include_prior"] == filters["prior"]]

    # Search filter
    if filters["search"]:
        search_term = filters["search"]
        mask = (
            filtered["path"].str.lower().str.contains(search_term, na=False) |
            filtered["goal"].str.lower().str.contains(search_term, na=False) |
            filtered["env"].str.lower().str.contains(search_term, na=False) |
            filtered["model"].str.lower().str.contains(search_term, na=False)
        )
        filtered = filtered[mask]

    # Paper matches filter
    if filters["matches_only"]:
        # Check if paper values exist
        baseline = filters["baseline"]
        ref_budget = filters["ref_budget"]

        if baseline == "gpt":
            if ref_budget == "0":
                filtered = filtered[filtered["paper_gpt4o_b0"].notna()]
            elif ref_budget == "10":
                filtered = filtered[filtered["paper_gpt4o_b10"].notna()]
            else:
                filtered = filtered[
                    filtered["paper_gpt4o_b0"].notna() |
                    filtered["paper_gpt4o_b10"].notna()
                ]
        else:
            if ref_budget == "0":
                filtered = filtered[filtered["paper_box_b0"].notna()]
            elif ref_budget == "10":
                filtered = filtered[filtered["paper_box_b10"].notna()]
            else:
                filtered = filtered[
                    filtered["paper_box_b0"].notna() |
                    filtered["paper_box_b10"].notna()
                ]

    return filtered


def get_paper_reference_value(
    row: pd.Series,
    baseline: str,
    ref_budget: Optional[str],
) -> Optional[float]:
    """Get the appropriate paper reference value for a row.

    Args:
        row: DataFrame row
        baseline: "gpt" or "box"
        ref_budget: "0", "10", or None (use row's budget)

    Returns:
        Paper reference value or None
    """
    if ref_budget is None:
        # Use same budget as the row
        budget = row.get("budget")
        if budget == 0:
            ref_budget = "0"
        elif budget == 10:
            ref_budget = "10"
        else:
            ref_budget = "10"  # Default to budget 10

    if baseline == "gpt":
        if ref_budget == "0":
            return row.get("paper_gpt4o_b0")
        return row.get("paper_gpt4o_b10")
    else:
        if ref_budget == "0":
            return row.get("paper_box_b0")
        return row.get("paper_box_b10")
