"""Table rendering components for BoxingGym dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .filters import get_paper_reference_value


def compute_delta_column(
    df: pd.DataFrame,
    baseline: str,
    ref_budget: str | None,
) -> pd.DataFrame:
    """Add delta column comparing z_mean to paper baseline.

    Args:
        df: DataFrame with z_mean and paper columns
        baseline: "gpt" or "box"
        ref_budget: "0", "10", or None

    Returns:
        DataFrame with paper_ref and delta columns added
    """
    df = df.copy()

    # Get paper reference values
    df["paper_ref"] = df.apply(
        lambda row: get_paper_reference_value(row, baseline, ref_budget),
        axis=1,
    )

    # Compute delta (lower is better, so negative delta = improvement)
    df["delta"] = df["z_mean"] - df["paper_ref"]

    # Don't show delta for paper baselines (comparing paper to paper is meaningless)
    # Convert to object type first to allow mixed values, then set paper rows to "—"
    if "model" in df.columns:
        paper_mask = df["model"].astype(str).str.contains("Paper", case=False, na=False)
        # Format non-paper deltas as strings, paper rows as "—"
        df["delta"] = df.apply(
            lambda row: "—" if paper_mask.loc[row.name] else f"{row['delta']:+.4f}",
            axis=1,
        )

    return df


def style_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply styling to the results DataFrame.

    Args:
        df: DataFrame to style

    Returns:
        Styled DataFrame
    """

    def color_delta(val):
        """Color delta values: green for negative (better), red for positive (worse)."""
        if pd.isna(val) or val == "—":
            return ""
        # Handle string-formatted deltas (e.g., "+0.5900", "-0.1234")
        try:
            num_val = float(val) if isinstance(val, str) else val
        except (ValueError, TypeError):
            return ""
        if num_val <= -0.1:
            return "background-color: rgba(52, 211, 153, 0.3); color: #34d399"
        elif num_val >= 0.1:
            return "background-color: rgba(251, 113, 133, 0.3); color: #fb7185"
        return "background-color: rgba(251, 191, 36, 0.15); color: #fbbf24"

    def color_z_mean(val):
        """Color z_mean values."""
        if pd.isna(val):
            return ""
        if val <= -0.5:
            return "color: #34d399"
        elif val >= 0.5:
            return "color: #fb7185"
        return "color: #f59e0b"

    def format_bool(val):
        """Format boolean values as checkmarks."""
        if val is True:
            return "✓"
        elif val is False:
            return "—"
        return val

    # Select columns for styling
    style_cols = []
    if "delta" in df.columns:
        style_cols.append("delta")
    if "z_mean" in df.columns:
        style_cols.append("z_mean")

    styled = df.style

    if "delta" in df.columns:
        styled = styled.map(color_delta, subset=["delta"])

    if "z_mean" in df.columns:
        styled = styled.map(color_z_mean, subset=["z_mean"])

    # Format numeric columns (delta is pre-formatted as string)
    format_dict = {
        "raw_mean": "{:.4f}",
        "raw_std": "{:.4f}",
        "z_mean": "{:+.4f}",
        "z_std": "{:.4f}",
        "sem": "{:.4f}",
        "count": "{:.0f}",
        "paper_ref": "{:.4f}",
        "budget": "{:.0f}",
        "seed": "{:.0f}",
    }

    # Only format columns that exist
    format_dict = {k: v for k, v in format_dict.items() if k in df.columns}
    styled = styled.format(format_dict, na_rep="—")

    return styled


def render_results_table(
    df: pd.DataFrame,
    baseline: str,
    ref_budget: str | None,
) -> None:
    """Render the main results table with filtering and styling.

    Args:
        df: DataFrame with results
        baseline: "gpt" or "box"
        ref_budget: "0", "10", or None
    """
    if df.empty:
        st.warning("No results match the current filters.")
        return

    # Add delta column
    df = compute_delta_column(df, baseline, ref_budget)

    # Detect aggregation mode (sem column present = aggregated)
    is_aggregated = "sem" in df.columns

    # Select and order columns for display
    if is_aggregated:
        display_cols = [
            "env",
            "model",
            "budget",
            "z_mean",
            "sem",
            "count",
            "paper_ref",
            "delta",
            "include_prior",
            "use_ppl",
            "goal",
        ]
    else:
        display_cols = [
            "env",
            "model",
            "budget",
            "z_mean",
            "z_std",
            "paper_ref",
            "delta",
            "raw_mean",
            "raw_std",
            "seed",
            "include_prior",
            "use_ppl",
            "goal",
            "experiment_type",
        ]

    # Only include columns that exist
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].copy()

    # Convert boolean columns to "✓" / "—" for cleaner display
    for bool_col in ["include_prior", "use_ppl"]:
        if bool_col in df_display.columns:
            df_display[bool_col] = df_display[bool_col].apply(lambda x: "✓" if x else "—")

    # Column configuration for Streamlit
    column_config = {
        "env": st.column_config.TextColumn("Environment", width="medium"),
        "model": st.column_config.TextColumn("Model", width="medium"),
        "budget": st.column_config.NumberColumn("Budget", format="%d", width="small"),
        "z_mean": st.column_config.NumberColumn(
            "Z Mean",
            format="%.4f",
            help="Z-score normalized (lower is better)",
        ),
        "z_std": st.column_config.NumberColumn("Z Std", format="%.4f", width="small"),
        "sem": st.column_config.NumberColumn(
            "SEM",
            format="%.4f",
            width="small",
            help="Standard Error of Mean",
        ),
        "count": st.column_config.NumberColumn(
            "N",
            format="%d",
            width="small",
            help="Number of seeds",
        ),
        "paper_ref": st.column_config.NumberColumn(
            "Paper",
            format="%.4f",
            help="Paper baseline value",
        ),
        "delta": st.column_config.TextColumn(
            "Δ",
            help="Difference from paper baseline (negative = better)",
        ),
        "raw_mean": st.column_config.NumberColumn("Raw Mean", format="%.4f"),
        "raw_std": st.column_config.NumberColumn("Raw Std", format="%.4f"),
        "seed": st.column_config.NumberColumn("Seed", format="%d", width="small"),
        "include_prior": st.column_config.TextColumn("Prior", width="small"),
        "use_ppl": st.column_config.TextColumn("PPL", width="small"),
        "goal": st.column_config.TextColumn("Goal", width="medium"),
        "experiment_type": st.column_config.TextColumn("Type", width="small"),
    }

    # Apply styling
    styled_df = style_dataframe(df_display)

    # Render with st.dataframe
    st.dataframe(
        styled_df,
        column_config=column_config,
        use_container_width=True,
        height=600,
        hide_index=True,
    )

    # Export and count
    col1, col2 = st.columns([1, 4])
    with col1:
        st.download_button(
            label="📥 Download CSV",
            data=df_display.to_csv(index=False).encode("utf-8"),
            file_name="benchmark_results.csv",
            mime="text/csv",
        )
    with col2:
        st.caption(f"Showing {len(df_display)} results")


def render_summary_stats(df: pd.DataFrame) -> None:
    """Render summary statistics row.

    Args:
        df: DataFrame with results
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Results", len(df))

    with col2:
        n_envs = df["env"].nunique()
        env_list = ", ".join(sorted(df["env"].unique())[:3])
        if n_envs > 3:
            env_list += f" +{n_envs - 3}"
        st.metric("Environments", n_envs)
        st.caption(env_list)

    with col3:
        n_models = df["model"].nunique()
        model_list = ", ".join(str(m)[:15] for m in sorted(df["model"].unique())[:3])
        if n_models > 3:
            model_list += f" +{n_models - 3}"
        st.metric("Models", n_models)
        st.caption(model_list)

    with col4:
        if "z_mean" in df.columns and not df["z_mean"].isna().all():
            best_idx = df["z_mean"].idxmin()
            best_row = df.loc[best_idx]
            best_z = best_row["z_mean"]
            best_env = str(best_row.get("env", "?"))
            best_model = str(best_row.get("model", "?"))[:15]
            st.metric("Best z_mean", f"{best_z:.3f}")
            st.caption(f"{best_env} / {best_model}")
        else:
            st.metric("Best z_mean", "—")
