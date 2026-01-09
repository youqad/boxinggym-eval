"""Plotly chart builders for BoxingGym sweep analysis."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Color constants
COLORS = {
    "positive": "#34d399",  # Green - good (low z_mean)
    "negative": "#fb7185",  # Red - bad (high z_mean)
    "neutral": "#f59e0b",   # Amber - neutral
    "accent": "#f59e0b",    # Primary accent
    "text": "#f1f5f9",
    "grid": "rgba(148, 163, 184, 0.1)",
}

# Dark mode tooltip styling (applied to each trace to override Plotly's trace-color default)
DARK_HOVERLABEL = dict(
    bgcolor="#1e293b",  # slate-800
    bordercolor="#334155",  # slate-700
    font=dict(color="#f1f5f9", size=12),  # slate-100
)

# Common layout settings
LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, system-ui", color=COLORS["text"]),
    margin=dict(l=20, r=20, t=40, b=20),
    hoverlabel=DARK_HOVERLABEL,
)


def build_importance_chart(importances: List[Dict]) -> go.Figure:
    """Build horizontal bar chart for parameter importance.

    Args:
        importances: List of dicts with 'parameter' and 'importance' keys

    Returns:
        Plotly figure
    """
    if not importances:
        return go.Figure().add_annotation(text="No importance data", showarrow=False)

    df = pd.DataFrame(importances).sort_values("importance", ascending=True).tail(10)

    fig = go.Figure(
        go.Bar(
            x=df["importance"].values,
            y=[p.replace("config/", "") for p in df["parameter"].values],
            orientation="h",
            marker_color=COLORS["accent"],
            text=[f"{v:.3f}" for v in df["importance"].values],
            textposition="outside",
            textfont=dict(size=11),
            hoverlabel=DARK_HOVERLABEL,
        )
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Parameter importance (random forest)", x=0),
        height=400,
        xaxis=dict(
            title="Importance",
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor=COLORS["grid"],
        ),
        yaxis=dict(title=None),
    )

    return fig


def build_model_ranking_chart(aggregates: List[Dict]) -> go.Figure:
    """Build horizontal bar chart for model rankings.

    Args:
        aggregates: List of dicts with model aggregation results

    Returns:
        Plotly figure
    """
    if not aggregates:
        return go.Figure().add_annotation(text="No model data", showarrow=False)

    # Get model column name
    model_col = None
    for col in ["config/llms", "model", "llms"]:
        if col in aggregates[0]:
            model_col = col
            break

    if not model_col:
        return go.Figure().add_annotation(text="No model column found", showarrow=False)

    df = pd.DataFrame(aggregates).sort_values("mean", ascending=True)

    # Clip display values for better visualization (cap at 5)
    actual_vals = df["mean"].values
    display_vals = np.clip(actual_vals, -2, 5)

    # Color bars by performance
    colors = [
        COLORS["positive"] if v < -0.5 else COLORS["negative"] if v > 0.5 else COLORS["neutral"]
        for v in actual_vals
    ]

    # Show actual value in label, but clip bar length
    text_labels = [f"{v:+.2f}" if abs(v) <= 5 else f"{v:+.1f}*" for v in actual_vals]

    # Get error bars if available
    error_x = None
    if "sem" in df.columns:
        error_x = dict(
            type="data",
            array=np.clip(df["sem"].values, 0, 1),
            visible=True,
            color="rgba(255,255,255,0.3)",
        )

    fig = go.Figure(
        go.Bar(
            x=display_vals,
            y=[str(m)[:20] for m in df[model_col].values],
            orientation="h",
            marker_color=colors,
            error_x=error_x,
            text=text_labels,
            textposition="outside",
            textfont=dict(size=11),
            hoverlabel=DARK_HOVERLABEL,
        )
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Model rankings (mean z_mean)", x=0),
        height=max(280, len(df) * 38),
        xaxis=dict(
            title="Mean z_mean (capped at 5)",
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.3)",
            range=[-2, 6],
        ),
        yaxis=dict(title=None, tickfont=dict(size=11)),
    )
    fig.update_layout(margin=dict(l=130, r=60, t=40, b=40))

    return fig


def build_heatmap_chart(
    runs: List[Dict],
    target_metric: str = "metric/eval/z_mean",
) -> go.Figure:
    """Build environment × model heatmap.

    Args:
        runs: List of run dicts with config and metrics
        target_metric: Metric column to visualize

    Returns:
        Plotly figure
    """
    if not runs:
        return go.Figure().add_annotation(text="No run data", showarrow=False)

    df = pd.DataFrame(runs)

    # Find column names
    env_col = None
    model_col = None

    for col in ["config/envs", "env", "envs"]:
        if col in df.columns:
            env_col = col
            break

    for col in ["config/llms", "model", "llms"]:
        if col in df.columns:
            model_col = col
            break

    metric_col = None
    for col in [target_metric, "z_mean", "metric/eval/z_mean"]:
        if col in df.columns:
            metric_col = col
            break

    if not env_col or not model_col or not metric_col:
        return go.Figure().add_annotation(
            text=f"Missing columns: env={env_col}, model={model_col}, metric={metric_col}",
            showarrow=False,
        )

    # Create pivot table
    pivot = df.pivot_table(
        index=env_col,
        columns=model_col,
        values=metric_col,
        aggfunc="mean",
    ).fillna(0)

    # Truncate labels
    x_labels = [str(c)[:18] for c in pivot.columns]
    y_labels = [str(r).replace("_direct", "").replace("_", " ")[:22] for r in pivot.index]

    # Create text matrix for annotations
    text_matrix = [[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in pivot.values]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=x_labels,
            y=y_labels,
            colorscale=[
                [0, COLORS["positive"]],
                [0.3, "#fbbf24"],
                [0.5, COLORS["neutral"]],
                [1, COLORS["negative"]],
            ],
            zmin=-1,
            zmax=2,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#1e1e1e"),  # dark text for contrast on colored cells
            hovertemplate="<b>%{y}</b><br>%{x}<br>z_mean: %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(title="z_mean", tickfont=dict(color=COLORS["text"])),
            hoverlabel=DARK_HOVERLABEL,
        )
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Environment × model performance", x=0),
        height=max(450, len(pivot.index) * 35),
        xaxis=dict(title="Model", tickangle=45, tickfont=dict(size=11)),
        yaxis=dict(title=None, tickfont=dict(size=11)),
    )
    fig.update_layout(margin=dict(l=150, r=20, t=40, b=100))

    return fig


def build_parameter_effect_chart(
    aggregates: List[Dict],
    parameter_name: str,
) -> go.Figure:
    """Build bar chart showing parameter effect on z_mean.

    Args:
        aggregates: List of dicts with parameter aggregation results
        parameter_name: Name of the parameter being analyzed

    Returns:
        Plotly figure
    """
    if not aggregates:
        return go.Figure().add_annotation(text="No data", showarrow=False)

    df = pd.DataFrame(aggregates).sort_values("mean", ascending=True)

    # Get parameter value column
    param_col = None
    for col in df.columns:
        if col not in ["mean", "std", "count", "sem"]:
            param_col = col
            break

    if not param_col:
        return go.Figure().add_annotation(text="No parameter column", showarrow=False)

    # Color bars by performance
    colors = [
        COLORS["positive"] if v < -0.5 else COLORS["negative"] if v > 0.5 else COLORS["neutral"]
        for v in df["mean"].values
    ]

    # Error bars
    error_x = None
    if "sem" in df.columns:
        error_x = dict(type="data", array=df["sem"].values, visible=True)

    fig = go.Figure(
        go.Bar(
            x=df["mean"].values,
            y=[str(v) for v in df[param_col].values],
            orientation="h",
            marker_color=colors,
            error_x=error_x,
            text=[f"{v:+.3f}" for v in df["mean"].values],
            textposition="outside",
            hoverlabel=DARK_HOVERLABEL,
        )
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"Effect of {parameter_name.lower()}", x=0),
        height=max(200, len(df) * 40),
        xaxis=dict(
            title="Mean z_mean",
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.3)",
        ),
        yaxis=dict(title=None),
    )

    return fig


# Paper comparison colors
PAPER_COLORS = {
    "Paper (GPT-4o)": "#60a5fa",  # blue
    "Paper (BOX)": "#fb923c",     # orange
}


def build_budget_progression_chart(
    paper_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    selected_envs: Optional[List[str]] = None,
    include_prior: bool = True,
) -> go.Figure:
    """Build faceted line chart showing z_mean vs experiment budget.

    Paper lines are dashed; sweep model lines are solid.

    Args:
        paper_df: DataFrame with paper baselines
        sweep_df: DataFrame with sweep results
        selected_envs: Optional list of environments to show
        include_prior: Filter to this prior setting

    Returns:
        Plotly figure with faceted subplots
    """
    # Filter by prior
    paper = paper_df[paper_df["include_prior"] == include_prior].copy()
    sweep = sweep_df[sweep_df["include_prior"] == include_prior].copy() if len(sweep_df) > 0 else pd.DataFrame()

    # Get environments to show
    if selected_envs:
        envs = selected_envs
    else:
        envs = sorted(paper["env"].unique())[:6]  # max 6 envs

    if len(envs) == 0:
        return go.Figure().add_annotation(text="No data for selected filters", showarrow=False)

    # Create subplots grid
    n_cols = min(3, len(envs))
    n_rows = (len(envs) + n_cols - 1) // n_cols

    # Shorten environment names for subplot titles
    short_envs = [e.replace("_direct", "").replace("_", " ")[:18] for e in envs]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=short_envs,
        horizontal_spacing=0.12,  # more space between columns
        vertical_spacing=0.18,    # more space between rows for titles
    )

    # Track which models we've added to legend
    legend_added = set()

    for idx, env in enumerate(envs):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Paper lines
        for source in ["Paper (GPT-4o)", "Paper (BOX)"]:
            data = paper[(paper["env"] == env) & (paper["source"] == source)]
            if len(data) == 0:
                continue

            data = data.sort_values("budget")
            show_legend = source not in legend_added
            legend_added.add(source)

            fig.add_trace(
                go.Scatter(
                    x=data["budget"],
                    y=data["z_mean"],
                    mode="lines+markers",
                    name=source,
                    line=dict(
                        color=PAPER_COLORS.get(source, COLORS["neutral"]),
                        dash="dash",
                        width=2,
                    ),
                    marker=dict(size=6),
                    showlegend=show_legend,
                    legendgroup=source,
                    hoverlabel=DARK_HOVERLABEL,
                ),
                row=row,
                col=col,
            )

        # Sweep model lines
        if len(sweep) > 0:
            sweep_env = sweep[sweep["env"] == env]
            models = sweep_env["model"].unique() if "model" in sweep_env.columns else []

            sweep_colors = ["#a78bfa", "#f472b6", "#4ade80", "#facc15"]  # purple, pink, green, yellow

            for m_idx, model in enumerate(models[:4]):
                data = sweep_env[sweep_env["model"] == model]
                # Aggregate by budget
                agg = data.groupby("budget")["z_mean"].mean().reset_index()
                agg = agg.sort_values("budget")

                show_legend = model not in legend_added
                legend_added.add(model)

                fig.add_trace(
                    go.Scatter(
                        x=agg["budget"],
                        y=agg["z_mean"],
                        mode="lines+markers",
                        name=str(model)[:20],
                        line=dict(
                            color=sweep_colors[m_idx % len(sweep_colors)],
                            width=2,
                        ),
                        marker=dict(size=8),
                        showlegend=show_legend,
                        legendgroup=model,
                        hoverlabel=DARK_HOVERLABEL,
                    ),
                    row=row,
                    col=col,
                )

    # Update layout
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"Budget progression (prior={'yes' if include_prior else 'no'})",
            x=0,
            y=0.98,
        ),
        height=320 * n_rows + 60,  # more height per row
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )
    fig.update_layout(margin=dict(l=50, r=20, t=100, b=40))  # more top margin

    # Style subplot titles (annotations)
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12, color=COLORS["text"])
        annotation['yshift'] = 10  # shift titles up slightly

    # Update axes - only show "Budget" on bottom row
    fig.update_xaxes(dtick=5, tickfont=dict(size=10))
    fig.update_yaxes(title_text="z_mean", zeroline=True, zerolinecolor=COLORS["grid"], tickfont=dict(size=10))

    return fig


def build_model_comparison_bars(
    paper_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    budget: int = 10,
    include_prior: bool = True,
) -> go.Figure:
    """Build grouped bar chart comparing models at specific budget.

    Args:
        paper_df: DataFrame with paper baselines
        sweep_df: DataFrame with sweep results
        budget: Budget level to compare (0 or 10)
        include_prior: Filter to this prior setting

    Returns:
        Plotly figure with grouped bars
    """
    # Filter data
    paper = paper_df[
        (paper_df["budget"] == budget) & (paper_df["include_prior"] == include_prior)
    ].copy()
    sweep = sweep_df[
        (sweep_df["budget"] == budget) & (sweep_df["include_prior"] == include_prior)
    ].copy() if len(sweep_df) > 0 else pd.DataFrame()

    if len(paper) == 0 and len(sweep) == 0:
        return go.Figure().add_annotation(text="No data for selected filters", showarrow=False)

    # Get environments present in data
    envs = sorted(paper["env"].unique())

    fig = go.Figure()

    # Paper GPT-4o bars
    gpt_data = paper[paper["source"] == "Paper (GPT-4o)"]
    if len(gpt_data) > 0:
        gpt_vals = []
        for env in envs:
            match = gpt_data[gpt_data["env"] == env]
            gpt_vals.append(match["z_mean"].values[0] if len(match) > 0 else None)

        fig.add_trace(
            go.Bar(
                x=envs,
                y=gpt_vals,
                name="Paper (GPT-4o)",
                marker_color=PAPER_COLORS["Paper (GPT-4o)"],
                text=[f"{v:.2f}" if v is not None else "" for v in gpt_vals],
                textposition="outside",
                textfont=dict(size=10),
                hoverlabel=DARK_HOVERLABEL,
            )
        )

    # Paper BOX bars
    box_data = paper[paper["source"] == "Paper (BOX)"]
    if len(box_data) > 0:
        box_vals = []
        for env in envs:
            match = box_data[box_data["env"] == env]
            box_vals.append(match["z_mean"].values[0] if len(match) > 0 else None)

        fig.add_trace(
            go.Bar(
                x=envs,
                y=box_vals,
                name="Paper (BOX)",
                marker_color=PAPER_COLORS["Paper (BOX)"],
                text=[f"{v:.2f}" if v is not None else "" for v in box_vals],
                textposition="outside",
                textfont=dict(size=10),
                hoverlabel=DARK_HOVERLABEL,
            )
        )

    # Sweep model bars
    if len(sweep) > 0:
        sweep_colors = ["#a78bfa", "#f472b6", "#4ade80", "#facc15"]
        models = sweep["model"].unique()[:4]

        for m_idx, model in enumerate(models):
            model_data = sweep[sweep["model"] == model]
            # Aggregate by env
            agg = model_data.groupby("env")["z_mean"].mean()

            sweep_vals = []
            for env in envs:
                sweep_vals.append(agg.get(env, None))

            fig.add_trace(
                go.Bar(
                    x=envs,
                    y=sweep_vals,
                    name=str(model)[:20],
                    marker_color=sweep_colors[m_idx % len(sweep_colors)],
                    text=[f"{v:.2f}" if v is not None else "" for v in sweep_vals],
                    textposition="outside",
                    textfont=dict(size=10),
                    hoverlabel=DARK_HOVERLABEL,
                )
            )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"Model comparison (budget={budget}, prior={'yes' if include_prior else 'no'})",
            x=0,
            y=0.98,
        ),
        height=450,
        barmode="group",
        xaxis=dict(
            title="Environment",
            tickangle=45,
        ),
        yaxis=dict(
            title="z_mean (lower is better)",
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.3)",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.15,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_layout(margin=dict(l=20, r=20, t=80, b=100))

    return fig
