"""Plotly chart builders for BoxingGym sweep analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "positive": "#34d399",  # Green - good (low z_mean)
    "negative": "#fb7185",  # Red - bad (high z_mean)
    "neutral": "#f59e0b",  # Amber - neutral
    "accent": "#f59e0b",  # Primary accent
    "text": "#f1f5f9",
    "grid": "rgba(148, 163, 184, 0.1)",
}

COLORS_ACCESSIBLE = {
    "good": "#2563eb",  # Blue - beats baseline (z < 0)
    "bad": "#ea580c",  # Orange - worse than baseline (z > 0.3)
    "neutral": "#6b7280",  # Gray - near baseline
}

DARK_HOVERLABEL = dict(
    bgcolor="#1e293b",  # slate-800
    bordercolor="#334155",  # slate-700
    font=dict(color="#f1f5f9", size=12),  # slate-100
)

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, system-ui", color=COLORS["text"]),
    margin=dict(l=20, r=20, t=40, b=20),
    hoverlabel=DARK_HOVERLABEL,
)


def build_importance_chart(importances: list[dict]) -> go.Figure:
    """Horizontal bar chart for parameter importance."""
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


def build_model_ranking_chart(aggregates: list[dict]) -> go.Figure:
    """Horizontal bar chart for model rankings."""
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

    actual_vals = df["mean"].values
    display_vals = np.clip(actual_vals, -2, 5)

    colors = [
        COLORS["positive"] if v < -0.5 else COLORS["negative"] if v > 0.5 else COLORS["neutral"]
        for v in actual_vals
    ]

    text_labels = [f"{v:+.2f}" if abs(v) <= 5 else f"{v:+.1f}*" for v in actual_vals]

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
    runs: list[dict],
    target_metric: str = "metric/eval/z_mean",
) -> go.Figure:
    """Environment x model heatmap."""
    if not runs:
        return go.Figure().add_annotation(text="No run data", showarrow=False)

    df = pd.DataFrame(runs)

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

    # Coerce metric to numeric to avoid pivot/np.isnan crashes on mixed types.
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    pivot = df.pivot_table(
        index=env_col,
        columns=model_col,
        values=metric_col,
        aggfunc="mean",
    )

    x_labels = [str(c)[:18] for c in pivot.columns]
    y_labels = [str(r).replace("_direct", "").replace("_", " ")[:22] for r in pivot.index]

    text_matrix = [[f"{v:.2f}" if not pd.isna(v) else "" for v in row] for row in pivot.values]

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
    aggregates: list[dict],
    parameter_name: str,
) -> go.Figure:
    """Bar chart showing parameter effect on z_mean."""
    if not aggregates:
        return go.Figure().add_annotation(text="No data", showarrow=False)

    df = pd.DataFrame(aggregates).sort_values("mean", ascending=True)

    param_col = None
    for col in df.columns:
        if col not in ["mean", "std", "count", "sem"]:
            param_col = col
            break

    if not param_col:
        return go.Figure().add_annotation(text="No parameter column", showarrow=False)

    colors = [
        COLORS["positive"] if v < -0.5 else COLORS["negative"] if v > 0.5 else COLORS["neutral"]
        for v in df["mean"].values
    ]

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


def build_budget_progression_sweep_chart(
    df: pd.DataFrame,
    metric_col: str,
    budget_col: str,
    model_col: str,
    max_models: int = 8,
) -> go.Figure:
    """Line chart of z_mean vs budget by model."""
    if (
        df.empty
        or metric_col not in df.columns
        or budget_col not in df.columns
        or model_col not in df.columns
    ):
        return go.Figure().add_annotation(text="No data for budget progression", showarrow=False)

    df_plot = df[[model_col, budget_col, metric_col]].copy()
    budget_num = pd.to_numeric(df_plot[budget_col], errors="coerce")
    if budget_num.notna().sum() >= 2:
        df_plot["_budget_key"] = budget_num
        df_plot = df_plot.dropna(subset=["_budget_key", metric_col, model_col])
        budget_key = "_budget_key"
        xaxis_type = "linear"
    else:
        df_plot = df_plot.dropna(subset=[budget_col, metric_col, model_col])
        df_plot["_budget_key"] = df_plot[budget_col].astype(str)
        budget_key = "_budget_key"
        xaxis_type = "category"

    agg = df_plot.groupby([model_col, budget_key])[metric_col].mean().reset_index()
    if agg.empty:
        return go.Figure().add_annotation(text="No data for budget progression", showarrow=False)

    models = agg[model_col].dropna().unique().tolist()
    models = models[:max_models]

    fig = go.Figure()

    palette = [
        "#60a5fa",
        "#f472b6",
        "#4ade80",
        "#facc15",
        "#f97316",
        "#a78bfa",
        "#22c55e",
        "#fb7185",
    ]

    for i, model in enumerate(models):
        model_data = agg[agg[model_col] == model].sort_values(budget_key)
        fig.add_trace(
            go.Scatter(
                x=model_data[budget_key],
                y=model_data[metric_col],
                mode="lines+markers",
                name=str(model)[:20],
                line=dict(color=palette[i % len(palette)], width=2),
                marker=dict(size=7),
                hovertemplate=f"{model}<br>Budget: %{{x}}<br>z_mean: %{{y:.3f}}<extra></extra>",
                hoverlabel=DARK_HOVERLABEL,
            )
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Budget progression by model", x=0),
        xaxis=dict(
            title="Budget",
            type=xaxis_type,
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor=COLORS["grid"],
        ),
        yaxis=dict(
            title="z_mean (lower is better)",
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor=COLORS["grid"],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=500,
    )
    fig.update_layout(margin=dict(l=50, r=20, t=40, b=80))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(148, 163, 184, 0.5)")

    return fig


PAPER_COLORS = {
    "Paper (GPT-4o)": "#60a5fa",  # blue
    "Paper (BOX)": "#fb923c",  # orange
}


def build_budget_progression_chart(
    paper_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    selected_envs: list[str] | None = None,
    include_prior: bool = True,
) -> go.Figure:
    """Faceted line chart: z_mean vs budget. Paper lines dashed, sweep lines solid."""
    paper = paper_df[paper_df["include_prior"] == include_prior].copy()
    sweep = (
        sweep_df[sweep_df["include_prior"] == include_prior].copy()
        if len(sweep_df) > 0
        else pd.DataFrame()
    )

    if selected_envs:
        envs = selected_envs
    else:
        envs = sorted(paper["env"].unique())[:6]  # max 6 envs

    if len(envs) == 0:
        return go.Figure().add_annotation(text="No data for selected filters", showarrow=False)

    n_cols = min(3, len(envs))
    n_rows = (len(envs) + n_cols - 1) // n_cols

    short_envs = [e.replace("_direct", "").replace("_", " ")[:18] for e in envs]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=short_envs,
        horizontal_spacing=0.12,  # more space between columns
        vertical_spacing=0.18,
    )

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

            sweep_colors = [
                "#a78bfa",
                "#f472b6",
                "#4ade80",
                "#facc15",
            ]  # purple, pink, green, yellow

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

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"Budget progression (prior={'yes' if include_prior else 'no'})",
            x=0,
            y=0.98,
        ),
        height=320 * n_rows + 60,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )
    fig.update_layout(margin=dict(l=50, r=20, t=100, b=40))

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=12, color=COLORS["text"])
        annotation["yshift"] = 10

    fig.update_xaxes(dtick=5, tickfont=dict(size=10))
    fig.update_yaxes(
        title_text="z_mean", zeroline=True, zerolinecolor=COLORS["grid"], tickfont=dict(size=10)
    )

    return fig


def build_model_comparison_bars(
    paper_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    budget: int = 10,
    include_prior: bool = True,
) -> go.Figure:
    """Grouped bar chart comparing models at specific budget."""
    paper = paper_df[
        (paper_df["budget"] == budget) & (paper_df["include_prior"] == include_prior)
    ].copy()
    sweep = (
        sweep_df[
            (sweep_df["budget"] == budget) & (sweep_df["include_prior"] == include_prior)
        ].copy()
        if len(sweep_df) > 0
        else pd.DataFrame()
    )

    if len(paper) == 0 and len(sweep) == 0:
        return go.Figure().add_annotation(text="No data for selected filters", showarrow=False)

    envs = sorted(paper["env"].unique())

    fig = go.Figure()

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

    if len(sweep) > 0:
        sweep_colors = ["#a78bfa", "#f472b6", "#4ade80", "#facc15"]
        models = sweep["model"].unique()[:4]

        for m_idx, model in enumerate(models):
            model_data = sweep[sweep["model"] == model]
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


def build_leaderboard_chart(rankings: list[dict]) -> go.Figure:
    """Horizontal bar chart with bootstrap CI error bars."""
    if not rankings:
        return go.Figure().add_annotation(text="No ranking data", showarrow=False)

    df = pd.DataFrame(rankings).sort_values("mean", ascending=True)

    colors = []
    for z in df["mean"].values:
        if z < 0:
            colors.append(COLORS_ACCESSIBLE["good"])
        elif z > 0.3:
            colors.append(COLORS_ACCESSIBLE["bad"])
        else:
            colors.append(COLORS_ACCESSIBLE["neutral"])

    error_minus = (df["mean"] - df["ci_low"]).clip(lower=0).values
    error_plus = (df["ci_high"] - df["mean"]).clip(lower=0).values

    labels = []
    for _, row in df.iterrows():
        sig = "*" if row.get("significant", False) else ""
        labels.append(f"{row['mean']:+.3f}{sig}")

    fig = go.Figure(
        go.Bar(
            x=df["mean"].values,
            y=[str(m) for m in df["model"].values],
            orientation="h",
            marker_color=colors,
            error_x=dict(
                type="data",
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                visible=True,
                color="rgba(255,255,255,0.4)",
                thickness=1.5,
            ),
            text=labels,
            textposition="outside",
            textfont=dict(size=11),
            cliponaxis=False,
            hovertemplate=("<b>%{y}</b><br>Mean z: %{x:+.3f}<br><extra></extra>"),
            hoverlabel=DARK_HOVERLABEL,
        )
    )

    x_max = (df["mean"] + error_plus).max()
    x_padding = max(0.08, x_max * 0.25)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Model rankings (mean z-score, 95% CI)", x=0),
        height=max(300, len(df) * 45),
        xaxis=dict(
            title="Mean z-score (lower is better)",
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.3)",
            range=[min(-0.05, df["mean"].min() - 0.05), x_max + x_padding],
        ),
        yaxis=dict(title=None, tickfont=dict(size=12)),
    )
    fig.update_layout(margin=dict(l=200, r=80, t=50, b=40))

    return fig


def build_env_champions_chart(champions: list[dict]) -> go.Figure:
    """Horizontal bar chart: best model per environment."""
    if not champions:
        return go.Figure().add_annotation(text="No champion data", showarrow=False)

    df = pd.DataFrame(champions).sort_values("z_mean", ascending=True)

    colors = []
    for z in df["z_mean"].values:
        if z < 0:
            colors.append(COLORS_ACCESSIBLE["good"])
        elif z > 0.3:
            colors.append(COLORS_ACCESSIBLE["bad"])
        else:
            colors.append(COLORS_ACCESSIBLE["neutral"])

    env_labels = [e.replace("_direct", "").replace("_", " ").title() for e in df["env"].values]

    fig = go.Figure(
        go.Bar(
            x=df["z_mean"].values,
            y=env_labels,
            orientation="h",
            marker_color=colors,
            hovertemplate=("<b>%{y}</b><br>z: %{x:+.3f}<br><extra></extra>"),
            hoverlabel=DARK_HOVERLABEL,
        )
    )

    for i, (_, row) in enumerate(df.iterrows()):
        z = row["z_mean"]
        label = f"{z:+.3f} ({row['model']})"
        if z >= 0:
            x_pos = z + 0.02
            anchor = "left"
        else:
            x_pos = 0.02
            anchor = "left"
        fig.add_annotation(
            x=x_pos,
            y=env_labels[i],
            text=label,
            showarrow=False,
            xanchor=anchor,
            font=dict(size=10, color=COLORS["text"]),
        )

    x_min = df["z_mean"].min()
    x_max = df["z_mean"].max()
    max_label_len = max(len(f"{r['z_mean']:+.3f} ({r['model']})") for _, r in df.iterrows())
    x_right_pad = max(0.5, max_label_len * 0.022)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Per-environment champions (best model)", x=0),
        height=max(380, len(df) * 44),
        xaxis=dict(
            title="Mean z-score (lower is better)",
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.3)",
            range=[min(x_min - 0.1, -0.1), x_max + x_right_pad],
        ),
        yaxis=dict(title=None, tickfont=dict(size=11)),
    )
    fig.update_layout(margin=dict(l=200, r=40, t=50, b=40))

    return fig
