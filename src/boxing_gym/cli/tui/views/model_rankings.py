"""Model Rankings view for TUI."""

from typing import Optional

import pandas as pd
from rich.table import Table

from boxing_gym.analysis.stats import bootstrap_ci, compare_models

from ..components.ascii_charts import colored_z
from . import BaseView


class ModelRankingsView(BaseView):
    @property
    def title(self) -> str:
        return "Model Rankings"

    def render(self) -> None:
        if self.metric not in self.df.columns:
            self.console.print(f"[yellow]Metric '{self.metric}' not found[/yellow]")
            return

        if "config/llms" not in self.df.columns:
            self.console.print("[yellow]No model (config/llms) column found[/yellow]")
            return

        agg = self.df.groupby("config/llms")[self.metric].agg(["mean", "std", "count"])
        agg = agg.sort_values("mean")
        agg = agg.reset_index()

        if agg.empty:
            self.console.print("[yellow]No data to rank[/yellow]")
            return

        ci_data = {}
        for model in agg["config/llms"]:
            scores = self.df[self.df["config/llms"] == model][self.metric].dropna().values
            if len(scores) < 2:
                continue
            ci_data[model] = bootstrap_ci(scores)

        # Only rank models with enough samples for a CI and valid p-values.
        agg_display = agg[agg["config/llms"].isin(ci_data.keys())].copy()
        if agg_display.empty:
            self.console.print("[yellow]Need at least 2 samples per model for rankings[/yellow]")
            return

        best_model = agg_display.iloc[0]["config/llms"]

        p_lookup: dict = {}
        if len(ci_data) >= 2:
            comparisons = compare_models(
                self.df,
                model_col="config/llms",
                score_col=self.metric,
                reference_model=best_model,
            )
            p_lookup = {c["model"]: c for c in comparisons["comparisons"]}

        table = Table(
            title="Model Rankings (lower z_mean = better)",
            border_style="cyan",
            header_style="bold magenta",
        )
        table.add_column("#", justify="right", width=4)
        table.add_column("Model", style="white", width=30)
        table.add_column("Mean z", justify="right", width=10)
        table.add_column("95% CI", justify="right", width=18)
        if p_lookup:
            table.add_column("vs #1", justify="right", width=10)
        table.add_column("n", justify="right", width=6)

        for rank, (_, row) in enumerate(agg_display.iterrows(), 1):
            model = str(row["config/llms"])
            mean = row["mean"]
            count = int(row["count"])
            ci = ci_data.get(model)
            if ci is None:
                continue

            z_str = colored_z(mean)
            ci_str = f"[{ci.ci_low:+.3f}, {ci.ci_high:+.3f}]"

            cells = [str(rank), model[:30], z_str, ci_str]

            if p_lookup:
                if model == best_model:
                    cells.append("-")
                else:
                    comp = p_lookup.get(model, {})
                    p_adj = comp.get("p_adjusted")
                    p_raw = comp.get("p_value")
                    p_val = p_adj if p_adj is not None else p_raw
                    sig = comp.get("significant_fdr", False)
                    if p_val is None or pd.isna(p_val):
                        cells.append("—")
                    else:
                        cells.append(f"[red]p={p_val:.2f}*[/red]" if sig else f"p={p_val:.2f}")

            cells.append(str(count))
            table.add_row(*cells)

        self.console.print(table)

        best_z = agg_display.iloc[0]["mean"]
        best_ci = ci_data.get(best_model)
        worst_model = agg_display.iloc[-1]["config/llms"]
        worst_z = agg_display.iloc[-1]["mean"]

        if best_ci:
            self.console.print(
                f"\n[green]Best:[/green]  {best_model} (z={best_z:+.3f}, "
                f"95% CI [{best_ci.ci_low:+.3f}, {best_ci.ci_high:+.3f}])"
            )
        else:
            self.console.print(f"\n[green]Best:[/green]  {best_model} (z={best_z:+.3f})")
        self.console.print(f"[red]Worst:[/red] {worst_model} (z={worst_z:+.3f})")
        self.console.print(f"[cyan]Spread:[/cyan] {worst_z - best_z:.3f}")
        displayed_models = set(agg_display["config/llms"].tolist())
        sig_worse = [
            m for m, c in p_lookup.items() if c.get("significant_fdr") and m in displayed_models
        ]
        if sig_worse:
            self.console.print(
                f"[red]Significantly worse than #1 (FDR<0.05):[/red] {', '.join(sig_worse)}"
            )

    def get_data(self) -> dict:
        if self.metric not in self.df.columns or "config/llms" not in self.df.columns:
            return {"rankings": []}

        agg = self.df.groupby("config/llms")[self.metric].agg(["mean", "std", "count"])
        agg = agg.sort_values("mean")
        agg = agg.reset_index()

        if agg.empty:
            return {"rankings": []}

        ci_data = {}
        for model in agg["config/llms"]:
            scores = self.df[self.df["config/llms"] == model][self.metric].dropna().values
            if len(scores) < 2:
                continue
            ci_data[model] = bootstrap_ci(scores)

        agg_display = agg[agg["config/llms"].isin(ci_data.keys())].copy()
        if agg_display.empty:
            return {"rankings": []}

        best_model = agg_display.iloc[0]["config/llms"]
        p_lookup: dict = {}
        if len(ci_data) >= 2:
            comparisons = compare_models(
                self.df,
                model_col="config/llms",
                score_col=self.metric,
                reference_model=best_model,
            )
            p_lookup = {c["model"]: c for c in comparisons["comparisons"]}

        rankings = []
        for rank, (_, row) in enumerate(agg_display.iterrows(), 1):
            model = str(row["config/llms"])
            ci = ci_data.get(model)
            if ci is None:
                continue
            comp = p_lookup.get(model, {})
            rankings.append(
                {
                    "rank": rank,
                    "model": model,
                    "mean": float(row["mean"]) if not pd.isna(row["mean"]) else None,
                    "ci_low": float(ci.ci_low),
                    "ci_high": float(ci.ci_high),
                    "p_adjusted": comp.get("p_adjusted"),
                    "significant_fdr": comp.get("significant_fdr", False),
                    "count": int(row["count"]),
                }
            )

        return {"rankings": rankings}

    def get_csv_rows(self) -> list:
        data = self.get_data()
        rankings = data.get("rankings", [])
        if not rankings:
            return []

        rows = [
            ["rank", "model", "mean", "ci_low", "ci_high", "p_adjusted", "significant_fdr", "count"]
        ]
        for r in rankings:
            rows.append(
                [
                    r.get("rank", ""),
                    r.get("model", ""),
                    r.get("mean", ""),
                    r.get("ci_low", ""),
                    r.get("ci_high", ""),
                    r.get("p_adjusted", ""),
                    r.get("significant_fdr", ""),
                    r.get("count", ""),
                ]
            )
        return rows

    def to_plotly(self) -> Optional["plotly.graph_objects.Figure"]:  # noqa: F821
        """Return Plotly bar chart with error bars for model rankings."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        data = self.get_data()
        if not data["rankings"]:
            return None

        rankings = data["rankings"]
        models = [r["model"] for r in rankings]
        means = [r["mean"] for r in rankings]
        ci_lows = [r["ci_low"] for r in rankings]
        ci_highs = [r["ci_high"] for r in rankings]
        counts = [r["count"] for r in rankings]
        significant = [r["significant_fdr"] for r in rankings]

        error_minus = [m - ci_l for m, ci_l in zip(means, ci_lows)]
        error_plus = [ci_h - m for m, ci_h in zip(means, ci_highs)]

        colors = ["#3b82f6"]  # blue for best
        for sig in significant[1:]:
            colors.append("#ef4444" if sig else "#6b7280")  # red if sig worse, gray otherwise

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=models,
                x=means,
                orientation="h",
                marker_color=colors,
                error_x={
                    "type": "data",
                    "array": error_plus,
                    "arrayminus": error_minus,
                    "visible": True,
                    "color": "#374151",
                    "thickness": 1.5,
                },
                text=[f"n={c}" for c in counts],
                textposition="outside",
                hovertemplate="Model: %{y}<br>z_mean: %{x:.3f}<br>95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>",
                customdata=list(zip(ci_lows, ci_highs)),
            )
        )

        fig.update_layout(
            title="Model Rankings (z_mean with 95% CI, lower is better)",
            xaxis_title="z_mean",
            yaxis_title="Model",
            yaxis={"categoryorder": "total ascending"},
            height=max(400, len(models) * 40 + 100),
            width=800,
            margin={"l": 200, "r": 100, "t": 80, "b": 50},
            showlegend=False,
        )

        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

        return fig
