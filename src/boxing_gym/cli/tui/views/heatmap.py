"""Environment × Model Heatmap view for TUI."""

from typing import Optional

import pandas as pd
from rich.table import Table

from ..components.ascii_charts import short_model_name, z_color
from . import BaseView


def _heat_color(z: float) -> str:
    if z <= -0.3:
        return "bold green"
    if z <= 0.0:
        return "green"
    if z <= 0.3:
        return "yellow"
    if z <= 1.0:
        return "red"
    return "bold red"


class HeatmapView(BaseView):
    @property
    def title(self) -> str:
        return "Environment × Model Heatmap"

    def render(self) -> None:
        if self.metric not in self.df.columns:
            self.console.print(f"[yellow]Metric '{self.metric}' not found[/yellow]")
            return

        if "config/llms" not in self.df.columns or "config/envs" not in self.df.columns:
            self.console.print("[yellow]Missing config/llms or config/envs columns[/yellow]")
            return

        pivot = self.df.pivot_table(
            values=self.metric,
            index="config/envs",
            columns="config/llms",
            aggfunc="mean",
        )

        if pivot.empty:
            self.console.print("[yellow]No data for heatmap[/yellow]")
            return

        # cap model count to fit terminal: 13 chars per column (10 content + 3 pad/border)
        max_models = max((self.console.width - 19) // 13, 3)
        coverage = pivot.notna().sum().sort_values(ascending=False)
        top_models = coverage.head(max_models).index
        pivot = pivot[top_models]

        pivot = pivot.dropna(how="all")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(na_position="last").index]

        table = Table(
            title=f"z_mean Heatmap (top {len(pivot.columns)} by coverage)",
            border_style="cyan",
            header_style="bold magenta",
            show_lines=True,
            expand=False,
        )

        table.add_column("Env", style="yellow", no_wrap=True)

        models = list(pivot.columns)
        for model in models:
            header = short_model_name(str(model))
            table.add_column(header, justify="center", min_width=len(header), no_wrap=True)

        for env, row in pivot.iterrows():
            cells = [str(env)[:15]]
            for model in models:
                val = row[model]
                if pd.isna(val):
                    cells.append("[dim]—[/dim]")
                else:
                    clamped = max(-99.0, min(99.0, val))
                    color = _heat_color(val)
                    cells.append(f"[{color}]{clamped:+.1f}[/{color}]")

            table.add_row(*cells)

        self.console.print(table)

        self.console.print(
            "[dim]Scale:[/dim] [bold green]≤-0.3[/bold green] "
            "[green]→0[/green] [yellow]→0.3[/yellow] "
            "[red]→1.0[/red] [bold red]>1.0[/bold red]"
        )

        self.console.print("[dim]Key:[/dim]")
        for m in models:
            short = short_model_name(str(m))
            if short != str(m):
                self.console.print(f"  [dim]{short}[/dim] = {m}")

        self.console.print("\n[cyan]Top 5 (lowest z):[/cyan]")
        stacked = pivot.stack().sort_values()
        for i, ((env, model), val) in enumerate(stacked.head(5).items()):
            clamped = max(-99.0, min(99.0, val))
            color = z_color(val)
            self.console.print(f"  {i + 1}. [{color}]{clamped:+.2f}[/{color}] {model} × {env}")

    def get_data(self) -> dict:
        if (
            self.metric not in self.df.columns
            or "config/llms" not in self.df.columns
            or "config/envs" not in self.df.columns
        ):
            return {"heatmap": {}}

        pivot = self.df.pivot_table(
            values=self.metric,
            index="config/envs",
            columns="config/llms",
            aggfunc="mean",
        )

        if pivot.empty:
            return {"heatmap": {}}

        pivot = pivot.dropna(how="all")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(na_position="last").index]

        heatmap_data = {}
        for env, row in pivot.iterrows():
            heatmap_data[str(env)] = {}
            for model in pivot.columns:
                val = row[model]
                heatmap_data[str(env)][str(model)] = None if pd.isna(val) else float(val)

        return {"heatmap": heatmap_data}

    def get_csv_rows(self) -> list:
        data = self.get_data()
        heatmap = data.get("heatmap", {})
        if not heatmap:
            return []

        models = set()
        for env_data in heatmap.values():
            models.update(env_data.keys())
        models = sorted(models)

        rows = [["environment"] + models]
        for env, env_data in sorted(heatmap.items()):
            row = [env]
            for model in models:
                val = env_data.get(model)
                row.append("" if val is None else val)
            rows.append(row)
        return rows

    def to_plotly(self) -> Optional["plotly.graph_objects.Figure"]:  # noqa: F821
        """Return Plotly heatmap figure."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if (
            self.metric not in self.df.columns
            or "config/llms" not in self.df.columns
            or "config/envs" not in self.df.columns
        ):
            return None

        pivot = self.df.pivot_table(
            values=self.metric,
            index="config/envs",
            columns="config/llms",
            aggfunc="mean",
        )

        if pivot.empty:
            return None

        pivot = pivot.dropna(how="all")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(na_position="last").index]
        coverage = pivot.notna().sum().sort_values(ascending=False)
        pivot = pivot[coverage.index]

        if pivot.empty:
            return None

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=[str(i) for i in pivot.index],
                colorscale=[
                    [0.0, "#22c55e"],  # green (z <= -0.3)
                    [0.08, "#22c55e"],  # green ends at z=-0.3: (-0.3 - (-0.5)) / 2.5 = 0.08
                    [0.08, "#84cc16"],  # lime starts at z=-0.3
                    [0.2, "#84cc16"],  # lime ends at z=0.0: (0.0 - (-0.5)) / 2.5 = 0.2
                    [0.2, "#eab308"],  # yellow starts at z=0.0
                    [0.32, "#eab308"],  # yellow ends at z=0.3: (0.3 - (-0.5)) / 2.5 = 0.32
                    [0.32, "#f97316"],  # orange starts at z=0.3
                    [0.6, "#ef4444"],  # red at z=1.0: (1.0 - (-0.5)) / 2.5 = 0.6
                    [1.0, "#dc2626"],  # dark red (z > 1.0)
                ],
                zmin=-0.5,
                zmax=2.0,
                text=pivot.values.round(2),
                texttemplate="%{text:+.2f}",
                textfont={"size": 10},
                hovertemplate="Model: %{x}<br>Env: %{y}<br>z_mean: %{z:.3f}<extra></extra>",
                colorbar={"title": "z_mean", "tickformat": "+.1f"},
            )
        )

        fig.update_layout(
            title="Environment × Model Heatmap (z_mean, lower is better)",
            xaxis_title="Model",
            yaxis_title="Environment",
            xaxis={"tickangle": 45},
            height=max(400, len(pivot.index) * 35 + 150),
            width=max(600, len(pivot.columns) * 80 + 200),
            margin={"l": 250, "r": 50, "t": 80, "b": 150},
            yaxis={"ticklabelposition": "outside", "ticksuffix": "  "},
        )

        return fig
