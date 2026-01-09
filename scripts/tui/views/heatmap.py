"""Environment × Model Heatmap view for TUI."""

import pandas as pd
from rich.console import Console
from rich.table import Table

from scripts.tui.views import BaseView
from scripts.tui.components.ascii_charts import z_color


class HeatmapView(BaseView):
    """Display environment × model heatmap."""

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

        # create pivot table
        pivot = self.df.pivot_table(
            values=self.metric,
            index="config/envs",
            columns="config/llms",
            aggfunc="mean",
        )

        if pivot.empty:
            self.console.print("[yellow]No data for heatmap[/yellow]")
            return

        # sort by average z_mean
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

        # create table
        table = Table(
            title="z_mean by Environment × Model",
            border_style="cyan",
            header_style="bold magenta",
            show_lines=True,
        )

        # add environment column
        table.add_column("Environment", style="yellow", width=25)

        # add model columns (truncate names)
        models = list(pivot.columns)
        for model in models:
            short_name = str(model)[:12]
            table.add_column(short_name, justify="center", width=10)

        # add rows
        for env, row in pivot.iterrows():
            cells = [str(env)[:25]]
            for model in models:
                val = row[model]
                if pd.isna(val):
                    cells.append("[dim]—[/dim]")
                else:
                    color = z_color(val)
                    cells.append(f"[{color}]{val:+.2f}[/{color}]")

            table.add_row(*cells)

        self.console.print(table)

        # color legend
        self.console.print("\n[dim]Legend: [/dim][green]< -0.3 (good)[/green] | [yellow]-0.3 to 0.3[/yellow] | [red]> 0.3 (poor)[/red]")

        # summary: best and worst combinations
        self.console.print("\n[cyan]Top 5 combinations:[/cyan]")
        stacked = pivot.stack().sort_values()
        for i, ((env, model), val) in enumerate(stacked.head(5).items()):
            color = z_color(val)
            self.console.print(f"  {i+1}. [{color}]{val:+.3f}[/{color}] {model} × {env}")

    def get_data(self) -> dict:
        """Return heatmap data as dict."""
        if self.metric not in self.df.columns or \
           "config/llms" not in self.df.columns or \
           "config/envs" not in self.df.columns:
            return {"heatmap": {}}

        # create pivot table
        pivot = self.df.pivot_table(
            values=self.metric,
            index="config/envs",
            columns="config/llms",
            aggfunc="mean",
        )

        if pivot.empty:
            return {"heatmap": {}}

        # sort by average z_mean
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

        # convert to nested dict: {env: {model: z_mean}}
        heatmap_data = {}
        for env, row in pivot.iterrows():
            heatmap_data[str(env)] = {}
            for model in pivot.columns:
                val = row[model]
                heatmap_data[str(env)][str(model)] = None if pd.isna(val) else float(val)

        return {"heatmap": heatmap_data}
