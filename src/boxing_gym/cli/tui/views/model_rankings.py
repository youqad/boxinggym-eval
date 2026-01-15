"""Model Rankings view for TUI."""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from . import BaseView
from ..components.ascii_charts import colored_z


class ModelRankingsView(BaseView):
    """Display model rankings by z_mean."""

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

        # aggregate by model
        agg = self.df.groupby("config/llms")[self.metric].agg(["mean", "std", "count"])
        agg = agg.sort_values("mean")
        agg["sem"] = agg["std"] / np.sqrt(agg["count"])
        agg = agg.reset_index()

        if agg.empty:
            self.console.print("[yellow]No data to rank[/yellow]")
            return

        table = Table(
            title="Model Rankings (lower z_mean = better)",
            border_style="cyan",
            header_style="bold magenta",
        )
        table.add_column("#", justify="right", width=4)
        table.add_column("Model", style="white", width=30)
        table.add_column("z_mean", justify="right", width=15)
        table.add_column("±SEM", justify="right", width=10)
        table.add_column("n", justify="right", width=6)

        for rank, (_, row) in enumerate(agg.iterrows(), 1):
            model = str(row["config/llms"])
            mean = row["mean"]
            sem = row["sem"]
            count = int(row["count"])

            z_str = colored_z(mean)

            table.add_row(
                str(rank),
                model[:30],
                z_str,
                f"±{sem:.3f}",
                str(count),
            )

        self.console.print(table)

        # summary stats
        best_model = agg.iloc[0]["config/llms"]
        best_z = agg.iloc[0]["mean"]
        worst_model = agg.iloc[-1]["config/llms"]
        worst_z = agg.iloc[-1]["mean"]

        self.console.print(f"\n[green]Best:[/green]  {best_model} ({best_z:+.3f})")
        self.console.print(f"[red]Worst:[/red] {worst_model} ({worst_z:+.3f})")
        self.console.print(f"[cyan]Spread:[/cyan] {worst_z - best_z:.3f}")

    def get_data(self) -> dict:
        """Return model rankings data as dict."""
        if self.metric not in self.df.columns or "config/llms" not in self.df.columns:
            return {"rankings": []}

        # aggregate by model
        agg = self.df.groupby("config/llms")[self.metric].agg(["mean", "std", "count"])
        agg = agg.sort_values("mean")
        agg["sem"] = agg["std"] / np.sqrt(agg["count"])
        agg = agg.reset_index()

        if agg.empty:
            return {"rankings": []}

        # convert to list of dicts with rank
        rankings = []
        for rank, (_, row) in enumerate(agg.iterrows(), 1):
            rankings.append({
                "rank": rank,
                "model": str(row["config/llms"]),
                "mean": float(row["mean"]) if not pd.isna(row["mean"]) else None,
                "std": float(row["std"]) if not pd.isna(row["std"]) else None,
                "sem": float(row["sem"]) if not pd.isna(row["sem"]) else None,
                "count": int(row["count"])
            })

        return {"rankings": rankings}
