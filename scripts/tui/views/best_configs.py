"""Best Configurations view for TUI."""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from scripts.tui.views import BaseView
from scripts.tui.components.ascii_charts import colored_z


class BestConfigsView(BaseView):
    """Display best configurations from the sweep."""

    @property
    def title(self) -> str:
        return "Best Configurations"

    def render(self) -> None:
        if self.metric not in self.df.columns:
            self.console.print(f"[yellow]Metric '{self.metric}' not found[/yellow]")
            return

        # get top 20 best runs
        best = self.df.nsmallest(20, self.metric)

        # key columns to display
        key_cols = ["config/llms", "config/envs", "config/include_prior", "config/use_ppl", "config/seed"]
        key_cols = [c for c in key_cols if c in best.columns]

        table = Table(
            title="Top 20 Configurations (lowest z_mean)",
            border_style="cyan",
            header_style="bold magenta",
            row_styles=["", "dim"],
        )

        table.add_column("#", justify="right", width=3)
        table.add_column("Model", width=20)
        table.add_column("Environment", width=25)
        table.add_column("Prior", justify="center", width=6)
        table.add_column("PPL", justify="center", width=5)
        table.add_column("Seed", justify="right", width=5)
        table.add_column("z_mean", justify="right", width=10)

        for rank, (_, row) in enumerate(best.iterrows(), 1):
            z_val = row.get(self.metric, float("nan"))
            z_str = colored_z(z_val)

            prior = "yes" if row.get("config/include_prior") else "no"
            ppl = "yes" if row.get("config/use_ppl") else "no"
            seed = str(int(row.get("config/seed", 0))) if pd.notna(row.get("config/seed")) else "?"

            table.add_row(
                str(rank),
                str(row.get("config/llms", "?"))[:20],
                str(row.get("config/envs", "?"))[:25],
                prior,
                ppl,
                seed,
                z_str,
            )

        self.console.print(table)

        # best per environment
        if "config/envs" in self.df.columns and "config/llms" in self.df.columns:
            self.console.print("\n[bold cyan]Best Model per Environment[/bold cyan]\n")

            best_per_env = self.df.loc[
                self.df.groupby("config/envs")[self.metric].idxmin()
            ].sort_values(self.metric)

            env_table = Table(border_style="cyan", header_style="magenta")
            env_table.add_column("Environment", style="yellow", width=30)
            env_table.add_column("Best Model", style="green", width=25)
            env_table.add_column("z_mean", justify="right", width=10)

            for _, row in best_per_env.iterrows():
                env = str(row.get("config/envs", "?"))
                model = str(row.get("config/llms", "?"))
                z = row.get(self.metric, float("nan"))

                env_table.add_row(env, model, colored_z(z))

            self.console.print(env_table)

    def get_data(self) -> dict:
        """Return best configurations data as dict."""
        if self.metric not in self.df.columns:
            return {"best_configs": [], "best_per_env": []}

        # get top 20 best runs
        best = self.df.nsmallest(20, self.metric)

        # key columns to extract
        key_cols = ["config/llms", "config/envs", "config/include_prior", "config/use_ppl", "config/seed"]
        key_cols = [c for c in key_cols if c in best.columns]

        # convert to list of dicts
        configs = []
        for rank, (_, row) in enumerate(best.iterrows(), 1):
            config = {
                "rank": rank,
                "z_mean": float(row.get(self.metric, float("nan"))),
            }
            for col in key_cols:
                key = col.replace("config/", "")
                val = row.get(col)
                # convert booleans and seeds to appropriate types
                if pd.isna(val):
                    config[key] = None
                elif isinstance(val, (bool, np.bool_)):
                    config[key] = bool(val)
                elif key == "seed":
                    config[key] = int(val)
                else:
                    config[key] = str(val)
            configs.append(config)

        # best per environment
        best_per_env = []
        if "config/envs" in self.df.columns and "config/llms" in self.df.columns:
            best_env_df = self.df.loc[
                self.df.groupby("config/envs")[self.metric].idxmin()
            ].sort_values(self.metric)

            for _, row in best_env_df.iterrows():
                best_per_env.append({
                    "environment": str(row.get("config/envs", "?")),
                    "best_model": str(row.get("config/llms", "?")),
                    "z_mean": float(row.get(self.metric, float("nan")))
                })

        return {
            "best_configs": configs,
            "best_per_env": best_per_env
        }
