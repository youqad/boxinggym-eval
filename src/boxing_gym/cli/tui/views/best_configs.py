"""Best Configurations view for TUI."""

from typing import Optional

import numpy as np
import pandas as pd
from rich.table import Table

from ..components.ascii_charts import colored_z
from . import BaseView


class BestConfigsView(BaseView):
    @property
    def title(self) -> str:
        return "Best Configurations"

    def render(self) -> None:
        if self.metric not in self.df.columns:
            self.console.print(f"[yellow]Metric '{self.metric}' not found[/yellow]")
            return

        best = self.df.nsmallest(20, self.metric)

        key_cols = [
            "config/llms",
            "config/envs",
            "config/include_prior",
            "config/use_ppl",
            "config/seed",
            "config/budget",
        ]
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
        table.add_column("Budget", justify="right", width=6)
        table.add_column("z_mean", justify="right", width=10)

        for rank, (_, row) in enumerate(best.iterrows(), 1):
            z_val = row.get(self.metric, float("nan"))
            z_str = colored_z(z_val)

            prior_val = row.get("config/include_prior")
            prior = "yes" if (pd.notna(prior_val) and prior_val) else "no"
            ppl_val = row.get("config/use_ppl")
            ppl = "yes" if (pd.notna(ppl_val) and ppl_val) else "no"
            seed = str(int(row.get("config/seed", 0))) if pd.notna(row.get("config/seed")) else "?"
            budget = (
                str(int(row.get("config/budget", 0))) if pd.notna(row.get("config/budget")) else "?"
            )

            table.add_row(
                str(rank),
                str(row.get("config/llms", "?"))[:20],
                str(row.get("config/envs", "?"))[:25],
                prior,
                ppl,
                seed,
                budget,
                z_str,
            )

        self.console.print(table)

        if "config/envs" in self.df.columns and "config/llms" in self.df.columns:
            self.console.print("\n[bold cyan]Best Model per Environment[/bold cyan]\n")

            idx = self.df.groupby("config/envs")[self.metric].idxmin().dropna()
            best_per_env = self.df.loc[idx].sort_values(self.metric)

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
        if self.metric not in self.df.columns:
            return {"best_configs": [], "best_per_env": []}

        best = self.df.nsmallest(20, self.metric)

        key_cols = [
            "config/llms",
            "config/envs",
            "config/include_prior",
            "config/use_ppl",
            "config/seed",
            "config/budget",
        ]
        key_cols = [c for c in key_cols if c in best.columns]

        configs = []
        for rank, (_, row) in enumerate(best.iterrows(), 1):
            config = {
                "rank": rank,
                "z_mean": float(row.get(self.metric, float("nan"))),
            }
            for col in key_cols:
                key = col.replace("config/", "")
                val = row.get(col)
                if pd.isna(val):
                    config[key] = None
                elif isinstance(val, (bool, np.bool_)):
                    config[key] = bool(val)
                elif key in ("seed", "budget"):
                    config[key] = int(val)
                else:
                    config[key] = str(val)
            configs.append(config)

        best_per_env = []
        if "config/envs" in self.df.columns and "config/llms" in self.df.columns:
            idx = self.df.groupby("config/envs")[self.metric].idxmin().dropna()
            best_env_df = self.df.loc[idx].sort_values(self.metric)

            for _, row in best_env_df.iterrows():
                best_per_env.append(
                    {
                        "environment": str(row.get("config/envs", "?")),
                        "best_model": str(row.get("config/llms", "?")),
                        "z_mean": float(row.get(self.metric, float("nan"))),
                    }
                )

        return {"best_configs": configs, "best_per_env": best_per_env}

    def get_csv_rows(self) -> list:
        data = self.get_data()
        configs = data.get("best_configs", [])
        best_per_env = data.get("best_per_env", [])

        rows = []

        if configs:
            all_keys = set()
            for c in configs:
                all_keys.update(c.keys())
            all_keys = sorted(all_keys)

            rows.append(all_keys)
            for c in configs:
                rows.append([c.get(k, "") for k in all_keys])

        if best_per_env:
            rows.append([])
            rows.append(["# Best per Environment"])
            rows.append(["environment", "best_model", "z_mean"])
            for b in best_per_env:
                rows.append(
                    [
                        b.get("environment", ""),
                        b.get("best_model", ""),
                        b.get("z_mean", ""),
                    ]
                )

        return rows

    def to_plotly(self) -> Optional["plotly.graph_objects.Figure"]:  # noqa: F821
        """Return Plotly bar chart of best z_mean per environment."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if self.metric not in self.df.columns:
            return None
        if "config/envs" not in self.df.columns or "config/llms" not in self.df.columns:
            return None

        idx = self.df.groupby("config/envs")[self.metric].idxmin().dropna()
        if idx.empty:
            return None
        best_env_df = self.df.loc[idx].sort_values(self.metric)

        if best_env_df.empty:
            return None

        envs = best_env_df["config/envs"].tolist()
        z_means = best_env_df[self.metric].tolist()
        models = best_env_df["config/llms"].tolist()

        colors = ["#22c55e" if z < 0 else "#ef4444" for z in z_means]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=envs,
                y=z_means,
                marker_color=colors,
                text=models,
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>Best Model: %{text}<br>z_mean: %{y:+.3f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="Best z_mean per Environment",
            xaxis_title="Environment",
            yaxis_title="z_mean (lower is better)",
            height=500,
            width=900,
            xaxis_tickangle=-45,
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        return fig
