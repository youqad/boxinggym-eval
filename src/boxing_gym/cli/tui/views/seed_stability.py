"""Seed Stability view for TUI."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from rich.table import Table

from ..components.ascii_charts import colored_z, horizontal_bar
from . import BaseView


class SeedStabilityView(BaseView):
    @property
    def title(self) -> str:
        return "Seed Stability Diagnostics"

    @staticmethod
    def _hashable_config_cols(df: pd.DataFrame) -> list[str]:
        """Return config columns that are safe to groupby (hashable values)."""
        noise_patterns = [
            "results",
            "ppl/",
            "ppl.",
            "hydra",
            "filename",
            "system_prompt",
            "_wandb",
            "_runtime",
            "seed",
        ]
        config_cols = [
            c
            for c in df.columns
            if c.startswith("config/") and not any(noise in c.lower() for noise in noise_patterns)
        ]

        def _series_is_hashable(s: pd.Series) -> bool:
            sample = s.dropna().head(50)
            for v in sample:
                try:
                    hash(v)
                except Exception:
                    return False
            return True

        return [c for c in config_cols if _series_is_hashable(df[c])]

    def render(self) -> None:
        if self.metric not in self.df.columns:
            self.console.print(f"[yellow]Metric '{self.metric}' not found[/yellow]")
            return

        has_multi_seed_format = (
            "summary/z_stderr" in self.df.columns or "z_stderr" in self.df.columns
        )

        if has_multi_seed_format:
            self._render_multi_seed_format()
        elif "config/seed" in self.df.columns:
            # old per-seed format
            self._render_per_seed_stats()
            self.console.print()
            self._render_per_config_variance()
        else:
            self.console.print("[yellow]No seed data available (config/seed or z_stderr)[/yellow]")

    def _render_multi_seed_format(self) -> None:
        stderr_col = "summary/z_stderr" if "summary/z_stderr" in self.df.columns else "z_stderr"

        self.console.print("[bold cyan]Multi-Seed Format Detected[/bold cyan]")
        self.console.print(
            "Each run aggregates 5 seeds internally. Showing z_stderr as seed variance.\n"
        )

        table = Table(
            title="Runs by Seed Variance (z_stderr)",
            border_style="cyan",
            header_style="bold magenta",
        )
        table.add_column("Environment", width=25)
        table.add_column("Model", width=20)
        table.add_column("z_mean", justify="right", width=10)
        table.add_column("z_stderr", justify="right", width=10)
        table.add_column("Variance Bar", width=15)

        df_sorted = (
            self.df.dropna(subset=[stderr_col]).sort_values(stderr_col, ascending=False).head(10)
        )
        max_stderr = df_sorted[stderr_col].max() if len(df_sorted) > 0 else 1

        for _, row in df_sorted.iterrows():
            env = row.get("config/envs", row.get("env", "—"))
            model = row.get("config/llms", row.get("model", "—"))
            z_mean = row.get(self.metric, row.get("z_mean", np.nan))
            z_stderr = row[stderr_col]

            z_str = colored_z(z_mean) if not pd.isna(z_mean) else "—"
            stderr_str = f"{z_stderr:.4f}" if not pd.isna(z_stderr) else "—"
            bar = horizontal_bar(z_stderr, max_stderr, width=12) if not pd.isna(z_stderr) else ""

            table.add_row(str(env), str(model), z_str, stderr_str, bar)

        self.console.print(table)

    def _render_per_seed_stats(self) -> None:
        seed_stats = self.df.groupby("config/seed")[self.metric].agg(["mean", "std", "count"])
        seed_stats = seed_stats.sort_values("mean")
        seed_stats = seed_stats.reset_index()

        if seed_stats.empty:
            self.console.print("[yellow]No seed data available[/yellow]")
            return

        max_mean = seed_stats["mean"].max()
        min_mean = seed_stats["mean"].min()
        range_mean = max_mean - min_mean

        table = Table(
            title="Per-Seed Performance (Dataset Difficulty)",
            border_style="cyan",
            header_style="bold magenta",
        )
        table.add_column("Seed", justify="right", width=10)
        table.add_column("z_mean", justify="right", width=15)
        table.add_column("Std Dev", justify="right", width=10)
        table.add_column("Count", justify="right", width=8)
        table.add_column("Difficulty", width=12)

        for _, row in seed_stats.iterrows():
            seed = int(row["config/seed"])
            mean = row["mean"]
            std = row["std"]
            count = int(row["count"])

            z_str = colored_z(mean)

            if range_mean > 0:
                norm_difficulty = (mean - min_mean) / range_mean
                bar = horizontal_bar(norm_difficulty, 1.0, width=10)
            else:
                bar = "─" * 10

            table.add_row(
                str(seed),
                z_str,
                f"{std:.3f}" if not np.isnan(std) else "N/A",
                str(count),
                bar,
            )

        self.console.print(table)

        global_std = self.df[self.metric].std()
        within_seed_std = seed_stats["std"].mean()
        between_seed_std = seed_stats["mean"].std()

        self.console.print(f"\n[cyan]Global Std Dev:[/cyan] {global_std:.3f}")
        self.console.print(
            f"[cyan]Between-Seed Std:[/cyan] {between_seed_std:.3f} (dataset difficulty variance)"
        )
        self.console.print(
            f"[cyan]Within-Seed Avg Std:[/cyan] {within_seed_std:.3f} (hyperparameter effects)"
        )

    def _render_per_config_variance(self) -> None:
        config_cols = self._hashable_config_cols(self.df)

        if not config_cols:
            self.console.print("[yellow]No config columns found for variance analysis[/yellow]")
            return

        variance_data = []
        for config_combo, group in self.df.groupby(config_cols):
            if len(group) < 2:  # need at least 2 seeds
                continue

            z_values = group[self.metric].dropna()
            if len(z_values) < 2:
                continue

            variance = z_values.var()
            mean = z_values.mean()
            std = z_values.std()
            count = len(z_values)

            config_dict = dict(
                zip(
                    config_cols, config_combo if isinstance(config_combo, tuple) else [config_combo]
                )
            )

            variance_data.append(
                {
                    "config": config_dict,
                    "variance": variance,
                    "std": std,
                    "mean": mean,
                    "count": count,
                }
            )

        if not variance_data:
            self.console.print(
                "[yellow]Not enough multi-seed configs for variance analysis[/yellow]"
            )
            return

        variance_df = pd.DataFrame(variance_data)
        variance_df = variance_df.sort_values("variance", ascending=False)

        max_var = variance_df["variance"].max()

        table = Table(
            title="Most Unstable Configs Across Seeds (Top 10)",
            border_style="cyan",
            header_style="bold magenta",
        )
        table.add_column("Config", style="white", width=50)
        table.add_column("Variance", justify="right", width=12)
        table.add_column("Std Dev", justify="right", width=10)
        table.add_column("Bar", width=12)
        table.add_column("Seeds", justify="right", width=8)

        for _, row in variance_df.head(10).iterrows():
            config = row["config"]
            variance = row["variance"]
            std = row["std"]
            count = int(row["count"])

            config_str = ", ".join(
                [
                    f"{k.replace('config/', '')[:15]}={str(v)[:10]}"
                    for k, v in config.items()
                    if k in ["config/llms", "config/envs", "config/exp"]
                ]
            )

            bar = horizontal_bar(variance, max_var, width=10)

            table.add_row(
                config_str[:50],
                f"{variance:.4f}",
                f"{std:.3f}",
                bar,
                str(count),
            )

        self.console.print(table)

        self.console.print("\n[dim]Interpretation:[/dim]")
        self.console.print(
            "[dim]  High variance = config performance heavily depends on seed/dataset[/dim]"
        )
        self.console.print("[dim]  Low variance = config performs consistently across seeds[/dim]")

    def get_data(self) -> dict[str, Any]:
        if self.metric not in self.df.columns:
            return {"error": "Missing metric column"}

        stderr_col = (
            "summary/z_stderr"
            if "summary/z_stderr" in self.df.columns
            else "z_stderr"
            if "z_stderr" in self.df.columns
            else None
        )
        if "config/seed" not in self.df.columns:
            if not stderr_col:
                return {"error": "Missing required columns"}
            df_sorted = (
                self.df.dropna(subset=[stderr_col])
                .sort_values(stderr_col, ascending=False)
                .head(10)
            )
            runs = []
            for _, row in df_sorted.iterrows():
                runs.append(
                    {
                        "environment": row.get("config/envs", row.get("env")),
                        "model": row.get("config/llms", row.get("model")),
                        "z_mean": row.get(self.metric),
                        "z_stderr": row.get(stderr_col),
                    }
                )
            stderr_mean = (
                float(self.df[stderr_col].dropna().mean())
                if self.df[stderr_col].notna().any()
                else 0.0
            )
            return {
                "format": "multi_seed",
                "stderr_column": stderr_col,
                "mean_z_stderr": stderr_mean,
                "top_runs_by_stderr": runs,
            }

        seed_stats = self.df.groupby("config/seed")[self.metric].agg(["mean", "std", "count"])
        seed_stats = seed_stats.sort_values("mean").reset_index()

        global_std = float(self.df[self.metric].std())
        between_seed_std = float(seed_stats["mean"].std())
        within_seed_std = float(seed_stats["std"].mean())

        config_cols = self._hashable_config_cols(self.df)

        variance_data = []
        if config_cols:
            for config_combo, group in self.df.groupby(config_cols):
                if len(group) >= 2:
                    z_values = group[self.metric].dropna()
                    if len(z_values) >= 2:
                        config_dict = dict(
                            zip(
                                config_cols,
                                config_combo if isinstance(config_combo, tuple) else [config_combo],
                            )
                        )
                        variance_data.append(
                            {
                                "config": config_dict,
                                "variance": float(z_values.var()),
                                "std": float(z_values.std()),
                                "mean": float(z_values.mean()),
                                "count": int(len(z_values)),
                            }
                        )

        variance_data.sort(key=lambda x: x["variance"], reverse=True)

        return {
            "per_seed_stats": seed_stats.to_dict(orient="records"),
            "variance_decomposition": {
                "global_std": global_std,
                "between_seed_std": between_seed_std,
                "within_seed_std": within_seed_std,
            },
            "unstable_configs": variance_data[:10],  # top 10 most unstable
        }

    def get_csv_rows(self) -> list:
        data = self.get_data()
        rows = []

        if data.get("format") == "multi_seed":
            rows.append(["# Multi-Seed Summary"])
            rows.append(["stderr_column", data.get("stderr_column", "")])
            rows.append(["mean_z_stderr", data.get("mean_z_stderr", "")])

            top_runs = data.get("top_runs_by_stderr", [])
            if top_runs:
                rows.append([])
                rows.append(["# Top Runs by z_stderr"])
                rows.append(["environment", "model", "z_mean", "z_stderr"])
                for r in top_runs:
                    rows.append(
                        [
                            r.get("environment", ""),
                            r.get("model", ""),
                            r.get("z_mean", ""),
                            r.get("z_stderr", ""),
                        ]
                    )
            return rows

        per_seed = data.get("per_seed_stats", [])
        if per_seed:
            rows.append(["# Per-Seed Performance"])
            rows.append(["seed", "mean", "std", "count"])
            for s in per_seed:
                rows.append(
                    [
                        s.get("config/seed", ""),
                        s.get("mean", ""),
                        s.get("std", ""),
                        s.get("count", ""),
                    ]
                )

        variance_decomp = data.get("variance_decomposition", {})
        if variance_decomp:
            rows.append([])
            rows.append(["# Variance Decomposition"])
            rows.append(["metric", "value"])
            rows.append(["global_std", variance_decomp.get("global_std", "")])
            rows.append(["between_seed_std", variance_decomp.get("between_seed_std", "")])
            rows.append(["within_seed_std", variance_decomp.get("within_seed_std", "")])

        unstable = data.get("unstable_configs", [])
        if unstable:
            rows.append([])
            rows.append(["# Most Unstable Configs"])
            rows.append(["config", "variance", "std", "mean", "count"])
            for u in unstable:
                rows.append(
                    [
                        str(u.get("config", {})),
                        u.get("variance", ""),
                        u.get("std", ""),
                        u.get("mean", ""),
                        u.get("count", ""),
                    ]
                )

        return rows

    def to_plotly(self) -> Optional["plotly.graph_objects.Figure"]:  # noqa: F821
        """Return Plotly box plot showing z_mean distribution per seed."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if self.metric not in self.df.columns:
            return None

        if "config/seed" not in self.df.columns:
            return None

        fig = go.Figure()

        seeds = sorted(self.df["config/seed"].dropna().unique())
        for seed in seeds:
            seed_data = self.df[self.df["config/seed"] == seed][self.metric].dropna()
            fig.add_trace(
                go.Box(
                    y=seed_data,
                    name=f"Seed {int(seed)}",
                    boxmean=True,
                    hovertemplate=f"Seed {int(seed)}<br>z_mean: %{{y:.3f}}<extra></extra>",
                )
            )

        fig.update_layout(
            title="z_mean Distribution by Seed (lower is better)",
            yaxis_title="z_mean",
            xaxis_title="Seed",
            height=500,
            width=800,
            showlegend=False,
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        return fig
