"""Seed Stability view for TUI."""

from typing import Any, Dict

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from . import BaseView
from ..components.ascii_charts import colored_z, horizontal_bar


class SeedStabilityView(BaseView):
    """Diagnostic view showing seed-driven variance in z_mean.

    Supports two formats:
    1. New multi-seed format: Each run contains aggregated results with z_stderr
    2. Old per-seed format: Each seed is a separate run

    This view shows seed variance for diagnostics without leaking test-set info.
    """

    @property
    def title(self) -> str:
        return "Seed Stability Diagnostics"

    def render(self) -> None:
        if self.metric not in self.df.columns:
            self.console.print(f"[yellow]Metric '{self.metric}' not found[/yellow]")
            return

        # detect format: new multi-seed runs have summary/z_stderr
        has_multi_seed_format = "summary/z_stderr" in self.df.columns or "z_stderr" in self.df.columns

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
        """Render for new multi-seed format where z_stderr is already computed."""
        stderr_col = "summary/z_stderr" if "summary/z_stderr" in self.df.columns else "z_stderr"

        self.console.print("[bold cyan]Multi-Seed Format Detected[/bold cyan]")
        self.console.print("Each run aggregates 5 seeds internally. Showing z_stderr as seed variance.\n")

        # create summary table of runs with highest seed variance
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

        # get top 10 by z_stderr
        df_sorted = self.df.dropna(subset=[stderr_col]).sort_values(stderr_col, ascending=False).head(10)
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
        """Show mean z_mean, std, and count for each seed."""
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

            # difficulty bar (higher mean = harder dataset instance)
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

        # summary
        global_std = self.df[self.metric].std()
        within_seed_std = seed_stats["std"].mean()
        between_seed_std = seed_stats["mean"].std()

        self.console.print(f"\n[cyan]Global Std Dev:[/cyan] {global_std:.3f}")
        self.console.print(f"[cyan]Between-Seed Std:[/cyan] {between_seed_std:.3f} (dataset difficulty variance)")
        self.console.print(f"[cyan]Within-Seed Avg Std:[/cyan] {within_seed_std:.3f} (hyperparameter effects)")

    def _render_per_config_variance(self) -> None:
        """Show configs with highest variance across seeds (unstable combinations)."""
        # identify config columns (exclude seed and noise)
        noise_patterns = ["results", "ppl/", "ppl.", "hydra", "filename", "system_prompt", "wandb", "_", "seed"]
        config_cols = [
            c for c in self.df.columns
            if c.startswith("config/")
            and not any(noise in c.lower() for noise in noise_patterns)
        ]

        if not config_cols:
            self.console.print("[yellow]No config columns found for variance analysis[/yellow]")
            return

        # group by config combination, calculate variance across seeds
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

            # create config dict
            config_dict = dict(zip(config_cols, config_combo if isinstance(config_combo, tuple) else [config_combo]))

            variance_data.append({
                "config": config_dict,
                "variance": variance,
                "std": std,
                "mean": mean,
                "count": count,
            })

        if not variance_data:
            self.console.print("[yellow]Not enough multi-seed configs for variance analysis[/yellow]")
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

            # compact config representation (key params only)
            config_str = ", ".join([
                f"{k.replace('config/', '')[:15]}={str(v)[:10]}"
                for k, v in config.items()
                if k in ["config/llms", "config/envs", "config/exp"]
            ])

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
        self.console.print("[dim]  High variance = config performance heavily depends on seed/dataset[/dim]")
        self.console.print("[dim]  Low variance = config performs consistently across seeds[/dim]")

    def get_data(self) -> Dict[str, Any]:
        """Return structured data for machine-readable output."""
        if self.metric not in self.df.columns or "config/seed" not in self.df.columns:
            return {"error": "Missing required columns"}

        # per-seed statistics
        seed_stats = self.df.groupby("config/seed")[self.metric].agg(["mean", "std", "count"])
        seed_stats = seed_stats.sort_values("mean").reset_index()

        # global variance breakdown
        global_std = float(self.df[self.metric].std())
        between_seed_std = float(seed_stats["mean"].std())
        within_seed_std = float(seed_stats["std"].mean())

        # per-config variance data
        noise_patterns = ["results", "ppl/", "ppl.", "hydra", "filename", "system_prompt", "wandb", "_", "seed"]
        config_cols = [
            c for c in self.df.columns
            if c.startswith("config/")
            and not any(noise in c.lower() for noise in noise_patterns)
        ]

        variance_data = []
        if config_cols:
            for config_combo, group in self.df.groupby(config_cols):
                if len(group) >= 2:
                    z_values = group[self.metric].dropna()
                    if len(z_values) >= 2:
                        config_dict = dict(zip(config_cols,
                                             config_combo if isinstance(config_combo, tuple) else [config_combo]))
                        variance_data.append({
                            "config": config_dict,
                            "variance": float(z_values.var()),
                            "std": float(z_values.std()),
                            "mean": float(z_values.mean()),
                            "count": int(len(z_values)),
                        })

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
