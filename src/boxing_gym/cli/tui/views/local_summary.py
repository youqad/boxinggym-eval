"""Local Results Summary view for TUI."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from . import BaseView
from ..components.ascii_charts import colored_z, horizontal_bar


class LocalSummaryView(BaseView):
    """
    Summary view for local results analysis.

    Shows:
    1. Overall model rankings by mean |z_mean|
    2. Per-environment top 3 models
    3. Data coverage (envs x models matrix)
    """

    @property
    def title(self) -> str:
        return "Local Results Summary"

    def _get_final_budget_df(self) -> pd.DataFrame:
        """Get DataFrame with only final budget rows per run.

        Uses run_id (filename) as the grouping key, which works correctly for both:
        - Multi-seed (new): one file = one aggregated run
        - Per-seed (legacy): one file = one seed's run

        Note: Avoids groupby on config/seed which causes pandas to silently drop
        rows with NaN seeds (multi-seed format) due to dropna=True default.

        Also normalizes n_seeds column (1 for legacy, n for multi-seed).
        """
        if "config/budget" not in self.df.columns:
            df = self.df.copy()
        else:
            # group by run_id (filename) to get max budget per result file
            idx = self.df.groupby("run_id")["config/budget"].idxmax()
            df = self.df.loc[idx].copy()

        # normalize n_seeds: default to 1 for legacy per-seed runs
        # also guard against n_seeds=0 which would cause ZeroDivisionError in np.average
        if "summary/n_seeds" not in df.columns:
            df["summary/n_seeds"] = 1
        bad_seeds = (df["summary/n_seeds"].fillna(0) <= 0).sum()
        if bad_seeds > 0:
            self.console.print(f"[yellow]⚠ {bad_seeds} runs with n_seeds≤0 — treating as 1[/yellow]")
        df["summary/n_seeds"] = df["summary/n_seeds"].fillna(1).clip(lower=1).astype(int)

        return df

    def render(self) -> None:
        """Render local results summary to console."""
        self.console.print()

        # 1. overall model rankings
        self._render_model_rankings()

        # 2. per-environment top models
        self._render_per_env_top3()

        # 3. data coverage matrix
        self._render_coverage()

    def _render_model_rankings(self) -> None:
        """Overall rankings by mean |z_mean| across all envs.

        Uses weighted mean when mixing multi-seed and per-seed runs:
        a multi-seed run with n_seeds=5 contributes 5x the weight.
        """
        final_df = self._get_final_budget_df()

        if self.metric not in final_df.columns:
            self.console.print(f"[yellow]Metric '{self.metric}' not found[/yellow]")
            return

        # weighted aggregation by model - multi-seed runs contribute more weight
        def weighted_agg(group):
            weights = group["summary/n_seeds"]
            z_vals = group[self.metric]
            w_mean = np.average(z_vals, weights=weights)
            # weighted std approximation (unweighted std of the observations)
            w_std = z_vals.std() if len(z_vals) > 1 else 0.0
            return pd.Series({
                "mean": w_mean,
                "std": w_std,
                "run_count": len(group),
                "effective_seeds": weights.sum(),
                "envs": group["config/envs"].nunique(),
            })

        agg = final_df.groupby("config/llms").apply(weighted_agg, include_groups=False)
        agg["abs_mean"] = agg["mean"].abs()
        agg = agg.sort_values("abs_mean")

        table = Table(
            title="Model Rankings (|z-score| - lower is better)",
            border_style="cyan",
            header_style="bold magenta",
            show_lines=False,
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Model", style="green", min_width=30)
        table.add_column("|z|", justify="right", width=10)
        table.add_column("Bar", width=15)
        table.add_column("Seeds", justify="right", width=6)
        table.add_column("Envs", justify="right", width=5)

        max_z = min(agg["abs_mean"].max(), 10.0)  # cap for bar display

        displayed_rank = 0
        for model, row in agg.iterrows():
            # use effective_seeds (sum of n_seeds) instead of run_count
            if row["effective_seeds"] < 3:
                continue

            displayed_rank += 1
            medal = "🥇" if displayed_rank == 1 else "🥈" if displayed_rank == 2 else "🥉" if displayed_rank == 3 else ""
            bar = horizontal_bar(min(row["abs_mean"], max_z), max_z, width=12)

            table.add_row(
                f"{medal}{displayed_rank}" if medal else str(displayed_rank),
                str(model),
                colored_z(row["mean"]),  # use signed z for correct coloring
                bar,
                str(int(row["effective_seeds"])),
                str(int(row["envs"])),
            )

        self.console.print(table)

    def _render_per_env_top3(self) -> None:
        """Show top 3 models per environment.

        Uses weighted mean when mixing multi-seed and per-seed runs.
        """
        final_df = self._get_final_budget_df()

        self.console.print()
        self.console.print("[bold]Per-Environment Top 3[/bold]")
        self.console.print()

        envs = sorted(final_df["config/envs"].dropna().unique())

        for env in envs:
            env_df = final_df[final_df["config/envs"] == env]

            # weighted aggregation by model
            def weighted_agg(group):
                weights = group["summary/n_seeds"]
                z_vals = group[self.metric]
                w_mean = np.average(z_vals, weights=weights)
                return pd.Series({
                    "mean": w_mean,
                    "effective_seeds": weights.sum(),
                })

            agg = env_df.groupby("config/llms").apply(weighted_agg, include_groups=False)
            agg["abs_mean"] = agg["mean"].abs()
            agg = agg[agg["effective_seeds"] >= 2]  # at least 2 effective seeds
            agg = agg.sort_values("abs_mean")

            if agg.empty:
                continue

            self.console.print(f"  [yellow]{env}[/yellow]")

            for i, (model, row) in enumerate(agg.head(3).iterrows(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                z_str = colored_z(row["mean"])  # use signed z for correct coloring
                self.console.print(f"    {medal} {z_str}  {model} (n={int(row['effective_seeds'])})")

            self.console.print()

    def _render_coverage(self) -> None:
        """Show envs x models coverage matrix."""
        final_df = self._get_final_budget_df()

        envs = sorted(final_df["config/envs"].dropna().unique())
        models = sorted(final_df["config/llms"].dropna().unique())

        # truncate model names for display
        model_short = {m: m[:12] + ".." if len(m) > 14 else m for m in models}

        table = Table(
            title="Data Coverage",
            border_style="dim",
            header_style="bold",
            show_lines=False,
        )
        table.add_column("Environment", style="cyan", min_width=15)

        for model in models:
            table.add_column(model_short[model], justify="center", width=8)

        for env in envs:
            row_data = [env]
            for model in models:
                has_data = (
                    (final_df["config/envs"] == env) &
                    (final_df["config/llms"] == model)
                ).any()
                row_data.append("[green]✓[/green]" if has_data else "[dim]-[/dim]")
            table.add_row(*row_data)

        self.console.print()
        self.console.print(table)

    def get_data(self) -> Dict[str, Any]:
        """Return structured summary data for JSON/CSV export.

        Uses weighted mean when mixing multi-seed and per-seed runs.
        """
        final_df = self._get_final_budget_df()

        # weighted aggregation by model
        def weighted_agg(group):
            weights = group["summary/n_seeds"]
            z_vals = group[self.metric]
            w_mean = np.average(z_vals, weights=weights)
            w_std = z_vals.std() if len(z_vals) > 1 else 0.0
            return pd.Series({
                "mean": w_mean,
                "std": w_std,
                "run_count": len(group),
                "effective_seeds": weights.sum(),
            })

        agg = final_df.groupby("config/llms").apply(weighted_agg, include_groups=False)
        agg["abs_mean"] = agg["mean"].abs()
        # filter same as display: require at least 3 effective seeds
        agg = agg[agg["effective_seeds"] >= 3]
        agg = agg.sort_values("abs_mean").reset_index()

        rankings = []
        for rank, row in enumerate(agg.to_dict("records"), 1):
            rankings.append({
                "rank": rank,
                "model": row["config/llms"],
                "z_mean": row["mean"],
                "abs_z_mean": row["abs_mean"],
                "std": row["std"],
                "run_count": int(row["run_count"]),
                "effective_seeds": int(row["effective_seeds"]),
            })

        # per-environment top 3 with weighted mean
        per_env: Dict[str, List[Dict]] = {}
        for env in final_df["config/envs"].dropna().unique():
            env_df = final_df[final_df["config/envs"] == env]

            def env_weighted_agg(group):
                weights = group["summary/n_seeds"]
                z_vals = group[self.metric]
                w_mean = np.average(z_vals, weights=weights)
                return pd.Series({
                    "mean": w_mean,
                    "effective_seeds": weights.sum(),
                })

            env_agg = env_df.groupby("config/llms").apply(env_weighted_agg, include_groups=False)
            env_agg["abs_mean"] = env_agg["mean"].abs()
            env_agg = env_agg.sort_values("abs_mean").head(3).reset_index()

            per_env[env] = [
                {
                    "model": row["config/llms"],
                    "z_mean": row["mean"],
                    "effective_seeds": int(row["effective_seeds"]),
                }
                for row in env_agg.to_dict("records")
            ]

        return {
            "rankings": rankings,
            "per_environment": per_env,
            "total_runs": len(final_df),
            "total_seeds": int(final_df["summary/n_seeds"].sum()),
            "total_models": final_df["config/llms"].nunique(),
            "total_environments": final_df["config/envs"].nunique(),
        }
