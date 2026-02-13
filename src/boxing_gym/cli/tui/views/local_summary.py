"""Local Results Summary view for TUI."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from rich.table import Table

from ..components.ascii_charts import colored_z, horizontal_bar, short_model_name
from . import BaseView


class LocalSummaryView(BaseView):
    @property
    def title(self) -> str:
        return "Local Results Summary"

    def _get_final_budget_df(self) -> pd.DataFrame:
        """Get final budget row per run, normalizing n_seeds (1 for legacy)."""
        if "config/budget" not in self.df.columns:
            df = self.df.copy()
        else:
            # group by run_id (filename) to get max budget per result file
            idx = self.df.groupby("run_id")["config/budget"].idxmax().dropna()
            df = self.df.loc[idx].copy()

        # normalize n_seeds: default to 1 for legacy per-seed runs
        # also guard against n_seeds=0 which would cause ZeroDivisionError in np.average
        if "summary/n_seeds" not in df.columns:
            df["summary/n_seeds"] = 1
        bad_seeds = (df["summary/n_seeds"].fillna(0) <= 0).sum()
        if bad_seeds > 0:
            self.console.print(
                f"[yellow]⚠ {bad_seeds} runs with n_seeds≤0 — treating as 1[/yellow]"
            )
        df["summary/n_seeds"] = df["summary/n_seeds"].fillna(1).clip(lower=1).astype(int)

        return df

    def render(self) -> None:
        self.console.print()

        self._render_model_rankings()

        self._render_per_env_top3()

        self._render_coverage()

    def _render_model_rankings(self) -> None:
        # weighted mean: multi-seed runs contribute n_seeds weight
        final_df = self._get_final_budget_df()

        if self.metric not in final_df.columns:
            self.console.print(f"[yellow]Metric '{self.metric}' not found[/yellow]")
            return

        def weighted_agg(group):
            weights = group["summary/n_seeds"]
            z_vals = group[self.metric]
            mask = ~z_vals.isna()
            if mask.sum() == 0:
                return pd.Series(
                    {"mean": np.nan, "std": np.nan, "run_count": 0, "effective_seeds": 0, "envs": 0}
                )
            w_mean = np.average(z_vals[mask], weights=weights[mask])
            w_std = z_vals[mask].std() if mask.sum() > 1 else 0.0
            return pd.Series(
                {
                    "mean": w_mean,
                    "std": w_std,
                    "run_count": len(group),
                    "effective_seeds": weights[mask].sum(),
                    "envs": group["config/envs"].nunique(),
                }
            )

        agg = final_df.groupby("config/llms").apply(weighted_agg, include_groups=False)
        agg["abs_mean"] = agg["mean"].abs()
        agg = agg.dropna(subset=["mean", "abs_mean"])
        agg = agg.sort_values("abs_mean")

        if agg.empty:
            self.console.print("[yellow]No valid metric data[/yellow]")
            return

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
            medal = (
                "🥇"
                if displayed_rank == 1
                else "🥈"
                if displayed_rank == 2
                else "🥉"
                if displayed_rank == 3
                else ""
            )
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
        final_df = self._get_final_budget_df()

        self.console.print()
        self.console.print("[bold]Per-Environment Top 3[/bold]")
        self.console.print()

        envs = sorted(final_df["config/envs"].dropna().unique())

        for env in envs:
            env_df = final_df[final_df["config/envs"] == env]

            def weighted_agg(group):
                weights = group["summary/n_seeds"]
                z_vals = group[self.metric]
                mask = ~z_vals.isna()
                if mask.sum() == 0:
                    return pd.Series({"mean": np.nan, "effective_seeds": 0})
                w_mean = np.average(z_vals[mask], weights=weights[mask])
                return pd.Series(
                    {
                        "mean": w_mean,
                        "effective_seeds": weights[mask].sum(),
                    }
                )

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
                self.console.print(
                    f"    {medal} {z_str}  {model} (n={int(row['effective_seeds'])})"
                )

            self.console.print()

    def _render_coverage(self) -> None:
        final_df = self._get_final_budget_df()

        envs = sorted(final_df["config/envs"].dropna().unique())

        # cap model count to fit terminal: 13 chars per column (10 content + 3 pad/border)
        max_models = max((self.console.width - 19) // 13, 3)

        # rank models by coverage (runs across environments)
        model_coverage = final_df.groupby("config/llms")["config/envs"].nunique()
        top_models = model_coverage.sort_values(ascending=False).head(max_models).index.tolist()

        table = Table(
            title=f"Data Coverage (top {len(top_models)} models by env coverage)",
            border_style="dim",
            header_style="bold",
            show_lines=False,
            expand=False,
        )
        table.add_column("Environment", style="cyan", no_wrap=True)

        for model in top_models:
            header = short_model_name(str(model))
            table.add_column(header, justify="center", min_width=len(header), no_wrap=True)

        for env in envs:
            row_data = [str(env)[:15]]
            for model in top_models:
                has_data = (
                    (final_df["config/envs"] == env) & (final_df["config/llms"] == model)
                ).any()
                row_data.append("[green]✓[/green]" if has_data else "[dim]-[/dim]")
            table.add_row(*row_data)

        self.console.print()
        self.console.print(table)

        # model legend
        self.console.print("[dim]Models:[/dim]", end=" ")
        legends = [f"[dim]{short_model_name(str(m))}[/dim]={str(m)}" for m in top_models]
        self.console.print(" | ".join(legends))

    def get_data(self) -> dict[str, Any]:
        """Return structured summary data for JSON/CSV export.

        Uses weighted mean when mixing multi-seed and per-seed runs.
        """
        final_df = self._get_final_budget_df()

        if self.metric not in final_df.columns:
            return {
                "rankings": [],
                "per_environment": {},
                "total_runs": 0,
                "total_seeds": 0,
                "total_models": 0,
                "total_environments": 0,
            }

        def weighted_agg(group):
            weights = group["summary/n_seeds"]
            z_vals = group[self.metric]
            mask = ~z_vals.isna()
            if mask.sum() == 0:
                return pd.Series(
                    {"mean": np.nan, "std": np.nan, "run_count": 0, "effective_seeds": 0}
                )
            w_mean = np.average(z_vals[mask], weights=weights[mask])
            w_std = z_vals[mask].std() if mask.sum() > 1 else 0.0
            return pd.Series(
                {
                    "mean": w_mean,
                    "std": w_std,
                    "run_count": len(group),
                    "effective_seeds": weights[mask].sum(),
                }
            )

        agg = final_df.groupby("config/llms").apply(weighted_agg, include_groups=False)
        agg["abs_mean"] = agg["mean"].abs()
        # filter same as display: require at least 3 effective seeds
        agg = agg[agg["effective_seeds"] >= 3]
        agg = agg.sort_values("abs_mean").reset_index()

        rankings = []
        for rank, row in enumerate(agg.to_dict("records"), 1):
            rankings.append(
                {
                    "rank": rank,
                    "model": row["config/llms"],
                    "z_mean": row["mean"],
                    "abs_z_mean": row["abs_mean"],
                    "std": row["std"],
                    "run_count": int(row["run_count"]),
                    "effective_seeds": int(row["effective_seeds"]),
                }
            )

        per_env: dict[str, list[dict]] = {}
        for env in final_df["config/envs"].dropna().unique():
            env_df = final_df[final_df["config/envs"] == env]

            def env_weighted_agg(group):
                weights = group["summary/n_seeds"]
                z_vals = group[self.metric]
                mask = ~z_vals.isna()
                if mask.sum() == 0:
                    return pd.Series({"mean": np.nan, "effective_seeds": 0})
                w_mean = np.average(z_vals[mask], weights=weights[mask])
                return pd.Series(
                    {
                        "mean": w_mean,
                        "effective_seeds": weights[mask].sum(),
                    }
                )

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

    def get_csv_rows(self) -> list:
        data = self.get_data()
        rows = []

        rankings = data.get("rankings", [])
        if rankings:
            rows.append(
                ["rank", "model", "z_mean", "abs_z_mean", "std", "run_count", "effective_seeds"]
            )
            for r in rankings:
                rows.append(
                    [
                        r.get("rank", ""),
                        r.get("model", ""),
                        r.get("z_mean", ""),
                        r.get("abs_z_mean", ""),
                        r.get("std", ""),
                        r.get("run_count", ""),
                        r.get("effective_seeds", ""),
                    ]
                )

        per_env = data.get("per_environment", {})
        if per_env:
            rows.append([])
            rows.append(["# Per-Environment Top 3"])
            rows.append(["environment", "model", "z_mean", "effective_seeds"])
            for env, models in sorted(per_env.items()):
                for m in models:
                    rows.append(
                        [
                            env,
                            m.get("model", ""),
                            m.get("z_mean", ""),
                            m.get("effective_seeds", ""),
                        ]
                    )

        rows.append([])
        rows.append(["# Summary"])
        rows.append(["total_runs", data.get("total_runs", "")])
        rows.append(["total_seeds", data.get("total_seeds", "")])
        rows.append(["total_models", data.get("total_models", "")])
        rows.append(["total_environments", data.get("total_environments", "")])

        return rows

    def to_plotly(self) -> Optional["plotly.graph_objects.Figure"]:  # noqa: F821
        """Return Plotly heatmap of data coverage (envs x models)."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        final_df = self._get_final_budget_df()

        if "config/envs" not in final_df.columns or "config/llms" not in final_df.columns:
            return None

        envs = sorted(final_df["config/envs"].dropna().unique())
        models = sorted(final_df["config/llms"].dropna().unique())

        if not envs or not models:
            return None

        matrix = []
        hover_text = []
        for env in envs:
            row = []
            hover_row = []
            for model in models:
                count = (
                    (final_df["config/envs"] == env) & (final_df["config/llms"] == model)
                ).sum()
                row.append(count)
                hover_row.append(f"{env}<br>{model}<br>Runs: {count}")
            matrix.append(row)
            hover_text.append(hover_row)

        model_labels = [short_model_name(m) for m in models]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=models,
                y=envs,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                colorscale=[[0, "#1f2937"], [0.01, "#3b82f6"], [1, "#22c55e"]],
                showscale=True,
                colorbar=dict(title="Runs"),
            )
        )

        fig.update_layout(
            title="Data Coverage (Environments × Models)",
            xaxis_title="Model",
            yaxis_title="Environment",
            height=max(400, len(envs) * 25),
            width=max(600, len(models) * 60),
            xaxis_tickangle=-45,
        )
        fig.update_xaxes(tickmode="array", tickvals=models, ticktext=model_labels)

        return fig
