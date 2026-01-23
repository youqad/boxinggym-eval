"""Model Rankings view for TUI."""

import pandas as pd
from rich.table import Table

from . import BaseView
from ..components.ascii_charts import colored_z
from boxing_gym.analysis.stats import bootstrap_ci, compare_models


class ModelRankingsView(BaseView):
    """Display model rankings by z_mean with bootstrap CIs and significance tests."""

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

        best_model = agg.iloc[0]["config/llms"]

        comparisons = compare_models(
            self.df, model_col="config/llms",
            score_col=self.metric, reference_model=best_model,
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
        table.add_column("vs #1", justify="right", width=10)
        table.add_column("n", justify="right", width=6)

        for rank, (_, row) in enumerate(agg.iterrows(), 1):
            model = str(row["config/llms"])
            mean = row["mean"]
            count = int(row["count"])
            ci = ci_data.get(model)
            if ci is None:
                continue

            z_str = colored_z(mean)
            ci_str = f"[{ci.ci_low:+.3f}, {ci.ci_high:+.3f}]"

            if model == best_model:
                p_str = "-"
            else:
                comp = p_lookup.get(model, {})
                p_val = comp.get("p_adjusted") or comp.get("p_value") or 1.0
                sig = comp.get("significant_fdr", False)
                p_str = f"[red]p={p_val:.2f}*[/red]" if sig else f"p={p_val:.2f}"

            table.add_row(str(rank), model[:30], z_str, ci_str, p_str, str(count))

        self.console.print(table)

        best_z = agg.iloc[0]["mean"]
        best_ci = ci_data[best_model]
        worst_model = agg.iloc[-1]["config/llms"]
        worst_z = agg.iloc[-1]["mean"]
        sig_worse = [c["model"] for c in comparisons["comparisons"] if c.get("significant_fdr")]

        self.console.print(
            f"\n[green]Best:[/green]  {best_model} (z={best_z:+.3f}, "
            f"95% CI [{best_ci.ci_low:+.3f}, {best_ci.ci_high:+.3f}])"
        )
        self.console.print(f"[red]Worst:[/red] {worst_model} (z={worst_z:+.3f})")
        self.console.print(f"[cyan]Spread:[/cyan] {worst_z - best_z:.3f}")
        if sig_worse:
            self.console.print(
                f"[red]Significantly worse than #1 (FDR<0.05):[/red] {', '.join(sig_worse)}"
            )

    def get_data(self) -> dict:
        """Return model rankings data as dict."""
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

        best_model = agg.iloc[0]["config/llms"]
        comparisons = compare_models(
            self.df, model_col="config/llms",
            score_col=self.metric, reference_model=best_model,
        )
        p_lookup = {c["model"]: c for c in comparisons["comparisons"]}

        rankings = []
        for rank, (_, row) in enumerate(agg.iterrows(), 1):
            model = str(row["config/llms"])
            ci = ci_data.get(model)
            if ci is None:
                continue
            comp = p_lookup.get(model, {})
            rankings.append({
                "rank": rank,
                "model": model,
                "mean": float(row["mean"]) if not pd.isna(row["mean"]) else None,
                "ci_low": float(ci.ci_low),
                "ci_high": float(ci.ci_high),
                "p_adjusted": comp.get("p_adjusted"),
                "significant_fdr": comp.get("significant_fdr", False),
                "count": int(row["count"]),
            })

        return {"rankings": rankings}
