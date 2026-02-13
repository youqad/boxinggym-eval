"""Budget Progression view for TUI."""

from typing import Optional

import pandas as pd
from rich.table import Table

from ..components.ascii_charts import colored_z, trend_indicator
from . import BaseView


class BudgetProgressionView(BaseView):
    @property
    def title(self) -> str:
        return "Budget Progression"

    @staticmethod
    def _format_budget(budget: float) -> str:
        try:
            b = float(budget)
        except (TypeError, ValueError):
            return str(budget)[:6]
        if b.is_integer():
            return str(int(b))
        return f"{b:.1f}".rstrip("0").rstrip(".")

    def render(self) -> None:
        if self.metric not in self.df.columns:
            self.console.print(f"[yellow]Metric '{self.metric}' not found[/yellow]")
            return

        budget_col = None
        for col in ["config/budget", "config/exp.budget", "metric/budget"]:
            if col in self.df.columns:
                budget_col = col
                break

        if budget_col is None:
            self.console.print("[yellow]No budget column found[/yellow]")
            self.console.print(
                "[dim]Looked for: config/budget, config/exp.budget, metric/budget[/dim]"
            )
            return

        if "config/llms" not in self.df.columns or "config/envs" not in self.df.columns:
            self.console.print("[yellow]Missing config/llms or config/envs columns[/yellow]")
            return

        df_local = self.df.copy()
        df_local["_budget_num"] = pd.to_numeric(df_local[budget_col], errors="coerce")
        df_local = df_local.dropna(subset=["_budget_num"])

        budgets = sorted(df_local["_budget_num"].unique())
        if len(budgets) < 2:
            self.console.print("[yellow]Need at least 2 budget values for progression[/yellow]")
            return

        df_local["env_model"] = (
            df_local["config/envs"].astype(str) + " | " + df_local["config/llms"].astype(str)
        )
        pivot = df_local.pivot_table(
            values=self.metric,
            index="env_model",
            columns="_budget_num",
            aggfunc="mean",
        )

        if pivot.empty:
            self.console.print("[yellow]No data for budget progression[/yellow]")
            return

        final_budget = budgets[-1]
        if final_budget in pivot.columns:
            pivot = pivot.sort_values(final_budget)

        table = Table(
            title=f"z_mean by Budget ({budget_col.replace('config/', '')})",
            border_style="cyan",
            header_style="bold magenta",
            show_lines=True,
        )

        table.add_column("Env | Model", style="yellow", width=35)

        display_budgets = budgets[:5] if len(budgets) > 5 else budgets
        for b in display_budgets:
            b_str = self._format_budget(b)
            table.add_column(f"B={b_str}", justify="center", width=8)

        table.add_column("Trend", justify="center", width=6)

        for env_model, row in pivot.head(20).iterrows():
            cells = [str(env_model)[:35]]

            first_val = None
            last_val = None

            for b in display_budgets:
                val = row.get(b, float("nan"))
                if pd.isna(val):
                    cells.append("[dim]—[/dim]")
                else:
                    cells.append(colored_z(val))
                    if first_val is None:
                        first_val = val
                    last_val = val

            if first_val is not None and last_val is not None:
                cells.append(trend_indicator(first_val, last_val))
            else:
                cells.append("[dim]—[/dim]")

            table.add_row(*cells)

        self.console.print(table)

        self.console.print(
            "\n[dim]Trend: [green]▼[/green]=improving, [red]▲[/red]=worsening, [dim]─[/dim]=stable[/dim]"
        )

        if len(budgets) >= 2:
            first_b, last_b = budgets[0], budgets[-1]
            if first_b in pivot.columns and last_b in pivot.columns:
                pivot["improvement"] = pivot[first_b] - pivot[last_b]
                pivot_sorted = pivot.dropna(subset=["improvement"]).sort_values(
                    "improvement", ascending=False
                )

                if len(pivot_sorted) > 0:
                    first_str = self._format_budget(first_b)
                    last_str = self._format_budget(last_b)
                    self.console.print(
                        f"\n[cyan]Top 5 most improved (B={first_str} → B={last_str}):[/cyan]"
                    )
                    for i, (env_model, row) in enumerate(pivot_sorted.head(5).iterrows()):
                        imp = row["improvement"]
                        color = "green" if imp > 0 else "red"
                        self.console.print(f"  {i + 1}. [{color}]{imp:+.3f}[/{color}] {env_model}")

    def get_data(self) -> dict:
        if self.metric not in self.df.columns:
            return {"progression": []}

        budget_col = None
        for col in ["config/budget", "config/exp.budget", "metric/budget"]:
            if col in self.df.columns:
                budget_col = col
                break

        if (
            budget_col is None
            or "config/llms" not in self.df.columns
            or "config/envs" not in self.df.columns
        ):
            return {"progression": []}

        df_local = self.df.copy()
        df_local["_budget_num"] = pd.to_numeric(df_local[budget_col], errors="coerce")
        df_local = df_local.dropna(subset=["_budget_num"])

        budgets = sorted(df_local["_budget_num"].unique())
        if len(budgets) < 2:
            return {"progression": []}

        df_local["env_model"] = (
            df_local["config/envs"].astype(str) + " | " + df_local["config/llms"].astype(str)
        )
        pivot = df_local.pivot_table(
            values=self.metric,
            index="env_model",
            columns="_budget_num",
            aggfunc="mean",
        )

        if pivot.empty:
            return {"progression": []}

        progression_data = []
        for env_model, row in pivot.iterrows():
            entry = {"env_model": str(env_model), "budgets": {}}
            for budget in budgets:
                val = row.get(budget, float("nan"))
                budget_key = int(budget) if float(budget).is_integer() else float(budget)
                entry["budgets"][budget_key] = None if pd.isna(val) else float(val)

            progression_data.append(entry)

        return {"progression": progression_data}

    def get_csv_rows(self) -> list:
        data = self.get_data()
        progression = data.get("progression", [])
        if not progression:
            return []

        budgets = set()
        for p in progression:
            budgets.update(p.get("budgets", {}).keys())
        budgets = sorted(budgets)

        rows = [["env_model"] + [f"budget_{b}" for b in budgets]]
        for p in progression:
            row = [p.get("env_model", "")]
            for b in budgets:
                val = p.get("budgets", {}).get(b)
                row.append("" if val is None else val)
            rows.append(row)
        return rows

    def to_plotly(self) -> Optional["plotly.graph_objects.Figure"]:  # noqa: F821
        """Return Plotly line chart showing z_mean progression across budgets."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if self.metric not in self.df.columns:
            return None

        budget_col = None
        for col in ["config/budget", "config/exp.budget", "metric/budget"]:
            if col in self.df.columns:
                budget_col = col
                break

        if budget_col is None or "config/llms" not in self.df.columns:
            return None

        df_local = self.df.copy()
        df_local["_budget_num"] = pd.to_numeric(df_local[budget_col], errors="coerce")
        df_local = df_local.dropna(subset=["_budget_num"])

        agg = df_local.groupby(["config/llms", "_budget_num"])[self.metric].mean().reset_index()

        if agg.empty:
            return None

        fig = go.Figure()

        models = agg["config/llms"].unique()
        for model in models:
            model_data = agg[agg["config/llms"] == model].sort_values("_budget_num")
            fig.add_trace(
                go.Scatter(
                    x=model_data["_budget_num"],
                    y=model_data[self.metric],
                    mode="lines+markers",
                    name=str(model),
                    hovertemplate=f"{model}<br>Budget: %{{x}}<br>z_mean: %{{y:.3f}}<extra></extra>",
                )
            )

        fig.update_layout(
            title="z_mean Progression by Budget (lower is better)",
            xaxis_title="Budget",
            yaxis_title="z_mean",
            height=500,
            width=900,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.3,
                "xanchor": "center",
                "x": 0.5,
            },
            hovermode="x unified",
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        return fig
