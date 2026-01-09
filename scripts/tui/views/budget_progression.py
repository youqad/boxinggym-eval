"""Budget Progression view for TUI."""

import pandas as pd
from rich.console import Console
from rich.table import Table

from scripts.tui.views import BaseView
from scripts.tui.components.ascii_charts import colored_z, trend_indicator


class BudgetProgressionView(BaseView):
    """Display z_mean progression across budget values."""

    @property
    def title(self) -> str:
        return "Budget Progression"

    def render(self) -> None:
        if self.metric not in self.df.columns:
            self.console.print(f"[yellow]Metric '{self.metric}' not found[/yellow]")
            return

        # check for budget column
        budget_col = None
        for col in ["config/budget", "config/exp.budget", "metric/budget"]:
            if col in self.df.columns:
                budget_col = col
                break

        if budget_col is None:
            self.console.print("[yellow]No budget column found[/yellow]")
            self.console.print("[dim]Looked for: config/budget, config/exp.budget, metric/budget[/dim]")
            return

        if "config/llms" not in self.df.columns or "config/envs" not in self.df.columns:
            self.console.print("[yellow]Missing config/llms or config/envs columns[/yellow]")
            return

        # get unique budgets
        budgets = sorted(self.df[budget_col].dropna().unique())
        if len(budgets) < 2:
            self.console.print("[yellow]Need at least 2 budget values for progression[/yellow]")
            return

        # create pivot: (env, model) × budget - use copy to avoid mutating shared df
        df_local = self.df.copy()
        df_local["env_model"] = df_local["config/envs"].astype(str) + " | " + df_local["config/llms"].astype(str)
        pivot = df_local.pivot_table(
            values=self.metric,
            index="env_model",
            columns=budget_col,
            aggfunc="mean",
        )

        if pivot.empty:
            self.console.print("[yellow]No data for budget progression[/yellow]")
            return

        # sort by final budget performance
        final_budget = budgets[-1]
        if final_budget in pivot.columns:
            pivot = pivot.sort_values(final_budget)

        # build table
        table = Table(
            title=f"z_mean by Budget ({budget_col.replace('config/', '')})",
            border_style="cyan",
            header_style="bold magenta",
            show_lines=True,
        )

        table.add_column("Env | Model", style="yellow", width=35)

        # add budget columns (show up to 5 budgets)
        display_budgets = budgets[:5] if len(budgets) > 5 else budgets
        for b in display_budgets:
            # format budget: use int if whole number, else keep decimal
            b_str = str(int(b)) if b == int(b) else f"{b:.1f}"
            table.add_column(f"B={b_str}", justify="center", width=8)

        table.add_column("Trend", justify="center", width=6)

        # add rows (limit to 20)
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

            # add trend
            if first_val is not None and last_val is not None:
                cells.append(trend_indicator(first_val, last_val))
            else:
                cells.append("[dim]—[/dim]")

            table.add_row(*cells)

        self.console.print(table)

        # legend
        self.console.print("\n[dim]Trend: [green]▼[/green]=improving, [red]▲[/red]=worsening, [dim]─[/dim]=stable[/dim]")

        # summary: biggest improvements
        if len(budgets) >= 2:
            first_b, last_b = budgets[0], budgets[-1]
            if first_b in pivot.columns and last_b in pivot.columns:
                pivot["improvement"] = pivot[first_b] - pivot[last_b]
                pivot_sorted = pivot.dropna(subset=["improvement"]).sort_values("improvement", ascending=False)

                if len(pivot_sorted) > 0:
                    first_str = str(int(first_b)) if first_b == int(first_b) else f"{first_b:.1f}"
                    last_str = str(int(last_b)) if last_b == int(last_b) else f"{last_b:.1f}"
                    self.console.print(f"\n[cyan]Top 5 most improved (B={first_str} → B={last_str}):[/cyan]")
                    for i, (env_model, row) in enumerate(pivot_sorted.head(5).iterrows()):
                        imp = row["improvement"]
                        color = "green" if imp > 0 else "red"
                        self.console.print(f"  {i+1}. [{color}]{imp:+.3f}[/{color}] {env_model}")

    def get_data(self) -> dict:
        """Return budget progression data as dict."""
        if self.metric not in self.df.columns:
            return {"progression": []}

        # check for budget column
        budget_col = None
        for col in ["config/budget", "config/exp.budget", "metric/budget"]:
            if col in self.df.columns:
                budget_col = col
                break

        if budget_col is None or "config/llms" not in self.df.columns or "config/envs" not in self.df.columns:
            return {"progression": []}

        # get unique budgets
        budgets = sorted(self.df[budget_col].dropna().unique())
        if len(budgets) < 2:
            return {"progression": []}

        # create pivot: (env, model) × budget
        df_local = self.df.copy()
        df_local["env_model"] = df_local["config/envs"].astype(str) + " | " + df_local["config/llms"].astype(str)
        pivot = df_local.pivot_table(
            values=self.metric,
            index="env_model",
            columns=budget_col,
            aggfunc="mean",
        )

        if pivot.empty:
            return {"progression": []}

        # convert to list of dicts
        progression_data = []
        for env_model, row in pivot.iterrows():
            entry = {
                "env_model": str(env_model),
                "budgets": {}
            }
            for budget in budgets:
                val = row.get(budget, float("nan"))
                # use int key if whole number, else float
                budget_key = int(budget) if budget == int(budget) else float(budget)
                entry["budgets"][budget_key] = None if pd.isna(val) else float(val)

            progression_data.append(entry)

        return {"progression": progression_data}
