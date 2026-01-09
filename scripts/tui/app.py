"""Main TUI application for sweep analysis."""

import csv
import json
import math
import sys
from typing import List, Optional, Dict, Type, Literal, Any

import numpy as np
import pandas as pd


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types and NaN values."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


from rich.console import Console
from rich.prompt import IntPrompt
from rich.rule import Rule

from scripts.tui.components.menu import render_menu, clear_screen, press_enter_to_continue
from scripts.tui.views import BaseView
from scripts.tui.views.parameter_importance import ParameterImportanceView
from scripts.tui.views.model_rankings import ModelRankingsView
from scripts.tui.views.heatmap import HeatmapView
from scripts.tui.views.best_configs import BestConfigsView
from scripts.tui.views.budget_progression import BudgetProgressionView
from scripts.tui.views.seed_stability import SeedStabilityView
from scripts.tui.views.local_summary import LocalSummaryView
from scripts.tui.views.call_logs import CallLogsView
from scripts.tui.views.ppl_examples import PPLExamplesView


# view name → view class mapping for non-interactive mode
VIEW_REGISTRY: Dict[str, Type[BaseView]] = {
    "parameter-importance": ParameterImportanceView,
    "model-rankings": ModelRankingsView,
    "heatmap": HeatmapView,
    "best-configs": BestConfigsView,
    "budget-progression": BudgetProgressionView,
    "seed-stability": SeedStabilityView,
    "local-summary": LocalSummaryView,
    "call-logs": CallLogsView,
    "ppl-examples": PPLExamplesView,
}

# all available view names (for CLI help text)
AVAILABLE_VIEWS = list(VIEW_REGISTRY.keys()) + ["all"]


class SweepTUI:
    """Interactive TUI for sweep analysis."""

    def __init__(
        self,
        df: pd.DataFrame,
        sweep_ids: List[str],
        metric: str = "metric/eval/z_mean",
        include_seed: bool = False,
        is_local: bool = False,
    ):
        self.df = df
        self.sweep_ids = sweep_ids
        self.metric = metric
        self.include_seed = include_seed
        self.is_local = is_local
        self.console = Console(width=120)

        # initialize views
        self.views = []

        # add local summary view first when using local data
        if is_local:
            self.views.append(LocalSummaryView(df, self.console, metric))

        self.views.extend([
            ParameterImportanceView(df, self.console, metric, include_seed=include_seed),
            ModelRankingsView(df, self.console, metric),
            HeatmapView(df, self.console, metric),
            BestConfigsView(df, self.console, metric),
            BudgetProgressionView(df, self.console, metric),
            SeedStabilityView(df, self.console, metric),
            CallLogsView(self.console, metric),
            PPLExamplesView(self.console, metric),
        ])

        # name → view instance mapping
        self._view_map = {v.title.lower().replace(" ", "-"): v for v in self.views}
        # also add canonical names for backwards compatibility
        self._view_map.update({
            "parameter-importance": next((v for v in self.views if isinstance(v, ParameterImportanceView)), None),
            "model-rankings": next((v for v in self.views if isinstance(v, ModelRankingsView)), None),
            "heatmap": next((v for v in self.views if isinstance(v, HeatmapView)), None),
            "best-configs": next((v for v in self.views if isinstance(v, BestConfigsView)), None),
            "budget-progression": next((v for v in self.views if isinstance(v, BudgetProgressionView)), None),
            "seed-stability": next((v for v in self.views if isinstance(v, SeedStabilityView)), None),
            "local-summary": next((v for v in self.views if isinstance(v, LocalSummaryView)), None),
            "call-logs": next((v for v in self.views if isinstance(v, CallLogsView)), None),
            "ppl-examples": next((v for v in self.views if isinstance(v, PPLExamplesView)), None),
        })
        # remove None entries
        self._view_map = {k: v for k, v in self._view_map.items() if v is not None}

    def _get_subtitle(self) -> str:
        """Build subtitle with sweep info."""
        sweep_str = ", ".join(self.sweep_ids[:3])
        if len(self.sweep_ids) > 3:
            sweep_str += f" +{len(self.sweep_ids) - 3} more"
        return f"Sweep: {sweep_str} | Runs: {len(self.df)}"

    def run(self) -> None:
        """Main TUI loop."""
        while True:
            clear_screen(self.console)

            # build menu options
            options = [
                (str(i + 1), view.title)
                for i, view in enumerate(self.views)
            ]
            options.append(("0", "Quit"))

            render_menu(
                self.console,
                title="BoxingGym Sweep Analysis - TUI",
                subtitle=self._get_subtitle(),
                options=options,
            )

            try:
                choice = IntPrompt.ask(
                    "[bold yellow]Select option[/bold yellow]",
                    console=self.console,
                    default=0,
                )
            except (KeyboardInterrupt, EOFError):
                break

            if choice == 0:
                self.console.print("\n[cyan]Goodbye![/cyan]")
                break

            if 1 <= choice <= len(self.views):
                clear_screen(self.console)
                view = self.views[choice - 1]
                self.console.print(f"\n[bold cyan]{view.title}[/bold cyan]\n")
                view.render()
                press_enter_to_continue(self.console)
            else:
                self.console.print("[red]Invalid option. Please try again.[/red]")
                press_enter_to_continue(self.console)

    def run_non_interactive(
        self,
        view_names: List[str],
        output_format: Literal["rich", "json", "csv"] = "rich"
    ) -> None:
        """Run specified views without user interaction.

        Renders views directly to stdout without menus or prompts.
        Suitable for piping output to other tools or AI agents.

        Args:
            view_names: List of view names to render. Use ["all"] for all views.
            output_format: Output format - "rich" (default), "json", or "csv"
        """
        # resolve "all" to all view names
        if "all" in view_names:
            view_names = list(self._view_map.keys())

        # handle machine-readable output formats
        if output_format == "json":
            self._output_json(view_names)
            return
        elif output_format == "csv":
            self._output_csv(view_names)
            return

        # rich format (default)
        # print header
        self.console.print(Rule("[bold cyan]BoxingGym Sweep Analysis[/bold cyan]"))
        self.console.print(f"[dim]{self._get_subtitle()}[/dim]\n")

        # render each requested view
        for i, name in enumerate(view_names):
            if name not in self._view_map:
                self.console.print(f"[red]Unknown view: {name}[/red]")
                self.console.print(f"[dim]Available: {', '.join(self._view_map.keys())}[/dim]\n")
                continue

            view = self._view_map[name]

            if i > 0:
                self.console.print()  # blank line between views

            self.console.print(Rule(f"[bold cyan]{view.title}[/bold cyan]"))
            view.render()
            self.console.print()  # spacing after view

    def _output_json(self, view_names: List[str]) -> None:
        """Output view data as JSON."""
        result = {}
        for name in view_names:
            if name not in self._view_map:
                continue
            view = self._view_map[name]
            result[name] = view.get_data()

        # output to stdout with custom encoder for NumPy types
        json.dump(result, sys.stdout, indent=2, cls=NumpyJSONEncoder)
        sys.stdout.write('\n')

    def _output_csv(self, view_names: List[str]) -> None:
        """Output view data as CSV.

        For views with nested data, creates one CSV per view.
        Flattens complex structures where possible.
        """
        writer = csv.writer(sys.stdout)

        for name in view_names:
            if name not in self._view_map:
                continue

            view = self._view_map[name]
            data = view.get_data()

            # write view header
            writer.writerow([f"# {view.title}"])

            # handle different view types
            if name == "parameter-importance":
                # list of parameters with importance/correlation
                params = data.get("parameters", [])
                if params:
                    writer.writerow(["parameter", "importance", "std", "correlation"])
                    for p in params:
                        writer.writerow([
                            p.get("parameter", ""),
                            p.get("importance", ""),
                            p.get("std", ""),
                            p.get("correlation", "")
                        ])

            elif name == "model-rankings":
                # list of models with rankings
                rankings = data.get("rankings", [])
                if rankings:
                    writer.writerow(["rank", "model", "mean", "std", "sem", "count"])
                    for r in rankings:
                        writer.writerow([
                            r.get("rank", ""),
                            r.get("model", ""),
                            r.get("mean", ""),
                            r.get("std", ""),
                            r.get("sem", ""),
                            r.get("count", "")
                        ])

            elif name == "heatmap":
                # pivot table: environments vs models
                heatmap = data.get("heatmap", {})
                if heatmap:
                    # get all models (columns)
                    models = set()
                    for env_data in heatmap.values():
                        models.update(env_data.keys())
                    models = sorted(models)

                    # header row
                    writer.writerow(["environment"] + models)

                    # data rows
                    for env, env_data in heatmap.items():
                        row = [env]
                        for model in models:
                            val = env_data.get(model)
                            row.append("" if val is None else val)
                        writer.writerow(row)

            elif name == "best-configs":
                # top configurations
                configs = data.get("best_configs", [])
                if configs:
                    # determine all keys
                    all_keys = set()
                    for c in configs:
                        all_keys.update(c.keys())
                    all_keys = sorted(all_keys)

                    writer.writerow(all_keys)
                    for c in configs:
                        writer.writerow([c.get(k, "") for k in all_keys])

                # best per environment (separate section)
                best_per_env = data.get("best_per_env", [])
                if best_per_env:
                    writer.writerow([])  # blank line
                    writer.writerow(["# Best per Environment"])
                    writer.writerow(["environment", "best_model", "z_mean"])
                    for b in best_per_env:
                        writer.writerow([
                            b.get("environment", ""),
                            b.get("best_model", ""),
                            b.get("z_mean", "")
                        ])

            elif name == "budget-progression":
                # progression data with budgets
                progression = data.get("progression", [])
                if progression:
                    # get all budgets
                    budgets = set()
                    for p in progression:
                        budgets.update(p.get("budgets", {}).keys())
                    budgets = sorted(budgets)

                    # header
                    writer.writerow(["env_model"] + [f"budget_{b}" for b in budgets])

                    # data rows
                    for p in progression:
                        row = [p.get("env_model", "")]
                        for b in budgets:
                            val = p.get("budgets", {}).get(b)
                            row.append("" if val is None else val)
                        writer.writerow(row)

            elif name == "seed-stability":
                # per-seed stats
                per_seed = data.get("per_seed_stats", [])
                if per_seed:
                    writer.writerow(["# Per-Seed Performance"])
                    writer.writerow(["seed", "mean", "std", "count"])
                    for s in per_seed:
                        writer.writerow([
                            s.get("config/seed", ""),
                            s.get("mean", ""),
                            s.get("std", ""),
                            s.get("count", "")
                        ])

                # variance decomposition
                variance_decomp = data.get("variance_decomposition", {})
                if variance_decomp:
                    writer.writerow([])  # blank line
                    writer.writerow(["# Variance Decomposition"])
                    writer.writerow(["metric", "value"])
                    writer.writerow(["global_std", variance_decomp.get("global_std", "")])
                    writer.writerow(["between_seed_std", variance_decomp.get("between_seed_std", "")])
                    writer.writerow(["within_seed_std", variance_decomp.get("within_seed_std", "")])

                # unstable configs
                unstable = data.get("unstable_configs", [])
                if unstable:
                    writer.writerow([])  # blank line
                    writer.writerow(["# Most Unstable Configs"])
                    writer.writerow(["config", "variance", "std", "mean", "count"])
                    for u in unstable:
                        config_str = str(u.get("config", {}))
                        writer.writerow([
                            config_str,
                            u.get("variance", ""),
                            u.get("std", ""),
                            u.get("mean", ""),
                            u.get("count", "")
                        ])

            elif name == "local-summary":
                # model rankings
                rankings = data.get("rankings", [])
                if rankings:
                    writer.writerow(["rank", "model", "z_mean", "abs_z_mean", "std", "run_count", "effective_seeds"])
                    for r in rankings:
                        writer.writerow([
                            r.get("rank", ""),
                            r.get("model", ""),
                            r.get("z_mean", ""),
                            r.get("abs_z_mean", ""),
                            r.get("std", ""),
                            r.get("run_count", ""),
                            r.get("effective_seeds", "")
                        ])

                # per-environment top 3
                per_env = data.get("per_environment", {})
                if per_env:
                    writer.writerow([])  # blank line
                    writer.writerow(["# Per-Environment Top 3"])
                    writer.writerow(["environment", "model", "z_mean", "effective_seeds"])
                    for env, models in sorted(per_env.items()):
                        for m in models:
                            writer.writerow([
                                env,
                                m.get("model", ""),
                                m.get("z_mean", ""),
                                m.get("effective_seeds", "")
                            ])

                # summary stats
                writer.writerow([])
                writer.writerow(["# Summary"])
                writer.writerow(["total_runs", data.get("total_runs", "")])
                writer.writerow(["total_models", data.get("total_models", "")])
                writer.writerow(["total_environments", data.get("total_environments", "")])

            elif name == "ppl-examples":
                # summary
                summary = data.get("summary", {})
                if summary:
                    writer.writerow(["# PPL Summary"])
                    writer.writerow(["metric", "value"])
                    writer.writerow(["total_runs", summary.get("total_runs", "")])
                    writer.writerow(["environments", summary.get("environments", "")])
                    writer.writerow(["models", summary.get("models", "")])
                    writer.writerow(["best_loo", summary.get("best_loo", "")])
                    writer.writerow(["total_programs", summary.get("total_programs", "")])

                # best per env
                best_per_env = data.get("best_per_env", [])
                if best_per_env:
                    writer.writerow([])
                    writer.writerow(["# Best PPL Model per Environment"])
                    writer.writerow(["env", "model", "z_mean", "best_loo"])
                    for b in best_per_env:
                        writer.writerow([
                            b.get("env", ""),
                            b.get("model", ""),
                            b.get("z_mean", ""),
                            b.get("best_loo", ""),
                        ])

                # all runs
                runs = data.get("runs", [])
                if runs:
                    writer.writerow([])
                    writer.writerow(["# All PPL Runs"])
                    writer.writerow(["run_id", "env", "model", "budget", "seed", "z_mean", "best_loo", "best_waic", "num_programs", "best_divergences", "best_max_rhat", "best_min_ess"])
                    for r in runs:
                        writer.writerow([
                            r.get("run_id", ""),
                            r.get("env", ""),
                            r.get("model", ""),
                            r.get("budget", ""),
                            r.get("seed", ""),
                            r.get("z_mean", ""),
                            r.get("best_loo", ""),
                            r.get("best_waic", ""),
                            r.get("num_programs", ""),
                            r.get("best_divergences", ""),
                            r.get("best_max_rhat", ""),
                            r.get("best_min_ess", ""),
                        ])

            # blank line between views
            writer.writerow([])
