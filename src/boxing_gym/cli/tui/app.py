"""Main TUI application for sweep analysis."""

import csv
import inspect
import json
import math
import sys
from typing import Any, Literal

import numpy as np
import pandas as pd


class NumpyJSONEncoder(json.JSONEncoder):
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

from boxing_gym.cli.quality_filter import apply_quality_filters

from .components.menu import clear_screen, press_enter_to_continue, render_menu
from .views import BaseView
from .views.best_configs import BestConfigsView
from .views.budget_progression import BudgetProgressionView
from .views.call_logs import CallLogsView
from .views.heatmap import HeatmapView
from .views.local_summary import LocalSummaryView
from .views.model_rankings import ModelRankingsView
from .views.parameter_importance import ParameterImportanceView
from .views.ppl_examples import PPLExamplesView
from .views.seed_stability import SeedStabilityView

VIEW_REGISTRY: dict[str, type[BaseView]] = {
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

AVAILABLE_VIEWS = list(VIEW_REGISTRY.keys()) + ["all"]


class SweepTUI:
    def __init__(
        self,
        df: pd.DataFrame,
        sweep_ids: list[str],
        metric: str = "metric/eval/z_mean",
        include_seed: bool = False,
        is_local: bool = False,
    ):
        self.original_df = df.copy()  # store unfiltered for dynamic filtering
        self.df = apply_quality_filters(df, z_col=metric)
        self.sweep_ids = sweep_ids
        self.metric = metric
        self.include_seed = include_seed
        self.is_local = is_local
        self.console = Console(width=120)
        self.current_env_filter: str | None = None
        self._init_views()

    def _init_views(self) -> None:
        self.views = []

        if self.is_local:
            self.views.append(LocalSummaryView(self.df, self.console, self.metric))

        self.views.extend(
            [
                ParameterImportanceView(
                    self.df, self.console, self.metric, include_seed=self.include_seed
                ),
                ModelRankingsView(self.df, self.console, self.metric),
                HeatmapView(self.df, self.console, self.metric),
                BestConfigsView(self.df, self.console, self.metric),
                BudgetProgressionView(self.df, self.console, self.metric),
                SeedStabilityView(self.df, self.console, self.metric),
                CallLogsView(self.console, self.metric),
                PPLExamplesView(self.console, self.metric),
            ]
        )

        # Canonical map: iterate/export only VIEW_REGISTRY keys.
        instance_by_type: dict[type[BaseView], BaseView] = {}
        for v in self.views:
            instance_by_type[type(v)] = v
        self._view_map = {
            key: instance_by_type.get(view_type)
            for key, view_type in VIEW_REGISTRY.items()
            if instance_by_type.get(view_type) is not None
        }
        # Title-based aliases for lookup only (not iteration).
        self._alias_map: dict[str, str] = {}
        for key, view in self._view_map.items():
            title_key = view.title.lower().replace(" ", "-")
            if title_key != key:
                self._alias_map[title_key] = key

    def _get_subtitle(self) -> str:
        sweep_str = ", ".join(self.sweep_ids[:3])
        if len(self.sweep_ids) > 3:
            sweep_str += f" +{len(self.sweep_ids) - 3} more"
        filter_str = f" | Env: {self.current_env_filter}" if self.current_env_filter else ""
        return f"Sweep: {sweep_str} | Runs: {len(self.df)}{filter_str}"

    def _get_env_column(self) -> str | None:
        for candidate in ["config/envs", "envs", "config/env", "env"]:
            if candidate in self.original_df.columns:
                return candidate
        return None

    def _get_environments(self) -> list[str]:
        env_col = self._get_env_column()
        if not env_col:
            return []
        return sorted(self.original_df[env_col].dropna().unique().tolist())

    def _filter_by_environment(self, env: str | None) -> None:
        self.current_env_filter = env
        env_col = self._get_env_column()

        if env is None or not env_col:
            base = self.original_df.copy()
        else:
            base = self.original_df[self.original_df[env_col] == env].copy()
        self.df = apply_quality_filters(base, z_col=self.metric)
        self._init_views()

    def _show_env_filter_menu(self) -> None:
        envs = self._get_environments()
        env_col = self._get_env_column()

        if not envs or not env_col:
            self.console.print("[yellow]No environment column found in data[/yellow]")
            press_enter_to_continue(self.console)
            return

        clear_screen(self.console)
        env_counts = self.original_df[env_col].value_counts().to_dict()
        options = [("0", f"All Environments ({len(self.original_df)} runs)")]
        for i, env in enumerate(envs, 1):
            count = env_counts.get(env, 0)
            marker = " *" if env == self.current_env_filter else ""
            options.append((str(i), f"{env} ({count} runs){marker}"))

        render_menu(
            self.console,
            title="Filter by Environment",
            subtitle=f"Current filter: {self.current_env_filter or 'None (all envs)'}",
            options=options,
        )

        try:
            choice = IntPrompt.ask(
                "[bold yellow]Select environment[/bold yellow]",
                console=self.console,
                default=0,
            )
        except (KeyboardInterrupt, EOFError):
            return

        if choice == 0:
            self._filter_by_environment(None)
            self.console.print("[green]Filter cleared - showing all environments[/green]")
        elif 1 <= choice <= len(envs):
            selected_env = envs[choice - 1]
            self._filter_by_environment(selected_env)
            self.console.print(f"[green]Filtered to: {selected_env}[/green]")
        else:
            self.console.print("[red]Invalid option[/red]")

        press_enter_to_continue(self.console)

    def run(self) -> None:
        while True:
            clear_screen(self.console)
            options = [(str(i + 1), view.title) for i, view in enumerate(self.views)]
            filter_idx = len(self.views) + 1
            filter_label = "Filter by Environment"
            if self.current_env_filter:
                filter_label += f" [{self.current_env_filter}]"
            options.append((str(filter_idx), filter_label))
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
            elif choice == filter_idx:
                self._show_env_filter_menu()
            else:
                self.console.print("[red]Invalid option. Please try again.[/red]")
                press_enter_to_continue(self.console)

    def run_non_interactive(
        self,
        view_names: list[str],
        output_format: Literal["rich", "json", "csv", "plotly"] = "rich",
        plotly_output_dir: str | None = None,
    ) -> None:
        if "all" in view_names:
            view_names = list(self._view_map.keys())

        if output_format == "json":
            self._output_json(view_names)
            return
        elif output_format == "csv":
            self._output_csv(view_names)
            return
        elif output_format == "plotly":
            self._output_plotly(view_names, plotly_output_dir)
            return

        self.console.print(Rule("[bold cyan]BoxingGym Sweep Analysis[/bold cyan]"))
        self.console.print(f"[dim]{self._get_subtitle()}[/dim]\n")

        for i, name in enumerate(view_names):
            canonical = self._alias_map.get(name, name)
            if canonical not in self._view_map:
                self.console.print(f"[red]Unknown view: {name}[/red]")
                self.console.print(f"[dim]Available: {', '.join(self._view_map.keys())}[/dim]\n")
                continue

            view = self._view_map[canonical]

            if i > 0:
                self.console.print()  # blank line between views

            self.console.print(Rule(f"[bold cyan]{view.title}[/bold cyan]"))
            render_sig = inspect.signature(view.render)
            if "interactive" in render_sig.parameters:
                view.render(interactive=False)
            else:
                view.render()
            self.console.print()  # spacing after view

    def _output_json(self, view_names: list[str]) -> None:
        result = {}
        for name in view_names:
            canonical = self._alias_map.get(name, name)
            if canonical not in self._view_map:
                continue
            view = self._view_map[canonical]
            result[canonical] = view.get_data()

        json.dump(result, sys.stdout, indent=2, cls=NumpyJSONEncoder)
        sys.stdout.write("\n")

    def _output_csv(self, view_names: list[str]) -> None:
        """Output view data as CSV. Each view implements get_csv_rows()."""
        writer = csv.writer(sys.stdout)

        for name in view_names:
            canonical = self._alias_map.get(name, name)
            if canonical not in self._view_map:
                continue

            view = self._view_map[canonical]
            writer.writerow([f"# {view.title}"])

            for row in view.get_csv_rows():
                writer.writerow(row)

            writer.writerow([])

    def _output_plotly(self, view_names: list[str], output_dir: str | None = None) -> None:
        from pathlib import Path

        output_path = Path(output_dir) if output_dir else Path.cwd()
        output_path.mkdir(parents=True, exist_ok=True)

        exported = []
        for name in view_names:
            canonical = self._alias_map.get(name, name)
            if canonical not in self._view_map:
                continue

            view = self._view_map[canonical]
            fig = view.to_plotly()

            if fig is None:
                self.console.print(f"[dim]{canonical}: no Plotly export available[/dim]")
                continue

            filename = f"{canonical.replace(' ', '_')}.html"
            filepath = output_path / filename
            fig.write_html(str(filepath), include_plotlyjs="cdn")
            exported.append((canonical, filepath))
            self.console.print(f"[green]✓[/green] {canonical} → {filepath}")

        if exported:
            self.console.print(
                f"\n[cyan]Exported {len(exported)} Plotly figures to {output_path}[/cyan]"
            )
        else:
            self.console.print(
                "[yellow]No Plotly figures exported (views may not support Plotly)[/yellow]"
            )

    def get_plotly_figures(self) -> dict[str, Any]:
        """Return dict of view name to Plotly Figure (or None if unsupported)."""
        return {name: view.to_plotly() for name, view in self._view_map.items()}
