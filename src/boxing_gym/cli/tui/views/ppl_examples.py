"""PPL Examples view for TUI - displays generated PyMC code from Box's Loop."""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import BaseView


def _truncate(text: str, max_len: int = 80) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _color_divergences(count: int) -> Text:
    if count == 0:
        return Text(str(count), style="green")
    elif count < 10:
        return Text(str(count), style="yellow")
    return Text(str(count), style="red")


def _color_rhat(rhat: float) -> Text:
    if rhat < 1.01:
        return Text(f"{rhat:.3f}", style="green")
    elif rhat < 1.1:
        return Text(f"{rhat:.3f}", style="yellow")
    return Text(f"{rhat:.3f}", style="red")


def _color_ess(ess: float) -> Text:
    if ess > 400:
        return Text(f"{ess:.0f}", style="green")
    elif ess > 100:
        return Text(f"{ess:.0f}", style="yellow")
    return Text(f"{ess:.0f}", style="red")


def _safe_get_config_value(
    config: dict, key: str, subkey: str, fallback_key: str, default="unknown"
) -> str:
    val = config.get(key, {})
    if isinstance(val, dict):
        return val.get(subkey, config.get(fallback_key, default))
    return str(val) if val else config.get(fallback_key, default)


class PPLExamplesView(BaseView):
    def __init__(
        self,
        console: Console,
        metric: str = "metric/eval/z_mean",
        results_dir: str = "results",
        entity: str | None = None,
        project: str | None = None,
        max_runs: int = 50,
    ):
        self._results_dir = Path(results_dir)
        self._entity = entity or os.environ.get("WANDB_ENTITY", "")
        self._project = project or os.environ.get("WANDB_PROJECT", "boxing-gym")
        self._max_runs = max_runs
        self._ppl_runs: list[dict[str, Any]] = []
        self._loaded = False
        self._load_error: str | None = None
        self._source: str = "none"
        super().__init__(pd.DataFrame(), console, metric)

    @property
    def title(self) -> str:
        return "PPL Examples"

    def _load_local_ppl_runs(self) -> bool:
        if not self._results_dir.exists():
            return False

        json_files = list(self._results_dir.rglob("*.json"))
        json_files = [f for f in json_files if not f.name.startswith("llm_calls")]

        count = 0
        for path in json_files:
            if count >= self._max_runs:
                break

            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            config = data.get("config", {})
            data_section = data.get("data", {})

            if not config.get("use_ppl", False):
                continue

            ppl_code = (
                data_section.get("best_program_code")
                or data_section.get("ppl", {}).get("best_program_code")
                or config.get("ppl/best_program_code")
                or ""
            )

            if not ppl_code:
                ppl_results = data_section.get("ppl_results", [])
                if isinstance(ppl_results, list) and ppl_results:
                    best_result = min(
                        (r for r in ppl_results if r.get("loo") is not None),
                        key=lambda r: r.get("loo", float("inf")),
                        default=None,
                    )
                    if best_result:
                        ppl_code = best_result.get("code", "")

            if not ppl_code:
                continue

            envs_config = config.get("envs", {})
            llms_config = config.get("llms", {})
            env_name = (
                envs_config.get("env_name") if isinstance(envs_config, dict) else str(envs_config)
            )
            model_name = (
                llms_config.get("model_name") if isinstance(llms_config, dict) else str(llms_config)
            )

            z_results = data_section.get("z_results", [])
            z_mean = None
            if z_results:
                final_result = z_results[-1] if isinstance(z_results, list) else z_results
                z_mean = final_result.get("z_mean") if isinstance(final_result, dict) else None

            ppl_data = data_section.get("ppl", {})
            best_loo = ppl_data.get("best_loo") or data_section.get("best_loo")
            best_waic = ppl_data.get("best_waic") or data_section.get("best_waic")
            num_programs = ppl_data.get("num_programs", 1)
            best_divergences = ppl_data.get("best_n_divergences", 0)
            best_max_rhat = ppl_data.get("best_max_rhat")
            best_min_ess = ppl_data.get("best_min_ess_bulk")

            self._ppl_runs.append(
                {
                    "run_id": str(path),
                    "run_name": path.stem,
                    "env": env_name or "unknown",
                    "model": model_name or "unknown",
                    "budget": config.get("budget", 0),
                    "seed": config.get("seed", 0),
                    "z_mean": z_mean,
                    "best_loo": best_loo,
                    "best_waic": best_waic,
                    "ppl_code": ppl_code,
                    "num_programs": num_programs,
                    "num_rounds": ppl_data.get("num_rounds", 1),
                    "best_divergences": best_divergences,
                    "best_max_rhat": best_max_rhat,
                    "best_min_ess": best_min_ess,
                    "url": None,
                }
            )
            count += 1

        return len(self._ppl_runs) > 0

    def _load_ppl_runs(self) -> None:
        if self._loaded:
            return

        if self._load_local_ppl_runs():
            self._source = "local"
            self._loaded = True
            return

        try:
            import wandb
        except ImportError:
            self._load_error = "No local PPL runs found and wandb not installed"
            self._loaded = True
            return

        try:
            api = wandb.Api(timeout=120)
            runs = api.runs(
                f"{self._entity}/{self._project}",
                filters={"config.use_ppl": True, "state": "finished"},
                order="-created_at",
            )

            count = 0
            for run in runs:
                if count >= self._max_runs:
                    break

                config = run.config
                summary = run.summary

                ppl_code = config.get("ppl/best_program_code", "") or summary.get(
                    "ppl/best_program_code", ""
                )
                if not ppl_code:
                    continue

                env_name = _safe_get_config_value(config, "envs", "name", "env_name")
                model_name = _safe_get_config_value(config, "llms", "model", "model_name")

                best_loo = summary.get("ppl/best_loo") or config.get("ppl/best_loo")
                best_waic = summary.get("ppl/best_waic") or config.get("ppl/best_waic")
                num_programs = summary.get("ppl/num_programs", 1)
                num_rounds = summary.get("ppl/num_rounds", 1)
                best_divergences = summary.get("ppl/best_n_divergences", 0)
                best_max_rhat = summary.get("ppl/best_max_rhat")
                best_min_ess = summary.get("ppl/best_min_ess_bulk")

                self._ppl_runs.append(
                    {
                        "run_id": run.id,
                        "run_name": run.name,
                        "env": env_name,
                        "model": model_name,
                        "budget": config.get("budget", summary.get("eval/budget", 0)),
                        "seed": config.get("seed", 0),
                        "z_mean": summary.get("eval/z_mean"),
                        "best_loo": best_loo,
                        "best_waic": best_waic,
                        "ppl_code": ppl_code,
                        "num_programs": num_programs,
                        "num_rounds": num_rounds,
                        "best_divergences": best_divergences,
                        "best_max_rhat": best_max_rhat,
                        "best_min_ess": best_min_ess,
                        "url": run.url,
                    }
                )
                count += 1

            if self._ppl_runs:
                self._source = "wandb"
            self._loaded = True

        except Exception as e:
            self._load_error = f"No local PPL runs found. WandB error: {e}"
            self._loaded = True

    def render(self) -> None:
        self._load_ppl_runs()

        if self._load_error:
            self.console.print(f"[yellow]{self._load_error}[/yellow]")
            self.console.print(f"\n[dim]Searched local: {self._results_dir}[/dim]")
            self.console.print("\n[dim]Run an experiment with PPL enabled:[/dim]")
            self.console.print(
                "[cyan]uv run python run_experiment.py use_ppl=true envs=dugongs_direct[/cyan]"
            )
            return

        if not self._ppl_runs:
            self.console.print("[yellow]No PPL runs found[/yellow]")
            self.console.print(f"\n[dim]Searched local: {self._results_dir}[/dim]")
            self.console.print("\n[dim]Run an experiment with PPL enabled:[/dim]")
            self.console.print(
                "[cyan]uv run python run_experiment.py use_ppl=true envs=dugongs_direct[/cyan]"
            )
            return

        self.console.print(f"[dim]Source: {self._source} ({len(self._ppl_runs)} PPL runs)[/dim]\n")
        self._render_summary()
        self.console.print()
        self._render_best_per_env()
        self.console.print()
        self._render_all_runs_table()

    def _render_summary(self) -> None:
        df = pd.DataFrame(self._ppl_runs)

        envs = df["env"].nunique()
        models = df["model"].nunique()
        best_loo = df["best_loo"].dropna().min() if "best_loo" in df else None
        total_programs = df["num_programs"].sum()

        summary_text = (
            f"[bold]PPL Runs:[/bold] {len(df)}  "
            f"[bold]Environments:[/bold] {envs}  "
            f"[bold]Models:[/bold] {models}  "
        )

        if pd.notna(best_loo):
            summary_text += f"[bold]Best LOO:[/bold] {best_loo:.2f}  "

        summary_text += f"[bold]Programs:[/bold] {int(total_programs)}"

        self.console.print(Panel(summary_text, title="Summary", border_style="cyan"))

    def _render_best_per_env(self) -> None:
        df = pd.DataFrame(self._ppl_runs)
        if "z_mean" not in df.columns:
            return

        valid_df = df.dropna(subset=["z_mean"])
        if len(valid_df) == 0:
            return

        self.console.print("[bold cyan]Best Models by Environment[/bold cyan]\n")

        best_per_env = valid_df.loc[valid_df.groupby("env")["z_mean"].idxmin()]

        for _, row in best_per_env.iterrows():
            loo_val = row.get("best_loo")
            loo_str = f"LOO={loo_val:.2f}" if loo_val is not None and pd.notna(loo_val) else ""
            header = f"{row['env']} → {row['model']} (z={row['z_mean']:.3f}{', ' + loo_str if loo_str else ''})"

            ppl_code = row.get("ppl_code", "")
            code_preview = ppl_code[:300] if ppl_code else "# No code available"
            if len(ppl_code) > 300:
                code_preview += "\n# ... (truncated)"

            syntax = Syntax(code_preview, "python", theme="monokai", line_numbers=False)
            self.console.print(Panel(syntax, title=header, border_style="green", padding=(0, 1)))
            self.console.print()

    def _render_all_runs_table(self) -> None:
        if not self._ppl_runs:
            return

        table = Table(
            title="All PPL Runs",
            border_style="blue",
            header_style="bold blue",
        )
        table.add_column("Env", width=16)
        table.add_column("Model", width=18)
        table.add_column("z_mean", justify="right", width=8)
        table.add_column("LOO", justify="right", width=8)
        table.add_column("Progs", justify="right", width=6)
        table.add_column("Div", justify="right", width=5)
        table.add_column("Rhat", justify="right", width=8)
        table.add_column("ESS", justify="right", width=8)

        sorted_runs = sorted(
            self._ppl_runs, key=lambda r: (r["env"], r.get("z_mean") or float("inf"))
        )

        for run in sorted_runs:
            z_mean = f"{run['z_mean']:.3f}" if pd.notna(run.get("z_mean")) else "—"
            loo = f"{run['best_loo']:.2f}" if pd.notna(run.get("best_loo")) else "—"
            progs = str(run.get("num_programs", 1))

            div_val = run.get("best_divergences")
            div = (
                _color_divergences(int(div_val))
                if div_val is not None and pd.notna(div_val)
                else Text("—", style="dim")
            )
            rhat_val = run.get("best_max_rhat")
            rhat = (
                _color_rhat(rhat_val)
                if rhat_val is not None and pd.notna(rhat_val)
                else Text("—", style="dim")
            )
            ess_val = run.get("best_min_ess")
            ess = (
                _color_ess(ess_val)
                if ess_val is not None and pd.notna(ess_val)
                else Text("—", style="dim")
            )

            table.add_row(
                _truncate(run["env"], 16),
                _truncate(run["model"], 18),
                z_mean,
                loo,
                progs,
                div,
                rhat,
                ess,
            )

        self.console.print(table)

    def get_data(self) -> dict[str, Any]:
        self._load_ppl_runs()

        if not self._ppl_runs:
            return {
                "entity": self._entity,
                "project": self._project,
                "error": self._load_error,
                "runs": [],
                "summary": {},
            }

        df = pd.DataFrame(self._ppl_runs)

        best_loo_val = df["best_loo"].dropna().min() if "best_loo" in df else None
        best_loo = None if pd.isna(best_loo_val) else best_loo_val
        valid_df = df.dropna(subset=["z_mean"])
        best_per_env = []
        if len(valid_df) > 0:
            for env in valid_df["env"].unique():
                env_df = valid_df[valid_df["env"] == env]
                best_row = env_df.loc[env_df["z_mean"].idxmin()]
                best_per_env.append(
                    {
                        "env": env,
                        "model": best_row["model"],
                        "z_mean": best_row["z_mean"],
                        "best_loo": best_row.get("best_loo"),
                        "ppl_code": best_row["ppl_code"],
                    }
                )

        return {
            "entity": self._entity,
            "project": self._project,
            "summary": {
                "total_runs": len(self._ppl_runs),
                "environments": df["env"].nunique(),
                "models": df["model"].nunique(),
                "best_loo": best_loo,
                "total_programs": int(df["num_programs"].sum()),
            },
            "best_per_env": best_per_env,
            "runs": [
                {
                    "run_id": r["run_id"],
                    "env": r["env"],
                    "model": r["model"],
                    "budget": r["budget"],
                    "seed": r["seed"],
                    "z_mean": r["z_mean"],
                    "best_loo": r["best_loo"],
                    "best_waic": r.get("best_waic"),
                    "num_programs": r["num_programs"],
                    "best_divergences": r.get("best_divergences"),
                    "best_max_rhat": r.get("best_max_rhat"),
                    "best_min_ess": r.get("best_min_ess"),
                    "ppl_code_preview": r.get("ppl_code", "")[:500],
                }
                for r in self._ppl_runs
            ],
        }

    def get_csv_rows(self) -> list:
        data = self.get_data()
        rows = []

        summary = data.get("summary", {})
        if summary:
            rows.append(["# PPL Summary"])
            rows.append(["metric", "value"])
            rows.append(["total_runs", summary.get("total_runs", "")])
            rows.append(["environments", summary.get("environments", "")])
            rows.append(["models", summary.get("models", "")])
            rows.append(["best_loo", summary.get("best_loo", "")])
            rows.append(["total_programs", summary.get("total_programs", "")])

        best_per_env = data.get("best_per_env", [])
        if best_per_env:
            rows.append([])
            rows.append(["# Best PPL Model per Environment"])
            rows.append(["env", "model", "z_mean", "best_loo"])
            for b in best_per_env:
                rows.append(
                    [
                        b.get("env", ""),
                        b.get("model", ""),
                        b.get("z_mean", ""),
                        b.get("best_loo", ""),
                    ]
                )

        runs = data.get("runs", [])
        if runs:
            rows.append([])
            rows.append(["# All PPL Runs"])
            rows.append(
                [
                    "run_id",
                    "env",
                    "model",
                    "budget",
                    "seed",
                    "z_mean",
                    "best_loo",
                    "best_waic",
                    "num_programs",
                    "best_divergences",
                    "best_max_rhat",
                    "best_min_ess",
                ]
            )
            for r in runs:
                rows.append(
                    [
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
                    ]
                )

        return rows
