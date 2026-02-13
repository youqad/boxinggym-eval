"""LLM Call Logs view for TUI."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text

from ..loaders.jsonl_loader import (
    compute_stats,
    find_jsonl_files,
    find_latest_jsonl,
    load_jsonl,
)
from . import BaseView


def _truncate(text: str, max_len: int = 60) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_cost(cost: float) -> str:
    if cost < 0.0001:
        return f"${cost:.6f}"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _format_latency(latency_ms: float) -> str:
    return f"{latency_ms:,.0f}ms"


class CallLogsView(BaseView):
    def __init__(
        self,
        console: Console,
        metric: str = "metric/eval/z_mean",
        jsonl_path: Path | None = None,
    ):
        self.jsonl_path = jsonl_path or find_latest_jsonl()
        self.available_files = find_jsonl_files()
        self.entries = []
        if self.jsonl_path and self.jsonl_path.exists():
            try:
                self.entries = load_jsonl(self.jsonl_path)
            except Exception:
                self.entries = []
        super().__init__(pd.DataFrame(), console, metric)

    @property
    def title(self) -> str:
        return "LLM Call Logs"

    def _load_file(self, path: Path) -> None:
        self.jsonl_path = path
        self.entries = []
        if path.exists():
            try:
                self.entries = load_jsonl(path)
            except Exception:
                pass

    def _show_file_selector(self) -> bool:
        if not self.available_files:
            self.console.print("[yellow]No JSONL call log files found[/yellow]")
            return False

        self.console.print("\n[bold cyan]Available Call Log Files[/bold cyan]\n")

        table = Table(border_style="dim", show_header=True, header_style="bold")
        table.add_column("#", justify="right", width=4)
        table.add_column("File", width=40)
        table.add_column("Size", justify="right", width=10)
        table.add_column("Modified", width=20)

        for i, path in enumerate(self.available_files[:20], 1):
            stat = path.stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

            marker = " *" if path == self.jsonl_path else ""
            table.add_row(
                str(i),
                path.name + marker,
                f"{size_kb:.1f} KB",
                mtime,
            )

        self.console.print(table)

        if len(self.available_files) > 20:
            self.console.print(f"[dim]... and {len(self.available_files) - 20} more files[/dim]")

        self.console.print("\n[dim]* = currently selected[/dim]")
        self.console.print("[dim]Enter 0 to go back without changing[/dim]\n")

        try:
            choice = IntPrompt.ask(
                "[bold yellow]Select file[/bold yellow]",
                console=self.console,
                default=0,
            )
        except (KeyboardInterrupt, EOFError):
            return False

        if choice == 0:
            return False

        if 1 <= choice <= min(20, len(self.available_files)):
            selected = self.available_files[choice - 1]
            self._load_file(selected)
            self.console.print(f"[green]Loaded: {selected.name}[/green]\n")
            return True

        self.console.print("[red]Invalid selection[/red]")
        return False

    def render(self, interactive: bool = True) -> None:
        # Disable prompts in non-TTY contexts (e.g., --view call-logs in CI/agents).
        interactive = interactive and sys.stdin.isatty()
        n_files = len(self.available_files)
        if interactive and n_files > 1:
            self.console.print(f"[cyan]{n_files} call log files available[/cyan]\n")
            self.console.print("[dim]  [1] View current file[/dim]")
            self.console.print("[dim]  [2] Select different file[/dim]\n")

            try:
                choice = IntPrompt.ask(
                    "[bold yellow]Choice[/bold yellow]",
                    console=self.console,
                    default=1,
                )
                if choice == 2:
                    self._show_file_selector()
            except (KeyboardInterrupt, EOFError):
                return

        if not self.jsonl_path:
            self.console.print("[yellow]No JSONL call log files found[/yellow]")
            self.console.print("[dim]Run an experiment with call recording enabled[/dim]")
            return

        if not self.entries:
            self.console.print(f"[yellow]No entries in {self.jsonl_path}[/yellow]")
            return

        stats = compute_stats(self.entries)

        self.console.print(f"[dim]File: {self.jsonl_path.name} ({len(self.entries)} calls)[/dim]\n")

        summary_table = Table(
            title="LLM Call Logs Summary",
            border_style="cyan",
            header_style="bold magenta",
        )
        summary_table.add_column("Agent", style="white", width=15)
        summary_table.add_column("Calls", justify="right", width=8)
        summary_table.add_column("Cost", justify="right", width=12)
        summary_table.add_column("Avg Latency", justify="right", width=12)
        summary_table.add_column("Tokens", justify="right", width=10)
        summary_table.add_column("Errors", justify="right", width=8)

        for agent_stat in stats.agents:
            error_style = "red" if agent_stat.error_count > 0 else "dim"
            summary_table.add_row(
                agent_stat.agent,
                str(agent_stat.call_count),
                _format_cost(agent_stat.total_cost),
                _format_latency(agent_stat.avg_latency),
                f"{agent_stat.total_tokens:,}",
                Text(str(agent_stat.error_count), style=error_style),
            )

        summary_table.add_section()
        error_style = "red" if stats.error_count > 0 else "dim"
        total_tokens = (
            stats.total_prompt_tokens + stats.total_completion_tokens + stats.total_reasoning_tokens
        )
        summary_table.add_row(
            "TOTAL",
            str(stats.total_calls),
            _format_cost(stats.total_cost),
            _format_latency(stats.avg_latency),
            f"{total_tokens:,}",
            Text(str(stats.error_count), style=error_style),
        )

        self.console.print(summary_table)
        self.console.print()

        if stats.models:
            models_str = ", ".join(stats.models)
            self.console.print(f"[cyan]Models:[/cyan] {models_str}\n")

        self._render_recent_calls()

        if stats.error_count > 0:
            self._render_errors()

    def _render_recent_calls(self, limit: int = 10) -> None:
        if not self.entries:
            return

        recent = self.entries[-limit:]

        table = Table(
            title=f"Recent Calls (last {min(limit, len(self.entries))})",
            border_style="blue",
            header_style="bold blue",
        )
        table.add_column("#", justify="right", width=4)
        table.add_column("Agent", width=12)
        table.add_column("Type", width=10)
        table.add_column("Latency", justify="right", width=10)
        table.add_column("Cost", justify="right", width=10)
        table.add_column("Prompt (truncated)", width=40)

        for i, entry in enumerate(recent, start=len(self.entries) - len(recent) + 1):
            prompt_preview = _truncate(entry.prompt.replace("\n", " "), 38)
            table.add_row(
                str(i),
                entry.agent,
                entry.call_type,
                _format_latency(entry.latency_ms),
                _format_cost(entry.cost_usd),
                prompt_preview,
            )

        self.console.print(table)

    def _render_errors(self) -> None:
        errors = [e for e in self.entries if e.error]
        if not errors:
            return

        self.console.print()
        self.console.print(
            Panel(
                f"[red]Found {len(errors)} error(s)[/red]",
                title="Errors",
                border_style="red",
            )
        )

        for i, entry in enumerate(errors[:5], 1):
            self.console.print(f"  {i}. [{entry.agent}] {_truncate(str(entry.error), 70)}")

        if len(errors) > 5:
            self.console.print(f"  ... and {len(errors) - 5} more")

    def get_data(self) -> dict[str, Any]:
        if not self.entries:
            return {
                "file": str(self.jsonl_path) if self.jsonl_path else None,
                "summary": {},
                "calls": [],
            }

        stats = compute_stats(self.entries)

        return {
            "file": str(self.jsonl_path),
            "summary": {
                "total_calls": stats.total_calls,
                "total_cost": stats.total_cost,
                "avg_latency_ms": stats.avg_latency,
                "error_count": stats.error_count,
                "total_prompt_tokens": stats.total_prompt_tokens,
                "total_completion_tokens": stats.total_completion_tokens,
                "total_reasoning_tokens": stats.total_reasoning_tokens,
                "models": stats.models,
            },
            "by_agent": [
                {
                    "agent": a.agent,
                    "call_count": a.call_count,
                    "total_cost": a.total_cost,
                    "avg_latency_ms": a.avg_latency,
                    "error_count": a.error_count,
                    "total_tokens": a.total_tokens,
                }
                for a in stats.agents
            ],
            "calls": [
                {
                    "agent": e.agent,
                    "model": e.model,
                    "call_type": e.call_type,
                    "latency_ms": e.latency_ms,
                    "cost_usd": e.cost_usd,
                    "prompt_tokens": e.prompt_tokens,
                    "completion_tokens": e.completion_tokens,
                    "reasoning_tokens": e.reasoning_tokens,
                    "has_reasoning": e.has_reasoning,
                    "step_idx": e.step_idx,
                    "timestamp": e.timestamp,
                    "error": e.error,
                    "prompt_preview": _truncate(e.prompt, 200),
                    "response_preview": _truncate(e.response, 200),
                }
                for e in self.entries
            ],
        }

    def get_csv_rows(self) -> list:
        data = self.get_data()
        rows = []

        summary = data.get("summary", {})
        if summary:
            rows.append(["# Call Logs Summary"])
            rows.append(["metric", "value"])
            rows.append(["total_calls", summary.get("total_calls", "")])
            rows.append(["total_cost", summary.get("total_cost", "")])
            rows.append(["avg_latency_ms", summary.get("avg_latency_ms", "")])
            rows.append(["error_count", summary.get("error_count", "")])
            rows.append(["total_prompt_tokens", summary.get("total_prompt_tokens", "")])
            rows.append(["total_completion_tokens", summary.get("total_completion_tokens", "")])
            rows.append(["models", ", ".join(summary.get("models", []))])

        by_agent = data.get("by_agent", [])
        if by_agent:
            rows.append([])
            rows.append(["# By Agent"])
            rows.append(
                [
                    "agent",
                    "call_count",
                    "total_cost",
                    "avg_latency_ms",
                    "error_count",
                    "total_tokens",
                ]
            )
            for a in by_agent:
                rows.append(
                    [
                        a.get("agent", ""),
                        a.get("call_count", ""),
                        a.get("total_cost", ""),
                        a.get("avg_latency_ms", ""),
                        a.get("error_count", ""),
                        a.get("total_tokens", ""),
                    ]
                )

        calls = data.get("calls", [])
        if calls:
            rows.append([])
            rows.append(["# All Calls"])
            rows.append(
                [
                    "agent",
                    "model",
                    "call_type",
                    "latency_ms",
                    "cost_usd",
                    "prompt_tokens",
                    "completion_tokens",
                    "step_idx",
                    "error",
                ]
            )
            for c in calls:
                rows.append(
                    [
                        c.get("agent", ""),
                        c.get("model", ""),
                        c.get("call_type", ""),
                        c.get("latency_ms", ""),
                        c.get("cost_usd", ""),
                        c.get("prompt_tokens", ""),
                        c.get("completion_tokens", ""),
                        c.get("step_idx", ""),
                        c.get("error", ""),
                    ]
                )

        return rows
