"""LLM Call Logs view for TUI."""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from scripts.tui.views import BaseView
from scripts.tui.loaders.jsonl_loader import (
    load_jsonl,
    compute_stats,
    find_latest_jsonl,
    CallLogEntry,
)


def _truncate(text: str, max_len: int = 60) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_cost(cost: float) -> str:
    """Format cost with appropriate precision."""
    if cost < 0.0001:
        return f"${cost:.6f}"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _format_latency(latency_ms: float) -> str:
    """Format latency with commas."""
    return f"{latency_ms:,.0f}ms"


class CallLogsView(BaseView):
    """Display LLM call logs from JSONL files."""

    def __init__(
        self,
        console: Console,
        metric: str = "metric/eval/z_mean",
        jsonl_path: Optional[Path] = None,
    ):
        self.jsonl_path = jsonl_path or find_latest_jsonl()
        self.entries = []
        if self.jsonl_path and self.jsonl_path.exists():
            try:
                self.entries = load_jsonl(self.jsonl_path)
            except Exception:
                pass
        super().__init__(pd.DataFrame(), console, metric)

    @property
    def title(self) -> str:
        return "LLM Call Logs"

    def render(self) -> None:
        if not self.jsonl_path:
            self.console.print("[yellow]No JSONL call log files found[/yellow]")
            self.console.print("[dim]Run an experiment with call recording enabled[/dim]")
            return

        if not self.entries:
            self.console.print(f"[yellow]No entries in {self.jsonl_path}[/yellow]")
            return

        stats = compute_stats(self.entries)

        self.console.print(
            f"[dim]File: {self.jsonl_path.name}[/dim]\n"
        )

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
            stats.total_prompt_tokens
            + stats.total_completion_tokens
            + stats.total_reasoning_tokens
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
        """Render most recent calls table."""
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
        """Render error summary."""
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

    def get_data(self) -> Dict[str, Any]:
        """Return structured data for JSON/CSV export."""
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
