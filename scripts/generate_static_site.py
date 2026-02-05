#!/usr/bin/env python3
"""Generate static HTML site with Plotly charts from sweep results.

Uses TUI view classes to generate interactive Plotly charts and exports them
as standalone HTML files for GitHub Pages deployment.

Usage:
    uv run python scripts/generate_static_site.py
    uv run python scripts/generate_static_site.py --output-dir _site
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from rich.console import Console


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load results from results directory or parquet cache."""
    # Prefer explicit results_dir if it exists and has content
    if results_dir.exists() and any(results_dir.rglob("*.json")):
        from boxing_gym.cli.tui.loaders.local_results import load_local_results

        return load_local_results(results_dir)

    # Fall back to parquet cache (from 'box sync')
    cache_dir = Path.home() / ".boxing-gym-cache"
    parquet_path = cache_dir / "runs.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    return pd.DataFrame()


def generate_chart_html(fig) -> str:
    """Generate standalone HTML for a Plotly figure."""
    return fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        config={"displayModeBar": True, "responsive": True},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate static site with Plotly charts")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_site"),
        help="Output directory for static site (default: _site)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory to load from (default: results)",
    )
    args = parser.parse_args()

    console = Console()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Loading results...[/cyan]")
    df = load_results(args.results_dir)

    if df.empty:
        console.print("[red]No results found. Run 'box sync --local results/' first.[/red]")
        return 1

    console.print(f"[green]Loaded {len(df)} runs[/green]")

    # filter to valid runs (budget >= 10, |z| < 100)
    metric_col = "metric/eval/z_mean"
    if metric_col in df.columns:
        df = df[df[metric_col].abs() < 100]

    budget_col = None
    for col in ["config/budget", "config/exp.budget", "metric/budget"]:
        if col in df.columns:
            budget_col = col
            break

    if budget_col:
        df = df[pd.to_numeric(df[budget_col], errors="coerce") >= 10]

    console.print(f"[green]After filtering: {len(df)} valid runs[/green]")

    if df.empty:
        console.print("[red]No valid runs after filtering[/red]")
        return 1

    # import view classes
    from boxing_gym.cli.tui.views.budget_progression import BudgetProgressionView
    from boxing_gym.cli.tui.views.heatmap import HeatmapView
    from boxing_gym.cli.tui.views.model_rankings import ModelRankingsView
    from boxing_gym.cli.tui.views.parameter_importance import ParameterImportanceView

    charts = []

    view_configs = [
        (
            "Model Rankings",
            "model_rankings.html",
            "z_mean with 95% CI",
            ModelRankingsView(df, console, metric_col),
        ),
        (
            "Heatmap",
            "heatmap.html",
            "Environment × Model z_mean",
            HeatmapView(df, console, metric_col),
        ),
        (
            "Budget Progression",
            "budget_progression.html",
            "z_mean by budget",
            BudgetProgressionView(df, console, metric_col),
        ),
        (
            "Parameter Importance",
            "parameter_importance.html",
            "Permutation importance",
            ParameterImportanceView(df, console, metric_col, include_seed=False),
        ),
    ]

    for title, filename, desc, view in view_configs:
        console.print(f"[cyan]Generating {title.lower()}...[/cyan]")
        try:
            fig = view.to_plotly()
            if fig:
                html = generate_chart_html(fig)
                chart_path = output_dir / filename
                chart_path.write_text(html)
                charts.append((title, filename, desc))
                console.print(f"  [green]✓[/green] {chart_path}")
        except Exception as e:
            console.print(f"  [yellow]Failed: {e}[/yellow]")

    if not charts:
        console.print("[red]No charts generated - check view classes[/red]")
        return 1

    # generate index.html
    console.print("[cyan]Generating index.html...[/cyan]")
    index_html = generate_index_html(charts, len(df))
    index_path = output_dir / "index.html"
    index_path.write_text(index_html)
    console.print(f"  [green]✓[/green] {index_path}")

    console.print(f"\n[bold green]Static site generated in {output_dir}/[/bold green]")
    console.print(f"  {len(charts)} charts + index.html")

    return 0


def generate_index_html(charts: list[tuple[str, str, str]], n_runs: int) -> str:
    """Generate index.html linking all charts."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    chart_links = "\n".join(
        f'        <li><a href="{filename}">{title}</a> — {desc}</li>'
        for title, filename, desc in charts
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BoxingGym Results Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{
            color: #fff;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 0.5rem;
        }}
        .stats {{
            background: #16213e;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }}
        .stats span {{
            color: #3b82f6;
            font-weight: bold;
        }}
        ul {{
            list-style: none;
            padding: 0;
        }}
        li {{
            margin: 1rem 0;
            padding: 1rem;
            background: #16213e;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
        }}
        a {{
            color: #60a5fa;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.1rem;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .footer {{
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #333;
            color: #888;
            font-size: 0.9rem;
        }}
        .footer a {{
            color: #888;
        }}
    </style>
</head>
<body>
    <h1>BoxingGym Results Dashboard</h1>

    <div class="stats">
        <p>Analysis of <span>{n_runs:,}</span> sweep runs</p>
        <p>Generated: {timestamp}</p>
    </div>

    <h2>Charts</h2>
    <ul>
{chart_links}
    </ul>

    <div class="footer">
        <p>
            <a href="https://github.com/youqad/boxing-gym-wip">GitHub</a> ·
            <a href="https://huggingface.co/spaces/youkad/boxing-gym-dashboard">HF Space</a> ·
            <a href="https://arxiv.org/abs/2501.01540">Paper</a>
        </p>
    </div>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
