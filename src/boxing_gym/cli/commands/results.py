"""Results command - view and explore results."""

import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def _adapt_columns_for_tui(df):
    mapping = {
        "model": "config/llms",
        "env": "config/envs",
        "seed": "config/seed",
        "budget": "config/budget",
        "z_mean": "metric/eval/z_mean",
        "z_std": "metric/eval/z_std",
        "experiment_type": "config/exp",
        "use_ppl": "config/use_ppl",
    }

    df = df.copy()
    for old_name, new_name in mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    return df


def run_results(
    view: str | None,
    tui: bool,
    web: bool,
    output_format: str,
    plotly_dir: str | None = None,
    include_outliers: bool = False,
):
    if web:
        _launch_web()
        return

    if tui:
        _launch_tui(include_outliers)
        return

    if view:
        _run_view(view, output_format, plotly_dir, include_outliers)
    elif output_format == "plotly":
        # export all views as Plotly when no specific view given
        _run_view("all", output_format, plotly_dir, include_outliers)
    else:
        _show_summary(include_outliers)


def _launch_web():
    app_path = (
        Path(__file__).parent.parent.parent.parent.parent / "scripts" / "streamlit_app" / "app.py"
    )
    if not app_path.exists():
        console.print(f"[red]Streamlit app not found at {app_path}[/red]")
        return

    console.print("[cyan]Launching Streamlit dashboard...[/cyan]")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def _launch_tui(include_outliers: bool = False):
    from boxing_gym.cli.commands.sync import get_cache_file

    cache_file = get_cache_file()
    if not cache_file.exists():
        console.print("[yellow]No cache found. Run sync first:[/yellow]")
        console.print("  box sync --local results/")
        return

    import pandas as pd

    from boxing_gym.cli.tui.app import SweepTUI

    df = pd.read_parquet(cache_file)

    if not include_outliers and "is_outlier" in df.columns:
        df = df[~df["is_outlier"].fillna(False)]

    df = _adapt_columns_for_tui(df)

    tui = SweepTUI(
        df=df,
        sweep_ids=["local"],
        metric="metric/eval/z_mean",
        is_local=True,
    )
    tui.run()


def _run_view(
    view: str, output_format: str, plotly_dir: str | None = None, include_outliers: bool = False
):
    from boxing_gym.cli.commands.sync import get_cache_file
    from boxing_gym.cli.tui.app import AVAILABLE_VIEWS, SweepTUI

    cache_file = get_cache_file()
    if not cache_file.exists():
        console.print("[yellow]No cache found. Run sync first:[/yellow]")
        console.print("  box sync --local results/")
        return

    if view not in AVAILABLE_VIEWS:
        console.print(f"[red]Unknown view: {view}[/red]")
        console.print(f"Available views: {', '.join(AVAILABLE_VIEWS)}")
        return

    import pandas as pd

    df = pd.read_parquet(cache_file)

    if not include_outliers and "is_outlier" in df.columns:
        df = df[~df["is_outlier"].fillna(False)]

    df = _adapt_columns_for_tui(df)

    tui = SweepTUI(
        df=df,
        sweep_ids=["local"],
        metric="metric/eval/z_mean",
        is_local=True,
    )

    fmt = "rich" if output_format == "rich" else output_format
    tui.run_non_interactive([view], output_format=fmt, plotly_output_dir=plotly_dir)


def _show_summary(include_outliers: bool = False):
    from boxing_gym.cli.commands.sync import get_cache_file

    cache_file = get_cache_file()
    console.print("[bold]BoxingGym Results[/bold]\n")

    if cache_file.exists():
        import pandas as pd

        df = pd.read_parquet(cache_file)

        n_total = len(df)
        if "is_outlier" in df.columns:
            n_flagged = int(df["is_outlier"].fillna(False).sum())
            n_valid = n_total - n_flagged
            console.print(
                f"[green]Cache loaded:[/green] {n_total:,} runs ({n_valid:,} valid, {n_flagged:,} flagged)"
            )
            if not include_outliers:
                df = df[~df["is_outlier"].fillna(False)]
        else:
            console.print(f"[green]Cache loaded:[/green] {n_total:,} runs")

        console.print(f"  Models: {df['model'].nunique()}")
        console.print(f"  Environments: {df['env'].nunique()}")
        console.print()
    else:
        console.print("[yellow]No cache. Run: box sync --local results/[/yellow]\n")

    console.print("[bold]Commands:[/bold]")
    console.print("  box results --view model-rankings")
    console.print("  box results --view heatmap")
    console.print("  box results --tui")
    console.print("  box results --web")
