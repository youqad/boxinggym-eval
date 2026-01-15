"""Sync command - cache WandB/local results to Parquet."""

import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

# resolve cache path relative to cwd at import time, but also support override
_CACHE_DIR_NAME = ".boxing-gym-cache"

# threshold for flagging numeric outliers (z-scores this extreme indicate pipeline failures)
Z_OUTLIER_THRESHOLD = 100


def get_cache_dir() -> Path:
    """Get cache directory, resolving to cwd."""
    return Path(os.getcwd()) / _CACHE_DIR_NAME


def get_cache_file() -> Path:
    """Get cache file path."""
    return get_cache_dir() / "runs.parquet"


# for backwards compatibility - but prefer the functions
CACHE_DIR = Path(_CACHE_DIR_NAME)
CACHE_FILE = CACHE_DIR / "runs.parquet"


def run_sync(local: str | None, sweep_id: str | None, refresh: bool, status: bool):
    """Sync results to local cache."""
    if status:
        _show_status()
        return

    if not local and not sweep_id:
        console.print("[yellow]Specify --local or --sweep-id[/yellow]")
        console.print("  box sync --local results/")
        console.print("  box sync --sweep-id 5h3g5cno")
        return

    if local:
        _sync_local(Path(local), refresh)
    elif sweep_id:
        sweep_ids = [s.strip() for s in sweep_id.split(",")]
        _sync_wandb(sweep_ids, refresh)


def _show_status():
    """Show cache status."""
    if not CACHE_FILE.exists():
        console.print("[yellow]No cache found.[/yellow]")
        console.print("Run: box sync --local results/")
        return

    import pandas as pd
    df = pd.read_parquet(CACHE_FILE)

    # calculate flagged outliers if column exists
    flagged_line = ""
    if "is_outlier" in df.columns:
        n_flagged = int(df["is_outlier"].fillna(False).sum())
        n_valid = len(df) - n_flagged
        flagged_line = f"[bold]Valid runs:[/bold] {n_valid:,} ({n_flagged:,} flagged as outliers)\n"

    console.print(Panel.fit(
        f"[bold]Cached runs:[/bold] {len(df):,}\n"
        f"{flagged_line}"
        f"[bold]Models:[/bold] {df['model'].nunique()}\n"
        f"[bold]Environments:[/bold] {df['env'].nunique()}\n"
        f"[bold]Cache file:[/bold] {CACHE_FILE}\n"
        f"[bold]Size:[/bold] {CACHE_FILE.stat().st_size / 1024:.1f} KB",
        title="Cache Status",
    ))


def _sync_local(local_dir: Path, refresh: bool):
    """Sync from local JSON results directory."""
    import sys
    # add src to path for imports
    src_path = Path(__file__).parent.parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from boxing_gym.agents.results_io import load_results_from_json_dir

    console.print(f"[cyan]Loading from {local_dir}...[/cyan]")

    results = load_results_from_json_dir(str(local_dir))

    if not results:
        console.print("[red]No results found.[/red]")
        return

    # convert to dataframe
    import pandas as pd
    from datetime import datetime

    rows = []
    for r in results:
        rows.append({
            "run_id": r.path or "unknown",
            "sweep_id": "local",
            "model": r.model,
            "env": r.env,
            "seed": r.seed,
            "budget": r.budget,
            "z_mean": r.z_mean,
            "z_std": r.z_std,
            "raw_mean": r.raw_mean,
            "raw_std": r.raw_std,
            "experiment_type": r.experiment_type,
            "use_ppl": r.use_ppl,
            "include_prior": r.include_prior,
            "goal": r.goal,
            "synced_at": datetime.now().isoformat(),
        })

    df = pd.DataFrame(rows)

    # flag outliers (do NOT drop - preserve for debugging)
    # NaN <= 100 is False in pandas, so explicit check is needed
    df["z_mean"] = pd.to_numeric(df["z_mean"], errors="coerce")
    is_nan = df["z_mean"].isna()
    is_numeric_outlier = df["z_mean"].abs() > Z_OUTLIER_THRESHOLD
    df["is_outlier"] = is_nan | is_numeric_outlier

    n_numeric = int(is_numeric_outlier.sum())
    n_nan = int(is_nan.sum())

    # save ALL rows (flagged, not filtered)
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_parquet(CACHE_FILE, index=False)

    console.print(f"[green]Cached {len(df):,} runs to {CACHE_FILE}[/green]")
    if n_numeric > 0 or n_nan > 0:
        console.print(f"[dim]Flagged {n_numeric:,} numeric outliers (|z| > {Z_OUTLIER_THRESHOLD}), {n_nan:,} NaNs[/dim]")


def _sync_wandb(sweep_ids: list[str], refresh: bool):
    """Sync from WandB API."""
    console.print("[yellow]WandB sync not yet implemented. Use --local for now.[/yellow]")
    # TODO: implement WandB sync
