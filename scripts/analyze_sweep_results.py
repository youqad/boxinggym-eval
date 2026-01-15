#!/usr/bin/env python3
"""Analyze WandB sweep results or local JSON results.

Uses Random Forest feature importance (like WandB's Parameter Importance panel)
to understand which hyperparameters most affect performance.

Usage:
    # Analyze WandB sweep with Rich CLI (default)
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID

    # Analyze LOCAL results (from results/ directory)
    uv run python scripts/analyze_sweep_results.py --local
    uv run python scripts/analyze_sweep_results.py --local --results-dir ./my_results

    # Analyze multiple sweeps
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID OTHER_SWEEP_ID

    # Launch interactive TUI dashboard
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID --tui
    uv run python scripts/analyze_sweep_results.py --local --tui

    # Non-interactive TUI (specific views, no prompts - suitable for AI agents)
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID --view model-rankings
    uv run python scripts/analyze_sweep_results.py --local --view local-summary
    uv run python scripts/analyze_sweep_results.py --local --view all

    # Machine-readable output formats (JSON and CSV)
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID --view all --format json
    uv run python scripts/analyze_sweep_results.py --local --view model-rankings --format csv

    # Launch Streamlit web dashboard
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID --web

    # Export to CSV
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID --output results.csv

    # Include seed in parameter importance (legacy per-seed runs only, deprecated for multi-seed)
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID --view parameter-importance --include-seed

    # Filter catastrophic runs (default: |z| > 100 filtered out)
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID --z-threshold 50

    # Exclude specific broken environments
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID --exclude-envs peregrines_direct location_finding_direct

    # Per-environment analysis: parameter importance for a single env
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID --filter-env hyperbolic_direct --view parameter-importance

    # Quality filtering (on by default): exclude incomplete runs (budget < 10, wall_time < 60s, missing z-score)
    # Z-scores at different budgets aren't comparable, so we filter by final budget reached
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID --min-budget 5 --min-wall-time 30

    # Disable all filtering (include everything)
    uv run python scripts/analyze_sweep_results.py --sweep-id YOUR_SWEEP_ID --no-filter

    # Streamlit app directly
    streamlit run scripts/streamlit_app/app.py

Available views for --view:
    parameter-importance  Random Forest feature importance analysis
    model-rankings        Model leaderboard sorted by z_mean
    heatmap               Environment × Model performance matrix
    best-configs          Top 20 configurations with details
    budget-progression    Performance trends across budget values
    local-summary         Local results summary (rankings + per-env top 3)
    all                   All views (default when --view used alone)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

# Rich imports for beautiful CLI output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn

console = Console(width=120)

try:
    import wandb
except ImportError:
    console.print("[red]wandb not installed. Run: uv pip install wandb[/red]")
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    console.print("[yellow]Warning: sklearn not available. Parameter importance will be skipped.[/yellow]")


def fetch_sweep_runs(sweep_id: str, entity: str = "", project: str = "") -> pd.DataFrame:
    """Fetch all runs from a WandB sweep with progress indication."""
    api = wandb.Api()
    sweep_path = f"{entity}/{project}/{sweep_id}"

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Fetching {sweep_path}...", total=100)

        progress.update(task, advance=20, description=f"Connecting to {sweep_path}...")
        sweep = api.sweep(sweep_path)
        runs = list(sweep.runs)
        total_runs = len(runs)

        progress.update(task, advance=10, description=f"Found {total_runs} runs, parsing...")

        data = []
        for i, run in enumerate(runs):
            if run.state != "finished":
                continue

            row = {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
            }

            # Extract config
            config = run.config
            for key, value in config.items():
                if not key.startswith("_"):
                    if isinstance(value, dict):
                        for k, v in value.items():
                            row[f"config/{key}.{k}"] = v
                    else:
                        row[f"config/{key}"] = value

            # Extract summary metrics
            summary = run.summary._json_dict
            for key, value in summary.items():
                if not key.startswith("_") and isinstance(value, (int, float)):
                    row[f"metric/{key}"] = value

            data.append(row)

            # Update progress
            progress.update(task, completed=30 + int(70 * (i + 1) / total_runs))

    console.print(f"  [green]Found {len(data)} finished runs[/green]")
    return pd.DataFrame(data)


def compute_parameter_importance(
    df: pd.DataFrame,
    target_metric: str = "metric/eval/z_mean",
    config_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Compute parameter importance using Random Forest (like WandB does)."""
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()

    if target_metric not in df.columns:
        console.print(f"[yellow]Warning: Target metric '{target_metric}' not found[/yellow]")
        return pd.DataFrame()

    if config_cols is None:
        # Filter to meaningful hyperparameters only (exclude noise)
        noise_patterns = ["results", "ppl/", "ppl.", "hydra", "filename", "system_prompt", "wandb", "_"]
        config_cols = [
            c for c in df.columns
            if c.startswith("config/")
            and not any(noise in c.lower() for noise in noise_patterns)
        ]

    X_data = df[config_cols].copy()
    y = df[target_metric].values

    valid_mask = ~np.isnan(y)
    X_data = X_data[valid_mask]
    y = y[valid_mask]

    if len(y) < 10:
        console.print("[yellow]Warning: Not enough data points for importance analysis[/yellow]")
        return pd.DataFrame()

    encoders = {}
    X_encoded = pd.DataFrame()

    for col in X_data.columns:
        if X_data[col].dtype == object or X_data[col].dtype.name == 'category':
            le = LabelEncoder()
            values = X_data[col].fillna("__NULL__").astype(str)
            X_encoded[col] = le.fit_transform(values)
            encoders[col] = le
        else:
            X_encoded[col] = X_data[col].fillna(0)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_encoded, y)

    importance = pd.DataFrame({
        "parameter": config_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    correlations = []
    for col in config_cols:
        try:
            if X_encoded[col].nunique() > 1 and len(y) > 1:
                corr = np.corrcoef(X_encoded[col].values.astype(float), y.astype(float))[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
        except Exception:
            corr = 0.0
        correlations.append(corr)

    importance["correlation"] = correlations
    return importance


def find_best_configurations(
    df: pd.DataFrame,
    target_metric: str = "metric/eval/z_mean",
    group_by: Optional[list] = None,
    minimize: bool = True,
) -> pd.DataFrame:
    """Find the best configurations, optionally grouped by environment/other factors."""
    if target_metric not in df.columns:
        console.print(f"[yellow]Warning: Target metric '{target_metric}' not found[/yellow]")
        return pd.DataFrame()

    key_configs = [
        "config/llms",
        "config/envs",
        "config/include_prior",
        "config/use_ppl",
        "config/seed",
    ]
    key_configs = [c for c in key_configs if c in df.columns]

    if group_by:
        if minimize:
            idx = df.groupby(group_by)[target_metric].idxmin()
        else:
            idx = df.groupby(group_by)[target_metric].idxmax()
        best = df.loc[idx]
    else:
        best = df.nsmallest(20, target_metric) if minimize else df.nlargest(20, target_metric)

    result_cols = key_configs + [target_metric, "run_id"]
    result_cols = [c for c in result_cols if c in best.columns]

    return best[result_cols].reset_index(drop=True)


def aggregate_by_parameter(
    df: pd.DataFrame,
    parameter: str,
    target_metric: str = "metric/eval/z_mean",
) -> pd.DataFrame:
    """Aggregate metric by a single parameter to see its effect."""
    if parameter not in df.columns or target_metric not in df.columns:
        return pd.DataFrame()

    agg = df.groupby(parameter)[target_metric].agg(["mean", "std", "count"])
    agg = agg.sort_values("mean")
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    return agg


def _z_style(z: float) -> str:
    """Get color style based on z_mean value."""
    if z < -0.5:
        return "bold green"
    elif z > 0.5:
        return "bold red"
    return "yellow"


def print_analysis_report_rich(df: pd.DataFrame, sweep_id: str):
    """Print a comprehensive analysis report with Rich formatting."""
    target = "metric/eval/z_mean"

    # Title panel
    title_panel = Panel(
        f"[bold cyan]Sweep Analysis: {sweep_id}[/bold cyan]\n[dim]{len(df)} finished runs[/dim]",
        border_style="cyan",
        expand=True
    )
    console.print(title_panel)

    if target not in df.columns:
        console.print(f"[red]Warning: Target metric '{target}' not found[/red]")
        console.print(f"[dim]Available metrics: {[c for c in df.columns if c.startswith('metric/')]}[/dim]")
        return

    # 1. Parameter Importance
    console.print("\n[bold cyan]Parameter Importance (Random Forest)[/bold cyan]")
    importance = compute_parameter_importance(df, target)
    if not importance.empty:
        imp_table = Table(border_style="cyan", show_header=True, header_style="magenta")
        imp_table.add_column("Parameter", style="green", width=30)
        imp_table.add_column("Importance", justify="right", width=12)
        imp_table.add_column("Correlation", justify="right", width=12)

        for _, row in importance.head(10).iterrows():
            param = row["parameter"].replace("config/", "")
            imp = row["importance"]
            corr = row["correlation"]
            corr_style = "green" if corr > 0.2 else "red" if corr < -0.2 else "yellow"
            imp_table.add_row(
                param,
                f"{imp:.3f}",
                f"[{corr_style}]{corr:+.3f}[/{corr_style}]"
            )
        console.print(imp_table)

    # 2. Best Configurations
    console.print("\n[bold cyan]Top 10 Configurations (lowest z_mean)[/bold cyan]")
    best = find_best_configurations(df, target, minimize=True)
    if not best.empty:
        best_table = Table(border_style="cyan", show_header=True, header_style="magenta", row_styles=["", "dim"])
        best_table.add_column("#", justify="right", width=3)
        best_table.add_column("Model", width=20)
        best_table.add_column("Environment", width=25)
        best_table.add_column("Prior", justify="center", width=6)
        best_table.add_column("PPL", justify="center", width=5)
        best_table.add_column("z_mean", justify="right", width=10)

        for idx, (_, row) in enumerate(best.head(10).iterrows(), 1):
            z_val = row.get(target, float("nan"))
            z_style = _z_style(z_val)
            prior = "yes" if row.get("config/include_prior") else "no"
            ppl = "yes" if row.get("config/use_ppl") else "no"

            best_table.add_row(
                str(idx),
                str(row.get("config/llms", "?"))[:20],
                str(row.get("config/envs", "?"))[:25],
                prior,
                ppl,
                f"[{z_style}]{z_val:+.3f}[/{z_style}]"
            )
        console.print(best_table)

    # 3. Parameter Effects
    console.print("\n[bold cyan]Parameter Effects[/bold cyan]")
    key_params = ["config/llms", "config/include_prior", "config/use_ppl"]

    for param in key_params:
        if param not in df.columns:
            continue

        agg = aggregate_by_parameter(df, param, target)
        if agg.empty:
            continue

        param_name = param.replace("config/", "").upper()
        console.print(f"\n  [yellow]{param_name}[/yellow]")

        param_table = Table(show_header=True, header_style="dim cyan", border_style="blue", padding=(0, 1))
        param_table.add_column("Value", style="green", width=25)
        param_table.add_column("z_mean", justify="right", width=20)
        param_table.add_column("n", justify="right", width=6)

        for idx, row in agg.iterrows():
            mean_val = row['mean']
            z_style = _z_style(mean_val)
            param_table.add_row(
                str(idx)[:25],
                f"[{z_style}]{mean_val:+.3f} +/- {row['sem']:.3f}[/{z_style}]",
                f"{int(row['count'])}"
            )
        console.print(param_table)

    # 4. Best Model per Environment
    if "config/envs" in df.columns and "config/llms" in df.columns:
        console.print("\n[bold cyan]Best Model per Environment[/bold cyan]")
        best_per_env = find_best_configurations(df, target, group_by=["config/envs"], minimize=True)
        if not best_per_env.empty:
            env_table = Table(border_style="cyan", show_header=True, header_style="magenta")
            env_table.add_column("Environment", style="yellow", width=35)
            env_table.add_column("Best Model", style="green", width=25)
            env_table.add_column("z_mean", justify="right", width=10)

            for _, row in best_per_env.iterrows():
                env = row.get("config/envs", "?")
                model = row.get("config/llms", "?")
                score = row.get(target, float("nan"))
                z_style = _z_style(score)
                env_table.add_row(env, model, f"[{z_style}]{score:+.3f}[/{z_style}]")

            console.print(env_table)

    # 5. Summary Statistics
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column(style="cyan", width=15)
    stats_table.add_column(justify="right", style="bold magenta")

    stats_table.add_row("Mean", f"{df[target].mean():+.3f}")
    stats_table.add_row("Std Dev", f"+/- {df[target].std():.3f}")
    stats_table.add_row("Best", f"[green]{df[target].min():+.3f}[/green]")
    stats_table.add_row("Worst", f"[red]{df[target].max():+.3f}[/red]")

    stats_panel = Panel(stats_table, title="Summary Statistics", border_style="blue")
    console.print(stats_panel)


def prepare_analysis_payload(
    combined_df: pd.DataFrame,
    sweep_ids: List[str],
    target_metric: str = "metric/eval/z_mean",
) -> Dict[str, Any]:
    """Prepare data payload for web dashboard."""
    importance = compute_parameter_importance(combined_df, target_metric)
    best_configs = find_best_configurations(combined_df, target_metric, minimize=True)

    # Aggregate by model
    agg_model = aggregate_by_parameter(combined_df, "config/llms", target_metric)
    agg_prior = aggregate_by_parameter(combined_df, "config/include_prior", target_metric)
    agg_ppl = aggregate_by_parameter(combined_df, "config/use_ppl", target_metric)

    return {
        "runs": combined_df.to_dict("records"),
        "importances": importance.to_dict("records") if not importance.empty else [],
        "best_configs": best_configs.to_dict("records") if not best_configs.empty else [],
        "aggregates": {
            "by_model": agg_model.reset_index().to_dict("records") if not agg_model.empty else [],
            "by_prior": agg_prior.reset_index().to_dict("records") if not agg_prior.empty else [],
            "by_ppl": agg_ppl.reset_index().to_dict("records") if not agg_ppl.empty else [],
        },
        "sweep_ids": sweep_ids,
        "target_metric": target_metric,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze WandB sweep results or local JSON results")
    parser.add_argument("--sweep-id", nargs="+", help="Sweep ID(s) to analyze (required unless --local)")
    parser.add_argument("--local", action="store_true", help="Analyze local results from results/ directory")
    parser.add_argument("--results-dir", default="results", help="Directory containing local JSON results (default: results)")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""), help="WandB entity")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", ""), help="WandB project")
    parser.add_argument("--output", help="Output CSV file path")
    parser.add_argument("--metric", default="metric/eval/z_mean", help="Target metric to analyze")
    parser.add_argument("--web", action="store_true", help="Launch interactive web dashboard")
    parser.add_argument("--tui", action="store_true", help="Launch interactive terminal dashboard")
    parser.add_argument(
        "--view",
        nargs="*",
        metavar="VIEW",
        help="Non-interactive mode: render specific view(s) without prompts. "
             "Views: parameter-importance, model-rankings, heatmap, best-configs, "
             "budget-progression, all. Use alone for 'all'.",
    )
    parser.add_argument(
        "--format",
        choices=["rich", "json", "csv"],
        default="rich",
        help="Output format for --view mode (default: rich). "
             "Use 'json' for JSON output or 'csv' for CSV tables."
    )
    parser.add_argument("--port", type=int, default=5003, help="Port for web dashboard")
    parser.add_argument(
        "--include-seed",
        action="store_true",
        help="[DEPRECATED for multi-seed runs] Include seed in RF parameter importance. "
             "With the new multi-seed format, seeds are handled internally per run and "
             "this flag has no effect. For legacy per-seed runs, seed is excluded by default "
             "because it causes RF to overfit to instance-specific patterns (test-set leakage).",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=100.0,
        help="Filter out runs with |z_mean| > threshold (default: 100). "
             "Catastrophic failures (e.g., peregrines z=6500) corrupt aggregations.",
    )
    parser.add_argument(
        "--exclude-envs",
        nargs="*",
        default=[],
        help="Environments to exclude (e.g., peregrines_direct location_finding_direct)",
    )
    parser.add_argument(
        "--filter-env",
        type=str,
        default=None,
        help="Analyze only this environment. Useful for per-env parameter importance. "
             "Example: --filter-env hyperbolic_direct",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable all filtering (include catastrophic runs and broken envs)",
    )
    parser.add_argument(
        "--min-budget",
        type=int,
        default=10,
        help="Minimum final budget reached (default: 10). "
             "Ensures fair comparison - z-scores at different budgets aren't comparable. "
             "Set to 0 to include incomplete runs.",
    )
    parser.add_argument(
        "--min-wall-time",
        type=float,
        default=60.0,
        help="Minimum wall time in seconds (default: 60). Set to 0 to include crashed runs.",
    )

    args = parser.parse_args()

    # Validate: need either --local or --sweep-id
    if not args.local and not args.sweep_id:
        parser.error("Either --local or --sweep-id is required")

    # Load data based on source
    if args.local:
        # Load local JSON results
        from boxing_gym.cli.tui.loaders import load_local_results

        console.print(f"[cyan]Loading local results from {args.results_dir}/...[/cyan]")
        combined_df = load_local_results(args.results_dir)
        sweep_ids = ["local"]
        is_local = True
    else:
        # Fetch from WandB
        all_dfs = []
        for sweep_id in args.sweep_id:
            df = fetch_sweep_runs(sweep_id, args.entity, args.project)
            df["sweep_id"] = sweep_id
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        sweep_ids = args.sweep_id
        is_local = False

    console.print(f"\n[cyan]Total runs: {len(combined_df)}[/cyan]")

    # Filter catastrophic runs and broken environments (unless --no-filter)
    if not args.no_filter:
        original_count = len(combined_df)

        # Find the metric column (handles both WandB and local formats)
        z_col = None
        for candidate in [args.metric, "metric/eval/z_mean", "z_mean", "summary/z_mean"]:
            if candidate in combined_df.columns:
                z_col = candidate
                break

        # Find the environment column
        env_col = None
        for candidate in ["config/envs", "envs", "config/env", "env"]:
            if candidate in combined_df.columns:
                env_col = candidate
                break

        # Filter by z-score threshold
        if z_col and args.z_threshold > 0:
            before = len(combined_df)
            combined_df = combined_df[combined_df[z_col].abs() <= args.z_threshold]
            filtered = before - len(combined_df)
            if filtered > 0:
                console.print(f"  [yellow]Filtered {filtered} runs with |z| > {args.z_threshold}[/yellow]")

        # Filter excluded environments
        if env_col and args.exclude_envs:
            before = len(combined_df)
            combined_df = combined_df[~combined_df[env_col].isin(args.exclude_envs)]
            filtered = before - len(combined_df)
            if filtered > 0:
                console.print(f"  [yellow]Excluded {filtered} runs from: {', '.join(args.exclude_envs)}[/yellow]")

        # Filter to single environment (for per-env analysis)
        if env_col and args.filter_env:
            before = len(combined_df)
            df_before = combined_df.copy()
            # Try exact match first
            combined_df = combined_df[combined_df[env_col] == args.filter_env]
            if len(combined_df) == 0:
                # Try partial match (e.g., "hyperbolic" matches "hyperbolic_direct")
                mask = df_before[env_col].str.contains(args.filter_env, case=False, na=False)
                combined_df = df_before[mask]
            console.print(f"  [cyan]Filtered to env '{args.filter_env}': {len(combined_df)} runs (was {before})[/cyan]")

        if len(combined_df) < original_count:
            console.print(f"  [green]After filtering: {len(combined_df)} runs[/green]")

        # Quality filtering: exclude incomplete runs (on by default, disable with --no-filter)
        original_strict = len(combined_df)
        strict_stats = {"missing_z": 0, "short_runs": 0, "incomplete": 0}

        # Find metric columns
        z_col = None
        for candidate in [args.metric, "metric/eval/z_mean", "z_mean", "metric/eval/z_mean_final"]:
            if candidate in combined_df.columns:
                z_col = candidate
                break

        wall_time_col = None
        for candidate in ["metric/eval/wall_time_sec", "wall_time_sec", "metric/wall_time"]:
            if candidate in combined_df.columns:
                wall_time_col = candidate
                break

        budget_col = None
        for candidate in ["metric/eval/budget", "eval/budget", "config/budget", "budget"]:
            if candidate in combined_df.columns:
                budget_col = candidate
                break

        # Filter missing z-score
        if z_col:
            before = len(combined_df)
            combined_df = combined_df[combined_df[z_col].notna()]
            strict_stats["missing_z"] = before - len(combined_df)

        # Filter short runs (wall_time < min_wall_time)
        if wall_time_col:
            before = len(combined_df)
            combined_df = combined_df[combined_df[wall_time_col] >= args.min_wall_time]
            strict_stats["short_runs"] = before - len(combined_df)

        # Filter incomplete runs (budget < min_budget)
        if budget_col and args.min_budget > 0:
            before = len(combined_df)
            combined_df = combined_df[combined_df[budget_col] >= args.min_budget]
            strict_stats["incomplete"] = before - len(combined_df)

        total_excluded = original_strict - len(combined_df)
        if total_excluded > 0:
            console.print("\n[bold cyan]╭─ Quality Filtering ─────────────────────────╮[/bold cyan]")
            console.print(f"[cyan]│[/cyan] Total before:           {original_strict:>6}")
            if strict_stats["missing_z"] > 0:
                console.print(f"[cyan]│[/cyan] [yellow]Missing z-score:        {strict_stats['missing_z']:>6}[/yellow]")
            if strict_stats["short_runs"] > 0:
                console.print(f"[cyan]│[/cyan] [yellow]Too short (<{args.min_wall_time}s):      {strict_stats['short_runs']:>6}[/yellow]")
            if strict_stats["incomplete"] > 0:
                console.print(f"[cyan]│[/cyan] [yellow]Incomplete (budget < {args.min_budget}): {strict_stats['incomplete']:>6}[/yellow]")
            console.print(f"[cyan]│[/cyan] [green]✓ Valid for analysis:   {len(combined_df):>6}[/green]")
            console.print("[bold cyan]╰─────────────────────────────────────────────╯[/bold cyan]\n")

    # non-interactive TUI mode (--view)
    if args.view is not None:
        from boxing_gym.cli.tui.app import SweepTUI, AVAILABLE_VIEWS

        # --view with no args means "all"
        view_names = args.view if args.view else ["all"]

        # validate view names
        invalid = [v for v in view_names if v not in AVAILABLE_VIEWS]
        if invalid:
            console.print(f"[red]Unknown view(s): {', '.join(invalid)}[/red]")
            console.print(f"[dim]Available: {', '.join(AVAILABLE_VIEWS)}[/dim]")
            sys.exit(1)

        tui = SweepTUI(combined_df, sweep_ids, args.metric, include_seed=args.include_seed, is_local=is_local)
        tui.run_non_interactive(view_names, output_format=args.format)
        sys.exit(0)

    if args.tui:
        from boxing_gym.cli.tui.app import SweepTUI
        tui = SweepTUI(combined_df, sweep_ids, args.metric, include_seed=args.include_seed, is_local=is_local)
        tui.run()
        sys.exit(0)

    if args.web:
        if is_local:
            console.print("[yellow]Warning: --web mode not yet supported for local results. Use --tui instead.[/yellow]")
            sys.exit(1)

        # Launch Streamlit web dashboard
        import subprocess

        console.print("[cyan]Launching Streamlit dashboard...[/cyan]")

        # Pass config via environment variables
        env = os.environ.copy()
        env["SWEEP_IDS"] = ",".join(sweep_ids)
        env["WANDB_ENTITY"] = args.entity
        env["WANDB_PROJECT"] = args.project
        env["TARGET_METRIC"] = args.metric

        # Locate the Streamlit app
        app_path = Path(__file__).parent / "streamlit_app" / "app.py"
        if not app_path.exists():
            console.print(f"[red]Streamlit app not found at {app_path}[/red]")
            sys.exit(1)

        # Launch Streamlit
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                str(app_path),
                "--server.port", str(args.port),
                "--server.headless", "true",
            ],
            env=env,
        )
    else:
        # CLI output with Rich formatting
        for sweep_id in sweep_ids:
            sweep_df = combined_df[combined_df["sweep_id"] == sweep_id]
            print_analysis_report_rich(sweep_df, sweep_id)

        if len(sweep_ids) > 1:
            console.print("\n")
            print_analysis_report_rich(combined_df, "COMBINED")

    # Export if requested
    if args.output:
        combined_df.to_csv(args.output, index=False)
        console.print(f"\n[green]Results exported to: {args.output}[/green]")


if __name__ == "__main__":
    main()
