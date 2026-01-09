#!/usr/bin/env python3
"""
Visualize W&B sweep benchmark results and compare to paper.

Usage:
    # From local logs
    uv run python scripts/visualize_sweep_results.py                     # Parse from logs/sweep_benchmark/
    uv run python scripts/visualize_sweep_results.py --logs-dir logs/sweep_20241212/

    # From W&B API (recommended)
    uv run python scripts/visualize_sweep_results.py --sweep-id gkl3dj89
    uv run python scripts/visualize_sweep_results.py --sweep-id ox/boxing-gym/gkl3dj89

    # List and compare sweeps
    uv run python scripts/visualize_sweep_results.py --list-sweeps
    uv run python scripts/visualize_sweep_results.py --all-sweeps        # Compare all sweeps
    uv run python scripts/visualize_sweep_results.py --all-sweeps --limit 5

    # Output options
    uv run python scripts/visualize_sweep_results.py --sweep-id gkl3dj89 --plot
    uv run python scripts/visualize_sweep_results.py --sweep-id gkl3dj89 --csv results.csv
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Import from shared module
from boxing_gym.agents.results_io import (
    TEST_ENVIRONMENTS,
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    AggregatedResult,
    aggregate_results,
    get_env_display_name,
    get_model_display_name,
    list_wandb_sweeps as list_sweeps_info,
    load_results_from_logs,
    load_results_from_wandb,
)
from boxing_gym.agents.results_io import PAPER_GPT4O_2D as PAPER_GPT4O


@dataclass
class SweepSummary:
    """Summary of a sweep's results."""

    sweep_id: str
    name: str
    state: str
    n_runs: int
    created_at: str
    avg_z_score: float
    best_model: str
    best_z: float


def list_wandb_sweeps(entity: str = DEFAULT_ENTITY, project: str = DEFAULT_PROJECT, limit: int = 20):
    """List all sweeps in a W&B project."""
    sweeps = list_sweeps_info(entity, project, limit)

    path = f"{entity}/{project}"
    print(f"\nSweeps in {path}:")
    print("=" * 90)
    print(f"{'ID':<12} {'Name':<35} {'State':<12} {'Runs':>6} {'Created':<20}")
    print("-" * 90)

    for s in sweeps:
        print(f"{s['id']:<12} {str(s['name'])[:34]:<35} {s['state']:<12} {s['n_runs']:>6} {s['created_at']:<20}")

    if len(sweeps) >= limit:
        print(f"\n... showing first {limit} sweeps. Use --limit to see more.")

    print("\nUse --sweep-id <id> to visualize a specific sweep")


def compare_all_sweeps(
    entity: str = DEFAULT_ENTITY, project: str = DEFAULT_PROJECT, limit: int = 10, budget: int = 5
) -> List[SweepSummary]:
    """Compare results across all sweeps."""
    try:
        import wandb
    except ImportError:
        print("wandb package required. Install with: uv add wandb", file=sys.stderr)
        sys.exit(1)

    api = wandb.Api()
    path = f"{entity}/{project}"

    print(f"\nComparing sweeps in {path} (Budget {budget}):")
    print("=" * 110)

    summaries: List[SweepSummary] = []

    try:
        # Get unique sweep IDs from runs with sweeps
        runs_with_sweeps = api.runs(path, filters={"sweep": {"$ne": None}}, per_page=500)

        sweep_ids = {}
        for run in runs_with_sweeps:
            if run.sweep and run.sweep.id not in sweep_ids:
                sweep_ids[run.sweep.id] = run.sweep

        for sweep_id, sweep in list(sweep_ids.items())[:limit]:
            try:
                full_sweep = api.sweep(f"{path}/{sweep_id}")
                runs = list(full_sweep.runs)
            except Exception:
                continue

            if not runs:
                continue

            results = load_results_from_wandb(sweep_id, entity, project)
            if not results:
                continue

            agg = aggregate_results(results, group_by=("env", "model", "budget"))
            model_scores = defaultdict(list)
            for key, r in agg.items():
                try:
                    b = int(getattr(r, "budget", -1))
                except Exception:
                    continue
                if b != budget:
                    continue
                try:
                    z = float(getattr(r, "z_mean", 0.0))
                except Exception:
                    continue
                if abs(z) >= 10:
                    continue
                model_scores[r.model].append(z)

            if not model_scores:
                continue

            model_avgs = {m: sum(s) / len(s) for m, s in model_scores.items() if s}
            best_model = min(model_avgs, key=model_avgs.get) if model_avgs else "N/A"
            best_z = model_avgs.get(best_model, 0.0) if model_avgs else 0.0
            overall_avg = (
                sum(sum(s) for s in model_scores.values()) / sum(len(s) for s in model_scores.values())
                if model_scores
                else 0.0
            )

            name = sweep.name if hasattr(sweep, "name") else sweep_id
            state = sweep.state if hasattr(sweep, "state") else "unknown"
            # Handle created_at as string or datetime
            created_raw = getattr(sweep, "created_at", None)
            if created_raw:
                created = (
                    str(created_raw)[:10]
                    if isinstance(created_raw, str)
                    else created_raw.strftime("%Y-%m-%d")
                    if hasattr(created_raw, "strftime")
                    else "N/A"
                )
            else:
                created = "N/A"

            summaries.append(
                SweepSummary(
                    sweep_id=sweep_id,
                    name=str(name)[:40],
                    state=state,
                    n_runs=len(runs),
                    created_at=created,
                    avg_z_score=overall_avg,
                    best_model=get_model_display_name(best_model),
                    best_z=best_z,
                )
            )

    except Exception as e:
        print(f"Error comparing sweeps: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return []

    # Print comparison table
    print(f"{'ID':<12} {'Name':<40} {'Runs':>5} {'Avg Z':>8} {'Best Model':<25} {'Best Z':>8}")
    print("-" * 110)

    # Sort by avg z-score (best first)
    summaries.sort(key=lambda s: s.avg_z_score)

    for i, s in enumerate(summaries):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(
            f"{medal}{s.sweep_id:<10} {s.name:<40} {s.n_runs:>5} {s.avg_z_score:>8.3f} {s.best_model:<25} {s.best_z:>8.3f}"
        )

    return summaries


def print_table_header(title: str, width: int = 100):
    """Print a formatted table header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_results_table(aggregated: Dict, budget: int):
    """Print a results table for a specific budget."""
    models = sorted(set(k[1] for k in aggregated.keys()))
    # Use canonical env names (normalized, without _direct suffix)
    envs = TEST_ENVIRONMENTS

    # Header
    print(f"\n{'Model':<28} ", end="")
    for env in envs:
        env_name = get_env_display_name(env)
        print(f"{env_name:>18} ", end="")
    print(f"{'Avg':>12}")
    print("-" * 90)

    # Rows with ranking
    model_avgs = []
    for model in models:
        scores = []
        row = f"{get_model_display_name(model):<28} "
        for env in envs:
            key = (env, model, budget)
            if key in aggregated:
                r = aggregated[key]
                score_str = f"{r.z_mean:>7.3f} ±{r.z_std:>5.2f}"
                scores.append(r.z_mean)
            else:
                score_str = f"{'N/A':>14}"
            row += f"{score_str:>18} "

        if scores:
            avg = sum(scores) / len(scores)
            row += f"{avg:>12.3f}"
            model_avgs.append((avg, model, row))
        else:
            row += f"{'N/A':>12}"
            model_avgs.append((float("inf"), model, row))

    # Sort by average and print with ranking
    model_avgs.sort()
    for i, (avg, model, row) in enumerate(model_avgs):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{medal} {row}")


def print_comparison_to_paper(aggregated: Dict, budget: int):
    """Print comparison of our results to paper's GPT-4o results."""
    print_table_header(f"COMPARISON TO PAPER (Budget {budget})", 95)

    models = sorted(set(k[1] for k in aggregated.keys()))
    # Use canonical env names (normalized, without _direct suffix)
    envs = TEST_ENVIRONMENTS

    # Header
    print(f"\n{'Model':<28} ", end="")
    for env in envs:
        env_name = get_env_display_name(env)
        print(f"{env_name:>14} ", end="")
    print()
    print("-" * 75)

    # Paper baseline row
    row = f"{'Paper GPT-4o':<28} "
    for env in envs:
        env_name = get_env_display_name(env)
        paper_key = (env_name, budget)
        if paper_key in PAPER_GPT4O:
            row += f"{PAPER_GPT4O[paper_key]:>14.3f} "
        else:
            row += f"{'N/A':>14} "
    print(f"📄 {row}")
    print("-" * 75)

    # Our results with delta
    for model in models:
        row = f"{get_model_display_name(model):<28} "
        deltas = []
        for env in envs:
            key = (env, model, budget)
            env_name = get_env_display_name(env)
            paper_key = (env_name, budget)

            if key in aggregated and paper_key in PAPER_GPT4O:
                our_z = aggregated[key].z_mean
                paper_z = PAPER_GPT4O[paper_key]
                delta = our_z - paper_z
                deltas.append(delta)

                # Color coding: negative delta (better) = green, positive = red
                if delta < -0.1:
                    symbol = "✓"  # significantly better
                elif delta > 0.1:
                    symbol = "✗"  # significantly worse
                else:
                    symbol = "≈"  # similar
                row += f"{delta:>+10.3f} {symbol:>2} "
            elif key in aggregated:
                row += f"{aggregated[key].z_mean:>10.3f} -- "
            else:
                row += f"{'N/A':>14} "

        # Summary
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            if avg_delta < -0.1:
                verdict = "BETTER"
            elif avg_delta > 0.1:
                verdict = "WORSE"
            else:
                verdict = "SIMILAR"
            print(f"   {row} [{verdict}]")
        else:
            print(f"   {row}")


def print_learning_improvement(aggregated: Dict):
    """Print improvement from budget 0 to budget 5."""
    print_table_header("LEARNING IMPROVEMENT (Budget 0 → 5)", 80)
    print("(More negative = learned more from experiments)\n")

    models = sorted(set(k[1] for k in aggregated.keys()))
    # Use canonical env names (normalized, without _direct suffix)
    envs = TEST_ENVIRONMENTS

    improvements = []
    for model in models:
        model_deltas = []
        for env in envs:
            key_0 = (env, model, 0)
            key_5 = (env, model, 5)
            if key_0 in aggregated and key_5 in aggregated:
                delta = aggregated[key_5].z_mean - aggregated[key_0].z_mean
                model_deltas.append(delta)

        if model_deltas:
            avg_improvement = sum(model_deltas) / len(model_deltas)
            improvements.append((avg_improvement, model, model_deltas))

    # Sort by improvement (most negative = best learner)
    improvements.sort()

    print(f"{'Model':<28} {'Avg Δ':>12} {'Dugongs':>12} {'Peregrines':>12} {'Lotka-V':>12}")
    print("-" * 80)

    for i, (avg, model, deltas) in enumerate(improvements):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        row = f"{medal} {get_model_display_name(model):<26} {avg:>12.3f}"
        for d in deltas:
            row += f" {d:>12.3f}"
        print(row)


def print_overall_ranking(aggregated: Dict, budget: int = 5):
    """Print overall model ranking."""
    print_table_header(f"OVERALL MODEL RANKING (Budget {budget})", 60)
    print("(Lower Z-score = better prediction accuracy)\n")

    models = sorted(set(k[1] for k in aggregated.keys()))
    # Use canonical env names (normalized, without _direct suffix)
    envs = TEST_ENVIRONMENTS

    rankings = []
    for model in models:
        scores = []
        n_runs = 0
        for env in envs:
            key = (env, model, budget)
            if key in aggregated:
                scores.append(aggregated[key].z_mean)
                n_runs += aggregated[key].n_seeds

        if scores:
            avg = sum(scores) / len(scores)
            rankings.append((avg, model, n_runs))

    rankings.sort()

    print(f"{'Rank':<6} {'Model':<28} {'Avg Z-Score':>14} {'Total Runs':>12}")
    print("-" * 62)

    for i, (avg, model, n_runs) in enumerate(rankings):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{medal} {i+1:<4} {get_model_display_name(model):<28} {avg:>14.4f} {n_runs:>12}")


def print_insights(aggregated: Dict, budget: int = 5):
    """Print dynamically computed insights from the benchmark."""
    print_table_header("KEY INSIGHTS", 80)

    models = sorted(set(k[1] for k in aggregated.keys()))
    # Use canonical env names (normalized, without _direct suffix)
    envs = TEST_ENVIRONMENTS

    # Compute model scores at specified budget
    model_scores = {}
    for model in models:
        scores = []
        for env in envs:
            key = (env, model, budget)
            if key in aggregated:
                scores.append(aggregated[key].z_mean)
        if scores:
            model_scores[model] = sum(scores) / len(scores)

    if not model_scores:
        print("\nInsufficient data to compute insights.\n")
        return

    best_model = min(model_scores, key=model_scores.get)
    worst_model = max(model_scores, key=model_scores.get)
    best_z = model_scores[best_model]
    worst_z = model_scores[worst_model]

    # 1. Best performer
    paper_avg = sum(PAPER_GPT4O.get((get_env_display_name(e), budget), 0) for e in envs) / len(envs)
    vs_paper = (
        "better than"
        if best_z < paper_avg
        else "similar to"
        if abs(best_z - paper_avg) < 0.1
        else "worse than"
    )
    print(
        f"""
1. BEST OVERALL: {get_model_display_name(best_model)}
   - Average Z-score at Budget {budget}: {best_z:.3f}
   - Performance {vs_paper} paper's GPT-4o baseline ({paper_avg:.3f})"""
    )

    # 2. Worst performer
    worst_reason = ""
    if (
        "speciale" in worst_model.lower()
        or "reasoner" in worst_model.lower()
        or "thinking" in worst_model.lower()
    ):
        worst_reason = "Reasoning/thinking models may struggle with this task"
    elif worst_z > 0.5:
        worst_reason = "Significant underperformance - may have formatting issues"
    else:
        worst_reason = "Moderate performance gap"
    print(
        f"""
2. WORST PERFORMER: {get_model_display_name(worst_model)}
   - Average Z-score at Budget {budget}: {worst_z:.3f}
   - {worst_reason}"""
    )

    # 3. Reasoning model comparison (if applicable)
    reasoning_models = [
        m for m in models if any(x in m.lower() for x in ["speciale", "reasoner", "thinking"])
    ]
    chat_models = [m for m in models if m not in reasoning_models and m in model_scores]

    if reasoning_models and chat_models:
        reasoning_avg = sum(model_scores.get(m, 0) for m in reasoning_models if m in model_scores) / len(
            [m for m in reasoning_models if m in model_scores]
        )
        chat_avg = sum(model_scores.get(m, 0) for m in chat_models) / len(chat_models)

        if reasoning_avg > chat_avg + 0.2:
            print(
                f"""
3. REASONING vs CHAT MODELS:
   - Reasoning models avg: {reasoning_avg:.3f}
   - Chat models avg: {chat_avg:.3f}
   - Finding: Chain-of-thought reasoning appears to HURT performance on this task"""
            )
        elif chat_avg > reasoning_avg + 0.2:
            print(
                f"""
3. REASONING vs CHAT MODELS:
   - Reasoning models avg: {reasoning_avg:.3f}
   - Chat models avg: {chat_avg:.3f}
   - Finding: Reasoning models OUTPERFORM chat models on this task"""
            )
        else:
            print(
                f"""
3. REASONING vs CHAT MODELS:
   - Reasoning models avg: {reasoning_avg:.3f}
   - Chat models avg: {chat_avg:.3f}
   - Finding: No significant difference between reasoning and chat models"""
            )

    # 4. Paper comparison summary
    models_vs_paper = {"better": 0, "similar": 0, "worse": 0}
    for model, score in model_scores.items():
        if score < paper_avg - 0.1:
            models_vs_paper["better"] += 1
        elif score > paper_avg + 0.1:
            models_vs_paper["worse"] += 1
        else:
            models_vs_paper["similar"] += 1

    print(
        f"""
4. PAPER COMPARISON SUMMARY:
   - Models better than paper: {models_vs_paper['better']}
   - Models similar to paper: {models_vs_paper['similar']}
   - Models worse than paper: {models_vs_paper['worse']}
"""
    )


def export_csv(aggregated: Dict, output_path: Path):
    """Export results to CSV."""
    import csv

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["env", "model", "budget", "z_mean", "z_std", "n_seeds", "paper_gpt4o", "delta_vs_paper"]
        )

        for key, r in sorted(aggregated.items()):
            env_name = get_env_display_name(r.env)
            paper_key = (env_name, r.budget)
            paper_val = PAPER_GPT4O.get(paper_key)
            delta = r.z_mean - paper_val if paper_val is not None else None

            writer.writerow(
                [
                    env_name,
                    get_model_display_name(r.model),
                    r.budget,
                    f"{r.z_mean:.4f}",
                    f"{r.z_std:.4f}",
                    r.n_seeds,
                    f"{paper_val:.4f}" if paper_val else "",
                    f"{delta:.4f}" if delta is not None else "",
                ]
            )

    print(f"\nExported results to: {output_path}")


def generate_plots(aggregated: Dict, output_dir: Path):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for plots. Install with:")
        print("  uv add matplotlib seaborn")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    # Prepare data for plotting
    import pandas as pd

    rows = []
    for key, r in aggregated.items():
        env_name = get_env_display_name(r.env)
        rows.append(
            {
                "env": env_name,
                "model": get_model_display_name(r.model),
                "budget": r.budget,
                "z_mean": r.z_mean,
                "z_std": r.z_std,
            }
        )

    df = pd.DataFrame(rows)

    # Plot 1: Bar chart comparison at Budget 5
    fig, ax = plt.subplots(figsize=(14, 7))
    df_b5 = df[df["budget"] == 5]

    env_order = TEST_ENVIRONMENTS

    sns.barplot(data=df_b5, x="env", y="z_mean", hue="model", order=env_order, ax=ax, errorbar=None)

    # Add paper reference lines
    for i, env in enumerate(env_order):
        paper_val = PAPER_GPT4O.get((env, 5))
        if paper_val:
            ax.axhline(
                y=paper_val, xmin=(i) / 3, xmax=(i + 1) / 3, color="red", linestyle="--", linewidth=2, alpha=0.7
            )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_title("Model Performance at Budget 5 (vs Paper GPT-4o baseline)")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Standardized Error (Z-score, lower is better)")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(output_dir / "budget5_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "budget5_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Learning curves (Budget 0 vs 5)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i, env in enumerate(env_order):
        ax = axes[i]
        df_env = df[df["env"] == env]

        sns.barplot(data=df_env, x="budget", y="z_mean", hue="model", ax=ax, errorbar=None)
        ax.set_title(env.replace("_", " ").title())
        ax.set_xlabel("Budget (# experiments)")
        if i == 0:
            ax.set_ylabel("Z-score (lower = better)")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        if i < 2:
            ax.get_legend().remove()

    axes[-1].legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    fig.suptitle("Learning from Experiments (Budget 0 → 5)", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize W&B sweep benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local logs
  %(prog)s                                    # Parse from default logs/sweep_benchmark/
  %(prog)s --logs-dir logs/sweep_20241212/    # Custom logs directory

  # W&B API (recommended)
  %(prog)s --sweep-id gkl3dj89                # Fetch specific sweep
  %(prog)s --sweep-id ox/boxing-gym/gkl3dj89  # Full sweep path

  # List and compare sweeps
  %(prog)s --list-sweeps                      # List all sweeps
  %(prog)s --all-sweeps                       # Compare all sweeps
  %(prog)s --all-sweeps --limit 5             # Compare top 5 sweeps

  # Output options
  %(prog)s --sweep-id gkl3dj89 --plot         # Generate plots
  %(prog)s --sweep-id gkl3dj89 --csv out.csv  # Export to CSV
        """,
    )

    # Data source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--sweep-id", type=str, help="W&B sweep ID (e.g., 'gkl3dj89' or 'entity/project/sweep_id')"
    )
    source_group.add_argument("--logs-dir", type=str, help="Directory containing agent_*.log files")
    source_group.add_argument("--list-sweeps", action="store_true", help="List all sweeps in the W&B project")
    source_group.add_argument("--all-sweeps", action="store_true", help="Compare results across all sweeps")

    # W&B options
    parser.add_argument(
        "--entity", type=str, default=DEFAULT_ENTITY, help=f"W&B entity (default: {DEFAULT_ENTITY})"
    )
    parser.add_argument(
        "--project", type=str, default=DEFAULT_PROJECT, help=f"W&B project (default: {DEFAULT_PROJECT})"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Max number of sweeps to list/compare (default: 20)"
    )

    # Analysis options
    parser.add_argument("--budget", type=int, default=5, help="Primary budget to analyze (default: 5)")
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=10.0,
        help="Z-score threshold for outlier filtering (default: 10.0)",
    )

    # Output options
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots")
    parser.add_argument("--csv", type=str, help="Export results to CSV file")

    args = parser.parse_args()

    # Handle list-sweeps
    if args.list_sweeps:
        list_wandb_sweeps(args.entity, args.project, args.limit)
        return

    # Handle all-sweeps comparison
    if args.all_sweeps:
        compare_all_sweeps(args.entity, args.project, args.limit, args.budget)
        return

    # Determine data source and parse results
    if args.sweep_id:
        # Fetch from W&B API
        print(f"Fetching results from W&B sweep: {args.sweep_id}")
        results = load_results_from_wandb(args.sweep_id, args.entity, args.project)
    else:
        # Parse from local logs
        logs_dir = Path(args.logs_dir if args.logs_dir else "logs/sweep_benchmark")
        if not logs_dir.exists():
            print(f"Error: Logs directory not found: {logs_dir}", file=sys.stderr)
            print(
                "Use --sweep-id to fetch from W&B or --list-sweeps to see available sweeps.", file=sys.stderr
            )
            sys.exit(1)

        print(f"Parsing results from: {logs_dir}")
        results = load_results_from_logs(str(logs_dir))

    print(f"Found {len(results)} individual run results")

    if not results:
        print("No results found. Check that the logs contain 'Budget X: z=...' lines.")
        sys.exit(1)

    # Aggregate - use simple (env, model, budget) grouping for CLI tables
    # This merges any goal/include_prior variants, which is fine for CLI comparison
    agg_result = aggregate_results(
        results,
        group_by=("env", "model", "budget"),
        outlier_threshold=args.outlier_threshold,
    )
    # Convert keys from tuples to (env, model, budget) format
    aggregated: Dict[Tuple[str, str, int], AggregatedResult] = {}
    for key, r in agg_result.items():
        new_key = (r.env, r.model, r.budget)
        aggregated[new_key] = r

    print(f"Aggregated into {len(aggregated)} (env, model, budget) combinations")

    # Print tables
    print_table_header(f"RESULTS BY ENVIRONMENT (Budget {args.budget})", 95)
    print_results_table(aggregated, args.budget)

    if 0 in set(k[2] for k in aggregated.keys()):
        print_table_header("RESULTS BY ENVIRONMENT (Budget 0)", 95)
        print_results_table(aggregated, 0)

    print_comparison_to_paper(aggregated, args.budget)
    print_learning_improvement(aggregated)
    print_overall_ranking(aggregated, args.budget)
    print_insights(aggregated, args.budget)

    # Export CSV
    if args.csv:
        export_csv(aggregated, Path(args.csv))

    # Generate plots
    if args.plot:
        generate_plots(aggregated, Path("outputs/sweep_plots"))


if __name__ == "__main__":
    main()
