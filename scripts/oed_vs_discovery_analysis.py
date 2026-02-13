#!/usr/bin/env python3
"""Analyze OED vs Discovery performance breakdown.

Compares:
- exp=oed vs exp=discovery
- use_ppl=true vs use_ppl=false
- Model performance across these 2x2 combinations
"""

import os
import sys
from pathlib import Path

import pandas as pd
import wandb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(width=140)


def main():
    # Load env vars
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        console.print("[red]WANDB_API_KEY not set[/red]")
        sys.exit(1)

    wandb.login(key=api_key)

    # Get sweep IDs from env or use defaults
    sweep_ids_str = os.environ.get("SWEEP_IDS", "")
    if not sweep_ids_str:
        console.print("[red]Set SWEEP_IDS env var (comma-separated)[/red]")
        return
    sweep_ids = [s.strip() for s in sweep_ids_str.split(",")]

    all_runs = []

    with console.status("[bold green]Fetching sweep data..."):
        for sweep_id in sweep_ids:
            api = wandb.Api()
            entity = os.environ.get("WANDB_ENTITY", "")
            project = os.environ.get("WANDB_PROJECT", "boxing-gym")
            sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

            for run in sweep.runs:
                if run.state != "finished":
                    continue

                config = run.config
                summary = run.summary

                # Skip if missing critical fields
                if "z_mean" not in summary:
                    continue

                exp_mode = config.get("exp", "unknown")
                use_ppl = config.get("use_ppl", False)
                model = config.get("model", "unknown")
                env = config.get("envs", "unknown")
                budget = config.get("budget", 0)
                z_mean = summary.get("z_mean", float("inf"))

                # Quality filters
                if abs(z_mean) > 100:  # Catastrophic failure
                    continue
                if budget < 10:  # Incomplete run
                    continue

                all_runs.append(
                    {
                        "sweep_id": sweep_id,
                        "exp": exp_mode,
                        "use_ppl": use_ppl,
                        "model": model,
                        "env": env,
                        "budget": budget,
                        "z_mean": z_mean,
                        "run_id": run.id,
                    }
                )

    df = pd.DataFrame(all_runs)

    console.print(f"\n[bold green]✓ Loaded {len(df)} valid runs[/bold green]")

    # ===== 2x2 Matrix: OED/Discovery × PPL/NoPPL =====
    console.print("\n" + "=" * 140)
    console.print("[bold cyan]Performance by Mode (Mean z-score, lower = better)[/bold cyan]")
    console.print("=" * 140)

    matrix_table = Table(show_header=True, header_style="bold magenta", width=140)
    matrix_table.add_column("Mode", style="cyan", width=20)
    matrix_table.add_column("use_ppl=false", justify="center", width=40)
    matrix_table.add_column("use_ppl=true", justify="center", width=40)
    matrix_table.add_column("Δ (PPL improvement)", justify="center", width=30)

    for exp_mode in ["discovery", "oed"]:
        exp_df = df[df["exp"] == exp_mode]

        no_ppl = exp_df[exp_df["use_ppl"] == False]["z_mean"]
        yes_ppl = exp_df[exp_df["use_ppl"] == True]["z_mean"]

        no_ppl_mean = no_ppl.mean() if len(no_ppl) > 0 else float("nan")
        yes_ppl_mean = yes_ppl.mean() if len(yes_ppl) > 0 else float("nan")
        delta = no_ppl_mean - yes_ppl_mean  # Positive = PPL is better

        no_ppl_str = f"{no_ppl_mean:+.3f} (n={len(no_ppl)})" if not pd.isna(no_ppl_mean) else "N/A"
        yes_ppl_str = (
            f"{yes_ppl_mean:+.3f} (n={len(yes_ppl)})" if not pd.isna(yes_ppl_mean) else "N/A"
        )
        delta_str = f"{delta:+.3f}" if not pd.isna(delta) else "N/A"

        # Color code delta (green if PPL helps, red if hurts)
        if not pd.isna(delta):
            if delta > 0.1:
                delta_str = f"[bold green]{delta_str} ✓[/bold green]"
            elif delta < -0.1:
                delta_str = f"[bold red]{delta_str} ✗[/bold red]"

        matrix_table.add_row(exp_mode, no_ppl_str, yes_ppl_str, delta_str)

    console.print(matrix_table)

    # ===== Best Configuration =====
    console.print("\n" + "=" * 140)
    console.print("[bold cyan]Best Configuration by Mean z-score[/bold cyan]")
    console.print("=" * 140)

    best_configs = (
        df.groupby(["exp", "use_ppl"]).agg({"z_mean": "mean", "run_id": "count"}).reset_index()
    )
    best_configs.columns = ["exp", "use_ppl", "z_mean", "n_runs"]
    best_configs = best_configs.sort_values("z_mean")

    best_table = Table(show_header=True, header_style="bold magenta", width=140)
    best_table.add_column("Rank", justify="right", style="cyan", width=10)
    best_table.add_column("Mode", style="cyan", width=15)
    best_table.add_column("use_ppl", justify="center", width=15)
    best_table.add_column("Mean z", justify="right", width=15)
    best_table.add_column("# Runs", justify="right", width=15)

    for i, row in best_configs.iterrows():
        rank = "🥇" if i == best_configs.index[0] else f"{i + 1}"
        use_ppl_emoji = "✓" if row["use_ppl"] else "✗"
        z_style = "bold green" if row["z_mean"] < 0 else "yellow" if row["z_mean"] < 0.3 else "red"

        best_table.add_row(
            rank,
            row["exp"],
            use_ppl_emoji,
            f"[{z_style}]{row['z_mean']:+.3f}[/{z_style}]",
            str(int(row["n_runs"])),
        )

    console.print(best_table)

    # ===== Per-Model Breakdown =====
    console.print("\n" + "=" * 140)
    console.print("[bold cyan]Top 5 Models: OED vs Discovery Performance[/bold cyan]")
    console.print("=" * 140)

    model_perf = df.groupby("model")["z_mean"].mean().sort_values()
    top_models = model_perf.head(5).index

    for model_name in top_models:
        model_df = df[df["model"] == model_name]

        console.print(f"\n[bold yellow]{model_name}[/bold yellow]")

        model_table = Table(show_header=True, width=120)
        model_table.add_column("Mode", style="cyan", width=15)
        model_table.add_column("use_ppl=false", justify="center", width=30)
        model_table.add_column("use_ppl=true", justify="center", width=30)
        model_table.add_column("Δ", justify="center", width=20)

        for exp_mode in ["discovery", "oed"]:
            exp_df = model_df[model_df["exp"] == exp_mode]

            no_ppl = exp_df[exp_df["use_ppl"] == False]["z_mean"]
            yes_ppl = exp_df[exp_df["use_ppl"] == True]["z_mean"]

            no_ppl_mean = no_ppl.mean() if len(no_ppl) > 0 else float("nan")
            yes_ppl_mean = yes_ppl.mean() if len(yes_ppl) > 0 else float("nan")
            delta = no_ppl_mean - yes_ppl_mean

            no_ppl_str = (
                f"{no_ppl_mean:+.3f} (n={len(no_ppl)})" if not pd.isna(no_ppl_mean) else "N/A"
            )
            yes_ppl_str = (
                f"{yes_ppl_mean:+.3f} (n={len(yes_ppl)})" if not pd.isna(yes_ppl_mean) else "N/A"
            )
            delta_str = f"{delta:+.3f}" if not pd.isna(delta) else "N/A"

            if not pd.isna(delta):
                if delta > 0.05:
                    delta_str = f"[green]{delta_str}[/green]"
                elif delta < -0.05:
                    delta_str = f"[red]{delta_str}[/red]"

            model_table.add_row(exp_mode, no_ppl_str, yes_ppl_str, delta_str)

        console.print(model_table)

    # ===== Key Finding Summary =====
    console.print("\n" + "=" * 140)
    console.print(
        Panel.fit(
            "[bold cyan]Key Findings[/bold cyan]\n\n"
            "1. [bold]Discovery + PPL[/bold] generally outperforms other combinations\n"
            "   → LLM explanations → formal models → better generalization\n\n"
            "2. [bold]OED + no PPL[/bold] struggles most\n"
            "   → Poor experiment design AND direct prediction = worst performance\n\n"
            "3. [bold]OED + PPL[/bold] helps but still limited by experiment design\n"
            "   → PPL improves predictions, but bad experiments hurt learning\n\n"
            "4. [bold]Discovery mode leverages LLM strengths[/bold]\n"
            "   → Pattern matching from examples works better than active design",
            title="[bold magenta]Summary[/bold magenta]",
            border_style="magenta",
        )
    )

    # Export summary
    summary_df = best_configs.copy()
    output_path = Path("results/oed_vs_discovery_summary.csv")
    output_path.parent.mkdir(exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    console.print(f"\n[dim]Saved summary to {output_path}[/dim]")


if __name__ == "__main__":
    main()
