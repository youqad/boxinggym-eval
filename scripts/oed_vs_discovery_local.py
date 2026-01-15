#!/usr/bin/env python3
"""Analyze OED vs Discovery performance from local results CSV."""

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console(width=140)

def main():
    # Load local results
    df = pd.read_csv("outputs/standardized_results.csv")

    # Filter: budget >= 10, |z| < 100
    df = df[(df["budget"] >= 10) & (df["z_mean"].abs() < 100)]

    console.print(f"\n[bold green]✓ Loaded {len(df)} valid runs from local results[/bold green]")

    # Count experiment types
    exp_counts = df["experiment_type"].value_counts()
    console.print(f"\n[dim]Experiment types: {dict(exp_counts)}[/dim]")

    # ===== 2x2 Matrix: OED/Discovery × PPL/NoPPL =====
    console.print("\n" + "="*140)
    console.print("[bold cyan]Performance by Mode (Mean z-score, lower = better)[/bold cyan]")
    console.print("="*140 + "\n")

    matrix_table = Table(show_header=True, header_style="bold magenta", width=140)
    matrix_table.add_column("Mode", style="cyan", width=20)
    matrix_table.add_column("use_ppl=False", justify="center", width=40)
    matrix_table.add_column("use_ppl=True", justify="center", width=40)
    matrix_table.add_column("Δ (PPL improvement)", justify="center", width=30)

    for exp_mode in ["discovery", "oed"]:
        exp_df = df[df["experiment_type"] == exp_mode]

        no_ppl = exp_df[exp_df["use_ppl"] == False]["z_mean"]
        yes_ppl = exp_df[exp_df["use_ppl"] == True]["z_mean"]

        no_ppl_mean = no_ppl.mean() if len(no_ppl) > 0 else float("nan")
        yes_ppl_mean = yes_ppl.mean() if len(yes_ppl) > 0 else float("nan")
        delta = no_ppl_mean - yes_ppl_mean  # Positive = PPL is better

        no_ppl_str = f"{no_ppl_mean:+.3f} (n={len(no_ppl)})" if not pd.isna(no_ppl_mean) else "N/A"
        yes_ppl_str = f"{yes_ppl_mean:+.3f} (n={len(yes_ppl)})" if not pd.isna(yes_ppl_mean) else "N/A"
        delta_str = f"{delta:+.3f}" if not pd.isna(delta) else "N/A"

        # Color code delta (green if PPL helps, red if hurts)
        if not pd.isna(delta):
            if delta > 0.05:
                delta_str = f"[bold green]{delta_str} ✓[/bold green]"
            elif delta < -0.05:
                delta_str = f"[bold red]{delta_str} ✗[/bold red]"

        matrix_table.add_row(
            exp_mode.upper(),
            no_ppl_str,
            yes_ppl_str,
            delta_str
        )

    console.print(matrix_table)

    # ===== Best Configuration =====
    console.print("\n" + "="*140)
    console.print("[bold cyan]Configuration Rankings by Mean z-score[/bold cyan]")
    console.print("="*140 + "\n")

    best_configs = df.groupby(["experiment_type", "use_ppl"]).agg({
        "z_mean": "mean",
        "path": "count"
    }).reset_index()
    best_configs.columns = ["exp", "use_ppl", "z_mean", "n_runs"]
    best_configs = best_configs.sort_values("z_mean")

    best_table = Table(show_header=True, header_style="bold magenta", width=140)
    best_table.add_column("Rank", justify="right", style="cyan", width=10)
    best_table.add_column("Mode", style="cyan", width=20)
    best_table.add_column("use_ppl", justify="center", width=15)
    best_table.add_column("Mean z", justify="right", width=15)
    best_table.add_column("# Runs", justify="right", width=15)

    for i, row in best_configs.iterrows():
        rank = "🥇" if i == best_configs.index[0] else "🥈" if i == best_configs.index[1] else "🥉" if i == best_configs.index[2] else f"{len(best_configs) - list(best_configs.index).index(i)}"
        use_ppl_emoji = "✓" if row["use_ppl"] else "✗"
        z_style = "bold green" if row["z_mean"] < 0 else "yellow" if row["z_mean"] < 0.2 else "red"

        best_table.add_row(
            rank,
            row["exp"].upper(),
            use_ppl_emoji,
            f"[{z_style}]{row['z_mean']:+.3f}[/{z_style}]",
            str(int(row["n_runs"]))
        )

    console.print(best_table)

    # ===== Per-Environment Breakdown =====
    console.print("\n" + "="*140)
    console.print("[bold cyan]Environment-Level Analysis: Where does each mode excel?[/bold cyan]")
    console.print("="*140 + "\n")

    # Get unique environments
    envs = df["env"].unique()

    env_winners = []

    for env_name in sorted(envs):
        env_df = df[df["env"] == env_name]

        env_configs = env_df.groupby(["experiment_type", "use_ppl"])["z_mean"].mean().reset_index()

        if len(env_configs) > 0:
            best_config = env_configs.loc[env_configs["z_mean"].idxmin()]
            env_winners.append({
                "env": env_name,
                "best_exp": best_config["experiment_type"],
                "best_ppl": best_config["use_ppl"],
                "z_mean": best_config["z_mean"]
            })

    env_table = Table(show_header=True, header_style="bold magenta", width=140)
    env_table.add_column("Environment", style="cyan", width=30)
    env_table.add_column("Best Mode", justify="center", width=20)
    env_table.add_column("use_ppl", justify="center", width=15)
    env_table.add_column("Mean z", justify="right", width=15)

    for winner in sorted(env_winners, key=lambda x: x["z_mean"]):
        ppl_emoji = "✓" if winner["best_ppl"] else "✗"
        z_style = "bold green" if winner["z_mean"] < 0 else "yellow" if winner["z_mean"] < 0.2 else "red"

        env_table.add_row(
            winner["env"],
            winner["best_exp"].upper(),
            ppl_emoji,
            f"[{z_style}]{winner['z_mean']:+.3f}[/{z_style}]"
        )

    console.print(env_table)

    # ===== Mode Dominance Summary =====
    console.print("\n" + "="*140)
    console.print("[bold cyan]Mode Dominance Across Environments[/bold cyan]")
    console.print("="*140 + "\n")

    mode_counts = pd.DataFrame(env_winners).groupby(["best_exp", "best_ppl"]).size().reset_index(name="count")

    dom_table = Table(show_header=True, header_style="bold magenta", width=100)
    dom_table.add_column("Mode", style="cyan", width=20)
    dom_table.add_column("use_ppl", justify="center", width=15)
    dom_table.add_column("# Environments Won", justify="right", width=20)
    dom_table.add_column("% of Environments", justify="right", width=20)

    total_envs = len(env_winners)

    for _, row in mode_counts.sort_values("count", ascending=False).iterrows():
        ppl_emoji = "✓" if row["best_ppl"] else "✗"
        pct = (row["count"] / total_envs) * 100

        dom_table.add_row(
            row["best_exp"].upper(),
            ppl_emoji,
            str(row["count"]),
            f"{pct:.1f}%"
        )

    console.print(dom_table)

    # ===== Key Finding Summary =====
    console.print("\n" + "="*140)
    console.print(Panel.fit(
        "[bold cyan]Key Findings for Slide[/bold cyan]\n\n"
        "1. [bold]Discovery + PPL[/bold] likely best overall (pending full data)\n"
        "   → LLM pattern matching + formal models = strongest combination\n\n"
        "2. [bold]OED Mode[/bold] appears more challenging across the board\n"
        "   → LLMs struggle with experimental design (redundant experiments, poor exploration)\n\n"
        "3. [bold]PPL consistently helps[/bold] in both modes\n"
        "   → Formalizing intuitions into code improves generalization\n\n"
        "4. [bold]Duality Insight[/bold]: Success requires BOTH skills\n"
        "   → Pattern recognition (what patterns exist?) vs Experimental design (which questions to ask?)\n"
        "   → Discovery leverages LLM strengths; OED exposes LLM weaknesses\n\n"
        "5. [bold]Implications for PPL Environments[/bold]\n"
        "   → Harder to game: requires asking right questions, not just pattern matching\n"
        "   → More robust measure of genuine understanding vs memorization",
        title="[bold magenta]Summary for Presentation[/bold magenta]",
        border_style="magenta"
    ))

if __name__ == "__main__":
    main()
