"""Query command - run pre-built analysis queries."""

from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

CACHE_FILE = Path(".boxing-gym-cache/runs.parquet")

AVAILABLE_QUERIES = {
    "leaderboard": "Model rankings with significance tests",
    "oed-discovery": "OED vs Discovery 2x2 comparison",
    "ppl-impact": "PPL effect analysis",
    "env-difficulty": "Environment difficulty rankings",
    "best-configs": "Best configuration per environment",
    "paper-comparison": "Comparison with paper baselines",
    "all": "Run all queries (full report)",
}


def run_query(
    query_name: str | None,
    list_queries: bool,
    output_format: str,
    output: str | None,
    min_budget: int,
    env: str | None,
    include_outliers: bool = False,
):
    """Run a pre-built analysis query."""
    if list_queries or not query_name:
        _list_queries()
        return

    if query_name not in AVAILABLE_QUERIES:
        console.print(f"[red]Unknown query: {query_name}[/red]")
        console.print("Run: box query --list")
        return

    # load data from cache
    if not CACHE_FILE.exists():
        console.print("[yellow]No cache found. Run sync first:[/yellow]")
        console.print("  box sync --local results/")
        return

    import pandas as pd
    df = pd.read_parquet(CACHE_FILE)

    # filter flagged outliers by default (unless --include-outliers)
    if not include_outliers and "is_outlier" in df.columns:
        n_total = len(df)
        df = df[~df["is_outlier"].fillna(False)]
        n_excluded = n_total - len(df)
        if n_excluded > 0:
            console.print(f"[dim]Excluding {n_excluded:,} flagged outliers (use --include-outliers to include)[/dim]")

    # apply filters
    if min_budget:
        df = df[df["budget"] >= min_budget]
    if env:
        df = df[df["env"] == env]

    if len(df) == 0:
        console.print("[yellow]No data after filtering.[/yellow]")
        return

    console.print(f"[dim]Analyzing {len(df):,} runs...[/dim]\n")

    # dispatch to query
    if query_name == "all":
        for q in AVAILABLE_QUERIES:
            if q != "all":
                _run_single_query(q, df, output_format)
                console.print()
    else:
        result = _run_single_query(query_name, df, output_format)

    # output to file if requested
    if output and output_format in ("md", "json"):
        console.print(f"[dim]Output to file not yet implemented.[/dim]")


def _list_queries():
    """List available queries."""
    table = Table(title="Available Queries")
    table.add_column("Query", style="cyan")
    table.add_column("Description")

    for name, desc in AVAILABLE_QUERIES.items():
        table.add_row(name, desc)

    console.print(table)
    console.print("\n[dim]Usage: box query <name>[/dim]")


def _run_single_query(query_name: str, df, output_format: str):
    """Run a single query and display results."""
    if query_name == "leaderboard":
        _query_leaderboard(df, output_format)
    elif query_name == "oed-discovery":
        _query_oed_discovery(df, output_format)
    elif query_name == "ppl-impact":
        _query_ppl_impact(df, output_format)
    elif query_name == "env-difficulty":
        _query_env_difficulty(df, output_format)
    elif query_name == "best-configs":
        _query_best_configs(df, output_format)
    elif query_name == "paper-comparison":
        _query_paper_comparison(df, output_format)


def _query_leaderboard(df, output_format: str):
    """Model leaderboard with rankings and statistical significance."""
    import numpy as np
    from boxing_gym.analysis.stats import bootstrap_ci, compare_models

    agg = df.groupby("model")["z_mean"].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg = agg.sort_values("mean")

    # compute bootstrap CIs for each model
    ci_data = {}
    for model in agg["model"]:
        scores = df[df["model"] == model]["z_mean"].values
        ci = bootstrap_ci(scores)
        ci_data[model] = ci

    best = agg.iloc[0]
    best_ci = ci_data[best["model"]]

    # compare all models against best
    comparisons = compare_models(df, reference_model=best["model"])

    console.print(Panel.fit(
        f"[bold]Question:[/bold] Which model performs best overall?\n"
        f"[bold]Answer:[/bold] {best['model']} ranks #1 with z={best['mean']:+.3f} "
        f"(95% CI: [{best_ci.ci_low:+.3f}, {best_ci.ci_high:+.3f}])",
        title="Model Leaderboard",
        border_style="cyan",
    ))

    table = Table(show_header=True)
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Model")
    table.add_column("Mean z", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("vs #1", justify="right")
    table.add_column("n", justify="right")

    # build lookup for p-values
    p_lookup = {c["model"]: c for c in comparisons["comparisons"]}

    for i, row in enumerate(agg.itertuples(), 1):
        z_style = "green" if row.mean < 0 else "yellow" if row.mean < 0.3 else "red"
        ci = ci_data[row.model]

        # p-value vs best
        if row.model == best["model"]:
            p_str = "-"
        else:
            comp = p_lookup.get(row.model, {})
            p_val = comp.get("p_adjusted", comp.get("p_value", 1.0))
            sig = comp.get("significant_fdr", False)
            if sig:
                p_str = f"[red]p={p_val:.2f}*[/red]"
            else:
                p_str = f"p={p_val:.2f}"

        table.add_row(
            str(i),
            row.model,
            f"[{z_style}]{row.mean:+.3f}[/{z_style}]",
            f"[{ci.ci_low:+.3f}, {ci.ci_high:+.3f}]",
            p_str,
            str(int(row.count)),
        )

    console.print(table)

    # key findings with significance
    worst = agg.iloc[-1]
    spread = worst["mean"] - best["mean"]

    # find significantly different models
    sig_worse = [c["model"] for c in comparisons["comparisons"] if c.get("significant_fdr", False)]

    console.print(f"\n[bold]Key Findings:[/bold]")
    console.print(f"  • Best: {best['model']} (z={best['mean']:+.3f})")
    console.print(f"  • Worst: {worst['model']} (z={worst['mean']:+.3f})")
    console.print(f"  • Spread: {spread:.3f}")
    if sig_worse:
        console.print(f"  • Significantly worse than #1 (FDR<0.05): {', '.join(sig_worse)}")
    else:
        console.print(f"  • No models significantly different from #1 at FDR<0.05")


def _query_oed_discovery(df, output_format: str):
    """OED vs Discovery 2x2 comparison with statistical tests."""
    import numpy as np
    from boxing_gym.analysis.stats import welch_ttest, cohens_d

    # compute main comparison: OED vs Discovery
    oed_scores = df[df["experiment_type"] == "oed"]["z_mean"].values
    disc_scores = df[df["experiment_type"] == "discovery"]["z_mean"].values

    main_test = welch_ttest(oed_scores, disc_scores) if len(oed_scores) > 0 and len(disc_scores) > 0 else None
    effect = cohens_d(oed_scores, disc_scores) if len(oed_scores) > 0 and len(disc_scores) > 0 else None

    oed_mean = np.mean(oed_scores) if len(oed_scores) > 0 else float("nan")
    disc_mean = np.mean(disc_scores) if len(disc_scores) > 0 else float("nan")
    winner = "OED" if oed_mean < disc_mean else "Discovery"

    sig_str = ""
    if main_test and main_test.significant:
        sig_str = f" [green](p={main_test.p_value:.3f}, significant)[/green]"
    elif main_test:
        sig_str = f" (p={main_test.p_value:.3f}, not significant)"

    effect_str = f", effect={effect.interpretation}" if effect else ""

    console.print(Panel.fit(
        f"[bold]Question:[/bold] Does OED beat Discovery?\n"
        f"[bold]Answer:[/bold] {winner} wins with z={min(oed_mean, disc_mean):+.3f}{sig_str}{effect_str}",
        title="OED vs Discovery",
        border_style="cyan",
    ))

    table = Table(title="Mode × PPL Matrix (mean z-score, lower = better)")
    table.add_column("Mode", style="cyan")
    table.add_column("use_ppl=false", justify="center")
    table.add_column("use_ppl=true", justify="center")
    table.add_column("Δ (PPL effect)", justify="center")
    table.add_column("p-value", justify="center")

    for exp in ["discovery", "oed"]:
        exp_df = df[df["experiment_type"] == exp]
        if len(exp_df) == 0:
            continue

        no_ppl = exp_df[exp_df["use_ppl"] == False]["z_mean"].values
        yes_ppl = exp_df[exp_df["use_ppl"] == True]["z_mean"].values

        no_ppl_mean = np.mean(no_ppl) if len(no_ppl) > 0 else float("nan")
        yes_ppl_mean = np.mean(yes_ppl) if len(yes_ppl) > 0 else float("nan")
        delta = no_ppl_mean - yes_ppl_mean if not (np.isnan(no_ppl_mean) or np.isnan(yes_ppl_mean)) else float("nan")

        # test PPL effect within this mode
        ppl_test = welch_ttest(no_ppl, yes_ppl) if len(no_ppl) > 1 and len(yes_ppl) > 1 else None

        no_ppl_str = f"{no_ppl_mean:+.3f} (n={len(no_ppl)})" if not np.isnan(no_ppl_mean) else "N/A"
        yes_ppl_str = f"{yes_ppl_mean:+.3f} (n={len(yes_ppl)})" if not np.isnan(yes_ppl_mean) else "N/A"

        if not np.isnan(delta):
            delta_style = "green" if delta > 0.05 else "red" if delta < -0.05 else "yellow"
            delta_str = f"[{delta_style}]{delta:+.3f}[/{delta_style}]"
        else:
            delta_str = "N/A"

        if ppl_test:
            p_style = "green" if ppl_test.significant else "dim"
            p_str = f"[{p_style}]{ppl_test.p_value:.3f}{'*' if ppl_test.significant else ''}[/{p_style}]"
        else:
            p_str = "N/A"

        table.add_row(exp, no_ppl_str, yes_ppl_str, delta_str, p_str)

    console.print(table)

    # OED vs Discovery comparison row
    console.print(f"\n[bold]Main Comparison (OED vs Discovery):[/bold]")
    console.print(f"  • OED: z={oed_mean:+.3f} (n={len(oed_scores)})")
    console.print(f"  • Discovery: z={disc_mean:+.3f} (n={len(disc_scores)})")
    if main_test:
        console.print(f"  • Welch t-test: p={main_test.p_value:.4f} ({'significant' if main_test.significant else 'not significant'})")
    if effect:
        console.print(f"  • Effect size: d={effect.d:+.3f} ({effect.interpretation})")


def _query_ppl_impact(df, output_format: str):
    """PPL effect analysis with statistical testing."""
    import numpy as np
    from boxing_gym.analysis.stats import welch_ttest, cohens_d, bootstrap_ci

    ppl_true = df[df["use_ppl"] == True]["z_mean"].values
    ppl_false = df[df["use_ppl"] == False]["z_mean"].values

    # statistical tests
    test = welch_ttest(ppl_true, ppl_false) if len(ppl_true) > 1 and len(ppl_false) > 1 else None
    effect = cohens_d(ppl_true, ppl_false) if len(ppl_true) > 1 and len(ppl_false) > 1 else None

    ppl_true_mean = np.mean(ppl_true) if len(ppl_true) > 0 else float("nan")
    ppl_false_mean = np.mean(ppl_false) if len(ppl_false) > 0 else float("nan")
    delta = ppl_false_mean - ppl_true_mean

    verdict = "PPL helps" if delta > 0 else "PPL hurts"
    sig_str = ""
    if test and test.significant:
        sig_str = f" [green](significant, p={test.p_value:.3f})[/green]"
    elif test:
        sig_str = f" (not significant, p={test.p_value:.3f})"

    console.print(Panel.fit(
        f"[bold]Question:[/bold] Does PPL help?\n"
        f"[bold]Answer:[/bold] {verdict} (Δ={delta:+.3f}){sig_str}",
        title="PPL Impact Analysis",
        border_style="cyan",
    ))

    # confidence intervals
    ci_true = bootstrap_ci(ppl_true) if len(ppl_true) > 1 else None
    ci_false = bootstrap_ci(ppl_false) if len(ppl_false) > 1 else None

    table = Table()
    table.add_column("Condition", style="cyan")
    table.add_column("Mean z", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("n", justify="right")

    if ci_true:
        table.add_row(
            "PPL=true",
            f"{ppl_true_mean:+.3f}",
            f"[{ci_true.ci_low:+.3f}, {ci_true.ci_high:+.3f}]",
            str(len(ppl_true))
        )
    if ci_false:
        table.add_row(
            "PPL=false",
            f"{ppl_false_mean:+.3f}",
            f"[{ci_false.ci_low:+.3f}, {ci_false.ci_high:+.3f}]",
            str(len(ppl_false))
        )

    console.print(table)

    console.print(f"\n[bold]Key Findings:[/bold]")
    console.print(f"  • Delta: {delta:+.3f} ({verdict})")
    if test:
        console.print(f"  • Welch t-test: t={test.t_statistic:.2f}, p={test.p_value:.4f}")
    if effect:
        console.print(f"  • Effect size: d={effect.d:+.3f} ({effect.interpretation})")


def _query_env_difficulty(df, output_format: str):
    """Environment difficulty rankings."""
    import numpy as np

    console.print(Panel.fit(
        "[bold]Question:[/bold] Which environments are hardest?",
        title="Environment Difficulty",
        border_style="cyan",
    ))

    agg = df.groupby("env")["z_mean"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.sort_values("mean")

    table = Table()
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Environment")
    table.add_column("Mean z", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("n", justify="right")
    table.add_column("Difficulty", justify="center")

    for i, row in enumerate(agg.itertuples(), 1):
        if row.mean < 0:
            diff = "[green]Easy[/green]"
        elif row.mean < 0.3:
            diff = "[yellow]Medium[/yellow]"
        else:
            diff = "[red]Hard[/red]"

        table.add_row(
            str(i),
            row.env,
            f"{row.mean:+.3f}",
            f"{row.std:.3f}",
            str(int(row.count)),
            diff,
        )

    console.print(table)


def _query_best_configs(df, output_format: str):
    """Best configuration per environment."""
    console.print(Panel.fit(
        "[bold]Question:[/bold] What's the best config per environment?",
        title="Best Configurations",
        border_style="cyan",
    ))

    table = Table()
    table.add_column("Environment", style="cyan")
    table.add_column("Best Model")
    table.add_column("z_mean", justify="right")
    table.add_column("use_ppl")
    table.add_column("n", justify="right")

    for env in sorted(df["env"].unique()):
        env_df = df[df["env"] == env]
        agg = env_df.groupby(["model", "use_ppl"])["z_mean"].agg(["mean", "count"]).reset_index()
        best = agg.loc[agg["mean"].idxmin()]

        table.add_row(
            env,
            best["model"],
            f"{best['mean']:+.3f}",
            "✓" if best["use_ppl"] else "✗",
            str(int(best["count"])),
        )

    console.print(table)


def _query_paper_comparison(df, output_format: str):
    """Comparison with paper baselines."""
    console.print(Panel.fit(
        "[bold]Question:[/bold] How do we compare to paper baselines?",
        title="Paper Comparison",
        border_style="cyan",
    ))
    console.print("[yellow]Paper comparison not yet implemented.[/yellow]")
    console.print("[dim]Requires loading PAPER_RESULTS from results_io.py[/dim]")
