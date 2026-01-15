"""BoxingGym CLI - unified command-line interface.

Usage:
    box sync [--local DIR | --sweep-id IDS] [--refresh]
    box query [leaderboard|oed-discovery|ppl-impact|...] [--format FORMAT]
    box results [--view VIEW] [--tui] [--web]
"""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="box")
def main():
    """BoxingGym benchmark analysis CLI.

    Analyze LLM performance on scientific discovery tasks.

    Quick start:

        box sync --local results/     # cache local results
        box query leaderboard         # model rankings
        box query oed-discovery       # OED vs Discovery comparison
    """
    pass


@main.command()
@click.option("--local", type=click.Path(exists=True), help="Load from local results directory")
@click.option("--sweep-id", help="WandB sweep ID(s), comma-separated (coming soon)")
@click.option("--refresh", is_flag=True, help="Force refresh, ignore cache")
@click.option("--status", is_flag=True, help="Show cache status")
def sync(local, sweep_id, refresh, status):
    """Sync results to local cache.

    Examples:

        box sync --local results/
        box sync --sweep-id 5h3g5cno,qqkzi3ln
        box sync --status
    """
    from boxing_gym.cli.commands.sync import run_sync
    run_sync(local=local, sweep_id=sweep_id, refresh=refresh, status=status)


@main.command()
@click.argument("query_name", required=False)
@click.option("--list", "list_queries", is_flag=True, help="List available queries")
@click.option("--format", "output_format", type=click.Choice(["rich", "md", "json"]), default="rich")
@click.option("--output", type=click.Path(), help="Output file for md/json formats (coming soon)")
@click.option("--min-budget", type=int, default=10, help="Minimum budget filter")
@click.option("--env", help="Filter to specific environment")
@click.option("--include-outliers", is_flag=True, help="Include flagged outliers in analysis")
def query(query_name, list_queries, output_format, output, min_budget, env, include_outliers):
    """Run pre-built analysis queries.

    Available queries:

        leaderboard      - Model rankings with significance tests
        oed-discovery    - OED vs Discovery 2x2 comparison
        ppl-impact       - PPL effect analysis
        env-difficulty   - Environment difficulty rankings
        best-configs     - Best configuration per environment
        paper-comparison - Comparison with paper baselines
        all              - Run all queries (full report)

    Examples:

        box query --list
        box query leaderboard
        box query oed-discovery --format md
        box query all --format md --output report.md
    """
    from boxing_gym.cli.commands.query import run_query
    run_query(
        query_name=query_name,
        list_queries=list_queries,
        output_format=output_format,
        output=output,
        min_budget=min_budget,
        env=env,
        include_outliers=include_outliers,
    )


@main.command()
@click.option("--view", help="Specific view (model-rankings, heatmap, best-configs, etc.)")
@click.option("--tui", is_flag=True, help="Launch interactive TUI")
@click.option("--web", is_flag=True, help="Launch Streamlit web dashboard")
@click.option("--format", "output_format", type=click.Choice(["rich", "json", "csv"]), default="rich")
@click.option("--include-outliers", is_flag=True, help="Include flagged outliers in analysis")
def results(view, tui, web, output_format, include_outliers):
    """View and explore results.

    Available views:

        model-rankings       - Model leaderboard
        heatmap              - Environment x Model matrix
        best-configs         - Top configurations
        budget-progression   - Performance by budget
        parameter-importance - Feature importance analysis
        seed-stability       - Cross-seed variance
        local-summary        - Quick summary of local results

    Examples:

        box results
        box results --view model-rankings
        box results --tui
        box results --web
    """
    from boxing_gym.cli.commands.results import run_results
    run_results(view=view, tui=tui, web=web, output_format=output_format, include_outliers=include_outliers)


if __name__ == "__main__":
    main()
