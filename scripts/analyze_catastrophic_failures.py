#!/usr/bin/env python3
"""Analyze catastrophic failures by correlating sweep results with Weave LLM traces.

This script:
1. Loads sweep results from WandB
2. Identifies catastrophic failures (e.g., MSE > 1e10)
3. Retrieves the corresponding LLM responses from Weave
4. Analyzes patterns in the failures

Usage:
    # Analyze specific sweep
    python scripts/analyze_catastrophic_failures.py --sweep-id YOUR_SWEEP_ID

    # Export detailed analysis
    python scripts/analyze_catastrophic_failures.py --sweep-id YOUR_SWEEP_ID --output analysis.json

    # Focus on specific environment
    python scripts/analyze_catastrophic_failures.py --sweep-id YOUR_SWEEP_ID --env hyperbolic
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import wandb
import weave


def setup_env():
    """Ensure environment variables are set."""
    if not os.environ.get("WANDB_PROJECT"):
        os.environ["WANDB_PROJECT"] = "boxing-gym"


def get_sweep_runs(sweep_id: str, entity: str = None, project: str = "boxing-gym"):
    """Get all runs from a sweep."""
    entity = entity or os.environ.get("WANDB_ENTITY", "")
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    runs_data = []
    for run in sweep.runs:
        run_data = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": run.created_at,
        }

        # Get config
        config = run.config
        run_data["model"] = config.get("llms", {}).get("model_name", "unknown")
        run_data["env"] = config.get("envs", "unknown")
        run_data["seed"] = config.get("seed", -1)

        # Get summary metrics
        summary = run.summary
        run_data["mse_final"] = summary.get("eval/mse_final")
        run_data["z_mean_final"] = summary.get("eval/z_mean_final")
        run_data["success_rate"] = summary.get("run/success_rate")
        run_data["total_cost_usd"] = summary.get("llm/total_cost_usd")

        runs_data.append(run_data)

    return pd.DataFrame(runs_data)


def identify_catastrophic_runs(df: pd.DataFrame, mse_threshold: float = 1e10):
    """Identify runs with catastrophically high MSE."""
    df["is_catastrophic"] = df["mse_final"] > mse_threshold
    return df


def get_run_llm_traces(client, run_id: str, limit: int = 1000):
    """Get LLM traces for a specific run from Weave."""
    # Note: Weave doesn't directly link to WandB run IDs in the current implementation
    # We need to filter by time window around the run's creation time

    print(f"  Retrieving Weave traces for run {run_id}...")

    # Get all recent LLM calls
    calls = client.get_calls(
        columns=["inputs", "output", "started_at", "ended_at", "op_name"],
        limit=limit,
    )

    llm_traces = []
    for call in calls:
        # Only process LiteLLM and OpenAI calls
        if not ("litellm" in call.op_name or "openai" in call.op_name):
            continue

        trace = {
            "id": call.id,
            "started_at": call.started_at.isoformat() if call.started_at else None,
        }

        # Extract model and prompt
        if hasattr(call, "inputs") and isinstance(call.inputs, dict):
            trace["model"] = call.inputs.get("model", "unknown")

            messages = call.inputs.get("messages", [])
            if messages:
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        trace["prompt"] = msg.get("content", "")
                        break

        # Extract response
        if hasattr(call, "output") and isinstance(call.output, dict):
            choices = call.output.get("choices", [])
            if choices:
                choice = choices[0] if isinstance(choices, list) else choices
                if isinstance(choice, dict):
                    message = choice.get("message", {})
                    if isinstance(message, dict):
                        trace["response"] = message.get("content", "")

        llm_traces.append(trace)

    return llm_traces


def analyze_llm_patterns(traces: list[dict]):
    """Analyze patterns in LLM responses."""
    if not traces:
        return {}

    analysis = {
        "total_traces": len(traces),
        "patterns": {},
    }

    # Check for common patterns
    patterns = {
        "empty_response": lambda r: not r.get("response"),
        "very_short": lambda r: len(str(r.get("response", ""))) < 50,
        "very_long": lambda r: len(str(r.get("response", ""))) > 10000,
        "contains_error": lambda r: "error" in str(r.get("response", "")).lower(),
        "contains_divergence": lambda r: "divergence" in str(r.get("response", "")).lower(),
        "contains_warning": lambda r: "warning" in str(r.get("response", "")).lower(),
        "mentions_nan": lambda r: "nan" in str(r.get("response", "")).lower(),
        "mentions_infinity": lambda r: "inf" in str(r.get("response", "")).lower(),
    }

    for pattern_name, pattern_func in patterns.items():
        count = sum(1 for trace in traces if pattern_func(trace))
        if count > 0:
            analysis["patterns"][pattern_name] = {
                "count": count,
                "percentage": count / len(traces) * 100,
            }

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze catastrophic failures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sweep-id",
        required=True,
        help="WandB sweep ID",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY", ""),
        help="WandB entity (from WANDB_ENTITY env var)",
    )
    parser.add_argument(
        "--project",
        default="boxing-gym",
        help="WandB project (default: boxing-gym)",
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=1e10,
        help="MSE threshold for catastrophic failure (default: 1e10)",
    )
    parser.add_argument(
        "--env",
        help="Filter by environment name",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Export results to JSON file",
    )

    args = parser.parse_args()

    # Setup environment
    setup_env()

    print(f"🔍 Analyzing sweep: {args.sweep_id}")
    print(f"   Entity: {args.entity}")
    print(f"   Project: {args.project}")
    print(f"   MSE threshold: {args.mse_threshold}")

    # Get sweep runs
    print("\n📊 Loading sweep runs...")
    df = get_sweep_runs(args.sweep_id, args.entity, args.project)

    # Filter by environment if specified
    if args.env:
        df = df[df["env"].str.contains(args.env, case=False, na=False)]
        print(f"   Filtered to {len(df)} runs for environment: {args.env}")

    # Identify catastrophic runs
    df = identify_catastrophic_runs(df, args.mse_threshold)
    catastrophic_runs = df[df["is_catastrophic"]]

    print(f"\n🔥 Found {len(catastrophic_runs)} catastrophic runs:")
    if len(catastrophic_runs) > 0:
        print(catastrophic_runs[["name", "model", "env", "mse_final", "z_mean_final"]].to_string())

        # Show statistics
        print("\n📈 Catastrophic run statistics:")
        print(f"   Mean MSE: {catastrophic_runs['mse_final'].mean():.2e}")
        print(f"   Max MSE: {catastrophic_runs['mse_final'].max():.2e}")

        # Group by model
        by_model = catastrophic_runs.groupby("model").size().sort_values(ascending=False)
        print("\n🤖 Catastrophic runs by model:")
        for model, count in by_model.items():
            print(f"   - {model}: {count}")

        # Group by environment
        by_env = catastrophic_runs.groupby("env").size().sort_values(ascending=False)
        print("\n🌍 Catastrophic runs by environment:")
        for env, count in by_env.items():
            print(f"   - {env}: {count}")

    # Initialize Weave to get LLM traces
    print("\n🔗 Connecting to Weave for LLM traces...")
    weave_client = weave.init(f"{args.entity}/{args.project}")

    # Analyze LLM traces for catastrophic runs (sample)
    detailed_analysis = []
    for idx, (_, run) in enumerate(catastrophic_runs.head(5).iterrows(), 1):
        print(f"\n📝 Analyzing run {idx}/{min(5, len(catastrophic_runs))}: {run['name']}")

        # Note: This is a simplified approach - in production you'd need to
        # match runs to traces by timestamp or other metadata
        traces = get_run_llm_traces(weave_client, run["id"])

        if traces:
            pattern_analysis = analyze_llm_patterns(traces)
            detailed_analysis.append(
                {
                    "run_id": run["id"],
                    "run_name": run["name"],
                    "model": run["model"],
                    "env": run["env"],
                    "mse_final": float(run["mse_final"]),
                    "llm_traces": len(traces),
                    "patterns": pattern_analysis.get("patterns", {}),
                    "sample_responses": traces[:3],  # Include first 3 responses
                }
            )

            print(f"   Found {len(traces)} LLM traces")
            if pattern_analysis.get("patterns"):
                print("   Patterns detected:")
                for pattern, info in pattern_analysis["patterns"].items():
                    print(f"     - {pattern}: {info['count']} ({info['percentage']:.1f}%)")

    # Export if requested
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "sweep_id": args.sweep_id,
            "analyzed_at": datetime.now().isoformat(),
            "total_runs": len(df),
            "catastrophic_runs": len(catastrophic_runs),
            "mse_threshold": args.mse_threshold,
            "summary": {
                "by_model": by_model.to_dict() if len(catastrophic_runs) > 0 else {},
                "by_env": by_env.to_dict() if len(catastrophic_runs) > 0 else {},
            },
            "detailed_analysis": detailed_analysis,
        }

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\n✅ Exported analysis to {output_file}")

    # Finish
    weave_client.finish()

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
