#!/usr/bin/env python3
"""Retrieve raw LLM responses from Weave traces.

Usage:
    # Get all LLM calls
    python scripts/retrieve_weave_traces.py

    # Filter by run ID
    python scripts/retrieve_weave_traces.py --run-id abc123

    # Filter by op name
    python scripts/retrieve_weave_traces.py --op-name llm_call

    # Export to JSON
    python scripts/retrieve_weave_traces.py --output traces.json

    # Search for specific content
    python scripts/retrieve_weave_traces.py --search "catastrophic"
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import weave
from weave.trace.weave_client import CallsFilter


def setup_env():
    """Ensure WANDB_PROJECT is set for Weave."""
    if not os.environ.get("WANDB_PROJECT"):
        os.environ["WANDB_PROJECT"] = "boxing-gym"


def get_llm_calls(
    client,
    op_name: str = "llm_call",
    run_id: str | None = None,
    limit: int | None = None,
    search: str | None = None,
):
    """Retrieve LLM call traces from Weave.

    Args:
        client: Weave client
        op_name: Operation name to filter by (default: "llm_call")
        run_id: WandB run ID to filter by
        limit: Maximum number of calls to retrieve
        search: Search string to filter outputs

    Returns:
        List of call objects with inputs and outputs
    """
    filter_kwargs = {}
    if op_name:
        filter_kwargs["op_names"] = [op_name]

    # Get calls with full inputs and outputs
    calls = client.get_calls(
        filter=CallsFilter(**filter_kwargs) if filter_kwargs else None,
        columns=["inputs", "output", "started_at", "ended_at", "summary"],
        limit=limit,
    )

    results = []
    for call in calls:
        call_data = {
            "id": call.id,
            "op_name": call.op_name,
            "started_at": call.started_at.isoformat() if call.started_at else None,
            "ended_at": call.ended_at.isoformat() if call.ended_at else None,
        }

        # Extract inputs (prompt, model, etc.)
        if hasattr(call, "inputs") and call.inputs:
            if isinstance(call.inputs, dict):
                call_data["inputs"] = call.inputs
            else:
                # If inputs is an object with attributes
                call_data["inputs"] = {
                    k: v
                    for k, v in call.inputs.__dict__.items()
                    if not k.startswith("_")
                }

        # Extract output (LLM response)
        if hasattr(call, "output"):
            if isinstance(call.output, dict):
                call_data["output"] = call.output
            elif call.output is not None:
                # Try to convert to string/dict
                try:
                    call_data["output"] = str(call.output)
                except Exception:
                    call_data["output"] = repr(call.output)

        # Extract summary stats (usage, latency, etc.)
        if hasattr(call, "summary") and call.summary:
            call_data["summary"] = call.summary

        # Filter by search term if provided
        if search:
            output_str = json.dumps(call_data.get("output", "")).lower()
            input_str = json.dumps(call_data.get("inputs", "")).lower()
            if search.lower() not in output_str and search.lower() not in input_str:
                continue

        results.append(call_data)

    return results


def print_summary(calls):
    """Print a summary of retrieved calls."""
    print(f"\n📊 Retrieved {len(calls)} LLM calls from Weave")
    print("=" * 80)

    if not calls:
        print("No calls found.")
        return

    # Group by model
    models = {}
    for call in calls:
        model = call.get("inputs", {}).get("model", "unknown")
        models[model] = models.get(model, 0) + 1

    print(f"\n🤖 Models used:")
    for model, count in sorted(models.items()):
        print(f"  - {model}: {count} calls")

    # Show time range
    timestamps = [
        datetime.fromisoformat(c["started_at"])
        for c in calls
        if c.get("started_at")
    ]
    if timestamps:
        print(f"\n📅 Time range:")
        print(f"  - First call: {min(timestamps)}")
        print(f"  - Last call: {max(timestamps)}")

    # Show sample outputs
    print(f"\n📝 Sample outputs (first 3):")
    for i, call in enumerate(calls[:3], 1):
        output = call.get("output", "")
        if isinstance(output, dict):
            output = json.dumps(output, indent=2)
        output_preview = str(output)[:200]
        print(f"\n  Call {i}:")
        print(f"    Model: {call.get('inputs', {}).get('model', 'unknown')}")
        print(f"    Output preview: {output_preview}...")


def export_to_file(calls, output_path: str):
    """Export calls to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(calls, f, indent=2)

    print(f"\n✅ Exported {len(calls)} calls to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve LLM call traces from Weave",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WEAVE_PROJECT", ""),
        help="Weave project name (from WEAVE_PROJECT env var)",
    )
    parser.add_argument(
        "--op-name",
        default="llm_call",
        help="Operation name to filter by (default: llm_call)",
    )
    parser.add_argument(
        "--run-id",
        help="Filter by WandB run ID",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of calls to retrieve",
    )
    parser.add_argument(
        "--search",
        help="Search string to filter outputs",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing summary",
    )

    args = parser.parse_args()

    # Setup environment
    setup_env()

    # Initialize Weave client
    print(f"🔍 Connecting to Weave project: {args.project}")
    client = weave.init(args.project)

    # Retrieve calls
    print(f"📡 Retrieving LLM calls (op_name={args.op_name})...")
    calls = get_llm_calls(
        client,
        op_name=args.op_name,
        run_id=args.run_id,
        limit=args.limit,
        search=args.search,
    )

    # Print summary
    if not args.no_summary:
        print_summary(calls)

    # Export if requested
    if args.output:
        export_to_file(calls, args.output)

    # Finish
    client.finish()


if __name__ == "__main__":
    main()
