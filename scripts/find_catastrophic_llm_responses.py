#!/usr/bin/env python3
"""Find and analyze LLM responses from catastrophic runs in Weave traces.

This script retrieves raw LLM responses from Weave for runs where the model
produced catastrophically bad results (e.g., MSE > 1e10).

Usage:
    # Find all catastrophic responses
    python scripts/find_catastrophic_llm_responses.py

    # Filter by time period (last 7 days)
    python scripts/find_catastrophic_llm_responses.py --days 7

    # Export to JSON
    python scripts/find_catastrophic_llm_responses.py --output catastrophic_traces.json

    # Search for specific pattern
    python scripts/find_catastrophic_llm_responses.py --search "divergences"
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import weave


def setup_env():
    """Ensure WANDB_PROJECT is set for Weave."""
    if not os.environ.get("WANDB_PROJECT"):
        os.environ["WANDB_PROJECT"] = "boxing-gym"


def extract_message_content(call_inputs):
    """Extract message content from call inputs."""
    if not isinstance(call_inputs, dict):
        return None

    messages = call_inputs.get("messages", [])
    if not messages:
        return None

    # Get the last user message (the prompt)
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")

    return None


def extract_response_content(call_output):
    """Extract response content from call output."""
    if not isinstance(call_output, dict):
        return str(call_output)

    # Handle OpenAI/LiteLLM response format
    choices = call_output.get("choices", [])
    if not choices:
        return json.dumps(call_output)

    # Get first choice
    choice = choices[0] if isinstance(choices, list) else choices
    if isinstance(choice, dict):
        message = choice.get("message", {})
        if isinstance(message, dict):
            return message.get("content", "")

    return json.dumps(call_output)


def get_llm_calls(
    client,
    start_time: datetime | None = None,
    search: str | None = None,
    limit: int | None = None,
):
    """Retrieve LLM call traces from Weave.

    Args:
        client: Weave client
        start_time: Only retrieve calls after this time
        search: Search string to filter responses
        limit: Maximum number of calls to retrieve

    Returns:
        List of call data dictionaries
    """
    print(f"📡 Retrieving LLM calls from Weave...")

    # Get all calls (we'll filter for LLM calls)
    calls = client.get_calls(
        columns=["inputs", "output", "started_at", "ended_at", "op_name", "summary"],
        limit=limit,
    )

    results = []
    for call in calls:
        # Only process LiteLLM and OpenAI calls
        if not ("litellm" in call.op_name or "openai" in call.op_name):
            continue

        # Filter by time if specified
        if start_time and call.started_at:
            if call.started_at < start_time:
                continue

        call_data = {
            "id": call.id,
            "op_name": call.op_name,
            "started_at": call.started_at.isoformat() if call.started_at else None,
            "ended_at": call.ended_at.isoformat() if call.ended_at else None,
        }

        # Extract model name
        if hasattr(call, "inputs") and isinstance(call.inputs, dict):
            call_data["model"] = call.inputs.get("model", "unknown")
            call_data["prompt"] = extract_message_content(call.inputs)

        # Extract response
        if hasattr(call, "output"):
            call_data["response"] = extract_response_content(call.output)

        # Extract usage stats
        if hasattr(call, "summary") and isinstance(call.summary, dict):
            weave_summary = call.summary.get("weave", {})
            if isinstance(weave_summary, dict):
                usage = weave_summary.get("usage", {})
                if isinstance(usage, dict):
                    call_data["usage"] = usage

        # Filter by search term if provided
        if search:
            response_str = str(call_data.get("response", "")).lower()
            prompt_str = str(call_data.get("prompt", "")).lower()
            if search.lower() not in response_str and search.lower() not in prompt_str:
                continue

        results.append(call_data)

    return results


def analyze_responses(calls):
    """Analyze LLM responses for patterns."""
    print(f"\n📊 Analyzing {len(calls)} LLM calls...")

    if not calls:
        print("No calls found.")
        return

    # Group by model
    models = {}
    for call in calls:
        model = call.get("model", "unknown")
        models[model] = models.get(model, 0) + 1

    print(f"\n🤖 Models used:")
    for model, count in sorted(models.items(), key=lambda x: -x[1]):
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

    # Analyze response patterns
    print(f"\n🔍 Response patterns:")

    # Check for common issues
    issues = {
        "empty_response": 0,
        "very_short": 0,
        "very_long": 0,
        "contains_error": 0,
        "contains_divergence": 0,
        "contains_warning": 0,
    }

    for call in calls:
        response = str(call.get("response", ""))

        if not response or response == "None":
            issues["empty_response"] += 1
        elif len(response) < 50:
            issues["very_short"] += 1
        elif len(response) > 10000:
            issues["very_long"] += 1

        response_lower = response.lower()
        if "error" in response_lower:
            issues["contains_error"] += 1
        if "divergence" in response_lower:
            issues["contains_divergence"] += 1
        if "warning" in response_lower:
            issues["contains_warning"] += 1

    for issue, count in issues.items():
        if count > 0:
            print(f"  - {issue}: {count} ({count/len(calls)*100:.1f}%)")


def show_sample_responses(calls, n=3):
    """Show sample responses."""
    print(f"\n📝 Sample responses (first {n}):")

    for i, call in enumerate(calls[:n], 1):
        print(f"\n{'=' * 80}")
        print(f"Call {i}/{len(calls)}")
        print(f"{'=' * 80}")
        print(f"Model: {call.get('model', 'unknown')}")
        print(f"Time: {call.get('started_at', 'unknown')}")

        # Show prompt
        prompt = call.get("prompt", "")
        if prompt:
            print(f"\n--- PROMPT ---")
            print(prompt[:500])
            if len(prompt) > 500:
                print(f"... ({len(prompt) - 500} more chars)")

        # Show response
        response = call.get("response", "")
        if response:
            print(f"\n--- RESPONSE ---")
            print(response[:1000])
            if len(response) > 1000:
                print(f"... ({len(response) - 1000} more chars)")

        # Show usage
        usage = call.get("usage", {})
        if usage:
            print(f"\n--- USAGE ---")
            print(f"Tokens: {usage.get('total_tokens', 'unknown')}")


def export_to_file(calls, output_path: str):
    """Export calls to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(calls, f, indent=2)

    print(f"\n✅ Exported {len(calls)} calls to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Find catastrophic LLM responses in Weave traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WEAVE_PROJECT", ""),
        help="Weave project name (from WEAVE_PROJECT env var)",
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Only retrieve calls from the last N days",
    )
    parser.add_argument(
        "--search",
        help="Search string to filter responses",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of calls to retrieve (default: 1000)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip analysis, just retrieve data",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample responses to show (default: 3)",
    )

    args = parser.parse_args()

    # Setup environment
    setup_env()

    # Calculate start time if days specified
    start_time = None
    if args.days:
        start_time = datetime.now() - timedelta(days=args.days)
        print(f"⏰ Filtering calls from last {args.days} days (since {start_time})")

    # Initialize Weave client
    print(f"🔍 Connecting to Weave project: {args.project}")
    client = weave.init(args.project)

    # Retrieve calls
    calls = get_llm_calls(
        client,
        start_time=start_time,
        search=args.search,
        limit=args.limit,
    )

    print(f"\n✅ Retrieved {len(calls)} LLM calls")

    # Analyze responses
    if not args.no_analysis and calls:
        analyze_responses(calls)
        show_sample_responses(calls, n=args.samples)

    # Export if requested
    if args.output:
        export_to_file(calls, args.output)

    # Finish
    client.finish()


if __name__ == "__main__":
    main()
