#!/usr/bin/env python3
"""
Extract the actual prediction values from extreme z-score runs.

Usage:
    # With command-line args
    uv run python scripts/extract_extreme_predictions.py --runs run1,run2

    # With environment variable
    EXTREME_RUN_IDS=run1,run2 uv run python scripts/extract_extreme_predictions.py
"""

import argparse
import json
import os
from pathlib import Path


def calculate_prediction_from_zscore(z_mean: float, mu: float = 10991.5464, sigma: float = 15725.115658658306) -> float:
    """Calculate the actual prediction value from z-score."""
    return z_mean * sigma + mu


def extract_run_details(sweep_data_path: Path, run_id: str):
    """Extract detailed information for a specific run."""
    with open(sweep_data_path, 'r') as f:
        data = json.load(f)

    runs = data['data']['project']['sweep']['runs']['edges']

    for edge in runs:
        run = edge['node']
        if run['id'] == run_id or run['name'] == run_id:
            metrics_str = run.get('summaryMetrics', '{}')
            metrics = json.loads(metrics_str)

            z_mean = metrics.get('eval/z_mean')
            z_std = metrics.get('eval/z_std')
            raw_mean = metrics.get('eval/mean')
            raw_std = metrics.get('eval/std')

            print(f"\n{'='*80}")
            print(f"RUN: {run['name']}")
            print(f"{'='*80}")
            entity = os.environ.get("WANDB_ENTITY", "")
            project = os.environ.get("WANDB_PROJECT", "boxing-gym")
            print(f"URL: https://wandb.ai/{entity}/{project}/runs/{run['name']}")
            print(f"\nMetrics:")
            print(f"  Z-score mean: {z_mean:,.2f}")
            print(f"  Z-score std: {z_std:,.2f}" if z_std else "  Z-score std: N/A")
            print(f"  Raw mean: {raw_mean:,.2f}" if raw_mean else "  Raw mean: N/A")
            print(f"  Raw std: {raw_std:,.2f}" if raw_std else "  Raw std: N/A")

            # Calculate reverse prediction
            if z_mean:
                predicted_value = calculate_prediction_from_zscore(z_mean)
                print(f"\n💡 Calculated prediction from z-score:")
                print(f"  Prediction ≈ {predicted_value:,.0f}")

            # Show other relevant metrics
            print(f"\nOther metrics:")
            for key, value in metrics.items():
                if any(keyword in key for keyword in ['pred', 'answer', 'mse', 'final']):
                    print(f"  {key}: {value}")

            return metrics

    print(f"❌ Run {run_id} not found")
    return None


def main():
    """Extract details for extreme runs."""
    parser = argparse.ArgumentParser(description="Extract extreme prediction details")
    parser.add_argument(
        "--runs",
        default=os.environ.get("EXTREME_RUN_IDS", ""),
        help="Comma-separated run IDs (or set EXTREME_RUN_IDS env var)",
    )
    parser.add_argument(
        "--sweep-data",
        default=str(Path(__file__).parent.parent / "data" / "wandb_queries" / "sweep_data.json"),
        help="Path to sweep JSON data",
    )
    args = parser.parse_args()

    if not args.runs:
        print("❌ No run IDs specified. Use --runs or set EXTREME_RUN_IDS env var")
        return

    sweep_path = Path(args.sweep_data)
    if not sweep_path.exists():
        print(f"❌ Sweep data not found at {sweep_path}")
        return

    print("=" * 80)
    print("🔍 EXTRACTING EXTREME RUN DETAILS")
    print("=" * 80)

    extreme_runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    all_details = {}

    for run_id in extreme_runs:
        metrics = extract_run_details(sweep_path, run_id)
        if metrics:
            all_details[run_id] = metrics

    # Save details
    output_path = Path(__file__).parent.parent / "data" / "extreme_run_details.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_details, f, indent=2)

    print(f"\n💾 Details saved to: {output_path}")

    # Print URLs for manual investigation
    print("\n" + "=" * 80)
    print("📝 NEXT STEPS: MANUAL INVESTIGATION")
    print("=" * 80)
    print("\n1. Visit these Weave URLs to see raw LLM responses:")
    entity = os.environ.get("WANDB_ENTITY", "")
    project = os.environ.get("WANDB_PROJECT", "boxing-gym")
    for run_id in extreme_runs:
        print(f"   https://wandb.ai/{entity}/{project}/runs/{run_id}/weave")

    print("\n2. On each page:")
    print("   - Look for LiteLLM traces")
    print("   - Find the completion content")
    print("   - Check if the answer was in proper tags or extracted")

    print("\n3. Check for downloaded files:")
    print("   - Go to Files tab on WandB run page")
    print("   - Download any .log or .json files")

    print("\n4. To reproduce locally:")
    print("   - Check the run config in WandB")
    print("   - Run with same seed/env/model")
    print("   - Compare output")


if __name__ == "__main__":
    main()
