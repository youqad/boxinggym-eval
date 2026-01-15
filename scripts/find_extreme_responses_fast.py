#!/usr/bin/env python3
"""
Fast targeted search for extreme z-score runs using direct API queries.
"""

import wandb
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def main():
    """Fast targeted investigation."""
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY", "")
    project = os.environ.get("WANDB_PROJECT", "boxing-gym")

    print("=" * 80)
    print("🔍 FAST INVESTIGATION: Extreme Z-Score Runs")
    print("=" * 80)

    # Strategy 1: Check specific sweeps (from env var)
    sweep_ids = os.environ.get("SWEEP_IDS", "").split(",") if os.environ.get("SWEEP_IDS") else []
    if not sweep_ids:
        print("⚠️  No sweep IDs provided. Set SWEEP_IDS env var (comma-separated).")
        return

    for sweep_id in sweep_ids:
        print(f"\n🔄 Checking sweep: {sweep_id}")
        try:
            sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
            print(f"  Sweep name: {sweep.name}")
            print(f"  Sweep config: {sweep.config}")

            # Get just the first 20 runs from this sweep
            runs = list(sweep.runs)[:20]
            print(f"  Found {len(runs)} runs (showing first 20)")

            extreme_found = []
            for run in runs:
                z_mean = run.summary.get("metric/eval/z_mean")
                env = run.config.get("envs", "?")
                llm = run.config.get("llms", "?")

                if z_mean:
                    print(f"    {run.name}: z={z_mean:.1f}, env={env}, llm={llm}")

                    if z_mean > 1000:
                        print(f"      ⚠️  EXTREME Z-SCORE!")
                        extreme_found.append({
                            "sweep_id": sweep_id,
                            "run_id": run.id,
                            "run_name": run.name,
                            "url": run.url,
                            "z_mean": z_mean,
                            "env": env,
                            "llm": llm,
                        })

            if extreme_found:
                print(f"\n  ✅ Found {len(extreme_found)} extreme runs in this sweep")
                for r in extreme_found:
                    print(f"    - {r['run_name']}: {r['url']}")

        except Exception as e:
            print(f"  ❌ Error checking sweep: {e}")

    # Strategy 2: Direct run ID query if we know specific run IDs
    # (You can add specific run IDs here if known)

    # Strategy 3: Query runs with specific metric filters using GraphQL-style filters
    print("\n🔍 Querying runs with summary metric filters...")
    try:
        # WandB API supports filtering by summary metrics
        runs = api.runs(
            f"{entity}/{project}",
            filters={
                "$and": [
                    {"config.envs": "peregrines_direct"},
                    {"summary_metrics.metric/eval/z_mean": {"$gt": 1000}}
                ]
            },
            per_page=50
        )

        run_list = list(runs)
        print(f"  Found {len(run_list)} runs with z_mean > 1000 and peregrines_direct")

        for run in run_list[:10]:  # Show first 10
            z_mean = run.summary.get("metric/eval/z_mean")
            llm = run.config.get("llms", "?")
            print(f"    {run.name}: z={z_mean:.1f}, llm={llm}")
            print(f"      URL: {run.url}")

    except Exception as e:
        print(f"  ❌ Metric filter query failed: {e}")
        print("  (This is expected - WandB API may not support this filter syntax)")

    # Strategy 4: Manual URL construction
    print("\n📝 Manual investigation URLs:")
    print("\nSweep pages:")
    for sweep_id in sweep_ids:
        print(f"  https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")

    print("\n💡 Next steps:")
    print("1. Visit sweep pages above")
    print("2. Filter by 'metric/eval/z_mean' column")
    print("3. Sort descending to find extreme values")
    print("4. Click on runs to view Weave traces and logs")


if __name__ == "__main__":
    main()
