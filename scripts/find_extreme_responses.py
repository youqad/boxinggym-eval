#!/usr/bin/env python3
"""
Find raw LLM responses for extreme z-score runs.

Investigates the catastrophic predictions (z-score > 1000) to understand
whether they came from properly formatted answers or emergency extraction.
"""

import wandb
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def find_extreme_runs(entity: str = None, project: str = "boxing-gym") -> List[Dict[str, Any]]:
    """Find runs with extreme z-scores."""
    entity = entity or os.environ.get("WANDB_ENTITY", "")
    api = wandb.Api()

    print("🔍 Searching for runs with extreme z-scores (>1000)...")

    # First, try with specific filters
    filters = {
        "config.envs": "peregrines_direct",
        "config.llms": "bedrock-qwen3-32b",
    }

    print(f"  Filter 1: {filters}")
    runs = api.runs(f"{entity}/{project}", filters=filters)

    extreme_runs = []
    checked_count = 0
    for run in runs:
        checked_count += 1
        # Check if z_mean is extreme
        z_mean = run.summary.get("metric/eval/z_mean")
        if z_mean is not None:
            print(f"    Run {run.name}: z_mean = {z_mean}")
            if z_mean > 1000:
                extreme_runs.append({
                    "id": run.id,
                    "name": run.name,
                    "z_mean": z_mean,
                    "state": run.state,
                    "created_at": run.created_at,
                    "config": run.config,
                    "summary": run.summary,
                    "url": run.url,
                })

    print(f"  Checked {checked_count} runs with filters")

    # If no results, try broader search
    if not extreme_runs:
        print("\n  No results with strict filters. Trying broader search...")
        print("  Filter 2: peregrines_direct only")
        runs = api.runs(f"{entity}/{project}", filters={"config.envs": "peregrines_direct"})

        checked_count = 0
        for run in runs:
            checked_count += 1
            z_mean = run.summary.get("metric/eval/z_mean")
            if z_mean is not None and z_mean > 1000:
                print(f"    Found extreme run: {run.name} (z_mean={z_mean})")
                extreme_runs.append({
                    "id": run.id,
                    "name": run.name,
                    "z_mean": z_mean,
                    "state": run.state,
                    "created_at": run.created_at,
                    "config": run.config,
                    "summary": run.summary,
                    "url": run.url,
                })

        print(f"  Checked {checked_count} runs with broader filter")

    # If still no results, try even broader
    if not extreme_runs:
        print("\n  Still no results. Trying broadest search (any z_mean > 100)...")
        runs = api.runs(f"{entity}/{project}")

        checked_count = 0
        for run in runs:
            checked_count += 1
            if checked_count % 100 == 0:
                print(f"    Checked {checked_count} runs...")

            z_mean = run.summary.get("metric/eval/z_mean")
            if z_mean is not None and z_mean > 100:
                env = run.config.get("envs", "unknown")
                llm = run.config.get("llms", "unknown")
                print(f"    Found high z-score: {run.name} (z_mean={z_mean}, env={env}, llm={llm})")

                if z_mean > 1000:
                    extreme_runs.append({
                        "id": run.id,
                        "name": run.name,
                        "z_mean": z_mean,
                        "state": run.state,
                        "created_at": run.created_at,
                        "config": run.config,
                        "summary": run.summary,
                        "url": run.url,
                    })

    print(f"\n✅ Found {len(extreme_runs)} runs with extreme z-scores (>1000)")
    return extreme_runs


def extract_run_artifacts(run_id: str, entity: str = None, project: str = "boxing-gym") -> Dict[str, Any]:
    """Extract all artifacts and logs from a specific run."""
    entity = entity or os.environ.get("WANDB_ENTITY", "")
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    result = {
        "run_id": run_id,
        "artifacts": [],
        "files": [],
        "logs": None,
        "history": None,
        "raw_responses": [],
    }

    print(f"\n📦 Extracting artifacts for run {run_id}...")

    # 1. Check artifacts
    try:
        artifacts = run.logged_artifacts()
        for artifact in artifacts:
            result["artifacts"].append({
                "name": artifact.name,
                "type": artifact.type,
                "description": artifact.description,
            })
            print(f"  - Artifact: {artifact.name} ({artifact.type})")
    except Exception as e:
        print(f"  ⚠️  Error fetching artifacts: {e}")

    # 2. Check files
    try:
        files = run.files()
        for file in files:
            result["files"].append({
                "name": file.name,
                "size": file.size,
                "url": file.url,
            })
            print(f"  - File: {file.name} ({file.size} bytes)")

            # Download and check specific files
            if file.name.endswith(('.log', '.json', '.txt', '.yaml')):
                try:
                    download_path = f"/tmp/wandb_{run_id}_{file.name}"
                    file.download(root="/tmp", replace=True)

                    # Read content
                    if os.path.exists(download_path):
                        with open(download_path, 'r') as f:
                            content = f.read()
                            result["files"][-1]["content_preview"] = content[:1000]
                except Exception as e:
                    print(f"    ⚠️  Error downloading {file.name}: {e}")
    except Exception as e:
        print(f"  ⚠️  Error fetching files: {e}")

    # 3. Check history for step-level data
    try:
        history = run.history()
        if not history.empty:
            # Look for any columns with "response" or "answer" in the name
            response_cols = [col for col in history.columns if any(
                keyword in col.lower() for keyword in ['response', 'answer', 'prediction', 'output']
            )]

            if response_cols:
                print(f"  📊 Found response columns: {response_cols}")
                result["history"] = history[response_cols].to_dict('records')
    except Exception as e:
        print(f"  ⚠️  Error fetching history: {e}")

    # 4. Check for Weave traces
    try:
        # Weave traces might be stored as artifacts or in a specific format
        weave_artifacts = [a for a in result["artifacts"] if "weave" in a["name"].lower()]
        if weave_artifacts:
            print(f"  🔗 Found Weave artifacts: {[a['name'] for a in weave_artifacts]}")
    except Exception as e:
        print(f"  ⚠️  Error checking Weave traces: {e}")

    return result


def check_sweep_runs(sweep_ids: List[str], entity: str = None, project: str = "boxing-gym") -> List[Dict[str, Any]]:
    """Check specific sweeps for extreme runs."""
    entity = entity or os.environ.get("WANDB_ENTITY", "")
    api = wandb.Api()

    all_extreme_runs = []

    for sweep_id in sweep_ids:
        print(f"\n🔄 Checking sweep {sweep_id}...")
        try:
            sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
            runs = sweep.runs

            for run in runs:
                z_mean = run.summary.get("metric/eval/z_mean")
                if z_mean and z_mean > 1000:
                    print(f"  ⚠️  Found extreme run: {run.name} (z_mean={z_mean})")
                    all_extreme_runs.append({
                        "sweep_id": sweep_id,
                        "run_id": run.id,
                        "run_name": run.name,
                        "z_mean": z_mean,
                        "url": run.url,
                    })
        except Exception as e:
            print(f"  ⚠️  Error checking sweep {sweep_id}: {e}")

    return all_extreme_runs


def search_for_specific_value(value: int = 67219139, entity: str = None, project: str = "boxing-gym") -> List[Dict[str, Any]]:
    """Search for runs that contain a specific prediction value."""
    entity = entity or os.environ.get("WANDB_ENTITY", "")
    api = wandb.Api()

    print(f"\n🎯 Searching for runs with prediction value: {value:,}...")

    # Get all peregrines_direct runs
    runs = api.runs(f"{entity}/{project}", filters={
        "config.envs": "peregrines_direct",
    })

    matching_runs = []
    for run in runs:
        # Check history for the specific value
        try:
            history = run.history()
            if not history.empty:
                # Look through all numeric columns
                for col in history.columns:
                    if history[col].dtype in ['int64', 'float64']:
                        if (history[col] == value).any():
                            print(f"  ✅ Found value in run {run.name}, column {col}")
                            matching_runs.append({
                                "run_id": run.id,
                                "run_name": run.name,
                                "column": col,
                                "url": run.url,
                            })
        except Exception as e:
            pass  # Skip runs with issues

    return matching_runs


def analyze_response_extraction(run_id: str, entity: str = None, project: str = "boxing-gym"):
    """Analyze how responses were extracted in a specific run."""
    entity = entity or os.environ.get("WANDB_ENTITY", "")
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    print(f"\n🔬 Analyzing response extraction for run {run_id}...")

    # Check config for relevant settings
    config = run.config
    print("\n📋 Relevant config:")
    print(f"  - LLM: {config.get('llms')}")
    print(f"  - Environment: {config.get('envs')}")
    print(f"  - Seed: {config.get('seed')}")

    # Check summary for extraction metrics
    summary = run.summary
    print("\n📊 Summary metrics:")
    for key, value in summary.items():
        if any(keyword in key.lower() for keyword in ['z_', 'pred', 'answer', 'response', 'extract']):
            print(f"  - {key}: {value}")


def main():
    """Main investigation workflow."""
    entity = os.environ.get("WANDB_ENTITY", "")
    project = os.environ.get("WANDB_PROJECT", "boxing-gym")

    print("=" * 80)
    print("🔍 INVESTIGATION: Finding Raw LLM Responses for Extreme Z-Score Runs")
    print("=" * 80)

    # Step 0: Check specific sweeps FIRST (faster)
    print("\n" + "=" * 80)
    print("🔄 Checking specific sweeps first...")
    print("=" * 80)

    sweep_ids_env = os.environ.get("SWEEP_IDS", "")
    sweep_ids = [s.strip() for s in sweep_ids_env.split(",") if s.strip()] if sweep_ids_env else []
    sweep_runs = check_sweep_runs(sweep_ids, entity, project)

    # Step 1: Find extreme runs
    extreme_runs = find_extreme_runs(entity, project)

    # Combine sweep runs with extreme runs
    all_extreme_runs = extreme_runs + [r for r in sweep_runs if r not in extreme_runs]

    if not all_extreme_runs:
        print("\n❌ No extreme runs found in either search or sweeps")
        print("\n💡 Tip: The runs might have been deleted or the metrics might be stored differently")
        return

    # Display summary
    print("\n📊 Summary of extreme runs:")
    df = pd.DataFrame(all_extreme_runs)
    if 'z_mean' in df.columns:
        print(df[['run_name' if 'run_name' in df.columns else 'name', 'z_mean']].to_string(index=False))
    else:
        print(df.to_string(index=False))

    # Step 3: Search for specific value
    specific_matches = search_for_specific_value(67219139, entity, project)

    if specific_matches:
        print(f"\n✅ Found {len(specific_matches)} runs with the exact value 67,219,139")
        for match in specific_matches:
            print(f"  - {match['run_name']}: {match['url']}")

    # Step 4: Extract detailed artifacts from first extreme run
    if all_extreme_runs:
        first_run = all_extreme_runs[0]
        run_name = first_run.get('run_name') or first_run.get('name')
        run_id = first_run.get('run_id') or first_run.get('id')

        print("\n" + "=" * 80)
        print(f"📦 Detailed artifact extraction for: {run_name}")
        print("=" * 80)

        artifacts = extract_run_artifacts(run_id, entity, project)
        analyze_response_extraction(run_id, entity, project)

        # Save results
        output_path = Path(__file__).parent.parent / "data" / "extreme_run_investigation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "extreme_runs": extreme_runs,
                "sweep_runs": sweep_runs,
                "specific_matches": specific_matches,
                "detailed_artifacts": artifacts,
            }, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_path}")

    # Step 5: Provide manual investigation instructions
    print("\n" + "=" * 80)
    print("📝 MANUAL INVESTIGATION STEPS")
    print("=" * 80)
    print("\nFor each extreme run, check:")
    print("1. WandB run page → Logs tab → Look for raw LLM responses")
    print("2. WandB run page → Weave tab → Check LiteLLM traces")
    print("3. WandB run page → Files tab → Download any .log or .json files")
    print("4. Run the code locally with same seed to reproduce")
    print("\nURLs to investigate:")
    for run in all_extreme_runs[:5]:  # Top 5
        url = run.get('url')
        if url:
            print(f"  - {url}")


if __name__ == "__main__":
    main()
