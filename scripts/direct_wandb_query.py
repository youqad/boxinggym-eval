#!/usr/bin/env python3
"""
Direct WandB API query using their web API.
"""

import requests
import json
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


def query_sweep_via_api(entity: str, project: str, sweep_id: str):
    """Query sweep data via WandB's GraphQL API."""
    api_key = os.getenv("WANDB_API_KEY")

    if not api_key:
        print("❌ WANDB_API_KEY not found in environment")
        return None

    # WandB GraphQL endpoint
    url = "https://api.wandb.ai/graphql"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Query for sweep and its runs
    query = """
    query SweepQuery($entity: String!, $project: String!, $sweep: String!) {
      project(name: $project, entityName: $entity) {
        sweep(sweepName: $sweep) {
          id
          name
          config
          runs(first: 100) {
            edges {
              node {
                id
                name
                state
                summaryMetrics
                config
                historyLineCount
              }
            }
          }
        }
      }
    }
    """

    variables = {
        "entity": entity,
        "project": project,
        "sweep": sweep_id,
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"query": query, "variables": variables},
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(response.text[:500])
            return None

    except Exception as e:
        print(f"❌ Request error: {e}")
        return None


def analyze_sweep_results(data: dict):
    """Analyze sweep data for extreme z-scores."""
    if not data or 'data' not in data:
        print("❌ No data in response")
        return []

    sweep_data = data['data']['project']['sweep']
    if not sweep_data:
        print("❌ Sweep not found")
        return []

    print(f"\n📊 Sweep: {sweep_data.get('name', 'unnamed')}")
    print(f"Config: {json.dumps(sweep_data.get('config', {}), indent=2)[:200]}...")

    runs = sweep_data.get('runs', {}).get('edges', [])
    print(f"\n🏃 Found {len(runs)} runs")

    extreme_runs = []

    for edge in runs:
        run = edge['node']
        metrics = run.get('summaryMetrics', '{}')

        # Parse metrics JSON
        try:
            if isinstance(metrics, str):
                metrics_dict = json.loads(metrics)
            else:
                metrics_dict = metrics

            # Try different metric keys
            z_mean = (
                metrics_dict.get('metric/eval/z_mean') or
                metrics_dict.get('eval/z_mean')
            )

            # Parse config JSON if it's a string
            config = run.get('config', '{}')
            if isinstance(config, str):
                config = json.loads(config)

            # Extract values from sweep config format
            env_val = config.get('envs', {})
            if isinstance(env_val, dict):
                env = env_val.get('value', '?')
            else:
                env = env_val

            llm_val = config.get('llms', {})
            if isinstance(llm_val, dict):
                llm = llm_val.get('value', '?')
            else:
                llm = llm_val

            if z_mean:
                print(f"  {run['name'][:40]:40} z={z_mean:8.2f} env={env:20} llm={llm}")

                if z_mean > 1000:
                    extreme_runs.append({
                        'run_id': run['id'],
                        'run_name': run['name'],
                        'z_mean': z_mean,
                        'env': env,
                        'llm': llm,
                        'state': run.get('state'),
                        'config': run.get('config', {}),
                    })
                    print(f"      ⚠️  EXTREME Z-SCORE: {z_mean:,.2f}")

        except Exception as e:
            pass  # Skip problematic runs

    return extreme_runs


def get_run_history(entity: str, project: str, run_id: str):
    """Get detailed history for a specific run."""
    api_key = os.getenv("WANDB_API_KEY")

    url = "https://api.wandb.ai/graphql"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    query = """
    query RunQuery($entity: String!, $project: String!, $name: String!) {
      project(name: $project, entityName: $entity) {
        run(name: $name) {
          id
          name
          summaryMetrics
          historyKeys
          files(first: 100) {
            edges {
              node {
                id
                name
                sizeBytes
                url
              }
            }
          }
        }
      }
    }
    """

    variables = {
        "entity": entity,
        "project": project,
        "name": run_id,
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"query": query, "variables": variables},
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get run history: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Main query workflow."""
    entity = os.environ.get("WANDB_ENTITY", "")
    project = os.environ.get("WANDB_PROJECT", "boxing-gym")
    sweep_ids = os.environ.get("SWEEP_IDS", "").split(",") if os.environ.get("SWEEP_IDS") else []

    if not sweep_ids:
        print("⚠️  No sweep IDs provided. Set SWEEP_IDS env var (comma-separated).")
        return

    print("=" * 80)
    print("🔍 DIRECT WANDB API QUERY")
    print("=" * 80)

    all_extreme = []

    for sweep_id in sweep_ids:
        print(f"\n{'=' * 80}")
        print(f"🔄 Querying sweep: {sweep_id}")
        print(f"{'=' * 80}")

        data = query_sweep_via_api(entity, project, sweep_id)

        if data:
            extreme = analyze_sweep_results(data)
            all_extreme.extend(extreme)

            # Save raw response
            output_dir = Path(__file__).parent.parent / "data" / "wandb_queries"
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / f"sweep_{sweep_id}.json", 'w') as f:
                json.dump(data, f, indent=2)

            print(f"\n💾 Saved raw data to: {output_dir}/sweep_{sweep_id}.json")

    # Summary
    print("\n" + "=" * 80)
    print(f"📊 SUMMARY: Found {len(all_extreme)} runs with extreme z-scores (>1000)")
    print("=" * 80)

    if all_extreme:
        for run in all_extreme:
            print(f"\n🚨 {run['run_name']}")
            print(f"   Z-score: {run['z_mean']:,.2f}")
            print(f"   Environment: {run['env']}")
            print(f"   Model: {run['llm']}")
            print(f"   URL: https://wandb.ai/{entity}/{project}/runs/{run['run_id']}")

            # Try to get detailed history
            print(f"   Fetching detailed data...")
            history = get_run_history(entity, project, run['run_id'])

            if history:
                run_data = history.get('data', {}).get('project', {}).get('run', {})
                files = run_data.get('files', {}).get('edges', [])

                print(f"   Files: {len(files)}")
                for file_edge in files[:5]:
                    file_node = file_edge['node']
                    print(f"     - {file_node['name']} ({file_node['sizeBytes']} bytes)")

        # Save summary
        summary_path = Path(__file__).parent.parent / "data" / "extreme_runs_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_extreme, f, indent=2)

        print(f"\n💾 Summary saved to: {summary_path}")

    else:
        print("\n❌ No extreme runs found")

    print("\n📝 Next steps:")
    print("1. Visit the WandB run URLs above")
    print("2. Click on 'Weave' tab to see LiteLLM traces")
    print("3. Look for raw LLM responses in traces")
    print("4. Check 'Files' tab for any saved logs")


if __name__ == "__main__":
    main()
