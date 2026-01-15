#!/usr/bin/env python3
"""
Quick analysis of completed MiniMax M2.1 benchmark results
Analyzes budget progression and cross-environment performance
"""

import json
from pathlib import Path
import statistics

def load_result(filepath):
    """Load and parse a result JSON file"""
    with open(filepath) as f:
        return json.load(f)

def extract_metrics(result):
    """Extract key metrics from a result file"""
    config = result['config']
    data = result['data']

    budgets = config.get('exp', {}).get('num_experiments')

    # Extract seed
    seed = config['seed']

    # Extract environment
    env = config['envs']['env_name']

    # Extract results (predictions vs ground truth)
    results_data = data.get('results', [])
    if not isinstance(results_data, list):
        results_data = []

    rows = []
    for i, entry in enumerate(results_data):
        if not entry or not isinstance(entry, list) or len(entry) < 1:
            continue
        metrics = entry[0]
        if not isinstance(metrics, list) or len(metrics) < 2:
            continue

        # Determine budget for this result entry
        budget = None
        if isinstance(budgets, list) and i < len(budgets):
            budget = budgets[i]
        elif isinstance(budgets, (int, float)):
            budget = int(budgets)
        else:
            budget = i

        mean_error = abs(metrics[0]) if metrics[0] is not None else None
        std_error = metrics[1] if len(metrics) > 1 else None

        rows.append({
            'env': env,
            'seed': seed,
            'budget': int(budget),
            'mean_error': mean_error,
            'std_error': std_error,
            'queries': data.get('queries', []),
            'observations': data.get('observations', []),
            'successes': data.get('successes', []),
        })

    return rows

def analyze_environment(results, env_name):
    """Analyze results for a specific environment"""
    env_results = [r for r in results if r['env'] == env_name]

    if not env_results:
        return None

    print(f"\n{'='*60}")
    print(f"📊 {env_name.upper()} ANALYSIS")
    print(f"{'='*60}")

    # Group by budget
    budgets = sorted(set(r['budget'] for r in env_results))

    for budget in budgets:
        budget_results = [r for r in env_results if r['budget'] == budget]
        errors = [r['mean_error'] for r in budget_results if r['mean_error'] is not None]

        if errors:
            avg_error = statistics.mean(errors)
            std_dev = statistics.stdev(errors) if len(errors) > 1 else 0.0

            print(f"\n🎯 Budget={budget} ({len(budget_results)} runs)")
            print(f"   Mean Error: {avg_error:.6f}")
            print(f"   Std Dev:    {std_dev:.6f}")
            print(f"   Seeds:      {sorted([r['seed'] for r in budget_results])}")

    # Budget progression analysis
    print("\n📈 BUDGET PROGRESSION:")
    budget_means = {}
    for budget in budgets:
        budget_results = [r for r in env_results if r['budget'] == budget]
        errors = [r['mean_error'] for r in budget_results if r['mean_error'] is not None]
        if errors:
            budget_means[budget] = statistics.mean(errors)

    if len(budget_means) >= 2:
        sorted_budgets = sorted(budget_means.keys())
        for i in range(len(sorted_budgets) - 1):
            b1, b2 = sorted_budgets[i], sorted_budgets[i+1]
            improvement = budget_means[b1] - budget_means[b2]
            pct_change = (improvement / budget_means[b1] * 100) if budget_means[b1] != 0 else 0

            arrow = "⬇️" if improvement > 0 else "⬆️"
            print(f"   {b1}→{b2}: {arrow} {improvement:+.6f} ({pct_change:+.1f}%)")

    return budget_means

def main():
    print("🥊 MiniMax M2.1 Benchmark Analysis")
    print("="*60)

    # Find all result files
    results_dir = Path("results")
    legacy_files = list(results_dir.glob("*/direct_openai_MiniMax-M2.1-boxloop_oed_*.json"))
    new_files = list(
        results_dir.glob(
            "*/env=*_goal=*_model=openai_MiniMax-M2.1-boxloop_exp=oed_prior=*_seed=*_*.json"
        )
    )
    result_files = sorted(set(legacy_files + new_files))

    print(f"\n📁 Found {len(result_files)} result files")

    # Load all results
    results = []
    for filepath in result_files:
        try:
            result = load_result(filepath)
            metrics_rows = extract_metrics(result)
            results.extend(metrics_rows)
        except Exception as e:
            print(f"⚠️  Error loading {filepath.name}: {e}")

    print(f"✅ Loaded {len(results)} results successfully")

    # Get unique environments
    environments = sorted(set(r['env'] for r in results))
    print(f"\n🌍 Environments: {', '.join(environments)}")

    # Analyze each environment
    all_budget_means = {}
    for env in environments:
        budget_means = analyze_environment(results, env)
        if budget_means:
            all_budget_means[env] = budget_means

    # Cross-environment comparison
    if len(all_budget_means) > 1:
        print(f"\n{'='*60}")
        print("🔀 CROSS-ENVIRONMENT COMPARISON")
        print(f"{'='*60}")

        # Find common budgets
        common_budgets = set.intersection(*[set(means.keys()) for means in all_budget_means.values()])

        for budget in sorted(common_budgets):
            print(f"\n💰 Budget={budget}:")
            for env in sorted(all_budget_means.keys()):
                if budget in all_budget_means[env]:
                    error = all_budget_means[env][budget]
                    print(f"   {env:15s}: {error:.6f}")

    # Success rate analysis
    print(f"\n{'='*60}")
    print("✅ SUCCESS RATES")
    print(f"{'='*60}")

    # Successes are per run (not per budget). De-duplicate by (env, seed).
    for env in environments:
        env_rows = [r for r in results if r['env'] == env]
        by_seed = {}
        for r in env_rows:
            by_seed[r['seed']] = r  # last row wins; successes are identical across budgets
        total_runs = len(by_seed)
        successful = sum(1 for r in by_seed.values() if all(r.get('successes', [])))
        success_rate = (successful / total_runs * 100) if total_runs > 0 else 0

        print(f"\n{env}:")
        print(f"   Total runs: {total_runs}")
        print(f"   Successful: {successful}")
        print(f"   Success rate: {success_rate:.1f}%")

    print(f"\n{'='*60}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
