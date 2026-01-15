#!/usr/bin/env python3
"""
Analyze test set consistency before and after the bug fix.

This script compares evaluation test points between fixed and unfixed results
to demonstrate that:
1. BEFORE FIX: Same seed with different budgets used DIFFERENT test sets (BUG!)
2. AFTER FIX: Same seed with different budgets use IDENTICAL test sets (CORRECT!)

Usage:
    uv run python scripts/analyze_test_set_consistency.py
    uv run python scripts/analyze_test_set_consistency.py --unfixed results_unfixed --fixed results
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _as_bool(v) -> bool:
    # OmegaConf values sometimes roundtrip as strings in JSON; be tolerant.
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


def extract_budgeted_test_points(result_data: Dict) -> Dict[int, List]:
    """Extract evaluation test points per budget from a results JSON blob.

    In the current (recommended) format, a single JSON file contains multiple
    budgets in `config.exp.num_experiments` and multiple result entries in
    `data.results` (one per budget).

    Each result entry is shaped like:
        [ [err_mean, err_std], questions, gts, predictions, ... ]
    and the test points are `questions` at index 1.
    """
    cfg = result_data.get("config", {}) if isinstance(result_data, dict) else {}
    exp_cfg = cfg.get("exp") if isinstance(cfg, dict) else {}
    budgets = exp_cfg.get("num_experiments") if isinstance(exp_cfg, dict) else None

    data = result_data.get("data", {}) if isinstance(result_data, dict) else {}
    results = data.get("results") if isinstance(data, dict) else None

    if not isinstance(results, list) or not results:
        return {}

    out: Dict[int, List] = {}
    for i, entry in enumerate(results):
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        questions = entry[1]
        if not questions:
            continue

        budget = None
        if isinstance(budgets, list) and i < len(budgets):
            budget = budgets[i]
        elif isinstance(budgets, (int, float)):
            budget = int(budgets)
        else:
            # Fallback: use index if budgets metadata is missing.
            budget = i

        try:
            budget_int = int(budget)
        except Exception:
            continue
        out[budget_int] = questions

    return out


def load_results_from_dir(results_dir: str) -> Dict[str, Dict]:
    """
    Load all result files from directory.

    Returns dict: {env_name: {filename: result_data}}
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return {}

    env_results = defaultdict(dict)

    for env_dir in results_path.iterdir():
        if not env_dir.is_dir():
            continue

        env_name = env_dir.name

        for result_file in env_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                env_results[env_name][result_file.name] = data
            except Exception as e:
                print(f"Warning: Failed to load {result_file}: {e}")

    return env_results


def analyze_test_set_consistency(results_dir: str) -> Dict[str, Dict]:
    """
    Analyze test set consistency within a single results directory.

    For each (env, model, seed), check if test points are consistent across budgets.

    Returns:
        {
            env_name: {
                (model, seed): {
                    'budgets': {budget: test_points},
                    'consistent': bool,
                    'details': str
                }
            }
        }
    """
    env_results = load_results_from_dir(results_dir)
    analysis = defaultdict(dict)

    for env_name, files in env_results.items():
        # Group by (goal, model, include_prior, use_ppl, seed)
        grouped = defaultdict(dict)

        for filename, data in files.items():
            cfg = data.get("config", {}) if isinstance(data, dict) else {}
            env_cfg = cfg.get("envs") if isinstance(cfg, dict) else {}
            exp_cfg = cfg.get("exp") if isinstance(cfg, dict) else {}
            llm_cfg = cfg.get("llms") if isinstance(cfg, dict) else {}

            exp_type = exp_cfg.get("experiment_type") if isinstance(exp_cfg, dict) else None
            if exp_type != "oed":
                continue  # Only analyze OED experiments with budgets

            goal_name = env_cfg.get("goal_name") if isinstance(env_cfg, dict) else None
            model_name = llm_cfg.get("model_name") if isinstance(llm_cfg, dict) else None
            include_prior = _as_bool(cfg.get("include_prior", False)) if isinstance(cfg, dict) else False
            use_ppl = _as_bool(cfg.get("use_ppl", False)) if isinstance(cfg, dict) else False
            seed = cfg.get("seed") if isinstance(cfg, dict) else None

            if not goal_name or not model_name or seed is None:
                continue

            budget_points = extract_budgeted_test_points(data)
            if not budget_points:
                continue

            key = (goal_name, model_name, include_prior, use_ppl, str(seed))
            for budget, test_points in budget_points.items():
                grouped[key][budget] = test_points

        # Check consistency for each (model, seed)
        for key, budgets_data in grouped.items():
            goal_name, model_name, include_prior, use_ppl, seed = key

            if len(budgets_data) <= 1:
                # Need at least 2 budgets to check consistency
                continue

            # Check if all test points are identical
            test_points_list = list(budgets_data.values())
            first_points = test_points_list[0]
            all_same = all(points == first_points for points in test_points_list)

            budgets_str = ", ".join(str(b) for b in sorted(budgets_data.keys()))

            analysis[env_name][key] = {
                'budgets': budgets_data,
                'consistent': all_same,
                'details': f"Budgets: {budgets_str}, Test points: {len(first_points)} points",
                'goal': goal_name,
                'model': model_name,
                'include_prior': include_prior,
                'use_ppl': use_ppl,
                'seed': seed,
            }

    return analysis


def compare_directories(unfixed_dir: str, fixed_dir: str) -> None:
    """
    Compare test set consistency between unfixed and fixed results.
    """
    print("=" * 80)
    print("TEST SET CONSISTENCY ANALYSIS")
    print("=" * 80)
    print()

    print(f"Analyzing unfixed results: {unfixed_dir}")
    unfixed_analysis = analyze_test_set_consistency(unfixed_dir)

    print(f"Analyzing fixed results: {fixed_dir}")
    fixed_analysis = analyze_test_set_consistency(fixed_dir)

    print()
    print("=" * 80)
    print("UNFIXED RESULTS (Before Bug Fix)")
    print("=" * 80)
    print()

    if not unfixed_analysis:
        print("❌ No unfixed results found!")
        print(f"   Directory: {unfixed_dir}")
        print(f"   Expected structure: {unfixed_dir}/ENV_NAME/*.json")
        print()
    else:
        for env_name, keys_data in sorted(unfixed_analysis.items()):
            print(f"Environment: {env_name}")
            print("-" * 40)

            for key, info in sorted(keys_data.items()):
                goal_name, model_name, include_prior, use_ppl, seed = key
                status = "✅ CONSISTENT" if info['consistent'] else "❌ INCONSISTENT"
                print(f"  Goal: {goal_name}  Prior: {include_prior}  PPL: {use_ppl}")
                print(f"  Model: {model_name}, Seed: {seed}")
                print(f"  Status: {status}")
                print(f"  {info['details']}")

                if not info['consistent']:
                    # Show the difference
                    budgets = info['budgets']
                    print("  Test point differences:")
                    for budget, points in sorted(budgets.items(), key=lambda x: int(x[0])):
                        print(f"    Budget {budget}: {points[:3]}... ({len(points)} points)")

                print()

    print()
    print("=" * 80)
    print("FIXED RESULTS (After Bug Fix)")
    print("=" * 80)
    print()

    if not fixed_analysis:
        print("❌ No fixed results found!")
        print(f"   Directory: {fixed_dir}")
        print("   Results are still being generated...")
        print()
    else:
        for env_name, keys_data in sorted(fixed_analysis.items()):
            print(f"Environment: {env_name}")
            print("-" * 40)

            for key, info in sorted(keys_data.items()):
                goal_name, model_name, include_prior, use_ppl, seed = key
                status = "✅ CONSISTENT" if info['consistent'] else "❌ INCONSISTENT"
                print(f"  Goal: {goal_name}  Prior: {include_prior}  PPL: {use_ppl}")
                print(f"  Model: {model_name}, Seed: {seed}")
                print(f"  Status: {status}")
                print(f"  {info['details']}")

                if not info['consistent']:
                    # Show the difference
                    budgets = info['budgets']
                    print("  Test point differences:")
                    for budget, points in sorted(budgets.items(), key=lambda x: int(x[0])):
                        print(f"    Budget {budget}: {points[:3]}... ({len(points)} points)")

                print()

    # Summary statistics
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    if unfixed_analysis:
        total_unfixed = sum(len(keys) for keys in unfixed_analysis.values())
        consistent_unfixed = sum(
            sum(1 for info in keys.values() if info['consistent'])
            for keys in unfixed_analysis.values()
        )
        print(f"UNFIXED: {consistent_unfixed}/{total_unfixed} consistent")
        print(f"  Expected: 0/{total_unfixed} (bug present in all)")
    else:
        print("UNFIXED: No results available")

    if fixed_analysis:
        total_fixed = sum(len(keys) for keys in fixed_analysis.values())
        consistent_fixed = sum(
            sum(1 for info in keys.values() if info['consistent'])
            for keys in fixed_analysis.values()
        )
        print(f"FIXED: {consistent_fixed}/{total_fixed} consistent")
        print(f"  Expected: {total_fixed}/{total_fixed} (bug fixed in all)")
    else:
        print("FIXED: No results available yet")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze test set consistency before/after bug fix"
    )
    parser.add_argument(
        "--unfixed",
        default="results_unfixed",
        help="Directory with unfixed results (default: results_unfixed)"
    )
    parser.add_argument(
        "--fixed",
        default="results",
        help="Directory with fixed results (default: results)"
    )

    args = parser.parse_args()

    compare_directories(args.unfixed, args.fixed)


if __name__ == "__main__":
    main()
