#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


def add_src_to_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def get_norm_factors(env_name: str) -> Tuple[float, float]:
    add_src_to_path()
    if env_name == "dugongs":
        from boxing_gym.envs import dugongs as mod
        env = mod.Dugongs()
        goal = mod.DirectGoal(env)
    elif env_name == "peregrines":
        from boxing_gym.envs import peregrines as mod
        env = mod.Peregrines()
        goal = mod.DirectGoal(env)
    elif env_name == "lotka_volterra":
        from boxing_gym.envs import lotka_volterra as mod
        env = mod.LotkaVolterra()
        goal = mod.DirectGoal(env)
    elif env_name == "hyperbolic_temporal_discount":
        from boxing_gym.envs import hyperbolic_temporal_discount as mod
        env = mod.TemporalDiscount()
        goal = mod.DirectGoal(env)
    elif env_name == "irt":
        from boxing_gym.envs import irt as mod
        env = mod.IRT()
        goal = mod.DirectCorrectness(env)
    elif env_name == "survival":
        from boxing_gym.envs import survival_analysis as mod
        env = mod.SurvivalAnalysis()
        goal = mod.DirectGoal(env)
    elif env_name == "location_finding":
        from boxing_gym.envs import location_finding as mod
        env = mod.Signal()
        goal = mod.DirectGoal(env)
    elif env_name == "death_process":
        from boxing_gym.envs import death_process as mod
        env = mod.DeathProcess()
        goal = mod.DirectDeath(env)
    elif env_name == "emotion":
        from boxing_gym.envs import emotion as mod
        env = mod.EmotionFromOutcome()
        goal = mod.DirectEmotionPrediction(env)
    elif env_name == "morals":
        from boxing_gym.envs import moral_machines as mod
        env = mod.MoralMachine()
        goal = mod.DirectPrediction(env)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")
    return float(goal.norm_mu), float(goal.norm_sigma)


PAPER_DISCOVERY10_PRIOR = {
    # These come from the flattened LaTeX tables (Discovery@10, with prior)
    "dugongs": (-0.06, 0.04),
    "peregrines": (-0.65, 0.02),
    "lotka_volterra": (-0.01, 0.12),
}


ENV_METRIC = {
    "dugongs": "MSE",
    "peregrines": "MSE",
    "lotka_volterra": "MAE",
    "hyperbolic_temporal_discount": "MSE",
    "irt": "MSE",
    "survival": "MSE",
    "location_finding": "MSE",
    "death_process": "MSE",
    "emotion": "MSE",
    "morals": "MSE",
}


def iter_json_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".json"):
                yield os.path.join(dirpath, fn)


def load_result(path: str) -> Optional[Dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def standardize(err_mean: float, err_std: float, mu0: float, sigma0: float) -> Tuple[float, float]:
    return (err_mean - mu0) / sigma0, err_std / sigma0


def _standardize_with_norm_factors(
    err_mean: float,
    err_std: float,
    norm_factors: Any,
) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(norm_factors, dict):
        return None, None
    mu0 = norm_factors.get("mu")
    sigma0 = norm_factors.get("sigma")
    if mu0 is None or sigma0 is None:
        return None, None
    try:
        sigma = float(sigma0)
        if sigma <= 0:
            return None, None
        mu = float(mu0)
        return (float(err_mean) - mu) / sigma, float(err_std) / sigma
    except Exception:
        return None, None


def verify_seed_reproducibility(root: str) -> Tuple[bool, List[str]]:
    """
    Verify that same seeds produce identical test sets across different budgets.

    Returns:
        (is_reproducible, issues_list)
        - is_reproducible: True if all seeds are reproducible, False otherwise
        - issues_list: List of issue descriptions if not reproducible
    """
    from collections import defaultdict

    # Group results by (env, model, seed)
    test_points_by_key = defaultdict(dict)
    issues = []

    for p in iter_json_files(root):
        blob = load_result(p)
        if not blob or not isinstance(blob, dict):
            continue

        cfg = blob.get("config", {})
        env_cfg = cfg.get("envs") or {}
        exp_cfg = cfg.get("exp") or {}
        env_name = env_cfg.get("env_name")
        model_name = (cfg.get("llms") or {}).get("model_name")
        seed = cfg.get("seed")
        experiment_type = exp_cfg.get("experiment_type")
        budgets = exp_cfg.get("num_experiments")

        if not env_name or not model_name or seed is None or experiment_type != "oed":
            continue

        # Extract test points
        all_res = (blob.get("data") or {}).get("results") or []
        for i, entry in enumerate(all_res):
            if not entry or not isinstance(entry, list) or len(entry) < 2:
                continue

            budget = None
            if isinstance(budgets, list) and i < len(budgets):
                budget = budgets[i]

            if budget is None:
                continue

            # Test points are in entry[1]
            test_points = entry[1]
            if not test_points:
                continue

            key = (env_name, model_name, seed)
            test_points_by_key[key][budget] = tuple(test_points)

    # Check consistency for each (env, model, seed)
    is_reproducible = True
    for key, budget_points in test_points_by_key.items():
        env_name, model_name, seed = key

        if len(budget_points) <= 1:
            continue

        # Check if all budgets have identical test points
        points_list = list(budget_points.values())
        first_points = points_list[0]

        if not all(points == first_points for points in points_list):
            is_reproducible = False
            budgets_str = ", ".join(str(b) for b in sorted(budget_points.keys()))
            issue = (
                f"INCONSISTENT test points: env={env_name}, model={model_name}, seed={seed}, "
                f"budgets=[{budgets_str}]"
            )
            issues.append(issue)

    return is_reproducible, issues


def main():
    ap = argparse.ArgumentParser(description="Aggregate and standardize BoxingGym results to CSV.")
    ap.add_argument("root", nargs="?", default="results", help="Results root (default: results)")
    ap.add_argument("--out", default="outputs/standardized_results.csv", help="Output CSV path")
    ap.add_argument("--skip-reproducibility-check", action="store_true",
                    help="Skip seed reproducibility verification")
    args = ap.parse_args()

    # Verify seed reproducibility first
    if not args.skip_reproducibility_check:
        print("=" * 80)
        print("VERIFYING SEED REPRODUCIBILITY...")
        print("=" * 80)
        is_reproducible, issues = verify_seed_reproducibility(args.root)

        if is_reproducible:
            print("✅ PASSED: All seeds produce identical test sets across budgets!")
        else:
            print("❌ FAILED: Found test set inconsistencies!")
            print("\nIssues detected:")
            for issue in issues:
                print(f"  - {issue}")
            print("\n⚠️  WARNING: Budget comparisons may be invalid due to different test sets!")
            print("    Consider running with --skip-reproducibility-check to proceed anyway.")
            print()
            return

        print()
        print("=" * 80)
        print("AGGREGATING RESULTS...")
        print("=" * 80)
        print()

    rows: List[Dict] = []
    for p in iter_json_files(args.root):
        blob = load_result(p)
        if not blob or not isinstance(blob, dict):
            continue
        cfg = blob.get("config", {})
        env_cfg = (cfg.get("envs") or {})
        exp_cfg = (cfg.get("exp") or {})
        env_name = env_cfg.get("env_name")
        goal_name = env_cfg.get("goal_name")
        com_limit = env_cfg.get("com_limit")
        include_prior = bool(cfg.get("include_prior", False))
        model_name = ((cfg.get("llms") or {}).get("model_name"))
        seed = cfg.get("seed")
        use_ppl = bool(cfg.get("use_ppl", False))
        experiment_type = exp_cfg.get("experiment_type")
        budgets = exp_cfg.get("num_experiments")

        if not env_name or not goal_name:
            continue

        metric = ENV_METRIC.get(env_name, "MSE")
        data = blob.get("data") or {}
        all_res = data.get("results") or []
        z_results_precomputed = data.get("z_results") or []
        norm_factors = data.get("norm_factors") or {}

        z_by_budget: Dict[int, Dict[str, Any]] = {}
        for zr in z_results_precomputed:
            if not isinstance(zr, dict):
                continue
            b = zr.get("budget")
            if b is None:
                continue
            try:
                z_by_budget[int(b)] = zr
            except Exception:
                continue

        mu0 = sigma0 = None

        for i, entry in enumerate(all_res):
            if not entry or not isinstance(entry, list) or not entry[0]:
                continue
            # Skip failed evaluations (entry[0] is a dict) or malformed entries
            if isinstance(entry[0], dict):
                continue
            try:
                err_mean, err_std = entry[0]
            except (ValueError, TypeError):
                continue

            budget = None
            if isinstance(budgets, list) and i < len(budgets):
                budget = budgets[i]
            elif isinstance(budgets, (int, float)):
                budget = int(budgets)

            z_entry = None
            if budget is not None and z_by_budget:
                try:
                    z_entry = z_by_budget.get(int(budget))
                except Exception:
                    z_entry = None
            if (
                z_entry is None
                and (budget is None or not z_by_budget)
                and z_results_precomputed
                and i < len(z_results_precomputed)
            ):
                z_entry = z_results_precomputed[i]

            z_mean = z_std = None
            if isinstance(z_entry, dict):
                z_mean = z_entry.get("z_mean")
                z_std = z_entry.get("z_std")
                if budget is None:
                    budget = z_entry.get("budget")

            if z_mean is None or z_std is None:
                z_mean, z_std = _standardize_with_norm_factors(
                    float(err_mean), float(err_std), norm_factors
                )

            if z_mean is None or z_std is None:
                if mu0 is None or sigma0 is None:
                    try:
                        mu0, sigma0 = get_norm_factors(env_name)
                    except Exception:
                        continue
                z_mean, z_std = standardize(float(err_mean), float(err_std), float(mu0), float(sigma0))

            paper_mean = paper_se = delta_vs_paper = None
            if (
                goal_name == "direct_naive"
                and include_prior
                and budget == 10
                and env_name in PAPER_DISCOVERY10_PRIOR
            ):
                paper_mean, paper_se = PAPER_DISCOVERY10_PRIOR[env_name]
                delta_vs_paper = z_mean - paper_mean

            rows.append(
                {
                    "path": p,
                    "env": env_name,
                    "goal": goal_name,
                    "experiment_type": experiment_type,
                    "include_prior": include_prior,
                    "use_ppl": use_ppl,
                    "model": model_name,
                    "seed": seed,
                    "budget": budget,
                    "metric": metric,
                    "raw_mean": f"{float(err_mean):.6g}",
                    "raw_std": f"{float(err_std):.6g}",
                    "z_mean": f"{z_mean:.6g}",
                    "z_std": f"{z_std:.6g}",
                    "paper_discovery10_mean": (
                        f"{paper_mean:.6g}" if paper_mean is not None else ""
                    ),
                    "paper_discovery10_se": (
                        f"{paper_se:.6g}" if paper_se is not None else ""
                    ),
                    "delta_vs_paper": (
                        f"{delta_vs_paper:.6g}" if delta_vs_paper is not None else ""
                    ),
                    "com_limit": com_limit,
                }
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = [
        "path",
        "env",
        "goal",
        "experiment_type",
        "include_prior",
        "use_ppl",
        "model",
        "seed",
        "budget",
        "metric",
        "raw_mean",
        "raw_std",
        "z_mean",
        "z_std",
        "paper_discovery10_mean",
        "paper_discovery10_se",
        "delta_vs_paper",
        "com_limit",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()

