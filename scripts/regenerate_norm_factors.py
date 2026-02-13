#!/usr/bin/env python
"""Regenerate baseline normalization factors for all BoxingGym environments.

Calls each Goal's get_norm_factors() method across multiple seeds and reports
the median. Use this to verify/update the hardcoded norm_mu, norm_sigma values
in each environment file.

Usage:
    uv run python scripts/regenerate_norm_factors.py
    uv run python scripts/regenerate_norm_factors.py --seeds 42 123 456
    uv run python scripts/regenerate_norm_factors.py --env death_process
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from statistics import median

import numpy as np

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from boxing_gym.envs.registry import get_environment_registry

logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)

DEFAULT_SEEDS = [42, 123, 456, 789, 101112]

# (env_name, goal_key) pairs for the primary direct goals
PRIMARY_GOALS = [
    ("death_process", "direct"),
    ("dugongs", "direct"),
    ("emotion", "direct"),
    ("hyperbolic_temporal_discount", "direct"),
    ("irt", "direct"),
    ("location_finding", "direct"),
    ("lotka_volterra", "direct"),
    ("morals", "direct"),
    ("peregrines", "direct"),
    ("survival", "direct"),
]


def compute_norm_factors(env_cls, goal_cls, seed: int):
    """Compute norm factors for a single env/goal/seed combination."""
    np.random.seed(seed)
    env = env_cls()
    goal = goal_cls(env)
    mu, sigma = goal.get_norm_factors()
    return float(mu), float(sigma)


def main():
    parser = argparse.ArgumentParser(description="Regenerate baseline norm factors")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds for reproducibility (default: 42 123 456 789 101112)",
    )
    parser.add_argument(
        "--env", type=str, default=None, help="Only regenerate for this environment"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON for scripting")
    args = parser.parse_args()

    nametoenv, nameenvtogoal = get_environment_registry()

    goals = PRIMARY_GOALS
    if args.env:
        goals = [(e, g) for e, g in goals if e == args.env]
        if not goals:
            print(f"error: unknown environment '{args.env}'", file=sys.stderr)
            print(f"available: {[e for e, _ in PRIMARY_GOALS]}", file=sys.stderr)
            sys.exit(1)

    results = {}

    for env_name, goal_key in goals:
        env_cls = nametoenv.get(env_name)
        goal_cls = nameenvtogoal.get((env_name, goal_key))

        if env_cls is None or goal_cls is None:
            print(f"SKIP {env_name}/{goal_key}: not found in registry")
            continue

        seed_results = []
        for seed in args.seeds:
            try:
                mu, sigma = compute_norm_factors(env_cls, goal_cls, seed)
                seed_results.append((mu, sigma))
                if not args.json:
                    print(f"  seed={seed}: mu={mu:.6f}, sigma={sigma:.6f}")
            except Exception as e:
                print(f"  seed={seed}: FAILED ({e})", file=sys.stderr)

        if not seed_results:
            print(f"FAILED {env_name}: no successful seeds")
            continue

        mus = [r[0] for r in seed_results]
        sigmas = [r[1] for r in seed_results]
        med_mu = median(mus)
        med_sigma = median(sigmas)

        results[env_name] = {
            "mu": med_mu,
            "sigma": med_sigma,
            "goal_key": goal_key,
            "n_seeds": len(seed_results),
            "per_seed": seed_results,
        }

        if not args.json:
            print(
                f"{env_name}: norm_mu={med_mu:.6f}, norm_sigma={med_sigma:.6f} "
                f"(median of {len(seed_results)} seeds)"
            )
            print()

    if args.json:
        import json

        print(json.dumps(results, indent=2))
    else:
        # summary table
        print("\n" + "=" * 70)
        print(f"{'Environment':<35} {'norm_mu':>12} {'norm_sigma':>12}")
        print("-" * 70)
        for env_name, data in results.items():
            print(f"{env_name:<35} {data['mu']:>12.6f} {data['sigma']:>12.6f}")
        print("=" * 70)

        # code snippet for updating env files
        print("\n# code to update environment files:")
        for env_name, data in results.items():
            print(f"# {env_name}: self.norm_mu, self.norm_sigma = ({data['mu']}, {data['sigma']})")


if __name__ == "__main__":
    main()
