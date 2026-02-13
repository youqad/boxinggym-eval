#!/usr/bin/env python
"""Re-normalize existing runs with updated baseline values.

Recomputes z-scores from stored raw_mean using the current norm_mu/norm_sigma
from each environment's Goal class. Adds baseline version tags for traceability.

Usage:
    uv run python scripts/renormalize_runs.py --dry-run     # preview changes
    uv run python scripts/renormalize_runs.py --execute      # apply in-place
    uv run python scripts/renormalize_runs.py --execute --backup  # backup before modifying
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from boxing_gym.data_quality.config import BASELINE_VERSION


def load_current_baselines() -> dict:
    """Load current norm factors from environment Goal classes."""
    from boxing_gym.envs.registry import get_environment_registry

    nametoenv, nameenvtogoal = get_environment_registry()

    baselines = {}
    for (env_name, goal_key), goal_cls in nameenvtogoal.items():
        if goal_key != "direct":
            continue
        try:
            env_cls = nametoenv[env_name]
            env = env_cls()
            goal = goal_cls(env)
            mu = getattr(goal, "norm_mu", None)
            sigma = getattr(goal, "norm_sigma", None)
            if mu is not None and sigma is not None:
                baselines[env_name] = {"mu": mu, "sigma": sigma, "version": BASELINE_VERSION}
        except Exception as e:
            print(f"warning: could not load baseline for {env_name}: {e}", file=sys.stderr)

    return baselines


def renormalize_file(path: Path, baselines: dict, dry_run: bool = True) -> dict | None:
    """Recompute z_mean for a single result file.

    Returns a summary dict if modified, None if skipped.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    config = data.get("config", {})
    envs_cfg = config.get("envs", {})
    env_name = envs_cfg.get("env_name") if isinstance(envs_cfg, dict) else None

    if not env_name or env_name not in baselines:
        return None

    baseline = baselines[env_name]
    data_section = data.get("data", {})
    z_results = data_section.get("z_results", [])

    if not z_results:
        return None

    # check if already at current version
    existing_nf = data_section.get("norm_factors", {})
    if isinstance(existing_nf, dict) and existing_nf.get("version") == baseline["version"]:
        return None

    mu = baseline["mu"]
    sigma = baseline["sigma"]
    changes = []

    for zr in z_results:
        if not isinstance(zr, dict):
            continue
        raw_mean = zr.get("raw_mean")
        if raw_mean is None or not isinstance(raw_mean, (int, float)):
            continue

        old_z = zr.get("z_mean")
        new_z = (float(raw_mean) - mu) / sigma if sigma > 0 else None

        if new_z is not None and old_z != new_z:
            changes.append(
                {
                    "budget": zr.get("budget"),
                    "old_z": old_z,
                    "new_z": new_z,
                    "raw_mean": raw_mean,
                }
            )

            if not dry_run:
                # preserve original z as v1
                if "z_mean_v1" not in zr:
                    zr["z_mean_v1"] = old_z
                zr["z_mean"] = new_z

                # recompute z_std if raw_std exists
                raw_std = zr.get("raw_std")
                if raw_std is not None and isinstance(raw_std, (int, float)) and sigma > 0:
                    if "z_std_v1" not in zr:
                        zr["z_std_v1"] = zr.get("z_std")
                    zr["z_std"] = float(raw_std) / sigma

    if not changes:
        # no z-score changes, but still stamp version if missing
        if not dry_run:
            needs_stamp = (
                not isinstance(existing_nf, dict)
                or existing_nf.get("version") != baseline["version"]
            )
            if needs_stamp:
                data_section["norm_factors"] = {
                    "mu": mu,
                    "sigma": sigma,
                    "version": baseline["version"],
                }
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                return {
                    "path": str(path),
                    "env": env_name,
                    "n_budgets_changed": 0,
                    "changes": [],
                }
        return None

    if not dry_run:
        data_section["norm_factors"] = {
            "mu": mu,
            "sigma": sigma,
            "version": baseline["version"],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    return {
        "path": str(path),
        "env": env_name,
        "n_budgets_changed": len(changes),
        "changes": changes,
    }


def main():
    parser = argparse.ArgumentParser(description="Re-normalize result z-scores")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview changes without modifying files (default)",
    )
    parser.add_argument("--execute", action="store_true", help="Apply changes in-place")
    parser.add_argument(
        "--backup", action="store_true", help="Create backup before modifying (used with --execute)"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Path to results directory"
    )
    args = parser.parse_args()

    dry_run = not args.execute

    if dry_run:
        print("DRY RUN: previewing changes (use --execute to apply)\n")
    else:
        print("EXECUTING: modifying result files in-place\n")

    print("loading baselines from environment classes...")
    baselines = load_current_baselines()
    print(f"loaded baselines for {len(baselines)} environments:")
    for env, bl in sorted(baselines.items()):
        print(f"  {env}: mu={bl['mu']:.6f}, sigma={bl['sigma']:.6f} ({bl['version']})")
    print()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    if args.backup and args.execute:
        backup_dir = results_dir.parent / f"{results_dir.name}_backup_prenorm"
        if not backup_dir.exists():
            print(f"creating backup: {backup_dir}")
            shutil.copytree(results_dir, backup_dir)
        else:
            print(f"backup already exists: {backup_dir}")

    json_files = list(results_dir.rglob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("llm_calls")]

    modified = 0
    skipped = 0
    total_budget_changes = 0

    for path in sorted(json_files):
        result = renormalize_file(path, baselines, dry_run=dry_run)
        if result is None:
            skipped += 1
            continue

        modified += 1
        total_budget_changes += result["n_budgets_changed"]

        print(f"{'WOULD MODIFY' if dry_run else 'MODIFIED'}: {result['path']}")
        print(f"  env={result['env']}, {result['n_budgets_changed']} budget(s) changed")
        for c in result["changes"][:3]:  # show first 3
            print(
                f"    budget={c['budget']}: z {c['old_z']:.4f} -> {c['new_z']:.4f} (raw={c['raw_mean']:.4f})"
            )
        if len(result["changes"]) > 3:
            print(f"    ... and {len(result['changes']) - 3} more")

    print(
        f"\nsummary: {modified} files {'would be ' if dry_run else ''}modified, "
        f"{skipped} skipped, {total_budget_changes} total budget changes"
    )


if __name__ == "__main__":
    main()
