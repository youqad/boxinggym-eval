#!/usr/bin/env python
"""Multi-layer validation of all BoxingGym result files.

Applies validation layers L0-L5 and generates reports:
  .scratch/validation_quarantine.json  - Auto-quarantine (L0/L1/L1.5/L2)
  .scratch/validation_review.json      - Manual review needed (L3/L4)
  .scratch/validation_summary.json     - Statistics by env/model

Usage:
    uv run python scripts/validate_all_runs.py
    uv run python scripts/validate_all_runs.py --results-dir results/
    uv run python scripts/validate_all_runs.py --quarantine   # auto-quarantine L0/L1/L1.5/L2 failures
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from boxing_gym.data_quality.config import (
    MAD_MULTIPLIER,
    SIGMA_FLOOR,
    ValidationLevel,
)
from boxing_gym.data_quality.quarantine import QuarantineManager
from boxing_gym.data_quality.rules import QualityValidator, ValidationResult


def _load_norm_factors() -> dict:
    """Load current norm factors from NORM_STATIC in results_io."""
    try:
        from boxing_gym.agents.results_io import NORM_STATIC

        return dict(NORM_STATIC)
    except ImportError:
        return {}


def _extract_z_values(path: Path) -> list[dict]:
    """Extract (env, model, budget, z_mean) tuples from a result file."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    config = data.get("config", {})
    envs_cfg = config.get("envs", {})
    env = envs_cfg.get("env_name", "unknown") if isinstance(envs_cfg, dict) else "unknown"
    llms = config.get("llms", {})
    model = llms.get("model_name", "unknown") if isinstance(llms, dict) else "unknown"

    results = []
    for zr in data.get("data", {}).get("z_results", []):
        if not isinstance(zr, dict):
            continue
        z = zr.get("z_mean")
        if isinstance(z, (int, float)) and math.isfinite(z):
            results.append(
                {
                    "env": env,
                    "model": model,
                    "budget": zr.get("budget", 0),
                    "z_mean": z,
                    "path": str(path),
                }
            )
    return results


def _compute_mad_outliers(
    all_z_entries: list[dict],
    multiplier: float = MAD_MULTIPLIER,
) -> list[dict]:
    """L4: flag z-scores > multiplier * MAD from median per (env, model, budget) group."""
    grouped = defaultdict(list)
    for entry in all_z_entries:
        key = (entry["env"], entry["model"], entry["budget"])
        grouped[key].append(entry)

    flagged = []
    for key, entries in grouped.items():
        values = [e["z_mean"] for e in entries]
        if len(values) < 3:
            continue

        med = statistics.median(values)
        mad = statistics.median([abs(v - med) for v in values])
        mad = max(mad, SIGMA_FLOOR)

        for entry in entries:
            deviation = abs(entry["z_mean"] - med)
            if deviation > multiplier * mad:
                flagged.append(
                    {
                        **entry,
                        "median": med,
                        "mad": mad,
                        "deviation": deviation,
                        "threshold": multiplier * mad,
                    }
                )
    return flagged


def _find_duplicates(paths: list[Path]) -> dict:
    """L5: find files with same (seed, env, model, budget, goal)."""
    seen = defaultdict(list)
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        config = data.get("config", {})
        envs_cfg = config.get("envs", {})
        env = envs_cfg.get("env_name") if isinstance(envs_cfg, dict) else None
        goal = envs_cfg.get("goal_name") if isinstance(envs_cfg, dict) else None
        llms = config.get("llms", {})
        model = llms.get("model_name") if isinstance(llms, dict) else None
        seed = config.get("seed")

        if env and model and seed is not None:
            key = (env, goal, model, seed)
            seen[key].append(str(path))

    return {str(k): v for k, v in seen.items() if len(v) > 1}


def main():
    parser = argparse.ArgumentParser(description="Validate all result files")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--quarantine", action="store_true", help="Auto-quarantine L0/L1/L1.5/L2 failures"
    )
    parser.add_argument(
        "--output-dir", type=str, default=".scratch", help="Directory for validation reports"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if not results_dir.exists():
        print(f"error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(results_dir.rglob("*.json"))
    json_files = [
        f
        for f in json_files
        if not f.name.startswith("llm_calls") and "comparative_benchmarks" not in str(f)
    ]
    print(f"found {len(json_files)} result files\n")

    # load norm factors for consistency checks
    norm_factors = _load_norm_factors()
    validator = QualityValidator(norm_factors=norm_factors)

    # L0-L2: per-file validation
    all_results: list[ValidationResult] = []
    quarantine_list = []
    review_list = []
    valid_count = 0

    for path in json_files:
        vr = validator.validate_file(path)
        all_results.append(vr)

        if vr.status == "QUARANTINE":
            quarantine_list.append(vr.to_dict())
        elif vr.status == "REVIEW":
            review_list.append(vr.to_dict())
        else:
            valid_count += 1

    print(
        f"L0-L2 results: {valid_count} valid, "
        f"{len(quarantine_list)} quarantine, {len(review_list)} review"
    )

    # L4: MAD outlier detection on valid files
    all_z_entries = []
    for path in json_files:
        all_z_entries.extend(_extract_z_values(path))

    mad_outliers = _compute_mad_outliers(all_z_entries)
    if mad_outliers:
        print(f"L4 MAD outliers: {len(mad_outliers)} flagged")
        for o in mad_outliers:
            review_list.append(
                {
                    "path": o["path"],
                    "env": o["env"],
                    "model": o["model"],
                    "status": "REVIEW",
                    "issues": [
                        {
                            "layer": "L4_MAD_OUTLIER",
                            "message": (
                                f"z={o['z_mean']:.3f} is {o['deviation']:.3f} from median "
                                f"({o['median']:.3f}), threshold={o['threshold']:.3f}"
                            ),
                            "budget": o["budget"],
                        }
                    ],
                }
            )

    # L5: duplicate detection
    duplicates = _find_duplicates(json_files)
    if duplicates:
        print(f"L5 duplicates: {len(duplicates)} groups")

    # write reports
    quarantine_path = output_dir / "validation_quarantine.json"
    review_path = output_dir / "validation_review.json"
    summary_path = output_dir / "validation_summary.json"

    with open(quarantine_path, "w") as f:
        json.dump(quarantine_list, f, indent=2)

    with open(review_path, "w") as f:
        json.dump(review_list, f, indent=2)

    # summary by env and layer
    env_counts = defaultdict(lambda: defaultdict(int))
    layer_counts = defaultdict(int)
    for vr in all_results:
        env_counts[vr.env][vr.status] += 1
        for issue in vr.issues:
            layer_counts[issue.layer.name] += 1

    summary = {
        "total_files": len(json_files),
        "valid": valid_count,
        "quarantine": len(quarantine_list),
        "review": len(review_list),
        "mad_outliers": len(mad_outliers),
        "duplicate_groups": len(duplicates),
        "by_layer": dict(layer_counts),
        "by_env": {env: dict(counts) for env, counts in sorted(env_counts.items())},
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nreports written to {output_dir}/:")
    print(f"  validation_quarantine.json ({len(quarantine_list)} files)")
    print(f"  validation_review.json ({len(review_list)} entries)")
    print("  validation_summary.json")

    # auto-quarantine if requested
    if args.quarantine and quarantine_list:
        qm = QuarantineManager()
        moved = 0
        for entry in quarantine_list:
            src = Path(entry["path"])
            if not src.exists():
                continue
            issues = entry.get("issues", [])
            layer_name = issues[0]["layer"] if issues else "UNKNOWN"
            try:
                layer = ValidationLevel[layer_name]
            except KeyError:
                layer = ValidationLevel.L0_SCHEMA

            reason = "; ".join(i["message"] for i in issues[:3])
            qm.quarantine(src, layer, reason)
            moved += 1

        print(f"\nquarantined {moved} files to .quarantine/")

    # print layer breakdown
    if layer_counts:
        print("\nissues by layer:")
        for layer_name, count in sorted(layer_counts.items()):
            print(f"  {layer_name}: {count}")


if __name__ == "__main__":
    main()
