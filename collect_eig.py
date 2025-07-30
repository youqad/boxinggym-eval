#!/usr/bin/env python
"""
collect_eig.py  –  aggregate EIG at obs 0,1,3,5,7,10 for one model,
skipping any seeds that are missing or bad.

Output
------
results/<MODEL>_oed_direct_eig_by_obs.csv
"""
import argparse, csv, json, math, statistics, sys, textwrap
from pathlib import Path
from collections import defaultdict
import re

# --------------------------------------------------------------------------- #
# Static grid (envs & observation list)
# --------------------------------------------------------------------------- #
ENV_DIRS = [
    "hyperbolic_temporal_discount", "location_finding", "death_process", "irt",
    "dugongs", "survival", "peregrines", "morals", "emotion",
]
EXP = "oed"
GOAL = "direct"
OBS_LIST = [0, 1, 3, 5, 7, 10]     # real observation indices we keep

# --------------------------------------------------------------------------- #
def ci95(vals):
    return 0.0 if len(vals) < 2 else 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))

def load_eigs(path: Path):
    try:
        data = json.loads(path.read_text())
        for key in ("eigs", "eigs_regret", "data.eigs"):
            obj = data
            for part in key.split("."):
                obj = obj.get(part) if isinstance(obj, dict) else None
            if isinstance(obj, list):
                return obj
        print(f"    ⚠️  No EIG array in {path}")
    except Exception as e:
        print(f"    ⚠️  Error reading {path}: {e}")
    return None

# --------------------------------------------------------------------------- #
def discover_seeds(env: str, model: str, root: Path):
    """Return sorted list of seeds that have at least one file (prior/np)."""
    pattern = re.compile(
        rf"regret_direct_{re.escape(model)}_{EXP}_(True|False)_(\d+)\.json$"
    )
    seeds = set()
    env_dir = root / env
    if not env_dir.is_dir():
        return []
    for fname in os.listdir(env_dir):
        m = pattern.match(fname)
        if m:
            seeds.add(int(m.group(2)))
    return sorted(seeds)

def collect_env(env: str, model: str, root: Path):
    rows = []
    print(f"\n🔍  Environment: {env}")

    # discover seeds present for this env / model
    seeds = discover_seeds(env, model, root)
    if not seeds:
        print("  ⚠️  No files for this environment & model.")
        return rows

    print(f"  Seeds detected: {seeds}")

    for prior in (True, False):
        tag = "WITH PRIOR" if prior else "NO PRIOR"
        print(f"  {tag}:")
        obs_vals = {idx: [] for idx in OBS_LIST}
        loaded = 0

        for seed in seeds:
            file = root / env / f"regret_direct_{model}_{EXP}_{prior}_{seed}.json"
            if file.exists():
                eigs = load_eigs(file)
                if eigs:
                    loaded += 1
                    for idx in OBS_LIST:
                        if idx < len(eigs):
                            obs_vals[idx].append(eigs[idx])
                    print(f"    ✓ LOADED  {file.name}")
                else:
                    print(f"    ⚠️  BAD    {file.name}")
            else:
                print(f"    ✗ MISSING {file.name}")

        print(f"    → {loaded}/{len(seeds)} seeds loaded")

        for idx in OBS_LIST:
            vals = obs_vals[idx]
            if vals:        # include row only if at least one seed contributed
                rows.append({
                    "Environment": env,
                    "Prior": prior,
                    "Obs_Index": idx,      # real observation count
                    "Mean_EIG": statistics.mean(vals),
                    "CI_95": ci95(vals),
                })
    return rows

# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Collect mean/CI95 EIG at obs 0,1,3,5,7,10 for one model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
          Files searched:
            results/<env>/regret_direct_<MODEL>_oed_<True|False>_<SEED>.json
          Seeds are detected automatically per environment; missing seeds are skipped.
        """))
    parser.add_argument("model", help="model name (must match filenames)")
    parser.add_argument("--results", default="results",
                        help="root results directory (default: ./results)")
    args = parser.parse_args()

    root = Path(args.results)
    rows = []
    for env in ENV_DIRS:
        rows.extend(collect_env(env, args.model, root))

    if not rows:
        sys.exit(f"\n❌  No usable files found for model '{args.model}'")

    out_path = root / f"{args.model}_oed_direct_eig_by_obs.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, ["Environment", "Prior", "Obs_Index", "Mean_EIG", "CI_95"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅  Wrote {out_path}  ({len(rows)} rows)")

if __name__ == "__main__":
    import os
    main()
