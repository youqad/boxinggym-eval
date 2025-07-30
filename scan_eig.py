#!/usr/bin/env python
"""
scan_eig.py  –  report missing regret-file combinations.

Checks for files
  results/<env>/regret_direct_<MODEL>_oed_<True|False>_<SEED>.json

For the prior-only environments **emotion** and **morals** we expect ONLY
`prior=True` files.
"""
import argparse, textwrap, os, sys
from pathlib import Path
from itertools import product

# --------------------------------------------------------------------------- #
# Default grid – edit here if your project changes
# --------------------------------------------------------------------------- #
DEFAULT_MODELS = [
    "qwen2.5-32b-instruct",
    #"qwen2.5-7b-instruct",
    "OpenThinker-7B",
    "OpenThinker-32B",
    "claude-3-7-sonnet-20250219",
]

ENV_DIRS = [
    "hyperbolic_temporal_discount", "location_finding", "death_process", "irt",
    "dugongs", "survival", "peregrines", "morals", "emotion",
]

PRIOR_ONLY_ENVS = {"emotion", "morals"}   # ← only prior=True expected
SEEDS = [1, 2, 3, 4, 5]
EXP = "oed"

# --------------------------------------------------------------------------- #
# Pretty colours
# --------------------------------------------------------------------------- #
G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"; C = "\033[36m"; N = "\033[0m"

# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Scan results directory for missing regret files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example:
              python scan_eig.py --models qwen2.5-7b-instruct,OpenThinker-7B
        """))
    parser.add_argument("--models", help="comma-separated list (default = all)")
    parser.add_argument("--results", default="results",
                        help="root results directory (default: ./results)")
    args = parser.parse_args()

    models = DEFAULT_MODELS if not args.models else args.models.split(",")
    root   = Path(args.results)

    missing = []
    scanned = 0

    for env, model, seed in product(ENV_DIRS, models, SEEDS):
        # Determine which prior settings to expect for this env
        priors = (True,) if env in PRIOR_ONLY_ENVS else (True, False)
        for prior in priors:
            file = root / env / f"regret_direct_{model}_{EXP}_{prior}_{seed}.json"
            scanned += 1
            if not file.is_file():
                missing.append(file)

    # --------------------------------------------------------------------- #
    print(f"\nScanned {scanned} expected files "
          f"across {len(ENV_DIRS)} envs, {len(models)} models, "
          f"{len(SEEDS)} seeds.")

    if not missing:
        print(f"{G}✅  All files present!{N}")
        return

    # summary by env / model
    print(f"{R}✗ Missing {len(missing)} files:{N}\n")
    summary = {}
    for p in missing:
        env = p.parents[0].name
        model = p.name.split("_")[2]          # regret_direct_<model>_...
        summary.setdefault((env, model), 0)
        summary[(env, model)] += 1

    width = max(len(e) for e in ENV_DIRS)
    for (env, model), cnt in sorted(summary.items()):
        print(f"{Y}{env:<{width}}{N} | {C}{model}{N} : {cnt}")

    # full list
    print("\nFull paths:")
    for p in missing:
        print(f"  {p}")

    print("\nDone.")

if __name__ == "__main__":
    main()
