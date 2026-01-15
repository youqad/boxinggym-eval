#!/usr/bin/env python3
"""
Run a minimal dugongs experiment across every LLM config using the fake-LLM mode.

This exercises Hydra config wiring and benchmark plumbing without hitting real APIs.
Set BOXINGGYM_FAKE_LLM=dugongs (default in this script) to stub LLM calls.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = REPO_ROOT / "conf" / "llms"


def list_models():
    """Return all llm config stems (e.g., gpt-5.1-mini)."""
    return sorted(p.stem for p in LLM_DIR.glob("*.yaml"))


def run_model(model: str, *, fake: bool, fake_hint: str):
    env = os.environ.copy()
    # Force-disable fake mode unless explicitly requested
    env["BOXINGGYM_FAKE_LLM"] = fake_hint if fake else "0"
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))

    run_dir = f"outputs/smoke_dugongs/{model}/${{now:%Y-%m-%d_%H-%M-%S}}"

    cmd = [
        "uv", "run", "python",
        "run_experiment.py",
        "envs=dugongs_direct_naive",
        f"llms={model}",
        "exp.num_experiments=[1]",
        "envs.num_evals=1",
        "include_prior=true",
        "use_ppl=false",
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=false",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Minimal dugongs smoke test per LLM config.")
    parser.add_argument("--all", action="store_true", help="Test all LLM configs")
    parser.add_argument("--model", action="append", help="Specific LLM config name (stem of conf/llms/*.yaml). Can be passed multiple times.")
    parser.add_argument("--fake", action="store_true", help="Enable offline fake LLM mode (no API calls)")
    parser.add_argument("--fake-hint", default="dugongs", help="Hint for fake mode content (default: dugongs)")
    args = parser.parse_args()

    if args.all:
        models = list_models()
    elif args.model:
        models = args.model
    else:
        models = ["gpt-4o"]  # default minimal real check

    if not models:
        print("No LLM configs selected.")
        sys.exit(1)

    failures = []
    for model in models:
        print(f"→ {model}")
        result = run_model(model, fake=args.fake, fake_hint=args.fake_hint)
        if result.returncode != 0:
            failures.append(
                (
                    model,
                    result.returncode,
                    result.stdout[-400:],
                    result.stderr[-400:],
                )
            )
            print("   FAILED")
        else:
            print("   ok")

    if failures:
        print("\nFailures:")
        for model, code, stdout_tail, stderr_tail in failures:
            print(f"- {model} (exit {code})")
            if stdout_tail:
                print("  stdout tail:\n", stdout_tail.strip())
            if stderr_tail:
                print("  stderr tail:\n", stderr_tail.strip())
        sys.exit(1)

    mode_str = "fake" if args.fake else "real"
    print(f"\nAll dugongs smoke tests passed ({mode_str} LLM mode).")


if __name__ == "__main__":
    main()
