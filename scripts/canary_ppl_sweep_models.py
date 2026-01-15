#!/usr/bin/env python3
"""
Canary runner for BoxLM / Box's Loop (`use_ppl=true`) across sweep models.

Purpose
-------
Before launching a full reproduction sweep, run a tiny end-to-end experiment for
each model listed in a W&B sweep YAML (default: sweeps/paper_replication_oed_ppl.yaml).

This exercises:
  - Hydra config wiring for each `llms=<model>` config
  - Real LLM calls (no fake mode)
  - Box's Loop proposal + execution
  - Prior-mode + posterior-mode PPL prediction paths (by using budgets [0, 1])

Notes
-----
- This *does* hit real provider APIs and will incur cost.
- We disable W&B and Weave by default for safety.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CanaryResult:
    model: str
    ok: bool
    returncode: int
    duration_s: float
    log_path: Path
    cmd: list[str]


def _load_sweep_models(sweep_yaml: Path) -> list[str]:
    data: dict[str, Any] = yaml.safe_load(sweep_yaml.read_text())
    try:
        models = data["parameters"]["llms"]["values"]
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Could not find parameters.llms.values in {sweep_yaml}") from exc
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        raise ValueError(f"Invalid llms.values in {sweep_yaml}: {models!r}")
    return models


def _make_env(
    *,
    box_loop_llm: str | None,
    extra_env: list[str],
) -> dict[str, str]:
    env = os.environ.copy()

    # Disable instrumentation side-effects (repo often has WANDB_PROJECT set via .env)
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WEAVE_DISABLED", "1")

    # Optional: force Box's Loop to use a specific model (LiteLLM-compatible string).
    if box_loop_llm:
        env["BOX_LOOP_LLM"] = box_loop_llm

    # Ensure imports work even if invoked from elsewhere.
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))

    # Apply user-provided env overrides (KEY=VALUE)
    for kv in extra_env:
        if "=" not in kv:
            raise ValueError(f"Invalid --env value (expected KEY=VALUE): {kv}")
        k, v = kv.split("=", 1)
        env[k] = v

    # Make Hydra print full traces on failure.
    env.setdefault("HYDRA_FULL_ERROR", "1")

    return env


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got {type(data).__name__}")
    return data


def _expected_results_json_path(
    *,
    env_cfg: str,
    llms_cfg: str,
    include_prior: bool,
    seed: int,
) -> Path:
    env_path = REPO_ROOT / "conf" / "envs" / f"{env_cfg}.yaml"
    llms_path = REPO_ROOT / "conf" / "llms" / f"{llms_cfg}.yaml"

    env_data = _load_yaml(env_path)
    llm_data = _load_yaml(llms_path)

    env_name = str(env_data.get("env_name", "")).strip().strip('"').strip("'")
    goal_name = str(env_data.get("goal_name", "")).strip().strip('"').strip("'")
    if not env_name or not goal_name:
        raise ValueError(f"Missing env_name/goal_name in {env_path}")

    model_name = llm_data.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"Missing model_name in {llms_path}")

    save_model_name = model_name.replace("/", "_") + "-boxloop"
    prior_tag = "true" if include_prior else "false"
    results_dir = REPO_ROOT / "results" / env_name

    def _pick_latest(paths: list[Path]) -> Path:
        try:
            return max(paths, key=lambda p: p.stat().st_mtime)
        except Exception:
            return sorted(paths)[-1]

    matches = sorted(
        results_dir.glob(
            f"env={env_name}_goal={goal_name}_model={save_model_name}_exp=oed_prior={prior_tag}_seed={int(seed)}_*.json"
        )
    )
    if matches:
        return _pick_latest(matches)

    # Fallback: multi-seed runs (seed label like s1-2-3-4-5)
    matches = sorted(
        results_dir.glob(
            f"env={env_name}_goal={goal_name}_model={save_model_name}_exp=oed_prior={prior_tag}_seed=*_*.json"
        )
    )
    if matches:
        return _pick_latest(matches)

    res_filename = f"{goal_name}_{save_model_name}_oed_{prior_tag}_{int(seed)}.json"
    return results_dir / res_filename


def _validate_results_json(
    *,
    results_path: Path,
    budgets: str,
) -> tuple[bool, str]:
    if not results_path.exists():
        return False, f"Missing results JSON: {results_path}"

    try:
        payload = json.loads(results_path.read_text())
    except Exception as exc:
        return False, f"Failed to parse results JSON {results_path}: {exc}"

    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    # For multi-seed runs, prefer per-seed all_data for validation.
    seed_runs = data.get("seed_runs", [])
    if isinstance(seed_runs, list) and seed_runs:
        first_seed = seed_runs[0]
        if isinstance(first_seed, dict):
            seed_all_data = first_seed.get("all_data")
            if isinstance(seed_all_data, dict):
                data = seed_all_data

    results = data.get("results", [])
    programs = data.get("programs", [])
    successes = data.get("successes", [])

    try:
        budget_list = yaml.safe_load(budgets)
    except Exception:
        budget_list = None
    if not isinstance(budget_list, list) or not all(isinstance(b, int) for b in budget_list):
        return False, f"Could not parse budgets list from {budgets!r}"

    if not isinstance(results, list) or len(results) != len(budget_list):
        return False, f"Expected {len(budget_list)} result entries, got {len(results) if isinstance(results, list) else type(results).__name__}"

    # Each element is [eval_score, questions]. eval_score should NOT be fallback dict.
    for i, r in enumerate(results):
        if not isinstance(r, list) or not r:
            return False, f"Result entry {i} is malformed: {type(r).__name__}"
        score = r[0]
        if isinstance(score, dict) and score.get("evaluation_failed"):
            return False, f"Budget {budget_list[i]} evaluation_failed: {score.get('error_message')}"

    if not isinstance(programs, list) or len(programs) < len(budget_list):
        return False, f"Expected >= {len(budget_list)} PPL programs, got {len(programs) if isinstance(programs, list) else type(programs).__name__}"

    # If we run any non-zero budget, we should have at least one successful observation.
    if budget_list and max(budget_list) > 0:
        if not isinstance(successes, list) or not any(bool(s) for s in successes):
            return False, "No successful observations recorded (successes has no True)"

    return True, "ok"


def _run_canary(
    *,
    model: str,
    env_cfg: str,
    budgets: str,
    include_prior: bool,
    num_evals: int,
    seed: int,
    timeout_s: int,
    out_dir: Path,
    box_loop_llm: str | None,
    extra_overrides: list[str],
    extra_env: list[str],
) -> CanaryResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    log_path = out_dir / f"{safe_model}.log"

    cmd = [
        "uv",
        "run",
        "python",
        "run_experiment.py",
        f"envs={env_cfg}",
        f"llms={model}",
        "exp=oed",
        f"exp.num_experiments={budgets}",
        f"envs.num_evals={int(num_evals)}",
        f"include_prior={'true' if include_prior else 'false'}",
        "use_ppl=true",
        f"seed={int(seed)}",
        # Keep cwd stable so python-dotenv finds .env and outputs are deterministic.
        "hydra.job.chdir=false",
        # Cut Hydra logging noise.
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]
    # removed llms.max_tokens=512 override - use model config values

    cmd.extend(extra_overrides or [])

    env = _make_env(box_loop_llm=box_loop_llm, extra_env=extra_env)

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        duration_s = time.time() - start
        ok = proc.returncode == 0

        # Additional correctness check: run_experiment can exit 0 even if the
        # PPL path fell back (evaluation_failed) for later budgets. Validate the
        # produced results JSON matches the requested budgets.
        results_json_path = None
        results_check_msg = ""
        try:
            results_json_path = _expected_results_json_path(
                env_cfg=env_cfg,
                llms_cfg=model,
                include_prior=include_prior,
                seed=seed,
            )
            ok2, msg2 = _validate_results_json(results_path=results_json_path, budgets=budgets)
            if not ok2:
                ok = False
                results_check_msg = f"RESULT_CHECK: FAIL - {msg2}"
            else:
                results_check_msg = "RESULT_CHECK: OK"
        except Exception as exc:
            ok = False
            results_check_msg = f"RESULT_CHECK: ERROR - {exc}"

        # Write full logs for post-mortem.
        log_path.write_text(
            "\n".join(
                [
                    f"MODEL: {model}",
                    f"CMD: {' '.join(cmd)}",
                    f"RETURNCODE: {proc.returncode}",
                    f"RESULTS_JSON: {results_json_path if results_json_path is not None else '<unknown>'}",
                    results_check_msg,
                    "",
                    "===== STDOUT =====",
                    proc.stdout or "",
                    "",
                    "===== STDERR =====",
                    proc.stderr or "",
                ]
            )
        )
        return CanaryResult(
            model=model,
            ok=ok,
            returncode=proc.returncode,
            duration_s=duration_s,
            log_path=log_path,
            cmd=cmd,
        )
    except subprocess.TimeoutExpired as exc:
        duration_s = time.time() - start
        log_path.write_text(
            "\n".join(
                [
                    f"MODEL: {model}",
                    f"CMD: {' '.join(cmd)}",
                    f"TIMEOUT_S: {timeout_s}",
                    "",
                    "===== STDOUT (partial) =====",
                    (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                    "",
                    "===== STDERR (partial) =====",
                    (exc.stderr or "") if isinstance(exc.stderr, str) else "",
                ]
            )
        )
        return CanaryResult(
            model=model,
            ok=False,
            returncode=124,
            duration_s=duration_s,
            log_path=log_path,
            cmd=cmd,
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Run a BoxLM canary across sweep models.")
    p.add_argument(
        "--sweep",
        default=str(REPO_ROOT / "sweeps" / "paper_replication_oed_ppl.yaml"),
        help="Path to sweep YAML (default: sweeps/paper_replication_oed_ppl.yaml)",
    )
    p.add_argument(
        "--env",
        default="dugongs_direct",
        help="Hydra env config to run (default: dugongs_direct)",
    )
    p.add_argument(
        "--budgets",
        default="[0,1]",
        help="Budgets list for exp.num_experiments (default: [0,1])",
    )
    p.add_argument("--seed", type=int, default=1, help="Seed (default: 1)")
    p.add_argument("--num-evals", type=int, default=1, help="envs.num_evals (default: 1)")
    p.add_argument(
        "--no-prior",
        action="store_true",
        help="Run with include_prior=false (default is include_prior=true)",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=30 * 60,
        help="Per-model timeout in seconds (default: 1800)",
    )
    p.add_argument(
        "--box-loop-llm",
        default=None,
        help="Optional: set BOX_LOOP_LLM for the run (default: use conf/box_loop.yaml)",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra overrides (repeatable), e.g. --override envs.num_evals=2",
    )
    p.add_argument(
        "--envvar",
        action="append",
        default=[],
        help="Extra environment variables KEY=VALUE (repeatable).",
    )
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "outputs" / "canary_ppl"),
        help="Directory to write per-model logs (default: outputs/canary_ppl)",
    )
    args = p.parse_args()

    sweep_path = Path(args.sweep)
    out_dir = Path(args.out_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")

    models = _load_sweep_models(sweep_path)
    include_prior = not bool(args.no_prior)

    print(f"Canary: envs={args.env} budgets={args.budgets} seed={args.seed} num_evals={args.num_evals}")
    if args.box_loop_llm:
        print(f"Canary: BOX_LOOP_LLM={args.box_loop_llm}")
    print(f"Models ({len(models)}): {models}")
    print(f"Logs: {out_dir}")

    results: list[CanaryResult] = []
    for m in models:
        print(f"\n→ {m}")
        r = _run_canary(
            model=m,
            env_cfg=args.env,
            budgets=args.budgets,
            include_prior=include_prior,
            num_evals=args.num_evals,
            seed=args.seed,
            timeout_s=args.timeout_s,
            out_dir=out_dir,
            box_loop_llm=args.box_loop_llm,
            extra_overrides=args.override,
            extra_env=args.envvar,
        )
        results.append(r)
        status = "ok" if r.ok else "FAILED"
        print(f"  {status} ({r.duration_s:.1f}s) log={r.log_path}")
        if not r.ok:
            # Print a short tail for quick debugging.
            try:
                tail = r.log_path.read_text()[-1200:]
            except Exception:
                tail = ""
            if tail:
                print("  --- log tail ---")
                print(tail)

    ok_count = sum(1 for r in results if r.ok)
    print("\nSummary:")
    for r in results:
        status = "ok" if r.ok else f"FAIL({r.returncode})"
        print(f"- {r.model}: {status} ({r.duration_s:.1f}s) -> {r.log_path}")
    print(f"\nPassed {ok_count}/{len(results)}")

    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
