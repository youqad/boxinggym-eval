"""
Shared data layer for BoxingGym benchmark results.

Both visualize_sweep_results.py (CLI) and bench_dashboard.py (web) import from here.

Example::

    from boxing_gym.agents.results_io import (
        load_results_from_wandb, aggregate_results, CANONICAL_ENVS
    )

    results = load_results_from_wandb("YOUR_SWEEP_ID")
    agg = aggregate_results(results, group_by=("env", "model", "budget"))
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT") or "boxing-gym"

class DataSource(Enum):
    """Indicates where benchmark data was loaded from."""

    LOCAL_JSON = "local_json"
    LOCAL_LOGS = "local_logs"
    LOCAL_WANDB = "local_wandb"
    WANDB_API = "wandb_api"
    PAPER_REFERENCE = "paper_reference"


@dataclass
class RunResult:
    """Single benchmark run result."""

    env: str
    model: str
    seed: int
    budget: int
    z_mean: float
    z_std: float
    raw_mean: float
    raw_std: float
    goal: Optional[str] = None
    include_prior: Optional[bool] = None
    use_ppl: Optional[bool] = None
    experiment_type: Optional[str] = None
    path: Optional[str] = None
    source: DataSource = DataSource.LOCAL_JSON


@dataclass
class AggregatedResult:
    """Aggregated results across seeds for a group key."""

    env: str
    model: str
    budget: int
    z_mean: float  # average across seeds
    z_std: float  # standard error of mean (SEM) across seeds
    n_seeds: int
    raw_results: List[RunResult] = field(default_factory=list)

    goal: Optional[str] = None
    include_prior: Optional[bool] = None


MODEL_NAMES: Dict[str, str] = {
    "gpt-4o": "GPT-4o",
    "gpt-5.1": "GPT-5.1",
    "gpt-5.1-mini": "GPT-5.1-Mini",
    "gpt-5-mini": "GPT-5.1-Mini",
    "gpt-5.1-codex-mini": "GPT-5.1-Codex-Mini",
    "gpt-oss-20b": "GPT-OSS-20B",
    "deepseek-v3.2": "DeepSeek-V3.2",
    "deepseek-v3.2-speciale": "DeepSeek-V3.2-Speciale",
    "deepseek-v3.2-thinking": "DeepSeek-V3.2-Thinking",
    "deepseek-chat": "DeepSeek-Chat",
    "deepseek-reasoner": "DeepSeek-Reasoner",
    "glm-4.7": "GLM-4.7",
    "minimax-m2.1": "MiniMax-M2.1",
    "kimi-k2": "Kimi-K2",
    "kimi-k2-thinking": "Kimi-K2-Thinking",
    "qwen3-coder-30b": "Qwen3-Coder-30B",
    # IDs as they appear in W&B runs (from litellm routing)
    "anthropic/glm-4.7": "GLM-4.7",
    "anthropic/kimi-for-coding": "Kimi-K2",
    "openai/MiniMax-M2.1": "MiniMax-M2.1",
    "openai/minimax-m2.1": "MiniMax-M2.1",
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-5.1": "GPT-5.1",
    "openai/gpt-5.1-mini": "GPT-5.1-Mini",
    "openai/gpt-5-mini": "GPT-5.1-Mini",
    "openai/gpt-5.1-codex-mini": "GPT-5.1-Codex-Mini",
    "deepseek/deepseek-chat": "DeepSeek-V3.2",
    "deepseek/deepseek-reasoner": "DeepSeek-V3.2-Speciale",
    "ollama/qwen3-coder:30b": "Qwen3-Coder-30B",
    "openai": "GPT-4o",
}

ENV_NAMES: Dict[str, str] = {
    "dugongs_direct": "dugongs",
    "peregrines_direct": "peregrines",
    "lotka_volterra_direct": "lotka_volterra",
    "morals": "moral_machines",
    "moral": "moral_machines",
}

TEST_ENVIRONMENTS: List[str] = ["dugongs", "peregrines", "lotka_volterra"]

# default goal for single-goal environments
# used when goal is not specified in config
DEFAULT_GOALS: Dict[str, str] = {
    "dugongs": "length",
    "peregrines": "population",
    "lotka_volterra": "population",
    "survival": "survival",
    "irt": "correctness",
    "emotion": "prediction",
    "moral_machines": "judgement",
    "hyperbolic_temporal_discount": "choice",
    "location_finding": "signal",
    "death_process": "num_infected",
}

# standardization constants: env -> (mean, std)
NORM_STATIC: Dict[str, Tuple[float, float]] = {
    "dugongs": (0.9058681693402041, 9.234192516908691),
    "peregrines": (10991.5464, 15725.115658658306),
    "lotka_volterra": (8.327445247142364, 17.548285564117467),
    "hyperbolic_temporal_discount": (0.25, 4.3),
    "irt": (0.5, 0.5),
    "survival": (0.2604, 0.43885286828275377),
    "location_finding": (176.9, 1247.7),  # DirectGoal ("signal"); matches location_finding.py
    "death_process": (0.2902838787350395, 1.756991075450312),
    "emotion": (1.58508525, 0.7237143937677741),
    "morals": (0.424, 0.494190246767376),
    "moral_machines": (0.424, 0.494190246767376),
}

# Paper reference values: (env, budget) -> z_mean
# from BoxingGym paper, GPT-4o Discovery with prior
PAPER_GPT4O_2D: Dict[Tuple[str, int], float] = {
    ("dugongs", 0): -0.04,
    ("dugongs", 5): -0.06,  # interpolated
    ("dugongs", 10): -0.08,
    ("peregrines", 0): 1.95,
    ("peregrines", 5): -0.50,  # interpolated
    ("peregrines", 10): -0.57,
    ("lotka_volterra", 0): 0.38,
    ("lotka_volterra", 5): -0.20,  # interpolated
    ("lotka_volterra", 10): -0.31,
}

# Paper reference values:
# (env, goal, include_prior, budget) -> z_mean
PAPER_RESULTS: Dict[str, Dict[Tuple[str, str, bool, int], float]] = {
    "gpt-4o": {
        # hyperbolic_temporal_discount
        ("hyperbolic_temporal_discount", "choice", True, 10): 0.87,
        ("hyperbolic_temporal_discount", "choice", False, 10): 1.04,
        ("hyperbolic_temporal_discount", "choice", True, 0): 0.32,
        ("hyperbolic_temporal_discount", "choice", False, 0): 0.96,
        ("hyperbolic_temporal_discount", "discount", True, 10): -0.06,
        ("hyperbolic_temporal_discount", "discount", True, 0): -0.06,
        # location_finding
        ("location_finding", "signal", True, 10): 0.59,
        ("location_finding", "signal", False, 10): 0.86,
        ("location_finding", "signal", True, 0): 0.30,
        ("location_finding", "signal", False, 0): 0.63,
        ("location_finding", "source_location", True, 10): -0.15,
        ("location_finding", "source_location", True, 0): 1.29,
        # death_process
        ("death_process", "num_infected", True, 10): -1.06,
        ("death_process", "num_infected", False, 10): -1.04,
        ("death_process", "num_infected", True, 0): 0.54,
        ("death_process", "num_infected", False, 0): -0.31,
        ("death_process", "infection_rate", True, 10): 1.64,
        ("death_process", "infection_rate", True, 0): 0.13,
        # irt
        ("irt", "correctness", True, 10): -0.24,
        ("irt", "correctness", False, 10): 0.00,
        ("irt", "correctness", True, 0): 0.12,
        ("irt", "correctness", False, 0): 0.08,
        # dugongs
        ("dugongs", "length", True, 10): -0.08,
        ("dugongs", "length", False, 10): -0.08,
        ("dugongs", "length", True, 0): -0.04,
        ("dugongs", "length", False, 0): -0.04,
        # peregrines
        ("peregrines", "population", True, 10): -0.57,
        ("peregrines", "population", False, 10): -0.65,
        ("peregrines", "population", True, 0): 1.95,
        ("peregrines", "population", False, 0): 1.30,
        # survival
        ("survival", "survival", True, 10): 0.36,
        ("survival", "survival", False, 10): 0.27,
        ("survival", "survival", True, 0): 0.04,
        ("survival", "survival", False, 0): 0.32,
        # lotka_volterra
        ("lotka_volterra", "population", True, 10): -0.31,
        ("lotka_volterra", "population", False, 10): -0.42,
        ("lotka_volterra", "population", True, 0): 0.38,
        ("lotka_volterra", "population", False, 0): 0.75,
        # emotion
        ("emotion", "prediction", True, 10): 1.22,
        ("emotion", "prediction", True, 0): 1.04,
        # moral_machines
        ("moral_machines", "judgement", True, 10): 0.36,
        ("moral_machines", "judgement", True, 0): 0.40,
    },
    "box": {
        # hyperbolic_temporal_discount
        ("hyperbolic_temporal_discount", "choice", True, 10): 1.17,
        ("hyperbolic_temporal_discount", "choice", False, 10): 0.91,
        ("hyperbolic_temporal_discount", "choice", True, 0): 0.66,
        ("hyperbolic_temporal_discount", "choice", False, 0): 0.66,
        # location_finding
        ("location_finding", "signal", True, 10): 1.45,
        ("location_finding", "signal", False, 10): 0.83,
        ("location_finding", "signal", True, 0): 0.99,
        ("location_finding", "signal", False, 0): 1.18,
        # death_process
        ("death_process", "num_infected", True, 10): -1.02,
        ("death_process", "num_infected", False, 10): -0.61,
        ("death_process", "num_infected", True, 0): 3.79,
        ("death_process", "num_infected", False, 0): -0.90,
        # irt
        ("irt", "correctness", True, 10): -0.12,
        ("irt", "correctness", False, 10): 0.12,
        ("irt", "correctness", True, 0): 0.44,
        ("irt", "correctness", False, 0): 0.12,
        # dugongs
        ("dugongs", "length", True, 10): -0.08,
        ("dugongs", "length", False, 10): -0.09,
        ("dugongs", "length", True, 0): 0.26,
        ("dugongs", "length", False, 0): 0.05,
        # peregrines
        ("peregrines", "population", True, 10): 0.04,
        ("peregrines", "population", False, 10): 0.95,
        ("peregrines", "population", True, 0): 2.71,
        ("peregrines", "population", False, 0): 1.62,
        # survival
        ("survival", "survival", True, 10): 0.55,
        ("survival", "survival", False, 10): 0.64,
        ("survival", "survival", True, 0): 0.14,
        ("survival", "survival", False, 0): 0.73,
        # moral_machines
        ("moral_machines", "judgement", True, 10): 0.89,
        ("moral_machines", "judgement", True, 0): 0.97,
        # lotka_volterra (from paper Table 2)
        ("lotka_volterra", "population", True, 10): -0.33,
        ("lotka_volterra", "population", False, 10): -0.42,
        ("lotka_volterra", "population", True, 0): 0.46,
        ("lotka_volterra", "population", False, 0): 0.59,
        # emotion (from paper Table 2)
        ("emotion", "prediction", True, 10): 1.15,
        ("emotion", "prediction", True, 0): 0.97,
    },
}


def get_model_display_name(model_key: str) -> str:
    """Get display name for a model."""
    if model_key in MODEL_NAMES:
        return MODEL_NAMES[model_key]
    lower_key = model_key.lower()
    if lower_key in MODEL_NAMES:
        return MODEL_NAMES[lower_key]
    return model_key


def get_env_display_name(env_key: str) -> str:
    """Get canonical environment name."""
    base = env_key.removesuffix("_direct")
    return ENV_NAMES.get(env_key, ENV_NAMES.get(base, base))


def get_goal_display_name(env_name: str, goal_key: Optional[str]) -> Optional[str]:
    if not goal_key:
        return DEFAULT_GOALS.get(env_name, goal_key)
    if goal_key in ("direct", "direct_naive", "direct_discovery"):
        return DEFAULT_GOALS.get(env_name, goal_key)
    if env_name == "location_finding" and goal_key == "source":
        return "source_location"
    if env_name == "death_process" and goal_key == "infection":
        return "infection_rate"
    return goal_key


_KNOWN_GOALS: Tuple[str, ...] = (
    "direct_naive",
    "direct_discovery",  # legacy alias for parsing old WandB runs
    "direct",
    "discount",
    "source",
    "infection",
    "best_student",
    "difficult_question",
    "discriminate_question",
)


def _parse_wandb_run_name(run_name: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"_(oed|discovery)_(.+)_seed(\d+)$", run_name)
    if not m:
        return None
    exp_type = m.group(1)
    model = m.group(2)
    seed = int(m.group(3))

    prefix = run_name[: m.start()]

    env_raw = prefix
    goal_raw = None
    for g in _KNOWN_GOALS:
        suffix = f"_{g}"
        if prefix.endswith(suffix):
            env_raw = prefix[: -len(suffix)]
            goal_raw = g
            break

    env = get_env_display_name(env_raw)
    goal = get_goal_display_name(env, goal_raw)

    return {
        "env": env,
        "goal": goal,
        "exp_type": exp_type,
        "model": model,
        "seed": seed,
    }


def standardize(
    err_mean: float,
    err_std: float,
    env_name: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Z-score standardization using static normalization constants."""
    mu0, sigma0 = NORM_STATIC.get(env_name, (None, None))
    if sigma0 in (None, 0):
        return None, None
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

def get_paper_value_4d(
    env: str,
    goal: str,
    include_prior: bool,
    budget: int,
    source: str = "gpt-4o",
) -> Optional[float]:
    """4D paper lookup for dashboard."""
    return PAPER_RESULTS.get(source, {}).get((env, goal, include_prior, budget))


def get_paper_value_2d(env: str, budget: int) -> Optional[float]:
    """2D paper lookup for CLI (legacy compatibility)."""
    return PAPER_GPT4O_2D.get((env, budget))


def _iter_json_files(root: str) -> Iterable[Path]:
    """Iterate over all JSON files in directory tree."""
    yield from Path(root).rglob("*.json")


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load a single JSON file, returning None on error."""
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_bool(val: Any) -> bool:
    """Parse boolean from various formats (handles string 'false' correctly)."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == "true"
    return bool(val)


def load_results_from_json_dir(root: str) -> List[RunResult]:
    """Parse all JSON result files from a directory."""
    results: List[RunResult] = []

    for path in _iter_json_files(root):
        blob = _load_json_file(path)
        if not blob or not isinstance(blob, dict):
            continue

        cfg = blob.get("config", {})
        env_cfg = cfg.get("envs") or {}
        exp_cfg = cfg.get("exp") or {}

        env_name = env_cfg.get("env_name")
        goal_name = env_cfg.get("goal_name")
        include_prior = _parse_bool(cfg.get("include_prior", False))
        model_name = (cfg.get("llms") or {}).get("model_name")
        seed = cfg.get("seed")
        use_ppl = _parse_bool(cfg.get("use_ppl", False))
        experiment_type = exp_cfg.get("experiment_type")
        budgets = exp_cfg.get("num_experiments")

        if not env_name or not goal_name:
            continue

        env_name = get_env_display_name(str(env_name))
        goal_name = get_goal_display_name(env_name, str(goal_name))

        data = blob.get("data") or {}
        raw_results = data.get("results") or []
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

        if not raw_results:
            continue

        for i, entry in enumerate(raw_results):
            if not entry or not isinstance(entry, list) or entry[0] is None:
                continue
            if isinstance(entry[0], dict):
                continue

            try:
                err_mean, err_std = entry[0]
            except (TypeError, ValueError):
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
                z_mean, z_std = standardize(float(err_mean), float(err_std), env_name)

            if z_mean is None or budget is None:
                continue

            results.append(
                RunResult(
                    env=env_name,
                    model=model_name or "unknown",
                    seed=int(seed) if seed is not None else 0,
                    budget=int(budget),
                    z_mean=float(z_mean),
                    z_std=float(z_std) if z_std is not None else 0.0,
                    raw_mean=float(err_mean),
                    raw_std=float(err_std),
                    goal=goal_name,
                    include_prior=include_prior,
                    use_ppl=use_ppl,
                    experiment_type=experiment_type,
                    path=str(path),
                    source=DataSource.LOCAL_JSON,
                )
            )

    return results


def load_results_from_logs(logs_dir: str) -> List[RunResult]:
    """Parse W&B sweep agent logs to extract results."""
    results: List[RunResult] = []
    logs_path = Path(logs_dir)

    for log_file in logs_path.glob("agent_*.log"):
        current_run: Optional[Dict[str, Any]] = None

        with open(log_file, "r") as f:
            for line in f:
                # Parse run identifier: "Syncing run dugongs_direct_oed_gpt-4o_seed1"
                if "Syncing run" in line:
                    match = re.search(r"Syncing run (\S+)", line)
                    if match:
                        parsed = _parse_wandb_run_name(match.group(1))
                        if parsed:
                            goal = parsed.get("goal")
                            if not goal:
                                goal = DEFAULT_GOALS.get(parsed["env"])
                            current_run = {
                                "env": parsed["env"],
                                "model": parsed["model"],
                                "seed": parsed["seed"],
                                "goal": goal,
                            }

                # Parse budget results: "Budget 0: z=0.013 +/- 0.109 (raw: 1.0247 +/- 1.0049)"
                elif "Budget" in line and current_run:
                    match = re.search(
                        r"Budget (\d+): z=([-\d.]+) \(raw: ([-\d.]+) [±+/-]+ ([\d.]+)\)",
                        line,
                    )
                    if match:
                        budget = int(match.group(1))
                        z_mean = float(match.group(2))
                        z_std = 0.0
                        raw_mean = float(match.group(3))
                        raw_std = float(match.group(4))
                    else:
                        match = re.search(
                            r"Budget (\d+): z=([-\d.]+) [±+/-]+ ([\d.]+) \(raw: ([-\d.]+) [±+/-]+ ([\d.]+)\)",
                            line,
                        )
                        if not match:
                            continue
                        budget = int(match.group(1))
                        z_mean = float(match.group(2))
                        z_std = float(match.group(3))
                        raw_mean = float(match.group(4))
                        raw_std = float(match.group(5))

                    results.append(
                        RunResult(
                            env=current_run["env"],
                            model=current_run["model"],
                            seed=current_run["seed"],
                            budget=budget,
                            z_mean=z_mean,
                            z_std=z_std,
                            raw_mean=raw_mean,
                            raw_std=raw_std,
                            goal=current_run.get("goal"),
                            include_prior=True,  # Default assumption
                            source=DataSource.LOCAL_LOGS,
                        )
                    )

    return results


def load_results_from_local_wandb(
    wandb_dir: str = "wandb-dir",
    sweep_id: Optional[str] = None,
) -> List[RunResult]:
    """Load results from local W&B run directories.

    Scans wandb-dir/run-*/files/wandb-summary.json or wandb-dir/wandb/run-*/files/wandb-summary.json
    for completed runs.
    Much faster than API calls when data is already synced locally.

    Args:
        wandb_dir: Path to wandb directory (default: "wandb-dir")
        sweep_id: Optional sweep ID to filter runs (not yet implemented)

    Returns:
        List of RunResult objects from local wandb runs
    """
    results: List[RunResult] = []
    wandb_path = Path(wandb_dir)
    if not wandb_path.exists():
        return results

    candidate_paths = []
    # Common layout: wandb-dir/wandb/run-*/files
    nested = wandb_path / "wandb"
    if nested.exists():
        candidate_paths.append(nested)
    candidate_paths.append(wandb_path)

    seen_paths = set()
    for base_path in candidate_paths:
        if base_path in seen_paths:
            continue
        seen_paths.add(base_path)

        for run_dir in base_path.glob("run-*/files"):
            summary_path = run_dir / "wandb-summary.json"
            if not summary_path.exists():
                continue

            try:
                with summary_path.open("r") as f:
                    summary = json.load(f)
            except Exception:
                continue

            if not isinstance(summary, dict):
                continue

            z_mean = summary.get("eval/z_mean")
            if z_mean is None or not isinstance(z_mean, (int, float)):
                continue

            env_name = summary.get("exp/env_name")
            if not env_name:
                continue

            env_name = get_env_display_name(str(env_name))
            goal_type = summary.get("exp/goal_type")
            goal = get_goal_display_name(env_name, str(goal_type) if goal_type else None)
            if not goal:
                goal = DEFAULT_GOALS.get(env_name)

            seed = summary.get("exp/seed")
            try:
                seed = int(seed) if seed is not None else 0
            except Exception:
                seed = 0

            budget = summary.get("budget")
            if budget is None:
                budget = summary.get("eval/budget")
            try:
                budget = int(budget) if budget is not None else 0
            except Exception:
                budget = 0

            z_std = summary.get("eval/z_std")
            if not isinstance(z_std, (int, float)):
                z_std = 0.0

            raw_mean = summary.get("eval/mean")
            if raw_mean is None:
                raw_mean = summary.get("eval/mse")
            if not isinstance(raw_mean, (int, float)):
                raw_mean = 0.0

            raw_std = summary.get("eval/std")
            if raw_std is None:
                raw_std = summary.get("eval/std_mse")
            if not isinstance(raw_std, (int, float)):
                raw_std = 0.0

            include_prior = summary.get("exp/include_prior", True)
            use_ppl = summary.get("exp/use_ppl", False)
            experiment_type = summary.get("exp/experiment_type")

            # extract model name from config.yaml or wandb-metadata.json
            model_name = None

            # try config.yaml first (has args)
            config_path = run_dir / "config.yaml"
            if config_path.exists():
                try:
                    import yaml
                    with config_path.open("r") as f:
                        config = yaml.safe_load(f)
                    if isinstance(config, dict):
                        wandb_meta = config.get("_wandb", {}).get("value", {})
                        e_data = wandb_meta.get("e", {})
                        for writer_data in e_data.values():
                            if isinstance(writer_data, dict):
                                args = writer_data.get("args", [])
                                for arg in args:
                                    if isinstance(arg, str) and arg.startswith("llms="):
                                        model_name = arg.split("=", 1)[1]
                                        break
                            if model_name:
                                break
                except Exception:
                    pass

            # try wandb-metadata.json if config didn't have it
            if not model_name:
                meta_path = run_dir / "wandb-metadata.json"
                if meta_path.exists():
                    try:
                        with meta_path.open("r") as f:
                            meta = json.load(f)
                        args = meta.get("args", [])
                        for arg in args:
                            if isinstance(arg, str) and arg.startswith("llms="):
                                model_name = arg.split("=", 1)[1]
                                break
                    except Exception:
                        pass

            # default to gpt-4o (matches Hydra default config)
            if not model_name:
                model_name = "gpt-4o"

            model_name = get_model_display_name(model_name)

            results.append(
                RunResult(
                    env=env_name,
                    model=model_name,
                    seed=seed,
                    budget=budget,
                    z_mean=float(z_mean),
                    z_std=float(z_std),
                    raw_mean=float(raw_mean),
                    raw_std=float(raw_std),
                    goal=goal,
                    include_prior=include_prior,
                    use_ppl=use_ppl,
                    experiment_type=experiment_type,
                    path=str(run_dir.parent),
                    source=DataSource.LOCAL_WANDB,
                )
            )

    return results


def _get_wandb_api():
    """Get wandb API instance, with helpful error on import failure."""
    try:
        import wandb

        return wandb.Api(timeout=120)  # Use 120s timeout for large sweeps
    except ImportError:
        raise ImportError("wandb package required. Install with: uv add wandb")


def load_results_from_wandb(
    sweep_path: str,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> List[RunResult]:
    api = _get_wandb_api()

    parts = sweep_path.split("/")
    if len(parts) == 1:
        sweep_id = parts[0]
    elif len(parts) == 3:
        entity, project, sweep_id = parts
    else:
        raise ValueError(
            f"Invalid sweep path: {sweep_path}. "
            "Expected: sweep_id or entity/project/sweep_id"
        )

    full_path = f"{entity}/{project}/{sweep_id}"

    import wandb

    try:
        sweep = api.sweep(full_path)
    except wandb.errors.CommError as e:
        raise ValueError(f"Error fetching sweep {full_path}: {e}")

    results: List[RunResult] = []
    runs = list(sweep.runs)

    for run in runs:
        summary = dict(run.summary)
        config = run.config or {}

        if isinstance(config.get("envs"), dict):
            env_cfg = config.get("envs", {})
            exp_cfg = config.get("exp", {})
            llm_cfg = config.get("llms", {})
            env = env_cfg.get("env_name")
            goal = env_cfg.get("goal_name")
            experiment_type = exp_cfg.get("experiment_type")
            model = llm_cfg.get("model_name")
            seed = config.get("seed")
        else:
            env = config.get("envs.env_name") or config.get("env_name")
            goal = config.get("envs.goal_name") or config.get("goal_name")
            experiment_type = config.get("exp.experiment_type") or config.get("experiment_type")
            model = config.get("llms.model_name") or config.get("llms") or config.get("model_name")
            seed = config.get("seed")

        if not env or not model or seed is None:
            parsed = _parse_wandb_run_name(str(run.name))
            if not parsed:
                continue
            env = parsed["env"]
            goal = parsed.get("goal")
            experiment_type = experiment_type or parsed.get("exp_type")
            model = model or parsed["model"]
            seed = seed if seed is not None else parsed["seed"]

        env = get_env_display_name(str(env))
        goal = get_goal_display_name(env, str(goal) if goal is not None else None)
        if not goal:
            goal = DEFAULT_GOALS.get(env)

        try:
            seed = int(seed)
        except Exception:
            seed = 0

        include_prior = config.get("include_prior", True)
        use_ppl = config.get("use_ppl", False)

        budget_rows: Dict[int, Dict[str, Any]] = {}
        history_keys = [
            "eval/budget",
            "eval/z_mean",
            "eval/z_std",
            "eval/mean",
            "eval/std",
            "eval/mse",
            "eval/std_mse",
        ]
        try:
            scan = run.scan_history(keys=history_keys)
            for row in scan:
                if not isinstance(row, dict):
                    continue
                b = row.get("eval/budget")
                z = row.get("eval/z_mean")
                if b is None or z is None:
                    continue
                try:
                    b_int = int(b)
                except Exception:
                    continue
                budget_rows[b_int] = row
        except Exception:
            try:
                hist = run.history(keys=history_keys, samples=10000)
                if hasattr(hist, "to_dict"):
                    rows = hist.to_dict("records")
                elif isinstance(hist, list):
                    rows = hist
                else:
                    rows = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    b = row.get("eval/budget")
                    z = row.get("eval/z_mean")
                    if b is None or z is None:
                        continue
                    try:
                        b_int = int(b)
                    except Exception:
                        continue
                    budget_rows[b_int] = row
            except Exception:
                budget_rows = {}

        if budget_rows:
            for budget, row in sorted(budget_rows.items()):
                z_mean = row.get("eval/z_mean")
                if not isinstance(z_mean, (int, float)):
                    continue
                raw_mean = row.get("eval/mean")
                if not isinstance(raw_mean, (int, float)):
                    raw_mean = row.get("eval/mse")
                raw_std = row.get("eval/std")
                if not isinstance(raw_std, (int, float)):
                    raw_std = row.get("eval/std_mse")
                z_std = row.get("eval/z_std")
                if not isinstance(z_std, (int, float)):
                    z_std = 0.0

                results.append(
                    RunResult(
                        env=env,
                        model=model,
                        seed=seed,
                        budget=budget,
                        z_mean=float(z_mean),
                        z_std=float(z_std),
                        raw_mean=float(raw_mean) if isinstance(raw_mean, (int, float)) else 0.0,
                        raw_std=float(raw_std) if isinstance(raw_std, (int, float)) else 0.0,
                        goal=goal,
                        include_prior=include_prior,
                        use_ppl=use_ppl,
                        experiment_type=experiment_type,
                        source=DataSource.WANDB_API,
                    )
                )
            continue

        if "eval/z_mean" in summary:
            z_mean = summary["eval/z_mean"]
            z_std = summary.get("eval/z_std", 0.0)
            budget = None
            try:
                budget = int(summary.get("eval/budget")) if summary.get("eval/budget") is not None else None
            except Exception:
                budget = None
            if budget is None:
                budgets_cfg = None
                if isinstance(config.get("exp"), dict):
                    budgets_cfg = (config.get("exp") or {}).get("num_experiments")
                else:
                    budgets_cfg = config.get("exp.num_experiments")
                try:
                    if isinstance(budgets_cfg, list) and budgets_cfg:
                        budget = int(max(int(b) for b in budgets_cfg))
                    elif budgets_cfg is not None:
                        budget = int(budgets_cfg)
                except Exception:
                    budget = None
            if budget is None:
                budget = 0

            if isinstance(z_mean, (int, float)):
                results.append(
                    RunResult(
                        env=env,
                        model=model,
                        seed=seed,
                        budget=budget,
                        z_mean=float(z_mean),
                        z_std=float(z_std) if isinstance(z_std, (int, float)) else 0.0,
                        raw_mean=float(summary.get("eval/mean", summary.get("eval/mean_final", 0.0)))
                        if isinstance(summary.get("eval/mean"), (int, float))
                        else 0.0,
                        raw_std=float(summary.get("eval/std", summary.get("eval/std_final", 0.0)))
                        if isinstance(summary.get("eval/std"), (int, float))
                        else 0.0,
                        goal=goal,
                        include_prior=include_prior,
                        use_ppl=use_ppl,
                        experiment_type=experiment_type,
                        source=DataSource.WANDB_API,
                    )
                )
            continue

        for key, value in summary.items():
            budget_match = re.search(r"budget[_/](\d+)[_/]?z_mean", key, re.IGNORECASE)
            if budget_match and isinstance(value, (int, float)):
                budget = int(budget_match.group(1))
                z_mean = float(value)

                std_key = key.replace("z_mean", "z_std").replace("mean", "std")
                z_std = summary.get(std_key, 0.0)
                if not isinstance(z_std, (int, float)):
                    z_std = 0.0

                results.append(
                    RunResult(
                        env=env,
                        model=model,
                        seed=seed,
                        budget=budget,
                        z_mean=z_mean,
                        z_std=float(z_std),
                        raw_mean=0.0,
                        raw_std=0.0,
                        goal=goal,
                        include_prior=include_prior,
                        use_ppl=use_ppl,
                        experiment_type=experiment_type,
                        source=DataSource.WANDB_API,
                    )
                )

    seen: set = set()
    unique_results: List[RunResult] = []
    for r in results:
        key = (r.env, r.model, r.seed, r.budget, r.goal, r.include_prior, r.use_ppl, r.experiment_type)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    return unique_results


def list_wandb_sweeps(
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """List all sweeps in a W&B project."""
    api = _get_wandb_api()
    path = f"{entity}/{project}"

    sweeps_info: List[Dict[str, Any]] = []

    runs = api.runs(path, filters={"sweep": {"$ne": None}}, per_page=500)

    sweep_ids: Dict[str, Any] = {}
    for run in runs:
        if run.sweep and run.sweep.id not in sweep_ids:
            sweep_ids[run.sweep.id] = run.sweep

    for sweep_id, sweep in list(sweep_ids.items())[:limit]:
        try:
            created_raw = getattr(sweep, "created_at", None)
            if created_raw:
                if isinstance(created_raw, str):
                    created = created_raw[:19]
                elif hasattr(created_raw, "strftime"):
                    created = created_raw.strftime("%Y-%m-%d %H:%M")
                else:
                    created = "N/A"
            else:
                created = "N/A"

            state = sweep.state if hasattr(sweep, "state") else "unknown"
            name = sweep.name if hasattr(sweep, "name") else sweep_id

            sweep_runs = api.runs(path, filters={"sweep": sweep_id})
            run_count = len(list(sweep_runs))

            sweeps_info.append(
                {
                    "id": sweep_id,
                    "name": str(name),
                    "state": state,
                    "n_runs": run_count,
                    "created_at": created,
                }
            )
        except Exception:
            sweeps_info.append(
                {
                    "id": sweep_id,
                    "name": "(error loading)",
                    "state": "unknown",
                    "n_runs": 0,
                    "created_at": "N/A",
                }
            )

    return sweeps_info


def aggregate_results(
    results: List[RunResult],
    group_by: Tuple[str, ...] = ("env", "model", "budget", "goal", "include_prior"),
    outlier_threshold: float = 10.0,
) -> Dict[Tuple, AggregatedResult]:
    """Aggregate results by specified keys, filtering outliers."""
    grouped: Dict[Tuple, List[RunResult]] = defaultdict(list)

    for r in results:
        if abs(r.z_mean) >= outlier_threshold:
            continue
        key = tuple(getattr(r, field) for field in group_by)
        grouped[key].append(r)

    aggregated: Dict[Tuple, AggregatedResult] = {}

    for key, runs in grouped.items():
        z_means = [r.z_mean for r in runs]
        avg_z = sum(z_means) / len(z_means)

        # compute standard error of mean (SEM) across seeds
        if len(z_means) > 1:
            variance = sum((z - avg_z) ** 2 for z in z_means) / (len(z_means) - 1)
            std_z = variance**0.5 / len(z_means)**0.5  # SEM = std / sqrt(n)
        else:
            std_z = 0.0

        first_run = runs[0]
        env_idx = group_by.index("env") if "env" in group_by else None
        model_idx = group_by.index("model") if "model" in group_by else None
        budget_idx = group_by.index("budget") if "budget" in group_by else None

        aggregated[key] = AggregatedResult(
            env=key[env_idx] if env_idx is not None else first_run.env,
            model=key[model_idx] if model_idx is not None else first_run.model,
            budget=key[budget_idx] if budget_idx is not None else first_run.budget,
            z_mean=avg_z,
            z_std=std_z,
            n_seeds=len(runs),
            raw_results=runs,
            goal=first_run.goal,
            include_prior=first_run.include_prior,
        )

    return aggregated

def generate_paper_reference_rows(source: str = "gpt-4o") -> List[RunResult]:
    """Generate RunResult objects for paper baseline values."""
    rows: List[RunResult] = []

    paper_data = PAPER_RESULTS.get(source, {})
    model_name = f"Paper ({source.upper() if source == 'box' else 'GPT-4o'})"

    for (env, goal, include_prior, budget), z_mean in paper_data.items():
        rows.append(
            RunResult(
                env=env,
                model=model_name,
                seed=0,
                budget=budget,
                z_mean=z_mean,
                z_std=0.0,
                raw_mean=0.0,
                raw_std=0.0,
                goal=goal,
                include_prior=include_prior,
                use_ppl=False,
                experiment_type="paper",
                path="(paper reference)",
                source=DataSource.PAPER_REFERENCE,
            )
        )

    return rows
