import copy
import logging
import random
from typing import Any

try:
    from omegaconf import ListConfig
except Exception:
    ListConfig = ()

import numpy as np
from omegaconf import OmegaConf

from boxing_gym.agents.agent import LMExperimenter
from boxing_gym.envs.registry import get_environment_registry
from boxing_gym.experiment.utils import compute_z_score, create_fallback_result

logger = logging.getLogger(__name__)


def build_env_goal(
    env_name: str,
    goal_name: str,
    env_params: dict[str, Any],
    include_prior: bool,
):
    nametoenv, nameenvtogoal = get_environment_registry()

    if env_name not in nametoenv:
        available = ", ".join(sorted(nametoenv.keys()))
        raise ImportError(
            f"Environment '{env_name}' is not available in this checkout. "
            f"Available environments: {available or 'none found'}."
        )
    if (env_name, goal_name) not in nameenvtogoal:
        raise ImportError(
            f"Goal '{goal_name}' for environment '{env_name}' is not registered. "
            "Check your config or add the goal mapping."
        )

    env = nametoenv[env_name](**env_params)
    env.include_prior = include_prior
    goal = nameenvtogoal[(env_name, goal_name)](env)
    return env, goal, nametoenv, nameenvtogoal


def pre_generate_eval_points(
    goal,
    nametoenv,
    nameenvtogoal,
    env_name: str,
    goal_name: str,
    env_params: dict[str, Any],
    include_prior: bool,
    num_evals: int,
    seed: int,
):
    print(f"Pre-generating {num_evals} evaluation test points with seed {seed}...")
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()

    goal_for_eval = None
    try:
        env_for_eval = nametoenv[env_name](**env_params)
        env_for_eval.include_prior = include_prior
        goal_for_eval = nameenvtogoal[(env_name, goal_name)](env_for_eval)
    except Exception as exc:
        logger.warning(
            "Fresh instantiation of goal/env for pre-generation failed. "
            "Falling back to lazy eval generation. Set envs.num_evals=0 or disable pre-gen if this is expected. "
            f"Error: {exc}"
        )

    if goal_for_eval is not None:
        try:
            for _ in range(num_evals):
                goal_for_eval.get_goal_eval_question(include_prior)
            goal.eval_points = copy.deepcopy(goal_for_eval.eval_points)
            goal.eval_pointer = 0
            print(f"✓ Pre-generated {len(goal.eval_points)} test points (experiment RNG restored)")
        except Exception as exc:
            logger.warning(f"Pre-generation loop failed: {exc}")
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
    else:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        print(
            "Skipped pre-generation because goal/env could not be instantiated; "
            "eval points will be generated lazily during evaluation."
        )


def create_agents(
    experiment_type: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    api_base: str,
    api_key: str,
    custom_llm_provider: str,
    extra_headers: dict = None,
):
    scientist_agent = LMExperimenter(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_base=api_base,
        api_key=api_key,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
    )
    naive_agent = None
    if experiment_type == "discovery":
        naive_agent = LMExperimenter(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
        )
    return scientist_agent, naive_agent


def compute_z_results(
    all_data,
    num_experiments,
    norm_mu: float,
    norm_sigma: float,
    env_name: str,
    goal_name: str,
):
    z_results = []

    def _normalize_budgets(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set, ListConfig)):
            budgets = []
            for b in value:
                try:
                    budgets.append(int(b))
                except Exception:
                    continue
            return sorted(set(budgets))
        try:
            return [int(value)]
        except Exception:
            return []

    budgets = _normalize_budgets(num_experiments)

    def _budget_for_z_result(idx: int):
        try:
            if budgets:
                return int(budgets[idx]) if idx < len(budgets) else int(budgets[-1])
            if num_experiments is not None:
                return int(num_experiments)
            return None
        except Exception:
            return idx + 1

    def _apply_env_transforms(raw_err, z_score, env, goal_type):
        transformed_raw = raw_err
        transformed_z = z_score

        if env == "location_finding" and "direct" in goal_type:
            if raw_err is not None:
                transformed_raw = min(max(raw_err, 0), 10000)
                if norm_mu is not None and norm_sigma is not None and norm_sigma > 0:
                    transformed_z = (transformed_raw - norm_mu) / norm_sigma

        return transformed_raw, transformed_z

    has_norm_factors = norm_mu is not None and norm_sigma is not None
    if has_norm_factors:
        print("\n📊 Z-Score Results (standardized for paper comparison):")
    else:
        print(
            f"⚠️  No normalization factors found for this goal (norm_mu={norm_mu}, norm_sigma={norm_sigma})"
        )

    for i, result_entry in enumerate(all_data[0]):
        budget = _budget_for_z_result(i)
        err_mean, err_std = None, None
        z_mean = None
        z_std = None

        err_stats = None
        try:
            if result_entry and len(result_entry) > 0:
                err_stats = result_entry[0]
        except Exception:
            err_stats = None

        if isinstance(err_stats, dict):
            err_mean = err_stats.get("mse", err_stats.get("accuracy", err_stats.get("score")))
            err_std = err_stats.get("std_mse", err_stats.get("std", err_stats.get("std_accuracy")))
        elif isinstance(err_stats, (list, tuple)):
            if len(err_stats) >= 1:
                err_mean = err_stats[0]
            if len(err_stats) >= 2:
                err_std = err_stats[1]

        if has_norm_factors and err_mean is not None:
            z_mean = compute_z_score(float(err_mean), float(norm_mu), float(norm_sigma))
            try:
                if err_std is not None and float(norm_sigma) > 0:
                    z_std = float(err_std) / float(norm_sigma)
                elif z_mean is not None:
                    z_std = 0.0
            except Exception:
                z_std = None
            err_mean, z_mean = _apply_env_transforms(float(err_mean), z_mean, env_name, goal_name)

        z_results.append(
            {
                "budget": budget,
                "raw_mean": float(err_mean) if err_mean is not None else None,
                "raw_std": float(err_std) if err_std is not None else None,
                "z_mean": float(z_mean) if z_mean is not None else None,
                "z_std": float(z_std) if z_std is not None else None,
            }
        )

        if has_norm_factors and z_mean is not None:
            print(
                f"  Budget {budget}: z={z_mean:.3f} "
                f"(raw: {float(err_mean):.4f} ± {float(err_std) if err_std else 0:.4f})"
            )

    return z_results


def aggregate_z_results_across_seeds(
    seed_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate z-results across multiple seeds.

    For each budget, computes:
    - z_mean: mean of z_means across seeds
    - z_stderr: std of z_means / sqrt(n_seeds) (standard error)
    - z_std: std of z_means across seeds (for reference)
    - raw_mean: mean of raw_means across seeds
    - raw_stderr: standard error of raw_means
    - n_seeds: number of seeds aggregated
    - per_seed: dict mapping seed -> z_mean for that seed

    Args:
        seed_results: List of dicts, each containing:
            - "seed": int
            - "z_results": list of z_result dicts (from compute_z_results)

    Returns:
        List of aggregated result dicts, one per budget.
    """
    if not seed_results:
        return []

    # collect all budgets across all seeds
    budgets = set()
    for sr in seed_results:
        for zr in sr.get("z_results", []):
            budget = zr.get("budget")
            if budget is not None:
                budgets.add(budget)

    aggregated = []
    for budget in sorted(budgets):
        z_means = []
        raw_means = []
        per_seed = {}

        for sr in seed_results:
            seed = sr.get("seed")
            for zr in sr.get("z_results", []):
                if zr.get("budget") == budget:
                    z_val = zr.get("z_mean")
                    raw_val = zr.get("raw_mean")
                    if z_val is not None:
                        z_means.append(z_val)
                        per_seed[seed] = z_val
                    if raw_val is not None:
                        raw_means.append(raw_val)
                    break

        n = len(z_means)
        if n == 0:
            aggregated.append(
                {
                    "budget": budget,
                    "z_mean": None,
                    "z_stderr": None,
                    "z_std": None,
                    "raw_mean": None,
                    "raw_stderr": None,
                    "n_seeds": 0,
                    "per_seed": {},
                }
            )
            continue

        z_mean_agg = float(np.mean(z_means))
        z_std_agg = float(np.std(z_means, ddof=1)) if n > 1 else 0.0
        z_stderr = z_std_agg / np.sqrt(n) if n > 0 else 0.0

        raw_mean_agg = float(np.mean(raw_means)) if raw_means else None
        raw_std = float(np.std(raw_means, ddof=1)) if len(raw_means) > 1 else 0.0
        raw_stderr = raw_std / np.sqrt(len(raw_means)) if raw_means else None

        aggregated.append(
            {
                "budget": budget,
                "z_mean": z_mean_agg,
                "z_stderr": float(z_stderr),
                "z_std": z_std_agg,
                "raw_mean": raw_mean_agg,
                "raw_stderr": float(raw_stderr) if raw_stderr is not None else None,
                "n_seeds": n,
                "per_seed": per_seed,
            }
        )

    return aggregated


def collect_results_payload(all_data):
    results = []
    for d in all_data[0]:
        new_d1, new_d2 = None, None
        if isinstance(d[0], np.ndarray):
            new_d1 = d[0].tolist()
        else:
            new_d1 = d[0]
        if isinstance(d[1], np.ndarray):
            new_d2 = d[1].tolist()
        else:
            new_d2 = d[1]

        if (
            isinstance(new_d1, (list, tuple))
            and len(new_d1) >= 2
            and (new_d1[0] is None or new_d1[1] is None)
        ):
            new_d1 = create_fallback_result("Evaluation returned None metrics")[0]
        results.append([new_d1, new_d2])
    return results


def collect_observations(all_data):
    observations = []
    for i in range(len(all_data[2])):
        new_obs = None
        if isinstance(all_data[2][i], np.ndarray):
            new_obs = all_data[2][i].tolist()
        else:
            new_obs = all_data[2][i]
        observations.append(new_obs)
    return observations


def build_results_payload(
    config,
    wandb_meta: dict[str, Any],
    results: list[Any],
    z_results: list[dict[str, Any]],
    norm_mu: float,
    norm_sigma: float,
    all_data,
    observations: list[Any],
    ppl_artifacts,
    scientist_messages: list[Any],
    naive_messages: list[Any],
):
    config_payload = OmegaConf.to_container(config, resolve=True)
    try:
        llm_cfg = config_payload.get("llms")
        if isinstance(llm_cfg, dict) and llm_cfg.get("api_key"):
            llm_cfg["api_key"] = "<redacted>"
    except Exception:
        pass
    return {
        "config": config_payload,
        "wandb": wandb_meta,
        "data": {
            "results": results,
            "z_results": z_results,
            "norm_factors": {"mu": norm_mu, "sigma": norm_sigma},
            "queries": all_data[1],
            "observations": observations,
            "successes": all_data[3],
            "explanations": all_data[4],
            "eigs": all_data[5],
            "programs": ppl_artifacts.only_programs,
            "programs_all": ppl_artifacts.programs_all_text,
            "programs_detailed": [
                {
                    "round": p.get("round"),
                    "program_idx": p.get("program_idx"),
                    "program_code": (p.get("program_code") or "")[:5000],
                    "loo": p.get("loo"),
                    "waic": p.get("waic"),
                    "summary_stats": (p.get("summary_stats") or "")[:1000],
                    "llm_response": (p.get("llm_response") or "")[:2000],
                    "n_divergences": p.get("n_divergences"),
                    "max_rhat": p.get("max_rhat"),
                    "min_ess_bulk": p.get("min_ess_bulk"),
                    "diagnostics": p.get("diagnostics"),
                }
                for p in ppl_artifacts.program_entries
            ],
            "round_stats": ppl_artifacts.round_stats_entries,
        },
        "scientist_messages": scientist_messages,
        "naive_messages": naive_messages,
    }
