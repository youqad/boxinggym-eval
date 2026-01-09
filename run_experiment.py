import argparse
import logging
import os
import sys
import importlib
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
logging.basicConfig(level=logging.WARNING)
for name in (
    "pytensor.tensor.blas",
    "pytensor.tensor.blas_headers",
):
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.ERROR)
    _logger.propagate = False

from dotenv import load_dotenv
import random
import hashlib

import numpy as np
from omegaconf import DictConfig, open_dict
import hydra

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boxing_gym.experiment.loop import iterative_experiment
from boxing_gym.experiment.step_logger import StepLogger
from boxing_gym.experiment.run_helpers import (
    aggregate_z_results_across_seeds,
    build_env_goal,
    build_results_payload,
    collect_observations,
    collect_results_payload,
    compute_z_results,
    create_agents,
    pre_generate_eval_points,
)
from boxing_gym.experiment.ppl_utils import extract_ppl_artifacts
from boxing_gym.agents.call_recorder import get_call_recorder
from boxing_gym.experiment.run_io import (
    build_output_filename,
    clear_checkpoint,
    ensure_output_dir,
    load_checkpoint,
    load_seed_data,
    save_checkpoint,
    write_results,
    write_run_artifact_index,
    write_seed_data,
)
from boxing_gym.experiment.wandb_logging import init_wandb, log_wandb_results

load_dotenv(override=True)
os.environ.setdefault("WANDB_DIR", str(REPO_ROOT / "wandb-dir"))

# import optional modules lazily (avoid side effects on --help)
# NOTE: if you have a local folder named 'wandb' or 'weave', rename it to avoid shadowing
wandb = None
weave = None

def _try_import(name: str, required_attr: Optional[str] = None):
    """Try to import a module, with optional attribute check and shadowing fallback."""
    try:
        mod = importlib.import_module(name)
        if required_attr is None or hasattr(mod, required_attr):
            return mod
    except Exception:
        mod = None

    if required_attr is None:
        return None

    # shadowing fallback: remove repo root / cwd from sys.path and retry.
    old_sys_path = list(sys.path)
    try:
        cwd = os.getcwd()
        filtered = [
            p for p in old_sys_path
            if p not in ("", ".", cwd, str(REPO_ROOT))
        ]
        sys.path = filtered
        sys.modules.pop(name, None)
        mod2 = importlib.import_module(name)
        if hasattr(mod2, required_attr):
            logging.getLogger(__name__).warning(
                f"Recovered real '{name}' package after detecting local shadowing."
            )
            return mod2
    except Exception:
        pass
    finally:
        sys.path = old_sys_path

    return None


def _check_model_expiration(llms_config) -> None:
    """Validate temporary model endpoints haven't expired."""
    from datetime import datetime, timezone

    is_temporary = llms_config.get("is_temporary", False)
    expiration_date = llms_config.get("expiration_date")

    if not is_temporary or not expiration_date:
        return

    try:
        exp_date = datetime.fromisoformat(expiration_date.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)

        if now > exp_date:
            raise RuntimeError(
                f"Model endpoint expired on {expiration_date}. "
                f"Update conf/llms config or use a different model."
            )

        days_remaining = (exp_date - now).days
        if days_remaining < 3:
            console = Console()
            console.print(
                f"[yellow]Warning: Model endpoint expires in {days_remaining} day(s) "
                f"({expiration_date})[/yellow]"
            )
    except ValueError as e:
        logging.getLogger(__name__).warning(f"Could not parse expiration_date: {e}")


CONF_ROOT = REPO_ROOT / "conf"


def _list_config_stems(subdir: str) -> List[str]:
    path = CONF_ROOT / subdir
    if not path.exists():
        return []
    return sorted(p.stem for p in path.glob("*.yaml"))


def _load_default_profiles() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        from omegaconf import OmegaConf
    except Exception:
        return None, None, None
    config_path = CONF_ROOT / "config.yaml"
    if not config_path.exists():
        return None, None, None
    try:
        cfg = OmegaConf.load(config_path)
    except Exception:
        return None, None, None
    default_env = None
    default_exp = None
    default_llm = None
    defaults = getattr(cfg, "defaults", None) or []
    for item in defaults:
        if isinstance(item, dict):
            if "envs" in item:
                default_env = item.get("envs")
            if "exp" in item:
                default_exp = item.get("exp")
            if "llms" in item:
                default_llm = item.get("llms")
    return default_env, default_exp, default_llm


def _is_naive_env(env_name: Optional[str]) -> bool:
    if not env_name:
        return False
    return str(env_name).endswith("_naive")


def _resolve_seeds(config) -> Tuple[List[int], bool]:
    """Resolve seeds from config, handling both legacy and multi-seed modes.

    Returns:
        Tuple of (seeds_list, is_multi_seed_mode)
        - If config.seeds is set: returns (seeds_list, True)
        - Otherwise: returns ([config.seed], False)
    """
    from omegaconf import ListConfig
    seeds_config = config.get("seeds", None)

    if seeds_config is not None and seeds_config:
        # multi-seed mode: config.seeds is set
        # handle OmegaConf ListConfig, Python list/tuple, or single value
        if isinstance(seeds_config, (list, tuple, ListConfig)):
            seeds = [int(s) for s in seeds_config]
        else:
            seeds = [int(seeds_config)]
        # dedupe while preserving order
        seen = set()
        unique_seeds = []
        for s in seeds:
            if s not in seen:
                seen.add(s)
                unique_seeds.append(s)
        return unique_seeds, True
    else:
        # legacy single-seed mode
        return [int(config.seed)], False


def _max_iters_from_num_experiments(num_experiments) -> int:
    """Compute the loop cap (max iterations) from a budget schedule."""
    try:
        from omegaconf import ListConfig
    except Exception:
        ListConfig = ()

    if num_experiments is None:
        return 0
    if isinstance(num_experiments, (list, tuple, set, ListConfig)):
        budgets = []
        for b in num_experiments:
            try:
                budgets.append(int(b))
            except Exception:
                continue
        return max(budgets) if budgets else 0
    try:
        return int(num_experiments)
    except Exception:
        return 0


def _print_rich_help(env_choices: List[str], exp_choices: List[str], llm_choices: List[str]) -> None:
    console = Console()
    console.print(Panel.fit("[bold]BoxingGym run_experiment[/]"))
    console.print("Usage:")
    console.print("  [cyan]python run_experiment.py[/] [flags] [hydra overrides]\n")

    table = Table(title="Options", show_header=True, header_style="bold")
    table.add_column("Flag", style="cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Default")
    table.add_row("--env", "Environment config (conf/envs/*.yaml)", "config default")
    table.add_row("--exp", "Experiment config (conf/exp/*.yaml)", "config default")
    table.add_row("--llm", "LLM config (conf/llms/*.yaml)", "config default")
    table.add_row("--seed", "Random seed", "config default")
    table.add_row("--num-experiments", "Budgets (e.g. 0 5 10)", "conf/exp/*.yaml")
    table.add_row("--num-evals", "Eval questions", "conf/envs/*.yaml")
    table.add_row("--include-prior / --no-include-prior", "Include the prior", "config default")
    table.add_row("--use-ppl / --no-use-ppl", "Use PPL/Box's Loop", "config default")
    table.add_row("--temperature", "LLM temperature (llms.temperature)", "conf/llms/*.yaml")
    table.add_row("--max-tokens", "LLM max tokens (llms.max_tokens)", "conf/llms/*.yaml")
    table.add_row("--api-base", "Override llms.api_base", "None")
    # --api-key removed: use .env or OPENAI_API_KEY env var instead (security)
    table.add_row("--custom-llm-provider", "Override llms.custom_llm_provider", "None")
    table.add_row("--com-limit", "Comms word limit (envs.com_limit)", "conf/envs/*.yaml")
    table.add_row("--env-param key=value", "Override envs.env_params.<key>", "None")
    table.add_row("--override key=value", "Raw Hydra override", "None")
    table.add_row("--list-envs / --list-exps / --list-llms", "List configs", "None")
    table.add_row("-h, --help", "Show this help and exit", "")
    console.print(table)

    console.print("\nExamples:")
    console.print("  [cyan]python run_experiment.py --env hyperbolic_direct --exp oed --llm gpt-4o[/]")
    console.print("  [cyan]python run_experiment.py --env dugongs_direct_naive --num-experiments 1 --num-evals 1 --no-use-ppl[/]")
    console.print("  [cyan]python run_experiment.py envs=dugongs_direct_naive exp.num_experiments=[1] envs.num_evals=1 use_ppl=false[/]")

    if env_choices:
        console.print(f"\nEnvs ({len(env_choices)}): {', '.join(env_choices)}")
    if exp_choices:
        console.print(f"Exps ({len(exp_choices)}): {', '.join(exp_choices)}")
    if llm_choices:
        console.print(f"LLMs ({len(llm_choices)}): {', '.join(llm_choices)}")


def _print_list(title: str, items: List[str]) -> None:
    console = Console()
    if not items:
        console.print(f"[yellow]No {title} found.[/]")
        return
    table = Table(title=title, show_header=False)
    table.add_column("Name", style="cyan")
    for item in items:
        table.add_row(item)
    console.print(table)


def _parse_int_list(values: Optional[List[str]]) -> Optional[List[int]]:
    if not values:
        return None
    out: List[int] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
    return out or None


def _parse_env_params(values: Optional[List[str]]) -> List[str]:
    overrides = []
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"--env-param must be key=value, got: {item}")
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            raise ValueError(f"--env-param key is empty: {item}")
        overrides.append(f"envs.env_params.{key}={val}")
    return overrides


def _build_arg_parser(env_choices: List[str], exp_choices: List[str], llm_choices: List[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--env", choices=env_choices or None)
    parser.add_argument("--exp", choices=exp_choices or None)
    parser.add_argument("--llm", choices=llm_choices or None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-experiments", nargs="+")
    parser.add_argument("--num-evals", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--api-base")
    # --api-key removed for security (visible in ps aux); use .env instead
    parser.add_argument("--custom-llm-provider")
    parser.add_argument("--com-limit", type=int)
    parser.add_argument("--env-param", action="append", default=[])
    parser.add_argument("--override", action="append", default=[])

    prior_group = parser.add_mutually_exclusive_group()
    prior_group.add_argument("--include-prior", action="store_true")
    prior_group.add_argument("--no-include-prior", action="store_true")

    ppl_group = parser.add_mutually_exclusive_group()
    ppl_group.add_argument("--use-ppl", action="store_true")
    ppl_group.add_argument("--no-use-ppl", action="store_true")

    parser.add_argument("--list-envs", action="store_true")
    parser.add_argument("--list-exps", action="store_true")
    parser.add_argument("--list-llms", action="store_true")
    return parser


def run_cli(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    env_choices = _list_config_stems("envs")
    exp_choices = _list_config_stems("exp")
    llm_choices = _list_config_stems("llms")
    default_env, default_exp, default_llm = _load_default_profiles()

    parser = _build_arg_parser(env_choices, exp_choices, llm_choices)
    args, unknown = parser.parse_known_args(argv)

    if args.help:
        _print_rich_help(env_choices, exp_choices, llm_choices)
        return 0

    if args.list_envs:
        _print_list("Envs (conf/envs)", env_choices)
        return 0
    if args.list_exps:
        _print_list("Experiments (conf/exp)", exp_choices)
        return 0
    if args.list_llms:
        _print_list("LLMs (conf/llms)", llm_choices)
        return 0

    exp_name = args.exp or default_exp
    env_name = args.env or default_env

    # Warn about mismatched env/exp configs (non-fatal)
    console = Console()
    if exp_name == "discovery" and env_name and not _is_naive_env(env_name):
        console.print(
            f"[yellow]Warning:[/] exp=discovery usually expects a *_naive env config. "
            f"Got env={env_name!r}. Consider --env {env_name}_naive or --exp oed."
        )
    if exp_name and exp_name != "discovery" and _is_naive_env(env_name):
        console.print(
            f"[yellow]Warning:[/] env={env_name!r} is a naive config but exp={exp_name!r}. "
            f"Consider --exp discovery or a non-naive env config."
        )

    overrides: List[str] = []
    if args.env:
        overrides.append(f"envs={args.env}")
    if args.exp:
        overrides.append(f"exp={args.exp}")
    if args.llm:
        overrides.append(f"llms={args.llm}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    num_experiments = _parse_int_list(args.num_experiments)
    if num_experiments is not None:
        overrides.append(f"exp.num_experiments={num_experiments}")
    if args.num_evals is not None:
        overrides.append(f"envs.num_evals={args.num_evals}")
    if args.temperature is not None:
        overrides.append(f"llms.temperature={args.temperature}")
    if args.max_tokens is not None:
        overrides.append(f"llms.max_tokens={args.max_tokens}")
    if args.api_base:
        overrides.append(f"llms.api_base={args.api_base}")
    # api_key override removed for security; use .env or env vars
    if args.custom_llm_provider:
        overrides.append(f"llms.custom_llm_provider={args.custom_llm_provider}")
    if args.com_limit is not None:
        overrides.append(f"envs.com_limit={args.com_limit}")
    if args.include_prior:
        overrides.append("include_prior=true")
    if args.no_include_prior:
        overrides.append("include_prior=false")
    if args.use_ppl:
        overrides.append("use_ppl=true")
    if args.no_use_ppl:
        overrides.append("use_ppl=false")
    overrides.extend(_parse_env_params(args.env_param))
    overrides.extend(args.override or [])

    # pass remaining args (Hydra overrides, -m, hydra.* options) through unchanged
    hydra_args = overrides + unknown
    sys.argv = [sys.argv[0]] + hydra_args

    # minimal rich run banner for clarity
    console = Console()
    if hydra_args:
        console.print(Panel.fit("[bold]Starting BoxingGym run[/]"))
        table = Table(show_header=False)
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        for item in hydra_args:
            table.add_row("override", item)
        console.print(table)

    main()
    return 0


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    import copy

    global wandb, weave
    if wandb is None:
        wandb = _try_import("wandb", required_attr="init")
    if weave is None:
        weave = _try_import("weave", required_attr="init")

    # resolve seeds: multi-seed mode (config.seeds) vs legacy single-seed (config.seed)
    seeds, is_multi_seed = _resolve_seeds(config)
    print(f"Seeds: {seeds} (multi-seed mode: {is_multi_seed})")

    # validate temporary endpoint hasn't expired
    _check_model_expiration(config.llms)

    model_name = config.llms.model_name
    temperature = config.llms.temperature
    max_tokens = config.llms.max_tokens
    api_base = config.llms.get("api_base", None)
    api_key = config.llms.get("api_key", None)
    custom_llm_provider = config.llms.get("custom_llm_provider", None)
    extra_headers = config.llms.get("extra_headers", None)
    if extra_headers is not None:
        from omegaconf import OmegaConf
        if hasattr(extra_headers, "_content"):
            extra_headers = OmegaConf.to_container(extra_headers, resolve=True)
    num_experiments = config.exp.num_experiments
    env_params = config.envs.env_params
    experiment_type = config.exp.experiment_type
    include_prior = config.include_prior
    num_evals = config.envs.num_evals
    env_name = config.envs.env_name
    goal_name = config.envs.goal_name
    com_limit = config.envs.com_limit
    check_eig = config.exp.get("check_eig", config.check_eig)
    use_ppl = config.use_ppl
    max_iters = _max_iters_from_num_experiments(num_experiments)

    with open_dict(config):
        config.check_eig = check_eig
        # store resolved seeds in config for WandB logging
        if is_multi_seed:
            config.seeds = seeds

    # seed label for run naming: "s1-2-3-4-5" for multi-seed (becomes "_seeds1-2-3-4-5" in run name),
    # just the seed number for single-seed (becomes "_seed1" in run name)
    primary_seed = seeds[0]
    seed_label_for_run = f"s{'-'.join(map(str, seeds))}" if is_multi_seed else str(primary_seed)

    wandb_ctx = init_wandb(
        config=config,
        model_name=model_name,
        env_name=env_name,
        goal_name=goal_name,
        experiment_type=experiment_type,
        seed=seed_label_for_run,
        wandb_module=wandb,
        weave_module=weave,
    )

    # pre-generate shared eval points ONCE with a fixed seed (for valid cross-seed comparison)
    # all seeds will see the SAME test questions
    eval_seed = 0
    print(f"\n📋 Pre-generating shared eval points with fixed seed {eval_seed}...")
    random.seed(eval_seed)
    np.random.seed(eval_seed)

    env_for_eval, goal_for_eval, nametoenv, nameenvtogoal = build_env_goal(
        env_name=env_name,
        goal_name=goal_name,
        env_params=env_params,
        include_prior=include_prior,
    )
    pre_generate_eval_points(
        goal=goal_for_eval,
        nametoenv=nametoenv,
        nameenvtogoal=nameenvtogoal,
        env_name=env_name,
        goal_name=goal_name,
        env_params=env_params,
        include_prior=include_prior,
        num_evals=num_evals,
        seed=eval_seed,
    )
    shared_eval_points = copy.deepcopy(goal_for_eval.eval_points)
    norm_mu = getattr(goal_for_eval, "norm_mu", None)
    norm_sigma = getattr(goal_for_eval, "norm_sigma", None)

    # create call recorder for entire run
    if wandb_ctx.meta.get("run_id"):
        run_id = wandb_ctx.meta.get("run_id")
    else:
        # stable, collision-resistant ID for non-WandB runs (supports resume without cross-run clobbering)
        run_key = f"{seed_label_for_run}|{env_name}|{goal_name}|{experiment_type}|{model_name}|{include_prior}|{use_ppl}"
        run_hash = hashlib.sha256(run_key.encode()).hexdigest()[:8]
        run_id = f"run_{seed_label_for_run}_{run_hash}"
    call_recorder = get_call_recorder(run_id=run_id, output_dir="results")

    # step logger for WandB
    step_logger = None
    if wandb_ctx.run is not None and wandb is not None:
        step_logger = StepLogger(
            wandb_module=wandb,
            wandb_run=wandb_ctx.run,
            start_time=wandb_ctx.start_time,
        )

    # checkpointing: load any existing checkpoint for crash recovery
    checkpoint = load_checkpoint(run_id) if is_multi_seed else {}
    completed_seeds = set(checkpoint.get("completed_seeds", []))

    # load FULL data for completed seeds (not just lightweight checkpoint)
    seed_results = []
    for completed_seed in sorted(completed_seeds):
        full_data = load_seed_data(run_id, completed_seed)
        if full_data:
            seed_results.append(full_data)
        else:
            # fallback to lightweight checkpoint data if full file missing
            for sr in checkpoint.get("seed_results", []):
                if sr["seed"] == completed_seed:
                    seed_results.append(sr)
                    break

    if completed_seeds:
        print(f"📦 Resuming from checkpoint: {len(completed_seeds)} seeds already completed")
        full_data_count = len([s for s in seed_results if s.get("all_data")])
        print(f"   Loaded full data for {full_data_count}/{len(completed_seeds)} seeds")
        if full_data_count < len(completed_seeds):
            print(f"   ⚠️  {len(completed_seeds) - full_data_count} seeds have only lightweight checkpoint data")

    # filter out already-completed seeds
    remaining_seeds = [s for s in seeds if s not in completed_seeds]

    # initialize variables that may not be set if all seeds already complete
    # (these are used by log_wandb_results after the loop)
    system_message = None
    scientist_agent = None
    naive_agent = None
    goal = None

    # if all seeds already complete, skip directly to aggregation
    if not remaining_seeds and seed_results:
        print(f"\n✅ All {len(seeds)} seeds already complete - skipping to aggregation")

    starting_completed_count = len(completed_seeds)
    for seed_idx, seed in enumerate(remaining_seeds):
        total_idx = starting_completed_count + seed_idx + 1
        print(f"\n{'='*60}")
        print(f"🌱 Starting seed {seed} ({total_idx}/{len(seeds)})")
        print(f"{'='*60}")

        if step_logger is not None and max_iters:
            step_offset = (starting_completed_count + seed_idx) * max_iters
            step_logger.set_step_offset(step_offset, reset_timing=True)

        # set RNG for this seed BEFORE creating env/goal/agents
        random.seed(seed)
        np.random.seed(int(seed))

        # fresh env/goal instantiation per seed (critical for isolation)
        env, goal, _, _ = build_env_goal(
            env_name=env_name,
            goal_name=goal_name,
            env_params=env_params,
            include_prior=include_prior,
        )
        # use shared eval points (same test questions for all seeds)
        goal.eval_points = copy.deepcopy(shared_eval_points)
        goal.eval_pointer = 0

        # fresh agents per seed (critical for clean message history)
        scientist_agent, naive_agent = create_agents(
            experiment_type=experiment_type,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
        )
        scientist_agent.set_call_recorder(call_recorder, agent_name=f"scientist_seed{seed}")
        if naive_agent is not None:
            naive_agent.set_call_recorder(call_recorder, agent_name=f"naive_seed{seed}")

        system_message = goal.get_system_message(include_prior)
        scientist_agent.set_system_message(system_message)

        print(f"Running {num_experiments} experiments for seed {seed}...")
        all_data = iterative_experiment(
            goal,
            scientist_agent,
            num_experiments,
            num_evals,
            include_prior,
            naive_agent,
            com_limit,
            check_eig,
            use_ppl,
            step_logger=step_logger,
            norm_mu=norm_mu,
            norm_sigma=norm_sigma,
            call_recorder=call_recorder,
        )

        z_results = compute_z_results(
            all_data=all_data,
            num_experiments=num_experiments,
            norm_mu=norm_mu,
            norm_sigma=norm_sigma,
            env_name=env_name,
            goal_name=goal_name,
        )

        scientist_messages = scientist_agent.all_messages
        naive_messages = naive_agent.all_messages if naive_agent else None

        seed_results.append({
            "seed": seed,
            "z_results": z_results,
            "all_data": all_data,
            "scientist_messages": scientist_messages,
            "naive_messages": naive_messages,
            "status": "ok",
        })

        # prepare per-seed metrics for WandB logging (log after successful checkpoint in multi-seed)
        wandb_log_payload = None
        if wandb_ctx.run is not None and wandb is not None and z_results:
            final_z = z_results[-1].get("z_mean")
            if final_z is not None:
                wandb_log_payload = {
                    f"seed_{seed}/z_mean": final_z,
                }

        # checkpoint: save after each seed (for crash recovery)
        # IMPORTANT: write full data BEFORE marking seed complete in checkpoint
        # (prevents data loss if process exits between checkpoint and full write)
        if is_multi_seed:
            # write FULL data first (all_data, messages) for investigation
            seed_data_path = write_seed_data(run_id, seed, seed_results[-1])
            if seed_data_path:
                print(f"   📁 Saved full data to {seed_data_path}")
                # only mark seed as complete if full data was successfully written
                completed_seeds.add(seed)
                save_checkpoint(run_id, list(completed_seeds), seed_results)
                if wandb_log_payload is not None:
                    wandb_log_payload["seeds_completed"] = len(completed_seeds)
                    wandb.log(wandb_log_payload)
            else:
                # write failed - DON'T mark complete, will retry on next run
                print(f"   ⚠️  Seed {seed} data NOT saved - will retry on next run")
        else:
            if wandb_log_payload is not None:
                wandb_log_payload["seeds_completed"] = total_idx
                wandb.log(wandb_log_payload)

        print(f"✓ Completed seed {seed}")

    # aggregate across seeds
    print(f"\n{'='*60}")
    print(f"📊 Aggregating results across {len(seeds)} seeds...")
    print(f"{'='*60}")

    if is_multi_seed and len(seed_results) > 1:
        z_results_aggregated = aggregate_z_results_across_seeds(seed_results)
        print("\n📊 Aggregated Z-Score Results:")
        for agg in z_results_aggregated:
            if agg.get("z_mean") is not None:
                print(
                    f"  Budget {agg['budget']}: z={agg['z_mean']:.3f} ± {agg['z_stderr']:.3f} "
                    f"(n={agg['n_seeds']} seeds)"
                )
    else:
        # single seed: just use the z_results directly
        z_results_aggregated = seed_results[0]["z_results"] if seed_results else []

    # build final payload with aggregated results + per-seed breakdown
    # use the last seed's data for backward-compatible fields
    last_seed_data = seed_results[-1] if seed_results else {}
    all_data = last_seed_data.get("all_data", ([], [], [], [], [], [], []))
    scientist_messages = last_seed_data.get("scientist_messages", [])
    naive_messages = last_seed_data.get("naive_messages")

    results = collect_results_payload(all_data)
    observations = collect_observations(all_data)
    ppl_artifacts = extract_ppl_artifacts(all_data, use_ppl)

    # build payload with aggregated z_results
    final_dict = build_results_payload(
        config=config,
        wandb_meta=wandb_ctx.meta,
        results=results,
        z_results=z_results_aggregated,
        norm_mu=norm_mu,
        norm_sigma=norm_sigma,
        all_data=all_data,
        observations=observations,
        ppl_artifacts=ppl_artifacts,
        scientist_messages=scientist_messages,
        naive_messages=naive_messages,
    )

    # add per-seed breakdown for multi-seed runs - include FULL data for investigation
    if is_multi_seed:
        final_dict["data"]["seed_runs"] = []
        for sr in seed_results:
            # convert all_data tuple to dict for JSON serialization
            all_data_tuple = sr.get("all_data", ([], [], [], [], [], [], []))
            all_data_dict = {
                "results": all_data_tuple[0] if len(all_data_tuple) > 0 else [],
                "queries": all_data_tuple[1] if len(all_data_tuple) > 1 else [],
                "observations": all_data_tuple[2] if len(all_data_tuple) > 2 else [],
                "successes": all_data_tuple[3] if len(all_data_tuple) > 3 else [],
                "explanations": all_data_tuple[4] if len(all_data_tuple) > 4 else [],
                "eigs": all_data_tuple[5] if len(all_data_tuple) > 5 else [],
                "programs": all_data_tuple[6] if len(all_data_tuple) > 6 else [],
            } if isinstance(all_data_tuple, (tuple, list)) else all_data_tuple

            final_dict["data"]["seed_runs"].append({
                "seed": sr["seed"],
                "z_results": sr["z_results"],
                "status": sr["status"],
                "all_data": all_data_dict,
                "scientist_messages": sr.get("scientist_messages"),
                "naive_messages": sr.get("naive_messages"),
            })
        final_dict["data"]["aggregation_mode"] = "multi_seed"
        final_dict["data"]["n_seeds"] = len(seeds)
    else:
        final_dict["data"]["aggregation_mode"] = "single_seed"

    # reuse seed_label_for_run computed earlier: "seeds1-2-3-4-5" or just seed number
    output_filename = build_output_filename(
        env_name=env_name,
        goal_name=goal_name,
        model_name=model_name,
        experiment_type=experiment_type,
        include_prior=include_prior,
        seed=seed_label_for_run,
        use_ppl=use_ppl,
        wandb_meta=wandb_ctx.meta,
    )
    ensure_output_dir(output_filename)

    write_results(final_dict, output_filename)
    print(f"Results saved to {output_filename}")

    # log aggregated summary metrics for multi-seed runs BEFORE log_wandb_results finishes the run
    if is_multi_seed and wandb_ctx.run is not None and wandb is not None:
        final_agg = z_results_aggregated[-1] if z_results_aggregated else {}
        wandb_ctx.run.summary.update({
            "eval/z_mean": final_agg.get("z_mean"),
            "eval/z_stderr": final_agg.get("z_stderr"),
            "eval/n_seeds": len(seeds),
            "run/aggregation_mode": "multi_seed",
            "run/seeds": seeds,
        })

    # log to WandB with aggregated metrics
    log_wandb_results(
        ctx=wandb_ctx,
        wandb_module=wandb,
        output_filename=output_filename,
        all_data=all_data,
        z_results=z_results_aggregated,
        num_experiments=num_experiments,
        env_name=env_name,
        goal_name=goal_name,
        experiment_type=experiment_type,
        include_prior=include_prior,
        seed=primary_seed,
        use_ppl=use_ppl,
        system_message=system_message,
        scientist_agent=scientist_agent,
        naive_agent=naive_agent,
        ppl_artifacts=ppl_artifacts,
        goal=goal,
        call_recorder_path=str(call_recorder.get_filepath()),
    )

    try:
        if wandb_ctx.run is not None:
            write_run_artifact_index(output_filename, wandb_ctx.meta)
    except Exception:
        pass

    # clear checkpoint after successful completion
    if is_multi_seed:
        clear_checkpoint(run_id)

    print(f"\n{model_name}")
    print(f"Completed {len(seeds)} seed(s)")

if __name__ == "__main__":
    raise SystemExit(run_cli())
