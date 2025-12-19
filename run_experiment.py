import argparse
import logging
import os
import sys
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

import numpy as np
from omegaconf import DictConfig
import hydra

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boxing_gym.experiment.utils import _import_real_module
from boxing_gym.experiment.loop import iterative_experiment
from boxing_gym.experiment.step_logger import StepLogger
from boxing_gym.experiment.run_helpers import (
    build_env_goal,
    build_results_payload,
    collect_observations,
    collect_results_payload,
    compute_z_results,
    create_agents,
    pre_generate_eval_points,
)
from boxing_gym.experiment.ppl_utils import extract_ppl_artifacts
from boxing_gym.experiment.run_io import (
    build_output_filename,
    ensure_output_dir,
    write_results,
    write_run_artifact_index,
)
from boxing_gym.experiment.wandb_logging import init_wandb, log_wandb_results

load_dotenv(override=True)
os.environ.setdefault("WANDB_DIR", str(REPO_ROOT / "wandb-dir"))

# import optional modules lazily (avoid side effects on --help)
wandb = None
weave = None

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


def _is_discovery_env(env_name: Optional[str]) -> bool:
    if not env_name:
        return False
    return str(env_name).endswith("_discovery")


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
    table.add_row("--api-key", "Override llms.api_key", "None")
    table.add_row("--custom-llm-provider", "Override llms.custom_llm_provider", "None")
    table.add_row("--com-limit", "Comms word limit (envs.com_limit)", "conf/envs/*.yaml")
    table.add_row("--env-param key=value", "Override envs.env_params.<key>", "None")
    table.add_row("--override key=value", "Raw Hydra override", "None")
    table.add_row("--list-envs / --list-exps / --list-llms", "List configs", "None")
    table.add_row("-h, --help", "Show this help and exit", "")
    console.print(table)

    console.print("\nExamples:")
    console.print("  [cyan]python run_experiment.py --env hyperbolic_direct --exp oed --llm gpt-4o[/]")
    console.print("  [cyan]python run_experiment.py --env dugongs_direct_discovery --num-experiments 1 --num-evals 1 --no-use-ppl[/]")
    console.print("  [cyan]python run_experiment.py envs=dugongs_direct_discovery exp.num_experiments=[1] envs.num_evals=1 use_ppl=false[/]")

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
    parser.add_argument("--api-key")
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

    # Warn about mismatched discovery configs (non-fatal)
    console = Console()
    if exp_name == "discovery" and env_name and not _is_discovery_env(env_name):
        console.print(
            f"[yellow]Warning:[/] exp=discovery usually expects a *_discovery env config. "
            f"Got env={env_name!r}. Consider --env {env_name}_discovery or --exp oed."
        )
    if exp_name and exp_name != "discovery" and _is_discovery_env(env_name):
        console.print(
            f"[yellow]Warning:[/] env={env_name!r} is a discovery config but exp={exp_name!r}. "
            f"Consider --exp discovery or a non-discovery env config."
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
    if args.api_key:
        overrides.append(f"llms.api_key={args.api_key}")
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
    global wandb, weave
    if wandb is None:
        wandb = _import_real_module("wandb", "init")
    if weave is None:
        weave = _import_real_module("weave", "init")
    seed = config.seed
    print(f"seed: {seed}")
    random.seed(seed)
    np.random.seed(int(seed))

    model_name = config.llms.model_name
    temperature = config.llms.temperature
    max_tokens = config.llms.max_tokens
    api_base = config.llms.get("api_base", None)
    api_key = config.llms.get("api_key", None)
    custom_llm_provider = config.llms.get("custom_llm_provider", None)
    num_experiments = config.exp.num_experiments
    env_params = config.envs.env_params
    experiment_type = config.exp.experiment_type
    include_prior = config.include_prior
    num_evals = config.envs.num_evals
    env_name = config.envs.env_name
    goal_name = config.envs.goal_name
    com_limit = config.envs.com_limit
    check_eig = False
    use_ppl = config.use_ppl

    wandb_ctx = init_wandb(
        config=config,
        model_name=model_name,
        env_name=env_name,
        goal_name=goal_name,
        experiment_type=experiment_type,
        seed=seed,
        wandb_module=wandb,
        weave_module=weave,
    )

    env, goal, nametoenv, nameenvtogoal = build_env_goal(
        env_name=env_name,
        goal_name=goal_name,
        env_params=env_params,
        include_prior=include_prior,
    )

    pre_generate_eval_points(
        goal=goal,
        nametoenv=nametoenv,
        nameenvtogoal=nameenvtogoal,
        env_name=env_name,
        goal_name=goal_name,
        env_params=env_params,
        include_prior=include_prior,
        num_evals=num_evals,
        seed=seed,
    )

    scientist_agent, naive_agent = create_agents(
        experiment_type=experiment_type,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_base=api_base,
        api_key=api_key,
        custom_llm_provider=custom_llm_provider,
    )

    system_message = goal.get_system_message(include_prior)
    scientist_agent.set_system_message(system_message)

    norm_mu = getattr(goal, "norm_mu", None)
    norm_sigma = getattr(goal, "norm_sigma", None)

    step_logger = None
    if wandb_ctx.run is not None and wandb is not None:
        step_logger = StepLogger(
            wandb_module=wandb,
            wandb_run=wandb_ctx.run,
            start_time=wandb_ctx.start_time,
        )

    print(f"running {num_experiments} experiments")
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
    naive_messages = None
    if experiment_type == "discovery":
        naive_messages = naive_agent.all_messages

    results = collect_results_payload(all_data)
    observations = collect_observations(all_data)

    ppl_artifacts = extract_ppl_artifacts(all_data, use_ppl)

    final_dict = build_results_payload(
        config=config,
        wandb_meta=wandb_ctx.meta,
        results=results,
        z_results=z_results,
        norm_mu=norm_mu,
        norm_sigma=norm_sigma,
        all_data=all_data,
        observations=observations,
        ppl_artifacts=ppl_artifacts,
        scientist_messages=scientist_messages,
        naive_messages=naive_messages,
    )

    output_filename = build_output_filename(
        env_name=env_name,
        goal_name=goal_name,
        model_name=model_name,
        experiment_type=experiment_type,
        include_prior=include_prior,
        seed=seed,
        use_ppl=use_ppl,
        wandb_meta=wandb_ctx.meta,
    )
    ensure_output_dir(output_filename)

    log_wandb_results(
        ctx=wandb_ctx,
        wandb_module=wandb,
        output_filename=output_filename,
        all_data=all_data,
        z_results=z_results,
        num_experiments=num_experiments,
        env_name=env_name,
        goal_name=goal_name,
        experiment_type=experiment_type,
        include_prior=include_prior,
        seed=seed,
        use_ppl=use_ppl,
        system_message=system_message,
        scientist_agent=scientist_agent,
        naive_agent=naive_agent,
        ppl_artifacts=ppl_artifacts,
        goal=goal,
    )

    write_results(final_dict, output_filename)
    print(f"Results saved to {output_filename}")
    try:
        if wandb_ctx.run is not None:
            write_run_artifact_index(output_filename, wandb_ctx.meta)
    except Exception:
        pass

    print(model_name)
    print("finished successfully :)")

if __name__ == "__main__":
    raise SystemExit(run_cli())
