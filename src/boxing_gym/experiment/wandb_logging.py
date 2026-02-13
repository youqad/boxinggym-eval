import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _wandb_unseeded_random():
    import random

    state = random.getstate()
    random.seed()
    try:
        yield
    finally:
        random.setstate(state)


@dataclass
class WandbContext:
    run: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
    artifacts_logged: list[dict[str, Any]] = field(default_factory=list)
    artifacts_failed: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = 0.0


def init_wandb(
    config,
    model_name: str,
    env_name: str,
    goal_name: str,
    experiment_type: str,
    seed: int,
    wandb_module,
    weave_module,
) -> WandbContext:
    ctx = WandbContext(start_time=time.time())
    if wandb_module is None:
        return ctx

    wandb_cfg = _get_wandb_cfg(config)
    wandb_enabled = _is_wandb_enabled(wandb_cfg)

    config_to_store = OmegaConf.to_container(config, resolve=True)
    _redact_llm_api_key(config_to_store)

    if not wandb_enabled:
        return ctx

    wandb_project, wandb_entity, wandb_group, wandb_tags = _get_wandb_project_info(wandb_cfg)
    run_name = _build_run_name(env_name, goal_name, experiment_type, model_name, seed)
    _ensure_wandb_env(wandb_project, wandb_entity)

    ctx.run = _init_wandb_run(
        wandb_module,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_tags=wandb_tags,
        run_name=run_name,
        config_to_store=config_to_store,
    )
    ctx.meta = _build_wandb_meta(ctx.run)
    if ctx.run and hasattr(ctx.run, "id"):
        os.environ["WANDB_RUN_ID"] = ctx.run.id

    _define_wandb_metrics(wandb_module)
    _init_weave(weave_module, wandb_project)

    return ctx


def _get_wandb_cfg(config):
    try:
        wandb_cfg = config.get("wandb", {}) or {}
    except Exception:
        wandb_cfg = {}
    if isinstance(wandb_cfg, bool):
        wandb_cfg = {"enabled": wandb_cfg}
    return wandb_cfg


def _is_wandb_enabled(wandb_cfg) -> bool:
    wandb_disabled = str(os.environ.get("WANDB_DISABLED", "")).lower() in ("1", "true", "yes")
    wandb_disabled = wandb_disabled or str(os.environ.get("WANDB_MODE", "")).lower() == "disabled"
    if wandb_disabled:
        return False
    if bool(os.environ.get("WANDB_PROJECT")):
        return True
    try:
        return bool(wandb_cfg.get("enabled", False))
    except Exception:
        return False


def _redact_llm_api_key(config_to_store):
    try:
        llm_cfg = config_to_store.get("llms")
        if isinstance(llm_cfg, dict) and llm_cfg.get("api_key"):
            llm_cfg["api_key"] = "<redacted>"
    except AttributeError:
        pass


def _get_wandb_project_info(wandb_cfg):
    wandb_project = wandb_cfg.get("project") or os.environ.get("WANDB_PROJECT") or "boxing-gym"
    wandb_entity = wandb_cfg.get("entity") or os.environ.get("WANDB_ENTITY")
    wandb_group = wandb_cfg.get("group") or os.environ.get("WANDB_GROUP")
    wandb_tags = wandb_cfg.get("tags") or []
    return wandb_project, wandb_entity, wandb_group, wandb_tags


def _build_run_name(env_name, goal_name, experiment_type, model_name, seed):
    display_model = model_name.split("/")[-1] if "/" in model_name else model_name
    return f"{env_name}_{goal_name}_{experiment_type}_{display_model}_seed{seed}"


def _ensure_wandb_env(wandb_project, wandb_entity):
    try:
        os.environ.setdefault("WANDB_PROJECT", str(wandb_project))
        if wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", str(wandb_entity))
    except Exception:
        pass


def _init_wandb_run(
    wandb_module,
    wandb_project,
    wandb_entity,
    wandb_group,
    wandb_tags,
    run_name,
    config_to_store,
):
    return wandb_module.init(
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        tags=wandb_tags if wandb_tags else None,
        name=run_name,
        config=config_to_store,
    )


def _build_wandb_meta(wandb_run):
    if wandb_run is None:
        return {}
    try:
        meta = {
            "run_id": getattr(wandb_run, "id", None),
            "run_name": getattr(wandb_run, "name", None),
            "project": getattr(wandb_run, "project", None),
            "entity": getattr(wandb_run, "entity", None),
            "dir": getattr(wandb_run, "dir", None),
            "url": getattr(wandb_run, "url", None),
        }
        if meta.get("entity") and meta.get("project") and meta.get("run_id"):
            meta["run_path"] = f"{meta['entity']}/{meta['project']}/{meta['run_id']}"
        return meta
    except Exception:
        return {}


def _define_wandb_metrics(wandb_module):
    try:
        wandb_module.define_metric("step/*", step_metric="step/idx")
        wandb_module.define_metric("cumulative/*", step_metric="step/idx")
        wandb_module.define_metric("llm/*", step_metric="step/idx")
        wandb_module.define_metric("eval/*", step_metric="eval/budget")
        wandb_module.define_metric("exit/*", step_metric="step/idx")
        wandb_module.define_metric("comm/*", step_metric="step/idx")
        wandb_module.define_metric("ppl/*", step_metric="eval/budget")
    except Exception as e:
        logger.debug(f"WandB define_metric failed: {e}")


def _init_weave(weave_module, wandb_project):
    if weave_module is None or os.environ.get("WEAVE_DISABLED", "0").lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        weave_project = os.environ.get("WEAVE_PROJECT") or wandb_project
        weave_settings = {}
        if os.environ.get("WEAVE_IMPLICITLY_PATCH_INTEGRATIONS") is None:
            weave_settings["implicitly_patch_integrations"] = False
        if weave_settings:
            weave_module.init(weave_project, settings=weave_settings)
        else:
            weave_module.init(weave_project)
    except Exception as e:
        logger.warning(f"Weave init failed: {e}")


_USAGE_KEYS = [
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "total_cost_usd",
    "call_count",
    "retry_count",
    "error_count",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_min_ms",
    "latency_max_ms",
]


def _preview_text(value, limit):
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    try:
        return str(value)[:limit]
    except Exception:
        return ""


def _safe_number(value):
    if value is None:
        return None
    try:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, str):
            if not value.strip():
                return None
            try:
                value = float(value)
            except Exception:
                return None
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value
        return None
    except Exception:
        return None


def _flatten_programs(programs):
    all_programs = []
    for round_programs in programs:
        if isinstance(round_programs, list):
            all_programs.extend(round_programs)
        elif round_programs:
            all_programs.append(round_programs)
    return all_programs


def _add_box_loop_stats(payload, result_entry):
    box_loop_stats = None
    try:
        if isinstance(result_entry, (list, tuple)) and len(result_entry) > 4:
            box_loop_stats = result_entry[4]
    except Exception:
        box_loop_stats = None
    if not isinstance(box_loop_stats, list) or not box_loop_stats:
        return
    payload["ppl/num_rounds"] = len(box_loop_stats)
    numeric_keys = set()
    for s in box_loop_stats:
        if not isinstance(s, dict):
            continue
        for k, v in s.items():
            if k == "round_idx":
                continue
            if isinstance(v, (int, float)):
                numeric_keys.add(k)
    for k in numeric_keys:
        vals = [
            float(s[k])
            for s in box_loop_stats
            if isinstance(s, dict) and isinstance(s.get(k), (int, float))
        ]
        if vals:
            payload[f"ppl/{k}_mean"] = sum(vals) / len(vals)
            payload[f"ppl/{k}_last"] = vals[-1]


def _build_eval_payload(result_entry, budget_i, z_by_budget, raw_by_budget=None):
    payload = {"eval/budget": budget_i}
    try:
        eval_score = result_entry[0]
    except Exception:
        return None

    mean_val = None
    std_val = None
    if isinstance(eval_score, dict):
        for k, v in eval_score.items():
            if isinstance(v, (int, float)) or v is None:
                payload[f"eval/{k}"] = v
        mean_val = eval_score.get("mse", eval_score.get("accuracy", eval_score.get("score")))
        std_val = eval_score.get("std_mse", eval_score.get("std", eval_score.get("std_accuracy")))
    else:
        try:
            mse, std_mse = eval_score
        except Exception:
            mse, std_mse = None, None
        mean_val, std_val = mse, std_mse
        payload["eval/mse"] = float(mse) if mse is not None else None
        payload["eval/std_mse"] = float(std_mse) if std_mse is not None else None

    payload["eval/mean"] = float(mean_val) if mean_val is not None else None
    payload["eval/std"] = float(std_val) if std_val is not None else None

    if raw_by_budget and budget_i in raw_by_budget:
        raw = raw_by_budget.get(budget_i) or {}
        raw_mean = raw.get("raw_mean")
        raw_std = raw.get("raw_std")
        if raw_mean is not None:
            payload["eval/mean"] = raw_mean
            payload["eval/mse"] = raw_mean
        if raw_std is not None:
            payload["eval/std"] = raw_std
            payload["eval/std_mse"] = raw_std

    zr = z_by_budget.get(budget_i)
    if zr:
        payload["eval/z_mean"] = zr.get("z_mean")
        payload["eval/z_std"] = zr.get("z_std")

    _add_box_loop_stats(payload, result_entry)
    return payload


def _log_eval_series(wandb_module, all_data, z_results, num_experiments):
    z_by_budget = {zr.get("budget"): zr for zr in z_results if isinstance(zr, dict)}
    raw_by_budget = {
        zr.get("budget"): zr
        for zr in z_results
        if isinstance(zr, dict) and zr.get("raw_mean") is not None
    }
    for i, result_entry in enumerate(all_data[0]):
        try:
            budget_i = (
                int(num_experiments[i]) if i < len(num_experiments) else int(num_experiments[-1])
            )
        except Exception:
            budget_i = i
        payload = _build_eval_payload(
            result_entry, budget_i, z_by_budget, raw_by_budget=raw_by_budget
        )
        if payload is None:
            continue
        wandb_module.log(payload)
    return z_by_budget


def _collect_usage_stats(agent, prefix):
    data = {}
    if agent is None or not hasattr(agent, "get_usage_stats"):
        return data
    usage = agent.get_usage_stats()
    for k in _USAGE_KEYS:
        data[f"{prefix}{k}"] = usage.get(k, 0)
    return data


def _build_summary_data(
    all_data,
    z_by_budget,
    num_experiments,
    env_name,
    goal_name,
    experiment_type,
    include_prior,
    seed,
    use_ppl,
    scientist_agent,
    naive_agent,
    start_time,
):
    successes = all_data[3] or []
    success_rate = (
        float(sum(bool(s) for s in successes)) / float(len(successes)) if successes else 0.0
    )
    summary_data = {
        "run/success_rate": success_rate,
        "run/num_queries": len(all_data[1] or []),
        "run/num_observations": len(all_data[2] or []),
        "run/wall_time_sec": time.time() - start_time,
    }

    summary_data.update(_collect_usage_stats(scientist_agent, "llm/"))
    naive_stats = _collect_usage_stats(naive_agent, "naive_llm/")
    summary_data.update(naive_stats)
    if naive_stats:
        try:
            summary_data["llm_total/total_tokens"] = summary_data.get(
                "llm/total_tokens", 0
            ) + summary_data.get("naive_llm/total_tokens", 0)
            summary_data["llm_total/total_cost_usd"] = summary_data.get(
                "llm/total_cost_usd", 0.0
            ) + summary_data.get("naive_llm/total_cost_usd", 0.0)
            summary_data["llm_total/call_count"] = summary_data.get(
                "llm/call_count", 0
            ) + summary_data.get("naive_llm/call_count", 0)
        except Exception:
            pass

    try:
        final_budget = None
        if hasattr(num_experiments, "__len__"):
            budgets = [int(b) for b in num_experiments]
            final_budget = max(budgets) if budgets else None
        else:
            final_budget = int(num_experiments) if num_experiments is not None else None

        final_entry = all_data[0][-1] if all_data and all_data[0] else None
        if final_entry is not None:
            final_score = final_entry[0]
            final_mean, final_std = None, None
            if isinstance(final_score, dict):
                final_mean = final_score.get(
                    "mse", final_score.get("accuracy", final_score.get("score"))
                )
                final_std = final_score.get(
                    "std_mse", final_score.get("std", final_score.get("std_accuracy"))
                )
            else:
                try:
                    final_mean, final_std = final_score
                except Exception:
                    final_mean, final_std = None, None
            summary_data["eval/mean_final"] = float(final_mean) if final_mean is not None else None
            summary_data["eval/std_final"] = float(final_std) if final_std is not None else None
            summary_data["eval/mse_final"] = summary_data["eval/mean_final"]
            summary_data["eval/std_mse_final"] = summary_data["eval/std_final"]

        if final_budget is not None and final_budget in z_by_budget:
            zf = z_by_budget.get(final_budget) or {}
            summary_data["eval/z_mean_final"] = zf.get("z_mean")
            summary_data["eval/z_mean"] = zf.get("z_mean")
            summary_data["eval/z_std_final"] = zf.get("z_std")
            summary_data["eval/z_std"] = zf.get("z_std")
    except Exception:
        pass

    summary_data.update(
        {
            "exp/env_name": env_name,
            "exp/goal_type": goal_name,
            "exp/experiment_type": experiment_type,
            "exp/use_ppl": use_ppl,
            "exp/include_prior": include_prior,
            "exp/seed": seed,
        }
    )

    programs = all_data[6] if len(all_data) > 6 else []
    if programs and use_ppl:
        flat_programs = [
            p
            for sublist in programs
            for p in (sublist if isinstance(sublist, list) else [sublist])
            if p
        ]
        summary_data["ppl/num_programs_generated"] = len(flat_programs)

    eigs = all_data[5] if len(all_data) > 5 else []
    if eigs:
        summary_data["eig/mean"] = sum(eigs) / len(eigs) if eigs else 0.0
        summary_data["eig/count"] = len(eigs)

    return summary_data


def _log_system_prompt(wandb_run, system_message):
    if not system_message:
        return
    try:
        wandb_run.config.update({"system_prompt": system_message}, allow_val_change=True)
    except Exception as e:
        logger.debug(f"W&B system prompt logging failed: {e}")


def _log_ppl_program_table(wandb_module, wandb_run, all_programs, detailed_entries):
    ppl_columns = [
        "round",
        "program_idx",
        "program_code",
        "loo_score",
        "waic_score",
        "n_divergences",
        "max_rhat",
        "min_ess_bulk",
        "summary_stats_preview",
        "llm_response_preview",
        "diagnostics",
    ]
    ppl_data = []
    if detailed_entries:
        source_entries = detailed_entries
        for prog in source_entries:
            diag = prog.get("diagnostics")
            if isinstance(diag, (list, tuple)):
                diag_text = ", ".join(str(d) for d in diag)
            else:
                diag_text = diag
            ppl_data.append(
                [
                    _safe_number(prog.get("round")),
                    _safe_number(prog.get("program_idx")),
                    _preview_text(prog.get("program_code"), 5000),
                    _safe_number(prog.get("loo")),
                    _safe_number(prog.get("waic")),
                    _safe_number(prog.get("n_divergences")),
                    _safe_number(prog.get("max_rhat")),
                    _safe_number(prog.get("min_ess_bulk")),
                    _preview_text(prog.get("summary_stats"), 1000),
                    _preview_text(prog.get("llm_response"), 500),
                    _preview_text(diag_text, 500),
                ]
            )
    else:
        for prog in all_programs:
            if not isinstance(prog, dict):
                continue
            diag = prog.get("diagnostics")
            if isinstance(diag, (list, tuple)):
                diag_text = ", ".join(str(d) for d in diag)
            else:
                diag_text = diag
            ppl_data.append(
                [
                    _safe_number(prog.get("round")),
                    _safe_number(prog.get("program_idx")),
                    _preview_text(prog.get("str_prob_prog"), 5000),
                    _safe_number(prog.get("loo")),
                    _safe_number(prog.get("waic")),
                    _safe_number(prog.get("n_divergences")),
                    _safe_number(prog.get("max_rhat")),
                    _safe_number(prog.get("min_ess_bulk")),
                    _preview_text(prog.get("summary_stats"), 1000),
                    _preview_text(prog.get("full_llm_response"), 500),
                    _preview_text(diag_text, 500),
                ]
            )
    if ppl_data:
        with _wandb_unseeded_random():
            ppl_table = wandb_module.Table(columns=ppl_columns, data=ppl_data)
            wandb_run.log({"ppl/programs": ppl_table})
    return len(ppl_data)


def _log_ppl_summary(wandb_run, all_programs, programs, ppl_data_len):
    loo_scores = [
        p.get("loo") for p in all_programs if isinstance(p, dict) and p.get("loo") not in ("", None)
    ]
    waic_scores = [
        p.get("waic")
        for p in all_programs
        if isinstance(p, dict) and p.get("waic") not in ("", None)
    ]

    ppl_summary = {
        "ppl/num_programs": len(all_programs),
        "ppl/num_rounds": len(programs),
        "ppl/num_programs_logged": ppl_data_len,
    }

    if loo_scores:
        try:
            numeric_loo = [float(s) for s in loo_scores if s is not None and s != ""]
            if numeric_loo:
                ppl_summary["ppl/best_loo"] = min(numeric_loo)
                ppl_summary["ppl/mean_loo"] = sum(numeric_loo) / len(numeric_loo)
        except (ValueError, TypeError):
            pass

    if waic_scores:
        try:
            numeric_waic = [float(s) for s in waic_scores if s is not None and s != ""]
            if numeric_waic:
                ppl_summary["ppl/best_waic"] = min(numeric_waic)
                ppl_summary["ppl/mean_waic"] = sum(numeric_waic) / len(numeric_waic)
        except (ValueError, TypeError):
            pass

    try:
        diag_count = 0
        divs = []
        rhat_vals = []
        ess_vals = []
        for prog in all_programs:
            if not isinstance(prog, dict):
                continue
            if prog.get("diagnostics"):
                diag_count += 1
            div = _safe_number(prog.get("n_divergences"))
            if div is not None:
                divs.append(div)
            rhat = _safe_number(prog.get("max_rhat"))
            if rhat is not None:
                rhat_vals.append(rhat)
            ess = _safe_number(prog.get("min_ess_bulk"))
            if ess is not None:
                ess_vals.append(ess)
        if divs:
            ppl_summary["ppl/total_divergences"] = int(sum(divs))
        if rhat_vals:
            ppl_summary["ppl/max_rhat_any"] = max(rhat_vals)
        if ess_vals:
            ppl_summary["ppl/min_ess_any"] = min(ess_vals)
        ppl_summary["ppl/diagnostic_warnings_count"] = diag_count
    except Exception:
        pass

    wandb_run.summary.update(ppl_summary)


def _log_ppl_round_summary(wandb_module, wandb_run, detailed_entries):
    if not detailed_entries:
        return
    round_summary = {}
    for entry in detailed_entries:
        round_idx = entry.get("round")
        if round_idx is None:
            continue
        stats = round_summary.setdefault(
            round_idx,
            {
                "num_programs": 0,
                "num_with_loo": 0,
                "best_loo": None,
                "best_waic": None,
                "num_diag_warnings": 0,
                "total_divergences": 0,
                "max_rhat": None,
                "min_ess_bulk": None,
            },
        )
        stats["num_programs"] += 1
        loo = _safe_number(entry.get("loo"))
        if loo is not None:
            stats["num_with_loo"] += 1
            if stats["best_loo"] is None or loo < stats["best_loo"]:
                stats["best_loo"] = loo
        waic = _safe_number(entry.get("waic"))
        if waic is not None:
            if stats["best_waic"] is None or waic < stats["best_waic"]:
                stats["best_waic"] = waic
        if entry.get("diagnostics"):
            stats["num_diag_warnings"] += 1
        div = _safe_number(entry.get("n_divergences"))
        if div is not None:
            stats["total_divergences"] += int(div)
        rhat = _safe_number(entry.get("max_rhat"))
        if rhat is not None:
            if stats["max_rhat"] is None or rhat > stats["max_rhat"]:
                stats["max_rhat"] = rhat
        ess = _safe_number(entry.get("min_ess_bulk"))
        if ess is not None:
            if stats["min_ess_bulk"] is None or ess < stats["min_ess_bulk"]:
                stats["min_ess_bulk"] = ess

    columns = [
        "round_idx",
        "num_programs",
        "num_with_loo",
        "best_loo",
        "best_waic",
        "num_diag_warnings",
        "total_divergences",
        "max_rhat",
        "min_ess_bulk",
    ]
    rows = []
    for round_idx in sorted(round_summary):
        stats = round_summary[round_idx]
        rows.append(
            [
                _safe_number(round_idx),
                _safe_number(stats["num_programs"]),
                _safe_number(stats["num_with_loo"]),
                _safe_number(stats["best_loo"]),
                _safe_number(stats["best_waic"]),
                _safe_number(stats["num_diag_warnings"]),
                _safe_number(stats["total_divergences"]),
                _safe_number(stats["max_rhat"]),
                _safe_number(stats["min_ess_bulk"]),
            ]
        )
    if rows:
        with _wandb_unseeded_random():
            round_summary_table = wandb_module.Table(columns=columns, data=rows)
            wandb_run.log({"ppl/round_summary": round_summary_table})


def _log_ppl_loo_progression(wandb_module, wandb_run, detailed_entries):
    if not detailed_entries:
        return
    points = []
    for entry in detailed_entries:
        round_idx = _safe_number(entry.get("round"))
        loo = _safe_number(entry.get("loo"))
        if round_idx is None or loo is None:
            continue
        points.append((int(round_idx), float(loo)))
    if not points:
        return
    import matplotlib.pyplot as plt

    rounds = [p[0] for p in points]
    loos = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(rounds, loos, alpha=0.5, label="All programs")
    best_by_round = {}
    for r, l in points:
        if r not in best_by_round or l < best_by_round[r]:
            best_by_round[r] = l
    ordered_rounds = sorted(best_by_round)
    ax.plot(
        ordered_rounds,
        [best_by_round[r] for r in ordered_rounds],
        "r-o",
        label="Best per round",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("LOO (lower is better)")
    ax.legend()
    wandb_run.log({"ppl/loo_progression": wandb_module.Image(fig)})
    plt.close(fig)


def _select_program_code(prog, detailed_entries):
    if not isinstance(prog, dict):
        return ""
    code = prog.get("str_prob_prog") or ""
    if code:
        return code
    round_idx = prog.get("round")
    prog_idx = prog.get("program_idx", 0)
    if round_idx is not None and detailed_entries:
        for entry in detailed_entries:
            if entry.get("round") == round_idx and entry.get("program_idx", 0) == prog_idx:
                code = entry.get("program_code") or ""
                if code:
                    return code
    llm_text = prog.get("full_llm_response") or ""
    return llm_text


def _log_ppl_best_program_artifact(
    ctx, wandb_run, wandb_module, all_programs, detailed_entries, wandb_meta
):
    valid_programs = [
        (p, float(p.get("loo")))
        for p in all_programs
        if isinstance(p, dict) and p.get("loo") not in ("", None)
    ]
    if not valid_programs:
        return
    best_prog, _ = min(valid_programs, key=lambda x: x[1])
    best_code = _select_program_code(best_prog, detailed_entries)
    if not best_code:
        return
    wandb_run.config.update({"ppl/best_program_code": best_code}, allow_val_change=True)
    try:
        wandb_run.summary.update(
            {
                "ppl/best_n_divergences": _safe_number(best_prog.get("n_divergences")),
                "ppl/best_max_rhat": _safe_number(best_prog.get("max_rhat")),
                "ppl/best_min_ess_bulk": _safe_number(best_prog.get("min_ess_bulk")),
                "ppl/has_ppc": bool(best_prog.get("posterior_predictive")),
            }
        )
    except Exception:
        pass
    try:
        if hasattr(wandb_module, "Artifact") and hasattr(wandb_run, "log_artifact"):
            run_id = getattr(wandb_run, "id", None) or (wandb_meta or {}).get("run_id")
            base_dir = getattr(wandb_run, "dir", None) or os.getcwd()
            filename = f"ppl_best_program_{run_id}.py" if run_id else "ppl_best_program.py"
            tmp_path = os.path.join(base_dir, filename)
            with open(tmp_path, "w") as tmp:
                tmp.write(best_code)
            artifact_name = f"ppl_best_program_{run_id}" if run_id else "ppl_best_program"
            with _wandb_unseeded_random():
                artifact = wandb_module.Artifact(artifact_name, type="ppl-program")
                artifact.add_file(tmp_path, name="best_program.py")
                artifact_result = wandb_run.log_artifact(artifact)
                try:
                    artifact_result.wait()
                except Exception:
                    pass
            ctx.artifacts_logged.append(
                {
                    "name": artifact.name,
                    "type": artifact.type,
                }
            )
    except Exception as e:
        ctx.artifacts_failed.append(
            {
                "name": "ppl_best_program",
                "type": "ppl-program",
                "error": str(e),
            }
        )
        logger.warning(f"PPL best program artifact logging failed: {e}")


def _log_ppl_llm_responses_artifact(
    ctx, wandb_run, wandb_module, ppl_artifacts, detailed_entries, wandb_meta
):
    if not hasattr(wandb_module, "Artifact") or not hasattr(wandb_run, "log_artifact"):
        return
    if not detailed_entries and not ppl_artifacts.programs_all_text:
        return
    run_id = getattr(wandb_run, "id", None) or (wandb_meta or {}).get("run_id")
    base_dir = getattr(wandb_run, "dir", None) or os.getcwd()
    filename = f"ppl_llm_responses_{run_id}.txt" if run_id else "ppl_llm_responses.txt"
    tmp_path = os.path.join(base_dir, filename)
    with open(tmp_path, "w") as tmp:
        if detailed_entries:
            for entry in detailed_entries:
                tmp.write(
                    f"=== round {entry.get('round')} program {entry.get('program_idx')} ===\n"
                )
                tmp.write(f"loo: {entry.get('loo')}\n")
                tmp.write(f"waic: {entry.get('waic')}\n")
                tmp.write(f"n_divergences: {entry.get('n_divergences')}\n")
                tmp.write(f"max_rhat: {entry.get('max_rhat')}\n")
                tmp.write(f"min_ess_bulk: {entry.get('min_ess_bulk')}\n")
                diag = entry.get("diagnostics")
                if isinstance(diag, (list, tuple)):
                    diag = ", ".join(str(d) for d in diag)
                tmp.write(f"diagnostics: {diag}\n\n")
                code = entry.get("program_code") or ""
                if code:
                    tmp.write("---- program_code ----\n")
                    tmp.write(code)
                    tmp.write("\n\n")
                llm_text = entry.get("llm_response") or ""
                if llm_text:
                    tmp.write("---- llm_response ----\n")
                    tmp.write(llm_text)
                    tmp.write("\n\n")
        else:
            for idx, text in enumerate(ppl_artifacts.programs_all_text):
                tmp.write(f"=== program {idx} ===\n")
                tmp.write(text or "")
                tmp.write("\n\n")
    artifact_name = f"ppl_llm_responses_{run_id}" if run_id else "ppl_llm_responses"
    with _wandb_unseeded_random():
        artifact = wandb_module.Artifact(artifact_name, type="ppl-text")
        artifact.add_file(tmp_path, name="ppl_llm_responses.txt")
        artifact_result = wandb_run.log_artifact(artifact)
        try:
            artifact_result.wait()
        except Exception:
            pass
    ctx.artifacts_logged.append(
        {
            "name": artifact.name,
            "type": artifact.type,
        }
    )


def _log_ppl_llm_response_table(wandb_module, wandb_run, ppl_artifacts, detailed_entries):
    llm_rows = []
    if detailed_entries:
        for entry in detailed_entries:
            llm_rows.append(
                [
                    _safe_number(entry.get("round")),
                    _safe_number(entry.get("program_idx")),
                    _preview_text(entry.get("program_code"), 20000),
                    _preview_text(entry.get("llm_response"), 20000),
                ]
            )
    elif ppl_artifacts.programs_all_text:
        for idx, text in enumerate(ppl_artifacts.programs_all_text):
            llm_rows.append(
                [
                    None,
                    _safe_number(idx),
                    "",
                    _preview_text(text, 20000),
                ]
            )
    if llm_rows:
        with _wandb_unseeded_random():
            llm_table = wandb_module.Table(
                columns=["round", "program_idx", "program_code", "llm_response"],
                data=llm_rows,
            )
            wandb_run.log({"ppl/llm_responses": llm_table})


def _log_ppl_trace_plots(wandb_module, wandb_run, ppl_artifacts, goal):
    if not ppl_artifacts.best_plot_candidates:
        return
    ppl_artifacts.best_plot_candidates.sort(key=lambda x: (x[0] is None, x[0]))
    _, trace_obj, round_idx, prog_idx, _, best_prog = ppl_artifacts.best_plot_candidates[0]
    import arviz as az
    import matplotlib.pyplot as plt

    az.plot_trace(trace_obj)
    fig = plt.gcf()
    wandb_run.log({f"ppl/trace_plot_round_{round_idx}_prog_{prog_idx}": wandb_module.Image(fig)})
    plt.close(fig)

    try:
        ppc = best_prog.get("posterior_predictive") if isinstance(best_prog, dict) else None
        if isinstance(ppc, dict) and ppc:
            key = "y_obs" if "y_obs" in ppc else next(iter(ppc.keys()))
            vals = ppc.get(key)
            if isinstance(vals, np.ndarray) and vals.ndim == 2:
                pred_mean = np.mean(vals, axis=0)
                obs = None
                try:
                    df = getattr(goal.env, "df", None)
                    if df is not None and hasattr(df, "columns"):
                        if hasattr(goal.env, "get_ordered_column_names") and hasattr(
                            goal.env, "get_ordered_features"
                        ):
                            cols = list(goal.env.get_ordered_column_names())
                            feats = list(goal.env.get_ordered_features())
                            outs = [c for c in cols if c not in feats]
                            if len(outs) == 1 and outs[0] in df.columns:
                                obs = df[outs[0]].to_numpy()
                        if obs is None:
                            if "Choice" in df.columns:
                                obs = df["Choice"].to_numpy()
                            elif key in df.columns:
                                obs = df[key].to_numpy()
                            else:
                                obs = df.iloc[:, -1].to_numpy()
                except Exception:
                    obs = None

                if obs is not None and len(obs) == len(pred_mean):
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].scatter(obs, pred_mean, alpha=0.6)
                    min_v = float(np.min(obs))
                    max_v = float(np.max(obs))
                    axes[0].plot([min_v, max_v], [min_v, max_v], "r--")
                    axes[0].set_xlabel("Observed")
                    axes[0].set_ylabel("Predicted (mean)")
                    axes[0].set_title("Posterior Predictive Check")
                    residuals = pred_mean - obs
                    axes[1].hist(residuals, bins=20, edgecolor="black")
                    axes[1].axvline(0, color="r", linestyle="--")
                    axes[1].set_title("Residuals")
                    wandb_run.log(
                        {
                            f"ppl/posterior_predictive_round_{round_idx}_prog_{prog_idx}": wandb_module.Image(
                                fig
                            )
                        }
                    )
                    plt.close(fig)
    except Exception as e:
        logger.warning(f"PPL posterior predictive logging failed: {e}")


def _log_ppl_round_stats_table(wandb_module, wandb_run, ppl_artifacts):
    if not ppl_artifacts.round_stats_entries:
        return
    all_keys = set()
    for row in ppl_artifacts.round_stats_entries:
        if isinstance(row, dict):
            all_keys.update(row.keys())
    if not all_keys:
        return
    columns = ["round_idx"] + sorted(k for k in all_keys if k != "round_idx")
    col_types = {}
    for col in columns:
        has_num = False
        has_str = False
        for row in ppl_artifacts.round_stats_entries:
            if not isinstance(row, dict):
                continue
            val = row.get(col)
            if val is None:
                continue
            if isinstance(val, np.generic):
                val = val.item()
            if isinstance(val, (int, float)) and not (
                isinstance(val, float) and not np.isfinite(val)
            ):
                has_num = True
            else:
                has_str = True
        if has_str:
            col_types[col] = "string"
        elif has_num:
            col_types[col] = "number"
        else:
            col_types[col] = "number"
    rows = []
    for row in ppl_artifacts.round_stats_entries:
        if isinstance(row, dict):
            row_vals = []
            for c in columns:
                val = row.get(c)
                if col_types.get(c) == "string":
                    row_vals.append(_preview_text(val, 1000))
                else:
                    row_vals.append(_safe_number(val))
            rows.append(row_vals)
    if rows:
        with _wandb_unseeded_random():
            round_table = wandb_module.Table(columns=columns, data=rows)
            wandb_run.log({"ppl/round_stats": round_table})


def _log_ppl_results(
    ctx, wandb_run, wandb_module, programs, use_ppl, ppl_artifacts, goal, wandb_meta
):
    if not programs or not use_ppl:
        return
    if ppl_artifacts is None:

        class _PPLStub:
            program_entries = []
            programs_all_text = []
            best_plot_candidates = []
            round_stats_entries = []

        ppl_artifacts = _PPLStub()
    all_programs = _flatten_programs(programs)
    detailed_entries = ppl_artifacts.program_entries if ppl_artifacts else []
    try:
        ppl_data_len = _log_ppl_program_table(
            wandb_module, wandb_run, all_programs, detailed_entries
        )
    except Exception as e:
        logger.warning(f"PPL program table logging failed: {e}")
        ppl_data_len = 0

    try:
        _log_ppl_summary(wandb_run, all_programs, programs, ppl_data_len)
    except Exception as e:
        logger.warning(f"PPL summary logging failed: {e}")

    try:
        _log_ppl_round_summary(wandb_module, wandb_run, detailed_entries)
    except Exception as e:
        logger.warning(f"PPL round summary logging failed: {e}")

    try:
        _log_ppl_loo_progression(wandb_module, wandb_run, detailed_entries)
    except Exception as e:
        logger.warning(f"PPL LOO progression plot failed: {e}")

    try:
        _log_ppl_best_program_artifact(
            ctx, wandb_run, wandb_module, all_programs, detailed_entries, wandb_meta
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"PPL best program selection failed: {e}")

    try:
        _log_ppl_llm_responses_artifact(
            ctx, wandb_run, wandb_module, ppl_artifacts, detailed_entries, wandb_meta
        )
    except Exception as e:
        ctx.artifacts_failed.append(
            {
                "name": "ppl_llm_responses",
                "type": "ppl-text",
                "error": str(e),
            }
        )
        logger.warning(f"PPL LLM response artifact logging failed: {e}")

    try:
        _log_ppl_llm_response_table(wandb_module, wandb_run, ppl_artifacts, detailed_entries)
    except Exception as e:
        logger.warning(f"PPL LLM response table logging failed: {e}")

    try:
        _log_ppl_trace_plots(wandb_module, wandb_run, ppl_artifacts, goal)
    except Exception as e:
        logger.warning(f"PPL trace plot logging failed: {e}")

    try:
        _log_ppl_round_stats_table(wandb_module, wandb_run, ppl_artifacts)
    except Exception as e:
        logger.warning(f"PPL round stats logging failed: {e}")


def _log_llm_calls_table(wandb_module, wandb_run, scientist_agent, naive_agent):
    """Log LLM call history as a WandB table (independent of Weave)."""
    columns = [
        "agent",
        "call_idx",
        "model",
        "prompt_preview",
        "response_preview",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cost_usd",
        "has_reasoning",
    ]
    rows = []

    for agent, agent_name in [(scientist_agent, "scientist"), (naive_agent, "naive")]:
        if agent is None or not hasattr(agent, "get_call_history"):
            continue
        call_history = agent.get_call_history()
        for call in call_history:
            rows.append(
                [
                    agent_name,
                    _safe_number(call.get("call_idx")),
                    call.get("model", ""),
                    _preview_text(call.get("prompt"), 2000),
                    _preview_text(call.get("response"), 2000),
                    _safe_number(call.get("latency_ms")),
                    _safe_number(call.get("prompt_tokens")),
                    _safe_number(call.get("completion_tokens")),
                    _safe_number(call.get("reasoning_tokens")),
                    _safe_number(call.get("cost_usd")),
                    1 if call.get("has_reasoning") else 0,
                ]
            )

    if rows:
        with _wandb_unseeded_random():
            llm_calls_table = wandb_module.Table(columns=columns, data=rows)
            wandb_run.log({"llm/calls": llm_calls_table})
        wandb_run.summary.update(
            {
                "llm/total_calls_logged": len(rows),
            }
        )
    return len(rows)


def _log_conversation_history_artifact(
    ctx, wandb_run, wandb_module, scientist_agent, naive_agent, wandb_meta
):
    """Log full conversation history as a WandB artifact (independent of Weave)."""
    if not hasattr(wandb_module, "Artifact") or not hasattr(wandb_run, "log_artifact"):
        return

    run_id = getattr(wandb_run, "id", None) or (wandb_meta or {}).get("run_id")
    base_dir = getattr(wandb_run, "dir", None) or os.getcwd()

    for agent, agent_name in [(scientist_agent, "scientist"), (naive_agent, "naive")]:
        if agent is None:
            continue

        messages = getattr(agent, "all_messages", [])
        if not messages:
            continue

        filename = (
            f"conversation_{agent_name}_{run_id}.txt"
            if run_id
            else f"conversation_{agent_name}.txt"
        )
        tmp_path = os.path.join(base_dir, filename)

        try:
            with open(tmp_path, "w") as f:
                for i, msg in enumerate(messages):
                    f.write(f"=== Message {i} ===\n")
                    f.write(str(msg))
                    f.write("\n\n")

            artifact_name = (
                f"conversation_{agent_name}_{run_id}" if run_id else f"conversation_{agent_name}"
            )
            with _wandb_unseeded_random():
                artifact = wandb_module.Artifact(artifact_name, type="conversation")
                artifact.add_file(tmp_path, name=f"conversation_{agent_name}.txt")
                artifact_result = wandb_run.log_artifact(artifact)
                try:
                    artifact_result.wait()
                except Exception:
                    pass
            ctx.artifacts_logged.append(
                {
                    "name": artifact.name,
                    "type": artifact.type,
                }
            )
        except Exception as e:
            ctx.artifacts_failed.append(
                {
                    "name": f"conversation_{agent_name}",
                    "type": "conversation",
                    "error": str(e),
                }
            )
            logger.warning(f"Conversation artifact logging failed for {agent_name}: {e}")


def _log_observations_table(wandb_module, wandb_run, all_data):
    """Log observations and queries as a WandB table."""
    queries = all_data[1] if len(all_data) > 1 else []
    observations = all_data[2] if len(all_data) > 2 else []
    successes = all_data[3] if len(all_data) > 3 else []

    if not queries and not observations:
        return 0

    columns = ["step_idx", "query", "observation", "success"]
    rows = []

    max_len = max(len(queries or []), len(observations or []))
    for i in range(max_len):
        query = queries[i] if queries and i < len(queries) else None
        obs = observations[i] if observations and i < len(observations) else None
        success = successes[i] if successes and i < len(successes) else None

        rows.append(
            [
                i,
                _preview_text(query, 5000),
                _preview_text(obs, 5000),
                1 if success else 0,
            ]
        )

    if rows:
        with _wandb_unseeded_random():
            obs_table = wandb_module.Table(columns=columns, data=rows)
            wandb_run.log({"experiment/observations": obs_table})
        wandb_run.summary.update(
            {
                "experiment/total_observations_logged": len(rows),
            }
        )
    return len(rows)


def _log_results_artifact(ctx, wandb_run, wandb_module, output_filename, wandb_meta):
    """Upload final results JSON as a WandB artifact."""
    if not hasattr(wandb_module, "Artifact") or not hasattr(wandb_run, "log_artifact"):
        return
    if not output_filename or not os.path.exists(output_filename):
        return

    run_id = getattr(wandb_run, "id", None) or (wandb_meta or {}).get("run_id")

    try:
        artifact_name = f"results_{run_id}" if run_id else "results"
        with _wandb_unseeded_random():
            artifact = wandb_module.Artifact(artifact_name, type="results")
            artifact.add_file(output_filename, name="results.json")
            artifact_result = wandb_run.log_artifact(artifact)
            try:
                artifact_result.wait()
            except Exception:
                pass
        ctx.artifacts_logged.append(
            {
                "name": artifact.name,
                "type": artifact.type,
            }
        )
    except Exception as e:
        ctx.artifacts_failed.append(
            {
                "name": "results",
                "type": "results",
                "error": str(e),
            }
        )
        logger.warning(f"Results artifact logging failed: {e}")


def _log_call_recorder_artifact(ctx, wandb_run, wandb_module, call_recorder_path, wandb_meta):
    """Upload the crash-resilient JSONL call log as a WandB artifact.

    This is the primary source of truth for LLM calls - written per-call to disk,
    survives crashes, includes both sync and async calls.
    """
    if not hasattr(wandb_module, "Artifact") or not hasattr(wandb_run, "log_artifact"):
        return
    if not call_recorder_path or not os.path.exists(call_recorder_path):
        return

    run_id = getattr(wandb_run, "id", None) or (wandb_meta or {}).get("run_id")

    try:
        artifact_name = f"llm_calls_jsonl_{run_id}" if run_id else "llm_calls_jsonl"
        with _wandb_unseeded_random():
            artifact = wandb_module.Artifact(artifact_name, type="llm_calls")
            artifact.add_file(str(call_recorder_path), name="llm_calls.jsonl")
            artifact_result = wandb_run.log_artifact(artifact)
            try:
                artifact_result.wait()
            except Exception:
                pass
        ctx.artifacts_logged.append(
            {
                "name": artifact.name,
                "type": artifact.type,
                "format": "jsonl",
                "source": "call_recorder",
            }
        )
        logger.info(f"Uploaded call recorder JSONL: {call_recorder_path}")
    except Exception as e:
        ctx.artifacts_failed.append(
            {
                "name": "llm_calls_jsonl",
                "type": "llm_calls",
                "error": str(e),
            }
        )
        logger.warning(f"Call recorder artifact logging failed: {e}")


def _log_llm_calls_artifact(ctx, wandb_run, wandb_module, scientist_agent, naive_agent, wandb_meta):
    """Log detailed LLM calls as a JSON artifact (full prompts/responses)."""
    if not hasattr(wandb_module, "Artifact") or not hasattr(wandb_run, "log_artifact"):
        return

    import json

    run_id = getattr(wandb_run, "id", None) or (wandb_meta or {}).get("run_id")
    base_dir = getattr(wandb_run, "dir", None) or os.getcwd()

    all_calls = []
    for agent, agent_name in [(scientist_agent, "scientist"), (naive_agent, "naive")]:
        if agent is None or not hasattr(agent, "get_call_history"):
            continue
        for call in agent.get_call_history():
            call_with_agent = dict(call)
            call_with_agent["agent"] = agent_name
            all_calls.append(call_with_agent)

    if not all_calls:
        return

    filename = f"llm_calls_{run_id}.json" if run_id else "llm_calls.json"
    tmp_path = os.path.join(base_dir, filename)

    try:
        with open(tmp_path, "w") as f:
            json.dump(all_calls, f, indent=2, default=str)

        artifact_name = f"llm_calls_{run_id}" if run_id else "llm_calls"
        with _wandb_unseeded_random():
            artifact = wandb_module.Artifact(artifact_name, type="llm-calls")
            artifact.add_file(tmp_path, name="llm_calls.json")
            artifact_result = wandb_run.log_artifact(artifact)
            try:
                artifact_result.wait()
            except Exception:
                pass
        ctx.artifacts_logged.append(
            {
                "name": artifact.name,
                "type": artifact.type,
            }
        )
    except Exception as e:
        ctx.artifacts_failed.append(
            {
                "name": "llm_calls",
                "type": "llm-calls",
                "error": str(e),
            }
        )
        logger.warning(f"LLM calls artifact logging failed: {e}")


def log_wandb_results(
    ctx: WandbContext,
    wandb_module,
    output_filename: str,
    all_data,
    z_results: list[dict[str, Any]],
    num_experiments,
    env_name: str,
    goal_name: str,
    experiment_type: str,
    include_prior: bool,
    seed: int,
    use_ppl: bool,
    system_message: str,
    scientist_agent,
    naive_agent,
    ppl_artifacts,
    goal,
    call_recorder_path: str | None = None,
):
    if ctx.run is None or wandb_module is None:
        return

    wandb_run = ctx.run
    wandb_meta = ctx.meta

    try:
        wandb_run.config.update({"results/filename": output_filename}, allow_val_change=True)
    except Exception:
        pass

    z_by_budget = _log_eval_series(wandb_module, all_data, z_results, num_experiments)

    try:
        summary_data = _build_summary_data(
            all_data=all_data,
            z_by_budget=z_by_budget,
            num_experiments=num_experiments,
            env_name=env_name,
            goal_name=goal_name,
            experiment_type=experiment_type,
            include_prior=include_prior,
            seed=seed,
            use_ppl=use_ppl,
            scientist_agent=scientist_agent,
            naive_agent=naive_agent,
            start_time=ctx.start_time,
        )

        # add custom endpoint info from config for easier filtering
        try:
            llm_cfg = wandb_run.config.get("llms", {})
            if isinstance(llm_cfg, dict):
                api_base = llm_cfg.get("api_base")
                custom_provider = llm_cfg.get("custom_llm_provider")
                if api_base:
                    summary_data["llm/api_base"] = api_base
                if custom_provider:
                    summary_data["llm/custom_provider"] = custom_provider
                summary_data["llm/is_custom_endpoint"] = bool(custom_provider or api_base)
        except Exception:
            pass

        wandb_run.summary.update(summary_data)
    except Exception as e:
        logger.debug(f"W&B summary update failed: {e}")

    _log_system_prompt(wandb_run, system_message)

    programs = all_data[6] if len(all_data) > 6 else []
    _log_ppl_results(
        ctx=ctx,
        wandb_run=wandb_run,
        wandb_module=wandb_module,
        programs=programs,
        use_ppl=use_ppl,
        ppl_artifacts=ppl_artifacts,
        goal=goal,
        wandb_meta=wandb_meta,
    )

    # comprehensive WandB logging (independent of Weave)
    try:
        _log_llm_calls_table(wandb_module, wandb_run, scientist_agent, naive_agent)
    except Exception as e:
        logger.warning(f"LLM calls table logging failed: {e}")

    try:
        _log_observations_table(wandb_module, wandb_run, all_data)
    except Exception as e:
        logger.warning(f"Observations table logging failed: {e}")

    try:
        _log_conversation_history_artifact(
            ctx, wandb_run, wandb_module, scientist_agent, naive_agent, wandb_meta
        )
    except Exception as e:
        logger.warning(f"Conversation history artifact logging failed: {e}")

    try:
        _log_llm_calls_artifact(
            ctx, wandb_run, wandb_module, scientist_agent, naive_agent, wandb_meta
        )
    except Exception as e:
        logger.warning(f"LLM calls artifact logging failed: {e}")

    # upload crash-resilient JSONL call log (primary source of truth)
    if call_recorder_path:
        try:
            _log_call_recorder_artifact(
                ctx, wandb_run, wandb_module, call_recorder_path, wandb_meta
            )
        except Exception as e:
            logger.warning(f"Call recorder JSONL artifact logging failed: {e}")

    # upload results JSON as artifact (file must exist - written by caller before this)
    try:
        _log_results_artifact(ctx, wandb_run, wandb_module, output_filename, wandb_meta)
    except Exception as e:
        logger.warning(f"Results artifact logging failed: {e}")

    try:
        wandb_meta["results_file"] = output_filename
        wandb_meta["artifacts_logged"] = list(ctx.artifacts_logged)
        if ctx.artifacts_failed:
            wandb_meta["artifacts_failed"] = list(ctx.artifacts_failed)
    except Exception:
        pass

    try:
        wandb_module.finish()
    except Exception as e:
        logger.debug(f"W&B finish failed: {e}")
