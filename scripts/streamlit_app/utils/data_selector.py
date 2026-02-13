"""Shared sidebar controls and loaders for sweep data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

try:
    import requests
except Exception:  # pragma: no cover - allow running without requests installed
    requests = None

from utils.data_loader import (
    get_entity_from_env,
    get_project_from_env,
    get_sweep_ids_from_env,
    load_local_results,
    load_wandb_results,
)


@dataclass
class DataSelection:
    data_source: str
    entity: str
    project: str
    sweep_ids: list[str]
    results_dir: str
    available_sweeps: list[tuple[str, str, str]]
    api_error: str | None
    refresh: bool


@st.cache_data(ttl=60, show_spinner=False)
def fetch_available_sweeps(entity: str, project: str):
    """Fetch all sweeps from the project."""
    if requests is None:
        return [], "requests not available in this environment."
    try:
        import wandb

        api = wandb.Api(timeout=60)
        full_path = f"{entity}/{project}"
        sweeps = api.project(full_path).sweeps()
        result = []
        for s in sweeps:
            result.append((s.id, s.name or s.id, s.state))
        return result, None
    except requests.exceptions.HTTPError as e:
        if "502" in str(e):
            return [], "WandB API temporarily unavailable (HTTP 502). Try again in a moment."
        return [], f"WandB API error: {str(e)}"
    except requests.exceptions.Timeout:
        return [], "WandB API timeout after 60s. The service may be slow or unavailable."
    except Exception as e:
        return [], str(e)


def render_data_source_selector(
    container,
    key_prefix: str = "sweep",
    default_source: str = "Local JSON Files",
    show_results_dir: bool = True,
    title: str = "🎯 Sweep Selection",
) -> DataSelection:
    """Render data source selector controls.

    Returns a DataSelection with user inputs and discovery state.
    """
    container.header(title)

    data_source = container.radio(
        "Data Source",
        options=["Local JSON Files", "WandB API"],
        index=0 if default_source == "Local JSON Files" else 1,
        help="Local files are much faster than WandB API",
        key=f"{key_prefix}_data_source",
    )

    entity = container.text_input(
        "Entity",
        value=get_entity_from_env(),
        key=f"{key_prefix}_entity",
    )
    project = container.text_input(
        "Project",
        value=get_project_from_env(),
        key=f"{key_prefix}_project",
    )

    available_sweeps: list[tuple[str, str, str]] = []
    api_error = None
    if data_source == "WandB API" and entity and project:
        with container.spinner("Discovering sweeps..."):
            available_sweeps, api_error = fetch_available_sweeps(entity, project)
        if api_error:
            container.warning(
                f"⚠️ WandB API error: {api_error[:200]}{'...' if len(api_error) > 200 else ''}"
            )
            container.info("Using manual sweep ID entry below.")

    sweep_ids: list[str] = []
    if data_source == "WandB API":
        if available_sweeps:
            container.markdown(f"**{len(available_sweeps)} sweeps found**")
            sweep_options = {
                f"{sid} ({name[:20]}) - {state}": sid for sid, name, state in available_sweeps
            }
            cli_sweep_ids = get_sweep_ids_from_env()
            default_indices = []
            if cli_sweep_ids:
                default_indices = [
                    opt for opt in sweep_options if any(sid in opt for sid in cli_sweep_ids)
                ]

            selected = container.multiselect(
                "Select Sweep(s)",
                options=list(sweep_options.keys()),
                default=default_indices if default_indices else list(sweep_options.keys())[:1],
                help="Select one or more sweeps to analyze (start with 1 for faster loading)",
                key=f"{key_prefix}_sweep_select",
            )
            sweep_ids = [sweep_options[s] for s in selected]
        else:
            cli_sweep_ids = get_sweep_ids_from_env()
            default_sweep = ",".join(cli_sweep_ids) if cli_sweep_ids else ""
            if not default_sweep:
                container.caption("💡 Enter your WandB sweep ID(s) above")
            sweep_input = container.text_input(
                "Sweep ID(s)",
                value=default_sweep,
                placeholder="e.g., abc123,def456",
                key=f"{key_prefix}_sweep_input",
            )
            sweep_ids = [s.strip() for s in sweep_input.split(",") if s.strip()]

    results_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "results")
    if show_results_dir:
        results_dir = container.text_input(
            "Results Directory",
            value=results_dir,
            key=f"{key_prefix}_results_dir",
        )

    container.divider()
    refresh = container.button(
        "🔄 Refresh Data", use_container_width=True, key=f"{key_prefix}_refresh"
    )
    if refresh:
        st.cache_data.clear()
        st.rerun()

    return DataSelection(
        data_source=data_source,
        entity=entity,
        project=project,
        sweep_ids=sweep_ids,
        results_dir=results_dir,
        available_sweeps=available_sweeps,
        api_error=api_error,
        refresh=refresh,
    )


def load_runs_from_selection(selection: DataSelection) -> list[dict]:
    """Load runs based on a DataSelection and return list of run dicts."""
    all_runs: list[dict] = []

    if selection.data_source == "Local JSON Files":
        results = load_local_results(selection.results_dir)
        for r in results:
            n_seeds = 1
            try:
                if r.path:
                    path_obj = Path(r.path)
                    candidates = [path_obj]
                    if not path_obj.is_absolute():
                        candidates.append(Path(selection.results_dir) / path_obj)
                    resolved = next((p for p in candidates if p.exists()), None)
                    if resolved:
                        with open(resolved) as f:
                            payload = json.load(f)
                        data_section = payload.get("data", {}) if isinstance(payload, dict) else {}
                        if data_section.get("aggregation_mode") == "multi_seed":
                            raw_n = data_section.get("n_seeds", 1)
                            n_seeds = int(raw_n) if raw_n else 1
            except Exception:
                n_seeds = 1
            run_id = r.path or f"local:{r.env}:{r.model}:{r.seed}:{r.budget}"
            run_dict = {
                "run_id": run_id,
                "config/llms": r.model,
                "config/envs": r.env,
                "config/seed": r.seed,
                "config/include_prior": r.include_prior,
                "config/use_ppl": r.use_ppl,
                "config/budget": r.budget,
                "config/goal": r.goal,
                "config/exp": r.experiment_type,
                "summary/n_seeds": n_seeds,
                "metric/eval/z_mean": r.z_mean,
                "metric/eval/z_std": r.z_std,
                "sweep_id": "local",
            }
            all_runs.append(run_dict)
    else:
        for sweep_id in selection.sweep_ids:
            runs = load_wandb_results(sweep_id, selection.entity, selection.project)
            for r in runs:
                # Use a per-run identifier; RunResult.path is often empty for W&B.
                run_id = getattr(r, "path", None) or (f"{sweep_id}:{r.env}:{r.model}:{r.seed}")
                run_dict = {
                    "run_id": run_id,
                    "config/llms": r.model,
                    "config/envs": r.env,
                    "config/seed": r.seed,
                    "config/include_prior": r.include_prior,
                    "config/use_ppl": r.use_ppl,
                    "config/budget": r.budget,
                    "config/goal": r.goal,
                    "config/exp": r.experiment_type,
                    "metric/eval/z_mean": r.z_mean,
                    "metric/eval/z_std": r.z_std,
                    "sweep_id": sweep_id,
                }
                all_runs.append(run_dict)

    return all_runs
