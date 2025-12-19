import json
import os
import time

import numpy as np


def build_output_filename(
    env_name: str,
    goal_name: str,
    model_name: str,
    experiment_type: str,
    include_prior: bool,
    seed: int,
    use_ppl: bool,
    wandb_meta: dict,
):
    save_model_name = model_name.replace("/", "_")
    if use_ppl:
        save_model_name = save_model_name + "-boxloop"
    prior_tag = "true" if bool(include_prior) else "false"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id_tag = wandb_meta.get("run_id") if isinstance(wandb_meta, dict) else None
    run_id_tag = run_id_tag or "no_wandb"
    return (
        f"results/{env_name}/env={env_name}_goal={goal_name}_model={save_model_name}"
        f"_exp={experiment_type}_prior={prior_tag}_seed={seed}_{timestamp}-{run_id_tag}.json"
    )


def ensure_output_dir(output_filename: str):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)


def write_results(final_dict, output_filename: str):
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    with open(output_filename, "w") as f:
        json.dump(final_dict, f, cls=NpEncoder, indent=2)


def write_run_artifact_index(results_path: str, wandb_meta: dict):
    if not results_path or not isinstance(results_path, str):
        return
    base, ext = os.path.splitext(results_path)
    index_path = f"{base}_artifacts.txt" if ext else f"{results_path}_artifacts.txt"
    lines = [f"results_file: {results_path}"]
    run_dir = None
    if isinstance(wandb_meta, dict):
        run_id = wandb_meta.get("run_id")
        run_path = wandb_meta.get("run_path")
        run_dir = wandb_meta.get("dir")
        if run_id:
            lines.append(f"wandb_run_id: {run_id}")
        if run_path:
            lines.append(f"wandb_run_path: {run_path}")
        if run_dir:
            lines.append(f"wandb_run_dir: {run_dir}")

    files_root = None
    if run_dir and os.path.isdir(run_dir):
        if os.path.basename(run_dir) == "files":
            files_root = run_dir
        else:
            candidate = os.path.join(run_dir, "files")
            files_root = candidate if os.path.isdir(candidate) else None

    if files_root:
        root_dir = os.getcwd()
        paths = []
        for root, _, files in os.walk(files_root):
            for name in files:
                full_path = os.path.join(root, name)
                try:
                    rel_path = os.path.relpath(full_path, root_dir)
                except Exception:
                    rel_path = full_path
                paths.append(rel_path)
        if paths:
            lines.append("artifacts:")
            for path in sorted(paths):
                lines.append(f"- {path}")

    try:
        with open(index_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass
