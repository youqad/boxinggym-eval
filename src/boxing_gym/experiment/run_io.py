import json
import os
import time

import numpy as np


class _NpEncoder(json.JSONEncoder):
    """JSON encoder for numpy types, pandas DataFrames, and unserializable objects."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # handle pandas DataFrames/Series
        try:
            import pandas as pd

            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            if isinstance(obj, pd.Series):
                return obj.tolist()
        except ImportError:
            pass
        # handle pymc InferenceData and other complex objects - skip with placeholder
        try:
            import arviz

            if isinstance(obj, arviz.InferenceData):
                return "<InferenceData - not serializable>"
        except ImportError:
            pass
        # last resort: try str() for any unserializable object
        try:
            return super().default(obj)
        except TypeError:
            try:
                return f"<{type(obj).__name__} - not serializable>"
            except Exception:
                return "<unknown - not serializable>"


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
    # use module-level _NpEncoder for numpy/pandas serialization
    with open(output_filename, "w") as f:
        json.dump(final_dict, f, cls=_NpEncoder, indent=2)


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


# checkpointing for multi-seed runs

CHECKPOINT_DIR = "checkpoints"


def _get_checkpoint_path(run_id: str) -> str:
    """Get path to checkpoint file for a run."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{run_id}_checkpoint.json")


def load_checkpoint(run_id: str) -> dict:
    """Load checkpoint if exists. Returns dict with completed_seeds and seed_results."""
    path = _get_checkpoint_path(run_id)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_checkpoint(run_id: str, completed_seeds: list, seed_results: list):
    """Save checkpoint with completed seeds and their results.

    We only save lightweight data (z_results, status) to keep checkpoint small.
    Full data will be reconstructed on final save.
    """
    path = _get_checkpoint_path(run_id)
    # keep only essential data for recovery
    lightweight_results = []
    for sr in seed_results:
        lightweight_results.append(
            {
                "seed": sr["seed"],
                "z_results": sr["z_results"],
                "status": sr.get("status", "ok"),
            }
        )

    checkpoint = {
        "completed_seeds": completed_seeds,
        "seed_results": lightweight_results,
    }

    try:
        with open(path, "w") as f:
            json.dump(checkpoint, f, cls=_NpEncoder, indent=2)
    except Exception:
        pass  # checkpoint is optional, don't fail the run


def clear_checkpoint(run_id: str):
    """Delete checkpoint file after successful completion."""
    path = _get_checkpoint_path(run_id)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# per-seed full data files (preserves all_data, messages for investigation)


def get_seed_data_path(run_id: str, seed: int) -> str:
    """Path to per-seed full data file."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{run_id}_seed{seed}_full.json")


def write_seed_data(run_id: str, seed: int, seed_result: dict) -> str:
    """Write full experiment data for a single seed (atomic write).

    Includes: all_data, scientist_messages, naive_messages, z_results.
    Called after each seed completes for crash recovery with full data.

    Uses atomic write pattern (temp file + rename) to prevent corruption
    if the process crashes mid-write.
    """
    path = get_seed_data_path(run_id, seed)
    temp_path = path + ".tmp"

    # all_data is a tuple of 7 lists - convert to dict for JSON
    all_data = seed_result.get("all_data", ([], [], [], [], [], [], []))
    data_dict = {
        "seed": seed,
        "z_results": seed_result["z_results"],
        "status": seed_result.get("status", "ok"),
        "all_data": {
            "results": all_data[0] if len(all_data) > 0 else [],
            "queries": all_data[1] if len(all_data) > 1 else [],
            "observations": all_data[2] if len(all_data) > 2 else [],
            "successes": all_data[3] if len(all_data) > 3 else [],
            "explanations": all_data[4] if len(all_data) > 4 else [],
            "eigs": all_data[5] if len(all_data) > 5 else [],
            "programs": all_data[6] if len(all_data) > 6 else [],
        },
        "scientist_messages": seed_result.get("scientist_messages", []),
        "naive_messages": seed_result.get("naive_messages"),
    }

    try:
        with open(temp_path, "w") as f:
            json.dump(data_dict, f, cls=_NpEncoder, indent=2)
        os.replace(temp_path, path)  # atomic on POSIX
        return path
    except Exception as e:
        # return None to signal failure - caller must check before marking complete
        print(f"⚠️  Failed to write seed data to {path}: {e}")
        # cleanup temp file if it exists
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return None


def load_seed_data(run_id: str, seed: int) -> dict:
    """Load full experiment data for a seed (on crash recovery)."""
    path = get_seed_data_path(run_id, seed)
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            # reconstruct all_data tuple from dict
            ad = data.get("all_data", {})
            if isinstance(ad, dict):
                data["all_data"] = (
                    ad.get("results", []),
                    ad.get("queries", []),
                    ad.get("observations", []),
                    ad.get("successes", []),
                    ad.get("explanations", []),
                    ad.get("eigs", []),
                    ad.get("programs", []),
                )
            return data
        except Exception:
            return {}
    return {}


def clear_seed_data(run_id: str, seeds: list):
    """Delete per-seed files after successful completion (optional cleanup)."""
    for seed in seeds:
        path = get_seed_data_path(run_id, seed)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
