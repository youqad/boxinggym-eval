"""Load local JSON result files into DataFrame for TUI analysis."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rich.console import Console

console = Console()

# filename patterns - use lookahead to handle multi-word values with underscores
# e.g., env=location_finding_goal=direct_model=gpt-4o_exp=oed_prior=true_seed=1
MODERN_PATTERN = re.compile(
    r"env=(?P<env>.+?)_goal=(?P<goal>.+?)_model=(?P<model>.+?)_exp=(?P<exp>\w+)_prior=(?P<prior>\w+)_seed=(?P<seed>\d+)"
)
LEGACY_PATTERN = re.compile(
    r"^(?:direct|discount)_(?P<model>.+?)(?:-boxloop)?_(?P<exp>\w+)_(?P<prior>\w+)_(?P<seed>\d+)\.json$"
)


def _extract_model_name(config: Dict[str, Any]) -> Optional[str]:
    """Extract model name from config."""
    llms = config.get("llms", {})
    if isinstance(llms, dict):
        model = llms.get("model_name", "")
    else:
        model = str(llms)

    return model if model else None


def _parse_filename_metadata(filename: str) -> Optional[Dict[str, Any]]:
    """Extract metadata from filename as fallback."""
    # try modern pattern first
    match = MODERN_PATTERN.search(filename)
    if match:
        return {
            "env": match.group("env"),
            "goal": match.group("goal"),
            "model": match.group("model"),
            "exp": match.group("exp"),
            "prior": match.group("prior").lower() == "true",
            "seed": int(match.group("seed")),
        }

    # try legacy pattern
    match = LEGACY_PATTERN.match(filename)
    if match:
        return {
            "model": match.group("model"),
            "exp": match.group("exp"),
            "prior": match.group("prior").lower() == "true",
            "seed": int(match.group("seed")),
        }

    return None


def _parse_json_file(path: Path, results_root: Path) -> List[Dict[str, Any]]:
    """Parse single JSON file, returning one row per budget checkpoint.

    Handles both formats:
    - Multi-seed (new): data.aggregation_mode == "multi_seed", uses aggregated z_results
    - Per-seed (legacy): individual seed runs, seed extracted from config/filename

    Args:
        path: Absolute path to the JSON file
        results_root: Root results directory for computing relative run_id
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    config = data.get("config", {})
    data_section = data.get("data", {})
    z_results = data_section.get("z_results", [])

    if not z_results:
        return []

    # extract base metadata
    model = _extract_model_name(config)
    envs_config = config.get("envs", {})
    env_name = envs_config.get("env_name") if isinstance(envs_config, dict) else None

    # fallback to filename parsing if config incomplete
    if not model or not env_name:
        file_meta = _parse_filename_metadata(path.name)
        if file_meta:
            model = model or file_meta.get("model")
            env_name = env_name or file_meta.get("env")

    if not model or not env_name:
        return []

    # check if boxloop run
    is_boxloop = "-boxloop" in path.name or "_boxloop" in path.name
    if is_boxloop and "(box)" not in model:
        model = f"{model} (box)"

    # detect multi-seed format (new) vs per-seed format (legacy)
    is_multi_seed = data_section.get("aggregation_mode") == "multi_seed"
    # handle both missing key AND explicit null: use `or 1` to catch None
    # cast to int for type safety (JSON could have string "10")
    n_seeds = int(data_section.get("n_seeds") or 1) if is_multi_seed else 1

    base_row = {
        "config/llms": model,
        "config/envs": env_name,
        "config/include_prior": config.get("include_prior", False),
        "config/use_ppl": config.get("use_ppl", False),
        # use relative path for portability while preserving uniqueness across subdirs
        "run_id": str(path.relative_to(results_root)) if path.is_relative_to(results_root) else str(path),
        "sweep_id": "local",
    }

    # for multi-seed: no config/seed (aggregated), add n_seeds
    # for per-seed: include config/seed
    if is_multi_seed:
        base_row["summary/n_seeds"] = n_seeds
    else:
        base_row["config/seed"] = config.get("seed")

    rows = []
    for zr in z_results:
        if not isinstance(zr, dict):
            continue

        z_mean = zr.get("z_mean")
        if z_mean is None or not isinstance(z_mean, (int, float)):
            continue

        row = base_row.copy()
        row["config/budget"] = zr.get("budget", 0)
        row["config/exp.budget"] = zr.get("budget", 0)
        row["metric/eval/z_mean"] = z_mean
        row["metric/eval/z_std"] = zr.get("z_std", 0.0)
        row["metric/eval/raw_mean"] = zr.get("raw_mean", 0.0)

        # multi-seed format includes z_stderr (std across seeds / sqrt(n))
        # use plain column name for seed_stability.py compatibility
        if is_multi_seed:
            row["z_stderr"] = zr.get("z_stderr")

        rows.append(row)

    return rows


def load_local_results(
    results_dir: str = "results",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load all JSON result files from results/ directory into DataFrame.

    Returns DataFrame with columns matching WandB sweep format:
        - config/llms: model name
        - config/envs: environment name
        - config/seed: random seed
        - config/include_prior: bool
        - config/budget: experiment budget
        - metric/eval/z_mean: z-score
        - run_id: filename
        - sweep_id: "local"
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # find all JSON files
    json_files = list(results_path.rglob("*.json"))

    # filter out non-result files
    json_files = [
        f for f in json_files
        if not f.name.startswith("llm_calls")
        and "comparative_benchmarks" not in str(f)
        and not f.name.endswith("_artifacts.txt")
    ]

    if verbose:
        console.print(f"[dim]Found {len(json_files)} JSON files[/dim]")

    all_rows = []
    errors = 0

    for path in json_files:
        try:
            rows = _parse_json_file(path, results_path)
            all_rows.extend(rows)
        except Exception as e:
            if verbose:
                console.print(f"[dim]Skipping {path.name}: {e}[/dim]")
            errors += 1

    if errors > 0 and verbose:
        console.print(f"[yellow]Skipped {errors} files with errors[/yellow]")

    if not all_rows:
        raise ValueError(f"No valid results found in {results_dir}")

    df = pd.DataFrame(all_rows)

    if verbose:
        n_models = df["config/llms"].nunique()
        n_envs = df["config/envs"].nunique()
        console.print(f"[green]Loaded {len(df)} records: {n_models} models, {n_envs} environments[/green]")

    return df
