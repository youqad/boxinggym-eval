"""Load local JSON result files into DataFrame for TUI analysis."""

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console

console = Console()

# patterns: env=location_finding_goal=direct_model=gpt-4o_exp=oed_prior=true_seed=1
MODERN_PATTERN = re.compile(
    r"env=(?P<env>.+?)_goal=(?P<goal>.+?)_model=(?P<model>.+?)_exp=(?P<exp>\w+)_prior=(?P<prior>\w+)_seed=(?P<seed>\d+)"
)
LEGACY_PATTERN = re.compile(
    r"^(?:direct|discount)_(?P<model>.+?)(?:-boxloop)?_(?P<exp>\w+)_(?P<prior>\w+)_(?P<seed>\d+)\.json$"
)


def _extract_model_name(config: dict[str, Any]) -> str | None:
    llms = config.get("llms", {})
    if isinstance(llms, dict):
        model = llms.get("model_name", "")
    else:
        model = str(llms)

    return model if model else None


def _parse_filename_metadata(filename: str) -> dict[str, Any] | None:
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

    match = LEGACY_PATTERN.match(filename)
    if match:
        return {
            "model": match.group("model"),
            "exp": match.group("exp"),
            "prior": match.group("prior").lower() == "true",
            "seed": int(match.group("seed")),
        }

    return None


def _parse_json_file(path: Path, results_root: Path) -> list[dict[str, Any]]:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    config = data.get("config", {})
    data_section = data.get("data", {})
    z_results = data_section.get("z_results", [])

    if not z_results:
        return []

    model = _extract_model_name(config)
    envs_config = config.get("envs", {})
    env_name = envs_config.get("env_name") if isinstance(envs_config, dict) else None

    if not model or not env_name:
        file_meta = _parse_filename_metadata(path.name)
        if file_meta:
            model = model or file_meta.get("model")
            env_name = env_name or file_meta.get("env")

    if not model or not env_name:
        return []

    is_boxloop = "-boxloop" in path.name or "_boxloop" in path.name
    if is_boxloop and "(box)" not in model:
        model = f"{model} (box)"

    is_multi_seed = data_section.get("aggregation_mode") == "multi_seed"
    if is_multi_seed:
        raw_n_seeds = data_section.get("n_seeds")
        try:
            n_seeds = int(raw_n_seeds) if raw_n_seeds else 1
        except (ValueError, TypeError):
            n_seeds = 1
    else:
        n_seeds = 1

    base_row = {
        "config/llms": model,
        "config/envs": env_name,
        "config/include_prior": config.get("include_prior", False),
        "config/use_ppl": config.get("use_ppl", False),
        "run_id": str(path.relative_to(results_root))
        if path.is_relative_to(results_root)
        else str(path),
        "sweep_id": "local",
    }

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

        if is_multi_seed:
            row["z_stderr"] = zr.get("z_stderr")

        rows.append(row)

    return rows


def _normalize_model_names(df: pd.DataFrame) -> pd.DataFrame:
    col = "config/llms"
    models = set(df[col].unique())
    renames = {}

    base_variants: dict = {}
    for m in models:
        if "/" not in m:
            continue
        base = m.rsplit("/", 1)[-1]
        if base in models:
            base_variants.setdefault(base, []).append(m)

    for base, prefixed in base_variants.items():
        candidates = prefixed + [base]
        canonical = max(candidates, key=lambda c: (df[col] == c).sum())
        for c in candidates:
            if c != canonical:
                renames[c] = canonical

    if renames:
        df = df.copy()
        df[col] = df[col].replace(renames)

    return df


def load_local_results(
    results_dir: str = "results",
    verbose: bool = True,
) -> pd.DataFrame:
    """Load JSON result files into DataFrame with WandB-compatible columns."""
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    json_files = list(results_path.rglob("*.json"))
    json_files = [
        f
        for f in json_files
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
    df = _normalize_model_names(df)

    if verbose:
        n_models = df["config/llms"].nunique()
        n_envs = df["config/envs"].nunique()
        console.print(
            f"[green]Loaded {len(df)} records: {n_models} models, {n_envs} environments[/green]"
        )

    return df
