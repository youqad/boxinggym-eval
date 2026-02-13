import json
from pathlib import Path

import pandas as pd
from scripts.refresh_canonical_results import (
    apply_canonical_filters,
    compute_champions_model_only,
    compute_champions_model_plus_ppl,
    render_readme_results_block,
    replace_marked_block,
)


def test_apply_canonical_filters_excludes_outliers_and_low_budget():
    df = pd.DataFrame(
        [
            {"model": "m1", "env": "e1", "budget": 10, "z_mean": 0.1, "is_outlier": False},
            {"model": "m1", "env": "e1", "budget": 5, "z_mean": 0.2, "is_outlier": False},
            {"model": "m2", "env": "e1", "budget": 10, "z_mean": 0.3, "is_outlier": True},
            {"model": "m3", "env": "e2", "budget": 10, "z_mean": None, "is_outlier": False},
        ]
    )

    filtered, stats = apply_canonical_filters(df, min_budget=10, exclude_outliers=True)

    assert len(filtered) == 1
    assert filtered.iloc[0]["model"] == "m1"
    assert stats["rows_total_before_filters"] == 4
    assert stats["rows_after_filters"] == 1


def test_champion_definitions_can_differ():
    df = pd.DataFrame(
        [
            # model m1 has one very good PPL=False run and one bad PPL=True run
            {"env": "e1", "model": "m1", "use_ppl": False, "z_mean": -0.20},
            {"env": "e1", "model": "m1", "use_ppl": True, "z_mean": +0.40},
            # model m2 is moderately good on average
            {"env": "e1", "model": "m2", "use_ppl": False, "z_mean": +0.05},
        ]
    )

    model_only = compute_champions_model_only(df)
    model_plus_ppl = compute_champions_model_plus_ppl(df)

    assert model_only[0]["model_raw"] == "m2"
    assert model_plus_ppl[0]["model_raw"] == "m1"
    assert model_plus_ppl[0]["use_ppl"] is False


def test_replace_marked_block_updates_only_marker_body(tmp_path: Path):
    src = (
        "before\n"
        "<!-- CANONICAL_RESULTS:START -->\n"
        "old\n"
        "<!-- CANONICAL_RESULTS:END -->\n"
        "after\n"
    )
    path = tmp_path / "README.md"
    path.write_text(src)

    replace_marked_block(path, "CANONICAL_RESULTS", "new content")

    out = path.read_text()
    assert "before" in out
    assert "after" in out
    assert "new content" in out
    assert "old" not in out


def test_render_readme_results_block_contains_both_champion_sections():
    metrics = {
        "leaderboard": [
            {
                "rank": 1,
                "model_display": "Model A",
                "mean_z": 0.1,
                "ci_low": 0.0,
                "ci_high": 0.2,
                "n": 10,
                "significant_vs_best": False,
                "p_adjusted_vs_best": None,
            }
        ],
        "champions_model_only": [
            {"env": "e1", "model_display": "Model A", "z_mean": -0.1}
        ],
        "champions_model_plus_ppl": [
            {"env": "e1", "model_display": "Model A", "use_ppl": True, "z_mean": -0.2}
        ],
        "key_findings": {
            "best_model": "Model A",
            "best_model_mean_z": 0.1,
            "n_env_beating_baseline_model_only": 1,
            "n_env_total": 1,
        },
    }
    metadata = {
        "generated_at_utc": "2026-02-13T00:00:00Z",
        "snapshot_version": "v1",
        "rows_after_filters": 1,
        "models_after_filters": 1,
        "envs_after_filters": 1,
        "filters": {"min_budget": 10, "exclude_outliers": True},
    }

    block = render_readme_results_block(metrics, metadata)
    assert "Per-Environment Champions (Best Model Only)" in block
    assert "Per-Environment Champions (Best Model + PPL)" in block


def test_metadata_json_roundtrip(tmp_path: Path):
    metadata = {
        "generated_at_utc": "2026-02-13T00:00:00Z",
        "snapshot_version": "v1",
        "filters": {"min_budget": 10, "exclude_outliers": True},
        "rows_total_before_filters": 100,
        "rows_after_filters": 80,
        "models_after_filters": 7,
        "envs_after_filters": 10,
    }
    p = tmp_path / "canonical_metadata.json"
    p.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    loaded = json.loads(p.read_text())
    assert loaded["snapshot_version"] == "v1"
    assert loaded["rows_after_filters"] == 80
