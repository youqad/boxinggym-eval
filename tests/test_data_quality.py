"""Tests for data_quality module: schema validation, non-finite checks, quarantine."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from boxing_gym.data_quality.config import ValidationLevel
from boxing_gym.data_quality.quarantine import QuarantineManager
from boxing_gym.data_quality.rules import QualityValidator, ValidationIssue, ValidationResult


def _make_valid_data(
    z_mean: float = 0.5,
    raw_mean: float = 0.3,
    z_std: float = 0.1,
    budget: int = 10,
    env_name: str = "dugongs",
) -> dict:
    return {
        "config": {"envs": {"env_name": env_name}, "llms": {"model_name": "test-model"}, "seed": 1},
        "data": {
            "z_results": [
                {"budget": budget, "z_mean": z_mean, "raw_mean": raw_mean, "z_std": z_std},
            ],
        },
    }


def _write_json(tmp_path: Path, data, name: str = "test.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


# -- enum ordering --


def test_validation_level_ordering():
    levels = sorted(ValidationLevel)
    names = [lv.name for lv in levels]
    assert names == [
        "L0_SCHEMA",
        "L1_NON_FINITE",
        "L1_5_CONSISTENCY",
        "L2_HARD_INVARIANTS",
        "L3_PAPER_BASELINE",
        "L4_MAD_OUTLIER",
        "L5_DUPLICATES",
    ]
    # numeric ordering preserved
    for i in range(len(levels) - 1):
        assert levels[i] < levels[i + 1]


# -- L0: schema tests --


def test_valid_file(tmp_path):
    p = _write_json(tmp_path, _make_valid_data())
    result = QualityValidator().validate_file(p)
    assert result.status == "VALID"
    assert len(result.issues) == 0


def test_top_level_not_dict(tmp_path):
    p = _write_json(tmp_path, [1, 2, 3])
    result = QualityValidator().validate_file(p)
    assert result.status == "QUARANTINE"
    assert any("not an object" in i.message for i in result.issues)


def test_data_not_dict(tmp_path):
    p = _write_json(tmp_path, {"config": {}, "data": "not_a_dict"})
    result = QualityValidator().validate_file(p)
    assert result.status == "QUARANTINE"
    assert any("'data' is not an object" in i.message for i in result.issues)


def test_z_results_not_list_none(tmp_path):
    data = {"config": {}, "data": {"z_results": None}}
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert result.status == "QUARANTINE"
    assert any("z_results is not a list" in i.message for i in result.issues)


def test_z_results_not_list_string(tmp_path):
    data = {"config": {}, "data": {"z_results": "foo"}}
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert result.status == "QUARANTINE"
    assert any("z_results is not a list" in i.message for i in result.issues)


def test_z_results_empty(tmp_path):
    data = {"config": {}, "data": {"z_results": []}}
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert result.status == "QUARANTINE"
    assert any("empty" in i.message for i in result.issues)


def test_z_results_no_valid_entries(tmp_path):
    data = {"config": {}, "data": {"z_results": [None, "garbage", 42]}}
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert result.status == "QUARANTINE"
    assert any("no valid entries" in i.message for i in result.issues)


def test_missing_fields(tmp_path):
    p = _write_json(tmp_path, {"config": {}})
    result = QualityValidator().validate_file(p)
    assert result.status == "QUARANTINE"
    assert any("missing required field 'data'" in i.message for i in result.issues)


# -- L1: non-finite tests --


def test_z_mean_null(tmp_path):
    data = _make_valid_data()
    data["data"]["z_results"][0]["z_mean"] = None
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert any(i.layer == ValidationLevel.L1_NON_FINITE and "z_mean is null" in i.message for i in result.issues)


def test_z_mean_non_numeric(tmp_path):
    data = _make_valid_data()
    data["data"]["z_results"][0]["z_mean"] = "abc"
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert any(i.layer == ValidationLevel.L1_NON_FINITE and "not numeric" in i.message for i in result.issues)


def test_z_mean_nan(tmp_path):
    data = _make_valid_data()
    data["data"]["z_results"][0]["z_mean"] = float("nan")
    # NaN doesn't survive JSON roundtrip, so write directly
    p = tmp_path / "nan_test.json"
    # write with NaN as null (JSON doesn't support NaN)
    # instead, test via non-finite float workaround
    data["data"]["z_results"][0]["z_mean"] = float("inf")
    p.write_text(json.dumps(data))
    result = QualityValidator().validate_file(p)
    assert any(i.layer == ValidationLevel.L1_NON_FINITE and "z_mean" in i.message for i in result.issues)


def test_raw_mean_null(tmp_path):
    data = _make_valid_data()
    data["data"]["z_results"][0]["raw_mean"] = None
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert any(i.layer == ValidationLevel.L1_NON_FINITE and "raw_mean is null" in i.message for i in result.issues)


def test_raw_mean_non_numeric(tmp_path):
    data = _make_valid_data()
    data["data"]["z_results"][0]["raw_mean"] = "abc"
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert any(
        i.layer == ValidationLevel.L1_NON_FINITE and "raw_mean is not numeric" in i.message for i in result.issues
    )


def test_z_std_non_positive(tmp_path):
    data = _make_valid_data()
    data["data"]["z_results"][0]["z_std"] = 0
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert any(i.layer == ValidationLevel.L1_NON_FINITE and "z_std" in i.message for i in result.issues)


def test_z_std_negative(tmp_path):
    data = _make_valid_data()
    data["data"]["z_results"][0]["z_std"] = -1.0
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert any(i.layer == ValidationLevel.L1_NON_FINITE and "z_std" in i.message for i in result.issues)


def test_z_std_non_numeric(tmp_path):
    data = _make_valid_data()
    data["data"]["z_results"][0]["z_std"] = "bad"
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert any(
        i.layer == ValidationLevel.L1_NON_FINITE and "z_std is not numeric" in i.message for i in result.issues
    )


# -- L1.5: consistency --


def test_z_consistency_mismatch(tmp_path):
    data = _make_valid_data(z_mean=5.0, raw_mean=0.3)
    # with norm_factors that produce a different z
    data["data"]["norm_factors"] = {"mu": 0.2, "sigma": 0.1}
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert any(i.layer == ValidationLevel.L1_5_CONSISTENCY for i in result.issues)


def test_z_consistency_correct(tmp_path):
    mu, sigma = 0.2, 0.1
    raw_mean = 0.3
    z_mean = (raw_mean - mu) / sigma  # = 1.0
    data = _make_valid_data(z_mean=z_mean, raw_mean=raw_mean)
    data["data"]["norm_factors"] = {"mu": mu, "sigma": sigma}
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert not any(i.layer == ValidationLevel.L1_5_CONSISTENCY for i in result.issues)


# -- L2: hard invariants --


def test_raw_mean_outside_bounds(tmp_path):
    data = _make_valid_data(raw_mean=-1.0, env_name="dugongs")
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert any(i.layer == ValidationLevel.L2_HARD_INVARIANTS and "outside" in i.message for i in result.issues)


def test_raw_mean_inf(tmp_path):
    data = _make_valid_data(env_name="dugongs")
    data["data"]["z_results"][0]["raw_mean"] = float("inf")
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert any(i.layer == ValidationLevel.L2_HARD_INVARIANTS for i in result.issues)


# -- env extraction --


def test_envs_as_string(tmp_path):
    data = _make_valid_data()
    data["config"]["envs"] = "test_env"
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert result.env == "test_env"


def test_envs_as_dict(tmp_path):
    data = _make_valid_data(env_name="dugongs")
    p = _write_json(tmp_path, data)
    result = QualityValidator().validate_file(p)
    assert result.env == "dugongs"


# -- quarantine --


def test_quarantine_moves_file(tmp_path):
    qm = QuarantineManager(str(tmp_path / ".quarantine"))
    src = tmp_path / "test.json"
    src.write_text('{"bad": true}')

    dest = qm.quarantine(src, ValidationLevel.L0_SCHEMA, "test reason")
    assert dest.exists()
    assert not src.exists()
    # audit log written
    log = qm.get_audit_log()
    assert len(log) == 1
    assert log[0]["action"] == "quarantine"


def test_quarantine_unique_names(tmp_path):
    qm = QuarantineManager(str(tmp_path / ".quarantine"))

    dests = []
    for i in range(3):
        src = tmp_path / f"test_{i}.json"
        src.write_text(f'{{"i": {i}}}')
        dest = qm.quarantine(src, ValidationLevel.L0_SCHEMA, f"collision test {i}")
        dests.append(dest)

    # all destinations are unique paths
    assert len(set(dests)) == 3


def test_restore_within_bounds(tmp_path):
    qm = QuarantineManager(str(tmp_path / ".quarantine"))
    results_root = tmp_path / "results"
    results_root.mkdir()

    src = tmp_path / "test.json"
    src.write_text('{"data": true}')
    quarantined = qm.quarantine(src, ValidationLevel.L0_SCHEMA, "test")

    dest_dir = results_root / "dugongs"
    restored = qm.restore(quarantined, dest_dir, results_root=results_root)
    assert restored.exists()
    assert not quarantined.exists()


def test_restore_outside_bounds_raises(tmp_path):
    qm = QuarantineManager(str(tmp_path / ".quarantine"))
    results_root = tmp_path / "results"
    results_root.mkdir()

    src = tmp_path / "test.json"
    src.write_text('{"data": true}')
    quarantined = qm.quarantine(src, ValidationLevel.L0_SCHEMA, "test")

    outside = tmp_path / "../../etc"
    with pytest.raises(ValueError, match="outside results root"):
        qm.restore(quarantined, outside, results_root=results_root)


def test_list_quarantined_includes_other(tmp_path):
    """list_quarantined should find files in the 'other' subdir."""
    qm = QuarantineManager(str(tmp_path / ".quarantine"))
    qm.setup()
    other_dir = qm.root / "other"
    other_dir.mkdir(exist_ok=True)
    (other_dir / "stray.json").write_text("{}")

    files = qm.list_quarantined()
    assert any("stray.json" in str(f) for f in files)
