"""Multi-layer validation rules for BoxingGym result files.

Validation layers (applied in order):
  L0: Schema integrity (malformed JSON, missing required fields)
  L1: Non-finite values (nan, inf, missing raw_mean, sigma <= 0)
  L1.5: Consistency (recomputed z != stored z)
  L2: Hard invariants (MSE < 0, error rate not in [0,1])
  L3: Paper baseline comparison (report only, no quarantine)
  L4: MAD outlier detection per (env, model, budget) group
  L5: Duplicate detection
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    PHYSICAL_BOUNDS,
    REQUIRED_DATA_FIELDS,
    REQUIRED_FIELDS,
    Z_CONSISTENCY_TOLERANCE,
    ValidationLevel,
)


@dataclass
class ValidationIssue:
    layer: ValidationLevel
    message: str
    budget: int | None = None
    details: dict[str, Any] | None = None


@dataclass
class ValidationResult:
    path: str
    env: str = "unknown"
    model: str = "unknown"
    seed: int | None = None
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def status(self) -> str:
        """QUARANTINE if L0/L1/L1.5/L2, REVIEW if L3/L4, else VALID."""
        quarantine_layers = {
            ValidationLevel.L0_SCHEMA,
            ValidationLevel.L1_NON_FINITE,
            ValidationLevel.L1_5_CONSISTENCY,
            ValidationLevel.L2_HARD_INVARIANTS,
        }
        for issue in self.issues:
            if issue.layer in quarantine_layers:
                return "QUARANTINE"
        if self.issues:
            return "REVIEW"
        return "VALID"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "env": self.env,
            "model": self.model,
            "seed": self.seed,
            "status": self.status,
            "issues": [
                {
                    "layer": issue.layer.name,
                    "message": issue.message,
                    "budget": issue.budget,
                    "details": issue.details,
                }
                for issue in self.issues
            ],
        }


class QualityValidator:
    """Applies multi-layer validation to result files."""

    def __init__(
        self,
        norm_factors: dict[str, tuple[float, float]] | None = None,
        paper_baselines: dict | None = None,
    ):
        self._norm_factors = norm_factors or {}
        self._paper_baselines = paper_baselines or {}

    def validate_file(self, path: Path) -> ValidationResult:
        """Apply L0-L2 validation to a single file."""
        result = ValidationResult(path=str(path))

        # L0: schema
        data = self._check_schema(path, result)
        if data is None:
            return result

        config = data.get("config", {})
        envs_cfg = config.get("envs", {})
        if isinstance(envs_cfg, dict):
            env = envs_cfg.get("env_name", "unknown")
        elif isinstance(envs_cfg, str):
            env = envs_cfg
        else:
            env = "unknown"
        model = "unknown"
        llms = config.get("llms", {})
        if isinstance(llms, dict):
            model = llms.get("model_name", "unknown")
        result.env = env
        result.model = model
        result.seed = config.get("seed")

        data_section = data.get("data")
        if not isinstance(data_section, dict):
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L0_SCHEMA,
                    message="'data' is not an object",
                )
            )
            return result

        z_results = data_section.get("z_results")
        if not isinstance(z_results, list):
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L0_SCHEMA,
                    message="data.z_results is not a list",
                )
            )
            return result

        if not z_results:
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L0_SCHEMA,
                    message="data.z_results is empty",
                )
            )
            return result

        norm_factors = data_section.get("norm_factors", {})

        valid_entries = 0
        for zr in z_results:
            if not isinstance(zr, dict):
                continue
            valid_entries += 1
            budget = zr.get("budget")
            z_mean = zr.get("z_mean")
            raw_mean = zr.get("raw_mean")
            z_std = zr.get("z_std")

            # L1: non-finite
            self._check_non_finite(result, budget, z_mean, raw_mean, z_std)

            # L1.5: consistency
            self._check_consistency(result, budget, z_mean, raw_mean, norm_factors, env)

            # L2: hard invariants
            self._check_hard_invariants(result, budget, raw_mean, env)

        if valid_entries == 0:
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L0_SCHEMA,
                    message="data.z_results has no valid entries",
                )
            )

        return result

    def check_duplicates(
        self,
        file_keys: dict[tuple, list[str]],
    ) -> list[tuple[str, list[str]]]:
        """L5: find duplicates by (seed, env, model, budget, goal).

        Returns list of (kept_path, [removed_paths]).
        """
        duplicates = []
        for key, paths in file_keys.items():
            if len(paths) <= 1:
                continue
            # keep the most recently modified file
            by_mtime = sorted(paths, key=lambda p: Path(p).stat().st_mtime, reverse=True)
            duplicates.append((by_mtime[0], by_mtime[1:]))
        return duplicates

    # -- internal checks --

    def _check_schema(self, path: Path, result: ValidationResult) -> dict | None:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L0_SCHEMA,
                    message=f"cannot parse: {e}",
                )
            )
            return None

        if not isinstance(data, dict):
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L0_SCHEMA,
                    message="top-level value is not an object",
                )
            )
            return None

        for field_name in REQUIRED_FIELDS:
            if field_name not in data:
                result.issues.append(
                    ValidationIssue(
                        layer=ValidationLevel.L0_SCHEMA,
                        message=f"missing required field '{field_name}'",
                    )
                )

        data_section = data.get("data")
        if not isinstance(data_section, dict):
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L0_SCHEMA,
                    message="'data' is not an object",
                )
            )
        else:
            for field_name in REQUIRED_DATA_FIELDS:
                if field_name not in data_section:
                    result.issues.append(
                        ValidationIssue(
                            layer=ValidationLevel.L0_SCHEMA,
                            message=f"missing data.{field_name}",
                        )
                    )

        if any(i.layer == ValidationLevel.L0_SCHEMA for i in result.issues):
            return None

        return data

    def _check_non_finite(
        self,
        result: ValidationResult,
        budget: Any,
        z_mean: Any,
        raw_mean: Any,
        z_std: Any,
    ) -> None:
        if z_mean is None:
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L1_NON_FINITE,
                    message="z_mean is null",
                    budget=budget,
                )
            )
            return

        if not isinstance(z_mean, (int, float)):
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L1_NON_FINITE,
                    message=f"z_mean is not numeric: {type(z_mean).__name__}",
                    budget=budget,
                )
            )
            return

        if math.isnan(z_mean) or math.isinf(z_mean):
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L1_NON_FINITE,
                    message=f"z_mean is {z_mean}",
                    budget=budget,
                )
            )

        if raw_mean is None:
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L1_NON_FINITE,
                    message="raw_mean is null",
                    budget=budget,
                )
            )
        elif not isinstance(raw_mean, (int, float)):
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L1_NON_FINITE,
                    message=f"raw_mean is not numeric: {type(raw_mean).__name__}",
                    budget=budget,
                )
            )

        if z_std is not None:
            if not isinstance(z_std, (int, float)):
                result.issues.append(
                    ValidationIssue(
                        layer=ValidationLevel.L1_NON_FINITE,
                        message=f"z_std is not numeric: {type(z_std).__name__}",
                        budget=budget,
                    )
                )
            elif z_std <= 0 or not math.isfinite(z_std):
                result.issues.append(
                    ValidationIssue(
                        layer=ValidationLevel.L1_NON_FINITE,
                        message=f"z_std is non-positive or non-finite: {z_std}",
                        budget=budget,
                    )
                )

    def _check_consistency(
        self,
        result: ValidationResult,
        budget: Any,
        z_mean: Any,
        raw_mean: Any,
        norm_factors: Any,
        env: str,
    ) -> None:
        """L1.5: verify stored z matches recomputed z from raw_mean + norm_factors."""
        if not isinstance(z_mean, (int, float)) or not isinstance(raw_mean, (int, float)):
            return
        if math.isnan(z_mean) or math.isinf(z_mean):
            return

        mu, sigma = None, None
        if isinstance(norm_factors, dict):
            mu = norm_factors.get("mu")
            sigma = norm_factors.get("sigma")

        if mu is None or sigma is None:
            nf = self._norm_factors.get(env)
            if nf:
                mu, sigma = nf

        if mu is None or sigma is None or sigma <= 0:
            return

        try:
            recomputed = (float(raw_mean) - float(mu)) / float(sigma)
        except (TypeError, ValueError, ZeroDivisionError):
            return

        diff = abs(float(z_mean) - recomputed)
        if diff > Z_CONSISTENCY_TOLERANCE:
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L1_5_CONSISTENCY,
                    message=f"stored z={z_mean:.4f} != recomputed z={recomputed:.4f} (delta={diff:.4f})",
                    budget=budget,
                    details={"stored": z_mean, "recomputed": recomputed, "delta": diff},
                )
            )

    def _check_hard_invariants(
        self,
        result: ValidationResult,
        budget: Any,
        raw_mean: Any,
        env: str,
    ) -> None:
        if raw_mean is None or not isinstance(raw_mean, (int, float)):
            return
        if math.isnan(raw_mean) or math.isinf(raw_mean):
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L2_HARD_INVARIANTS,
                    message=f"raw_mean is {raw_mean}",
                    budget=budget,
                )
            )
            return

        bounds = PHYSICAL_BOUNDS.get(env)
        if bounds is None:
            return

        lo, hi = bounds
        if not (lo <= raw_mean <= hi):
            result.issues.append(
                ValidationIssue(
                    layer=ValidationLevel.L2_HARD_INVARIANTS,
                    message=f"raw_mean={raw_mean} outside [{lo}, {hi}] for {env}",
                    budget=budget,
                    details={"raw_mean": raw_mean, "bounds": [lo, hi]},
                )
            )
