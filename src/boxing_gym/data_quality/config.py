"""Thresholds and constants for data quality validation."""

from enum import IntEnum

# baseline versioning
BASELINE_VERSION = "v2_2026-01-27"

# z-score threshold: values beyond this are pipeline failures, not bad runs
Z_OUTLIER_THRESHOLD = 100

# robust statistics
SIGMA_FLOOR = 0.01  # minimum sigma to prevent division-by-zero
MAD_MULTIPLIER = 2.0  # flag runs > 2 MAD from median

# L1.5 consistency: max allowed discrepancy between stored and recomputed z
Z_CONSISTENCY_TOLERANCE = 0.01


class ValidationLevel(IntEnum):
    """Validation layers, applied in order."""

    L0_SCHEMA = 0
    L1_NON_FINITE = 10
    L1_5_CONSISTENCY = 15
    L2_HARD_INVARIANTS = 20
    L3_PAPER_BASELINE = 30
    L4_MAD_OUTLIER = 40
    L5_DUPLICATES = 50


# physical bounds on raw_mean per environment
# only hard invariants: MSE >= 0, error_rate in [0, 1]
# large-but-finite z is a valid terrible run, not corruption
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    # MSE-based (lower bound = 0, upper bound = generous max)
    "death_process": (0, 2500),  # max pop 50, so max MSE = 50^2
    "dugongs": (0, 100),  # length ~3m, generous upper
    "lotka_volterra": (0, 1000),  # MAE on population counts
    "location_finding": (0, float("inf")),  # MSE on signal, unbounded
    "peregrines": (0, 1e8),  # MSE on population counts
    # error-rate-based (must be in [0, 1])
    "hyperbolic_temporal_discount": (0, 1),
    "irt": (0, 1),
    "moral_machines": (0, 1),
    "morals": (0, 1),  # alias
    "survival": (0, 1),
    # MAE-based
    "emotion": (0, 8),  # MAE on 1-9 scale
}

# required top-level JSON fields
REQUIRED_FIELDS = ("config", "data")

# required fields inside data section
REQUIRED_DATA_FIELDS = ("z_results",)
