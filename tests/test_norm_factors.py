"""Tests for normalization factor plausibility.

Verifies that hardcoded norm_mu/norm_sigma values in each environment
are physically plausible and consistent with the metric type.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from boxing_gym.data_quality.config import PHYSICAL_BOUNDS

# (env_name, goal_key, expected_mu_range, expected_sigma_range)
NORM_EXPECTATIONS = [
    # MSE-based: mu >= 0, sigma > 0
    ("death_process", "direct", (0, 2500), (0.01, 5000)),
    ("dugongs", "direct", (0, 100), (0.01, 200)),
    ("lotka_volterra", "direct", (0, 1000), (0.01, 2000)),
    ("peregrines", "direct", (0, 1e8), (0.01, 1e8)),
    ("location_finding", "direct", (0, 1e6), (0.01, 1e7)),
    # error-rate-based: mu in [0, 1], sigma in (0, 1]
    ("hyperbolic_temporal_discount", "direct", (0, 1), (0.01, 1)),
    ("irt", "direct", (0, 1), (0.01, 1)),
    ("morals", "direct", (0, 1), (0.01, 1)),
    ("survival", "direct", (0, 1), (0.01, 1)),
    # MAE-based: mu in [0, 8] (1-9 scale)
    ("emotion", "direct", (0, 8), (0.01, 10)),
]


def _get_norm_values(env_name: str, goal_key: str):
    """Load norm_mu, norm_sigma from the Goal class."""
    from boxing_gym.envs.registry import get_environment_registry

    nametoenv, nameenvtogoal = get_environment_registry()
    env_cls = nametoenv.get(env_name)
    goal_cls = nameenvtogoal.get((env_name, goal_key))

    if env_cls is None or goal_cls is None:
        pytest.skip(f"env/goal not found: {env_name}/{goal_key}")

    env = env_cls()
    goal = goal_cls(env)
    return goal.norm_mu, goal.norm_sigma


@pytest.mark.parametrize(
    "env_name,goal_key,mu_range,sigma_range",
    NORM_EXPECTATIONS,
    ids=[f"{e[0]}" for e in NORM_EXPECTATIONS],
)
def test_norm_mu_in_range(env_name, goal_key, mu_range, sigma_range):
    mu, sigma = _get_norm_values(env_name, goal_key)

    lo, hi = mu_range
    assert mu is not None, f"{env_name}: norm_mu is None"
    assert math.isfinite(mu), f"{env_name}: norm_mu={mu} is not finite"
    assert lo <= mu <= hi, f"{env_name}: norm_mu={mu} outside [{lo}, {hi}]"


@pytest.mark.parametrize(
    "env_name,goal_key,mu_range,sigma_range",
    NORM_EXPECTATIONS,
    ids=[f"{e[0]}" for e in NORM_EXPECTATIONS],
)
def test_norm_sigma_positive(env_name, goal_key, mu_range, sigma_range):
    mu, sigma = _get_norm_values(env_name, goal_key)

    lo, hi = sigma_range
    assert sigma is not None, f"{env_name}: norm_sigma is None"
    assert math.isfinite(sigma), f"{env_name}: norm_sigma={sigma} is not finite"
    assert sigma > 0, f"{env_name}: norm_sigma={sigma} must be positive"
    assert lo <= sigma <= hi, f"{env_name}: norm_sigma={sigma} outside [{lo}, {hi}]"


def test_physical_bounds_cover_all_envs():
    """Every env with norm expectations should have physical bounds defined."""
    for env_name, _, _, _ in NORM_EXPECTATIONS:
        assert env_name in PHYSICAL_BOUNDS, (
            f"{env_name} missing from PHYSICAL_BOUNDS in data_quality/config.py"
        )


def test_norm_static_matches_goal_classes():
    """NORM_STATIC in results_io should match current Goal class values."""
    try:
        from boxing_gym.agents.results_io import NORM_STATIC
    except ImportError:
        pytest.skip("results_io not importable")

    from boxing_gym.envs.registry import get_environment_registry

    nametoenv, nameenvtogoal = get_environment_registry()

    for env_name in NORM_STATIC:
        goal_cls = nameenvtogoal.get((env_name, "direct"))
        if goal_cls is None:
            continue

        env_cls = nametoenv.get(env_name)
        if env_cls is None:
            continue

        env = env_cls()
        goal = goal_cls(env)
        static_mu, static_sigma = NORM_STATIC[env_name]

        # allow some tolerance for float differences
        assert abs(goal.norm_mu - static_mu) < 0.1, (
            f"{env_name}: Goal.norm_mu={goal.norm_mu} != NORM_STATIC mu={static_mu}"
        )
        # note: this test will fail after updating env files but before updating NORM_STATIC
        # that's intentional — it catches desync between the two sources
