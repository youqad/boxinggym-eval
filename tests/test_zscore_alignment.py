import importlib
import json
import sys
import types
from pathlib import Path

from boxing_gym.agents.results_io import load_results_from_json_dir


def _ensure_experiment_package():
    # Avoid importing boxing_gym.experiment.__init__ (pulls in heavy deps).
    import boxing_gym  # noqa: F401

    pkg_name = "boxing_gym.experiment"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [
            str(Path(__file__).resolve().parents[1] / "src" / "boxing_gym" / "experiment")
        ]
        sys.modules[pkg_name] = pkg

    # Stub LMExperimenter to avoid importing litellm in tests.
    agent_mod = "boxing_gym.agents.agent"
    if agent_mod not in sys.modules:
        stub_agent = types.ModuleType(agent_mod)

        class LMExperimenter:  # noqa: N801 - matches class name used in prod
            pass

        stub_agent.LMExperimenter = LMExperimenter
        sys.modules[agent_mod] = stub_agent


def _load_compute_z_results():
    _ensure_experiment_package()
    module = importlib.import_module("boxing_gym.experiment.run_helpers")
    return module.compute_z_results


def _load_build_eval_payload():
    _ensure_experiment_package()
    module = importlib.import_module("boxing_gym.experiment.wandb_logging")
    return module._build_eval_payload


def test_load_results_prefers_budget_keyed_z_results_over_index(tmp_path):
    blob = {
        "config": {
            "envs": {"env_name": "dugongs", "goal_name": "direct"},
            "exp": {"experiment_type": "discovery", "num_experiments": [0, 5, 10]},
            "include_prior": True,
            "llms": {"model_name": "unit-test"},
            "seed": 1,
            "use_ppl": False,
        },
        "data": {
            "results": [
                [[0.0, 0.1], []],
                [[5.0, 0.2], []],
                [[10.0, 0.3], []],
            ],
            "z_results": [
                {"budget": 0, "z_mean": 42.0, "z_std": 0.01},
                {"budget": 10, "z_mean": 100.0, "z_std": 0.02},
            ],
            "norm_factors": {"mu": 0.0, "sigma": 1.0},
        },
    }

    (tmp_path / "one.json").write_text(json.dumps(blob))

    results = load_results_from_json_dir(str(tmp_path))
    by_budget = {r.budget: r for r in results}

    assert by_budget[0].z_mean == 42.0
    assert by_budget[5].z_mean == 5.0
    assert by_budget[10].z_mean == 100.0


def test_compute_z_results_handles_dict_scores():
    compute_z_results = _load_compute_z_results()
    all_data = [
        [
            [{"accuracy": 0.8, "std_accuracy": 0.2}, []],
            [{"mse": 2.0, "std_mse": 0.6}, []],
            [{"score": 3.0, "std": 0.4}, []],
        ]
    ]

    z_results = compute_z_results(
        all_data=all_data,
        num_experiments=[0, 5, 10],
        norm_mu=1.0,
        norm_sigma=2.0,
        env_name="moral",
        goal_name="direct",
    )

    by_budget = {row["budget"]: row for row in z_results}

    assert by_budget[0]["raw_mean"] == 0.8
    assert by_budget[0]["raw_std"] == 0.2
    assert by_budget[0]["z_mean"] == (0.8 - 1.0) / 2.0
    assert by_budget[0]["z_std"] == 0.2 / 2.0

    assert by_budget[5]["raw_mean"] == 2.0
    assert by_budget[5]["raw_std"] == 0.6
    assert by_budget[5]["z_mean"] == (2.0 - 1.0) / 2.0
    assert by_budget[5]["z_std"] == 0.6 / 2.0

    assert by_budget[10]["raw_mean"] == 3.0
    assert by_budget[10]["raw_std"] == 0.4
    assert by_budget[10]["z_mean"] == (3.0 - 1.0) / 2.0
    assert by_budget[10]["z_std"] == 0.4 / 2.0


def test_build_eval_payload_includes_z_mean_for_dict_scores():
    build_eval_payload = _load_build_eval_payload()
    result_entry = [{"accuracy": 0.8, "std_accuracy": 0.2}, []]
    z_by_budget = {5: {"budget": 5, "z_mean": -0.1, "z_std": 0.1}}

    payload = build_eval_payload(result_entry, 5, z_by_budget)

    assert payload["eval/z_mean"] == -0.1
    assert payload["eval/z_std"] == 0.1
    assert payload["eval/mean"] == 0.8
    assert payload["eval/std"] == 0.2


def test_compute_z_results_handles_tuple_scores():
    compute_z_results = _load_compute_z_results()
    all_data = [[[[2.5, 0.5], []]]]

    z_results = compute_z_results(
        all_data=all_data,
        num_experiments=[3],
        norm_mu=1.0,
        norm_sigma=2.0,
        env_name="dugongs",
        goal_name="direct",
    )

    assert z_results[0]["budget"] == 3
    assert z_results[0]["raw_mean"] == 2.5
    assert z_results[0]["raw_std"] == 0.5
    assert z_results[0]["z_mean"] == (2.5 - 1.0) / 2.0
    assert z_results[0]["z_std"] == 0.5 / 2.0


def test_compute_z_results_applies_location_finding_transform():
    compute_z_results = _load_compute_z_results()
    all_data = [[[[20000.0, 1.0], []]]]

    z_results = compute_z_results(
        all_data=all_data,
        num_experiments=[10],
        norm_mu=0.0,
        norm_sigma=100.0,
        env_name="location_finding",
        goal_name="direct",
    )

    assert z_results[0]["raw_mean"] == 10000.0
    assert z_results[0]["raw_std"] == 1.0
    assert z_results[0]["z_mean"] == 100.0
    assert z_results[0]["z_std"] == 0.01


def test_build_summary_data_includes_z_std():
    _ensure_experiment_package()
    module = importlib.import_module("boxing_gym.experiment.wandb_logging")

    all_data = [
        [[{"mse": 1.0, "std_mse": 0.2}, []]],
        [],
        [],
        [],
        [],
        [],
    ]
    z_by_budget = {10: {"budget": 10, "z_mean": -0.1, "z_std": 0.1}}

    summary = module._build_summary_data(
        all_data=all_data,
        z_by_budget=z_by_budget,
        num_experiments=[10],
        env_name="dugongs",
        goal_name="direct",
        experiment_type="discovery",
        include_prior=True,
        seed=1,
        use_ppl=False,
        scientist_agent=None,
        naive_agent=None,
        start_time=0.0,
    )

    assert summary["eval/z_std_final"] == 0.1
    assert summary["eval/z_std"] == 0.1
