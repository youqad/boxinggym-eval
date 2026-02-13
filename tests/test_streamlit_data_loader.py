import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _install_streamlit_stub():
    """Install a tiny streamlit stub so tests don't depend on streamlit importability."""
    if "streamlit" in sys.modules:
        return

    def cache_data(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    sys.modules["streamlit"] = SimpleNamespace(cache_data=cache_data)


def load_data_loader_module():
    _install_streamlit_stub()
    module_path = Path("scripts/streamlit_app/utils/data_loader.py").resolve()
    spec = importlib.util.spec_from_file_location("streamlit_data_loader_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_results_path_prefers_canonical_in_demo_mode(tmp_path: Path):
    module = load_data_loader_module()

    project_root = tmp_path / "repo"
    demo_dir = project_root / "scripts" / "streamlit_app" / "demo_data"
    cache_dir = project_root / ".boxing-gym-cache"
    demo_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    (demo_dir / "canonical_runs.parquet").write_bytes(b"canon")
    (demo_dir / "demo_runs.parquet").write_bytes(b"demo")
    (cache_dir / "runs.parquet").write_bytes(b"cache")

    path, source = module.resolve_results_path(
        parquet_path=None,
        project_root=project_root,
        demo_data_dir=demo_dir,
        demo_mode=True,
    )

    assert path == demo_dir / "canonical_runs.parquet"
    assert source == "canonical_snapshot"


def test_resolve_results_path_prefers_local_cache_outside_demo(tmp_path: Path):
    module = load_data_loader_module()

    project_root = tmp_path / "repo"
    demo_dir = project_root / "scripts" / "streamlit_app" / "demo_data"
    cache_dir = project_root / ".boxing-gym-cache"
    demo_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    (demo_dir / "canonical_runs.parquet").write_bytes(b"canon")
    (cache_dir / "runs.parquet").write_bytes(b"cache")

    path, source = module.resolve_results_path(
        parquet_path=None,
        project_root=project_root,
        demo_data_dir=demo_dir,
        demo_mode=False,
    )

    assert path == cache_dir / "runs.parquet"
    assert source == "local_cache"


def test_resolve_results_path_honors_explicit_override(tmp_path: Path):
    module = load_data_loader_module()

    explicit = tmp_path / "explicit.parquet"
    explicit.write_bytes(b"x")

    path, source = module.resolve_results_path(parquet_path=str(explicit))
    assert path == explicit
    assert source == "explicit_path"


def test_resolve_results_path_demo_falls_back_to_bundled_demo(tmp_path: Path):
    module = load_data_loader_module()

    project_root = tmp_path / "repo"
    demo_dir = project_root / "scripts" / "streamlit_app" / "demo_data"
    cache_dir = project_root / ".boxing-gym-cache"
    demo_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    (demo_dir / "demo_runs.parquet").write_bytes(b"demo")
    (cache_dir / "runs.parquet").write_bytes(b"cache")

    path, source = module.resolve_results_path(
        parquet_path=None,
        project_root=project_root,
        demo_data_dir=demo_dir,
        demo_mode=True,
    )

    assert path == demo_dir / "demo_runs.parquet"
    assert source == "bundled_demo_fallback"


def test_resolve_results_path_returns_none_for_missing_explicit_file(tmp_path: Path):
    module = load_data_loader_module()
    missing = tmp_path / "missing.parquet"
    path, source = module.resolve_results_path(parquet_path=str(missing))
    assert path is None
    assert source == "none"


def test_get_active_results_source_passthrough(monkeypatch):
    module = load_data_loader_module()

    def fake_resolve_results_path(parquet_path=None, **_kwargs):
        assert parquet_path == "x.parquet"
        return Path("/tmp/x.parquet"), "explicit_path"

    monkeypatch.setattr(module, "resolve_results_path", fake_resolve_results_path)
    assert module.get_active_results_source("x.parquet") == "explicit_path"


def test_filter_to_default_goals_only_keeps_matching_default_goal():
    module = load_data_loader_module()
    df = pd.DataFrame(
        [
            {"env": "dugongs", "goal": "length", "z_mean": 0.1},
            {"env": "dugongs", "goal": "population", "z_mean": 0.2},
            {"env": "irt", "goal": "correctness", "z_mean": 0.3},
        ]
    )
    filtered = module.filter_to_default_goals(df)
    assert len(filtered) == 2
    assert set(filtered["goal"].tolist()) == {"length", "correctness"}
