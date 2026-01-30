"""Smoke tests for TUI views."""

from io import StringIO

import pandas as pd
import pytest
from rich.console import Console

from boxing_gym.cli.tui.views.best_configs import BestConfigsView
from boxing_gym.cli.tui.views.budget_progression import BudgetProgressionView
from boxing_gym.cli.tui.views.call_logs import CallLogsView
from boxing_gym.cli.tui.views.heatmap import HeatmapView
from boxing_gym.cli.tui.views.local_summary import LocalSummaryView
from boxing_gym.cli.tui.views.model_rankings import ModelRankingsView
from boxing_gym.cli.tui.views.parameter_importance import ParameterImportanceView
from boxing_gym.cli.tui.views.ppl_examples import PPLExamplesView
from boxing_gym.cli.tui.views.seed_stability import SeedStabilityView


@pytest.fixture
def console():
    return Console(file=StringIO(), force_terminal=True, width=120)


@pytest.fixture
def minimal_df():
    return pd.DataFrame(
        {
            "run_id": ["run_001", "run_002", "run_003"],
            "config/llms": ["gpt-4o", "gpt-4o", "claude-3"],
            "config/envs": ["dugongs", "peregrines", "dugongs"],
            "config/seed": [1, 2, 1],
            "config/budget": [10, 10, 20],
            "metric/eval/z_mean": [-0.5, 0.2, -0.3],
            "summary/z_stderr": [0.1, 0.05, 0.08],
            "summary/n_seeds": [1, 1, 1],
        }
    )


@pytest.fixture
def empty_df():
    return pd.DataFrame()


class TestParameterImportanceView:
    def test_instantiation(self, minimal_df, console):
        view = ParameterImportanceView(minimal_df, console, "metric/eval/z_mean")
        assert view.title == "Parameter Importance"

    def test_render_no_crash(self, minimal_df, console):
        view = ParameterImportanceView(minimal_df, console, "metric/eval/z_mean")
        view.render()

    def test_get_data_returns_dict(self, minimal_df, console):
        view = ParameterImportanceView(minimal_df, console, "metric/eval/z_mean")
        data = view.get_data()
        assert isinstance(data, dict)

    def test_get_csv_rows_returns_list(self, minimal_df, console):
        view = ParameterImportanceView(minimal_df, console, "metric/eval/z_mean")
        rows = view.get_csv_rows()
        assert isinstance(rows, list)

    def test_empty_df_no_crash(self, empty_df, console):
        view = ParameterImportanceView(empty_df, console, "metric/eval/z_mean")
        view.render()
        view.get_data()
        view.get_csv_rows()


class TestModelRankingsView:
    def test_instantiation(self, minimal_df, console):
        view = ModelRankingsView(minimal_df, console, "metric/eval/z_mean")
        assert view.title == "Model Rankings"

    def test_render_no_crash(self, minimal_df, console):
        view = ModelRankingsView(minimal_df, console, "metric/eval/z_mean")
        view.render()

    def test_get_data_returns_dict(self, minimal_df, console):
        view = ModelRankingsView(minimal_df, console, "metric/eval/z_mean")
        data = view.get_data()
        assert isinstance(data, dict)

    def test_get_csv_rows_returns_list(self, minimal_df, console):
        view = ModelRankingsView(minimal_df, console, "metric/eval/z_mean")
        rows = view.get_csv_rows()
        assert isinstance(rows, list)

    def test_empty_df_no_crash(self, empty_df, console):
        view = ModelRankingsView(empty_df, console, "metric/eval/z_mean")
        view.render()
        view.get_data()
        view.get_csv_rows()


class TestHeatmapView:
    def test_instantiation(self, minimal_df, console):
        view = HeatmapView(minimal_df, console, "metric/eval/z_mean")
        assert view.title == "Environment × Model Heatmap"

    def test_render_no_crash(self, minimal_df, console):
        view = HeatmapView(minimal_df, console, "metric/eval/z_mean")
        view.render()

    def test_get_data_returns_dict(self, minimal_df, console):
        view = HeatmapView(minimal_df, console, "metric/eval/z_mean")
        data = view.get_data()
        assert isinstance(data, dict)

    def test_get_csv_rows_returns_list(self, minimal_df, console):
        view = HeatmapView(minimal_df, console, "metric/eval/z_mean")
        rows = view.get_csv_rows()
        assert isinstance(rows, list)

    def test_empty_df_no_crash(self, empty_df, console):
        view = HeatmapView(empty_df, console, "metric/eval/z_mean")
        view.render()
        view.get_data()
        view.get_csv_rows()


class TestBestConfigsView:
    def test_instantiation(self, minimal_df, console):
        view = BestConfigsView(minimal_df, console, "metric/eval/z_mean")
        assert view.title == "Best Configurations"

    def test_render_no_crash(self, minimal_df, console):
        view = BestConfigsView(minimal_df, console, "metric/eval/z_mean")
        view.render()

    def test_get_data_returns_dict(self, minimal_df, console):
        view = BestConfigsView(minimal_df, console, "metric/eval/z_mean")
        data = view.get_data()
        assert isinstance(data, dict)

    def test_get_csv_rows_returns_list(self, minimal_df, console):
        view = BestConfigsView(minimal_df, console, "metric/eval/z_mean")
        rows = view.get_csv_rows()
        assert isinstance(rows, list)

    def test_empty_df_no_crash(self, empty_df, console):
        view = BestConfigsView(empty_df, console, "metric/eval/z_mean")
        view.render()
        view.get_data()
        view.get_csv_rows()


class TestBudgetProgressionView:
    def test_instantiation(self, minimal_df, console):
        view = BudgetProgressionView(minimal_df, console, "metric/eval/z_mean")
        assert view.title == "Budget Progression"

    def test_render_no_crash(self, minimal_df, console):
        view = BudgetProgressionView(minimal_df, console, "metric/eval/z_mean")
        view.render()

    def test_get_data_returns_dict(self, minimal_df, console):
        view = BudgetProgressionView(minimal_df, console, "metric/eval/z_mean")
        data = view.get_data()
        assert isinstance(data, dict)

    def test_get_csv_rows_returns_list(self, minimal_df, console):
        view = BudgetProgressionView(minimal_df, console, "metric/eval/z_mean")
        rows = view.get_csv_rows()
        assert isinstance(rows, list)

    def test_empty_df_no_crash(self, empty_df, console):
        view = BudgetProgressionView(empty_df, console, "metric/eval/z_mean")
        view.render()
        view.get_data()
        view.get_csv_rows()


class TestSeedStabilityView:
    def test_instantiation(self, minimal_df, console):
        view = SeedStabilityView(minimal_df, console, "metric/eval/z_mean")
        assert view.title == "Seed Stability Diagnostics"

    def test_render_no_crash(self, minimal_df, console):
        view = SeedStabilityView(minimal_df, console, "metric/eval/z_mean")
        view.render()

    def test_get_data_returns_dict(self, minimal_df, console):
        view = SeedStabilityView(minimal_df, console, "metric/eval/z_mean")
        data = view.get_data()
        assert isinstance(data, dict)

    def test_get_csv_rows_returns_list(self, minimal_df, console):
        view = SeedStabilityView(minimal_df, console, "metric/eval/z_mean")
        rows = view.get_csv_rows()
        assert isinstance(rows, list)

    def test_empty_df_no_crash(self, empty_df, console):
        view = SeedStabilityView(empty_df, console, "metric/eval/z_mean")
        view.render()
        view.get_data()
        view.get_csv_rows()


class TestLocalSummaryView:
    def test_instantiation(self, minimal_df, console):
        view = LocalSummaryView(minimal_df, console, "metric/eval/z_mean")
        assert view.title == "Local Results Summary"

    def test_render_no_crash(self, minimal_df, console):
        view = LocalSummaryView(minimal_df, console, "metric/eval/z_mean")
        view.render()

    def test_get_data_returns_dict(self, minimal_df, console):
        view = LocalSummaryView(minimal_df, console, "metric/eval/z_mean")
        data = view.get_data()
        assert isinstance(data, dict)

    def test_get_csv_rows_returns_list(self, minimal_df, console):
        view = LocalSummaryView(minimal_df, console, "metric/eval/z_mean")
        rows = view.get_csv_rows()
        assert isinstance(rows, list)

    def test_sparse_data_no_crash(self, console):
        # LocalSummaryView requires run_id + config columns; test with minimal valid data
        sparse_df = pd.DataFrame(
            {
                "run_id": ["run_001"],
                "config/llms": ["gpt-4o"],
                "config/envs": ["dugongs"],
                "metric/eval/z_mean": [0.0],
                "summary/n_seeds": [1],
            }
        )
        view = LocalSummaryView(sparse_df, console, "metric/eval/z_mean")
        view.render()
        view.get_data()
        view.get_csv_rows()


class TestCallLogsView:
    def test_instantiation(self, console):
        view = CallLogsView(console, "metric/eval/z_mean")
        assert view.title == "LLM Call Logs"

    def test_render_no_crash(self, console):
        view = CallLogsView(console, "metric/eval/z_mean")
        view.render(interactive=False)

    def test_get_data_returns_dict(self, console):
        view = CallLogsView(console, "metric/eval/z_mean")
        data = view.get_data()
        assert isinstance(data, dict)

    def test_get_csv_rows_returns_list(self, console):
        view = CallLogsView(console, "metric/eval/z_mean")
        rows = view.get_csv_rows()
        assert isinstance(rows, list)


class TestPPLExamplesView:
    def test_instantiation(self, console):
        view = PPLExamplesView(console, "metric/eval/z_mean")
        assert view.title == "PPL Examples"

    def test_render_no_crash(self, console):
        view = PPLExamplesView(console, "metric/eval/z_mean")
        view.render()

    def test_get_data_returns_dict(self, console):
        view = PPLExamplesView(console, "metric/eval/z_mean")
        data = view.get_data()
        assert isinstance(data, dict)

    def test_get_csv_rows_returns_list(self, console):
        view = PPLExamplesView(console, "metric/eval/z_mean")
        rows = view.get_csv_rows()
        assert isinstance(rows, list)


# ============================================================================
# ASCII Chart Utilities Tests
# ============================================================================

from boxing_gym.cli.tui.components.ascii_charts import (
    colored_z,
    horizontal_bar,
    short_model_name,
    sparkline,
    trend_indicator,
    z_color,
)


class TestHorizontalBar:
    """Tests for horizontal_bar() NaN/inf guards and normal behavior."""

    def test_normal_case(self):
        # 50% filled
        result = horizontal_bar(5, 10, width=10)
        assert result == "█████░░░░░"

    def test_full_bar(self):
        result = horizontal_bar(10, 10, width=10)
        assert result == "██████████"

    def test_empty_bar(self):
        result = horizontal_bar(0, 10, width=10)
        assert result == "░░░░░░░░░░"

    def test_nan_value_returns_empty(self):
        result = horizontal_bar(float("nan"), 10, width=10)
        assert result == "░░░░░░░░░░"

    def test_inf_value_returns_empty(self):
        result = horizontal_bar(float("inf"), 10, width=10)
        assert result == "░░░░░░░░░░"

    def test_neg_inf_value_returns_empty(self):
        result = horizontal_bar(float("-inf"), 10, width=10)
        assert result == "░░░░░░░░░░"

    def test_nan_max_value_returns_empty(self):
        result = horizontal_bar(5, float("nan"), width=10)
        assert result == "░░░░░░░░░░"

    def test_inf_max_value_returns_empty(self):
        result = horizontal_bar(5, float("inf"), width=10)
        assert result == "░░░░░░░░░░"

    def test_zero_max_value_returns_empty(self):
        result = horizontal_bar(5, 0, width=10)
        assert result == "░░░░░░░░░░"

    def test_negative_max_value_returns_empty(self):
        result = horizontal_bar(5, -10, width=10)
        assert result == "░░░░░░░░░░"

    def test_negative_value_clamps_to_zero(self):
        # negative value should produce empty bar (clamped to 0)
        result = horizontal_bar(-5, 10, width=10)
        assert result == "░░░░░░░░░░"

    def test_value_exceeds_max_clamps_to_full(self):
        result = horizontal_bar(15, 10, width=10)
        assert result == "██████████"

    def test_custom_width(self):
        result = horizontal_bar(5, 10, width=20)
        assert len(result) == 20
        assert result.count("█") == 10


class TestColoredZ:
    """Tests for colored_z() NaN guard and color markup."""

    def test_nan_returns_dim_nan(self):
        result = colored_z(float("nan"))
        assert result == "[dim]NaN[/dim]"

    def test_negative_z_green(self):
        result = colored_z(-0.5)
        assert "[green]" in result
        assert "-0.500" in result

    def test_positive_z_red(self):
        result = colored_z(0.5)
        assert "[red]" in result
        assert "+0.500" in result

    def test_near_zero_yellow(self):
        result = colored_z(0.0)
        assert "[yellow]" in result
        assert "+0.000" in result

    def test_custom_thresholds(self):
        # z=-0.2 with low=-0.1 should be green (below threshold)
        result = colored_z(-0.2, low=-0.1, high=0.1)
        assert "[green]" in result


class TestZColor:
    """Tests for z_color() threshold logic."""

    def test_below_low_is_green(self):
        assert z_color(-0.5) == "green"

    def test_above_high_is_red(self):
        assert z_color(0.5) == "red"

    def test_in_range_is_yellow(self):
        assert z_color(0.0) == "yellow"

    def test_at_low_boundary(self):
        # exactly at -0.3 should be yellow (not < low)
        assert z_color(-0.3) == "yellow"

    def test_at_high_boundary(self):
        # exactly at 0.3 should be yellow (not > high)
        assert z_color(0.3) == "yellow"


class TestShortModelName:
    """Tests for short_model_name() truncation and abbreviation logic."""

    def test_empty_string(self):
        assert short_model_name("") == "???"

    def test_none_like_empty(self):
        assert short_model_name("") == "???"

    def test_provider_prefix_stripped(self):
        assert short_model_name("openai/gpt-4o") == "gpt-4o"

    def test_nested_slashes(self):
        # rsplit takes last component
        assert short_model_name("provider/org/model") == "model"

    def test_deepseek_shortening(self):
        result = short_model_name("deepseek-chat")
        assert result == "ds-chat"

    def test_codex_mini_shortening(self):
        # "codex-mini" → "codex", but result is still 13 chars so gets truncated
        result = short_model_name("gpt-5.1-codex-mini")
        assert "codex" in result or result.endswith("…")  # truncated or shortened

    def test_codex_mini_shortening_with_enough_space(self):
        # with longer max_len, the full replacement is visible
        result = short_model_name("gpt-5.1-codex-mini", max_len=20)
        assert result == "gpt-5.1-codex"

    def test_qwen_normalization(self):
        result = short_model_name("qwen3-32b-v1:0")
        assert result == "qwen3-32b"

    def test_box_suffix_short_name(self):
        result = short_model_name("gpt-4o (box)")
        assert result == "gpt-4o†"

    def test_box_suffix_long_name_truncates(self):
        # "very-long-model (box)" should truncate with …†
        result = short_model_name("very-long-model (box)", max_len=12)
        assert len(result) <= 12
        assert result.endswith("†")
        assert "…" in result

    def test_truncation_without_box(self):
        result = short_model_name("a-very-long-model-name", max_len=12)
        assert len(result) == 12
        assert result.endswith("…")

    def test_exact_max_len_no_truncation(self):
        # 12 chars exactly should not truncate
        result = short_model_name("exactly12chr", max_len=12)
        assert result == "exactly12chr"

    def test_custom_max_len(self):
        result = short_model_name("very-long-model-name", max_len=8)
        assert len(result) == 8


class TestTrendIndicator:
    """Tests for trend_indicator() arrow logic."""

    def test_big_improvement(self):
        result = trend_indicator(1.0, 0.5, threshold=0.1)  # diff=-0.5
        assert "▼▼" in result
        assert "green" in result

    def test_slight_improvement(self):
        result = trend_indicator(1.0, 0.85, threshold=0.1)  # diff=-0.15
        assert "▼" in result
        assert "▼▼" not in result

    def test_stable(self):
        result = trend_indicator(1.0, 1.05, threshold=0.1)  # diff=0.05
        assert "─" in result

    def test_slight_worsening(self):
        result = trend_indicator(1.0, 1.15, threshold=0.1)  # diff=0.15
        assert "▲" in result
        assert "▲▲" not in result

    def test_big_worsening(self):
        result = trend_indicator(1.0, 1.5, threshold=0.1)  # diff=0.5
        assert "▲▲" in result
        assert "red" in result


class TestSparkline:
    """Tests for sparkline() edge cases."""

    def test_empty_list(self):
        result = sparkline([])
        assert result == "──────────"

    def test_single_value(self):
        result = sparkline([5])
        # uniform values use middle block
        assert result == "▄"

    def test_all_same_value(self):
        result = sparkline([5, 5, 5, 5, 5])
        assert result == "▄▄▄▄▄"

    def test_ascending_values(self):
        result = sparkline([1, 2, 3, 4, 5])
        # should show progression from low to high blocks
        assert len(result) == 5
        # first should be lower than last
        assert result[0] < result[-1] or result[0] == " "

    def test_longer_than_width_samples(self):
        values = list(range(100))
        result = sparkline(values, width=10)
        assert len(result) == 10


class TestModelRankingsSingleModel:
    """Test model_rankings.py guard for single-model case."""

    def test_single_model_no_crash(self, console):
        single_model_df = pd.DataFrame(
            {
                "run_id": ["run_001", "run_002", "run_003"],
                "config/llms": ["gpt-4o", "gpt-4o", "gpt-4o"],  # only one model
                "config/envs": ["dugongs", "peregrines", "dugongs"],
                "config/seed": [1, 2, 3],
                "metric/eval/z_mean": [-0.5, 0.2, -0.3],
                "summary/z_stderr": [0.1, 0.05, 0.08],
            }
        )
        view = ModelRankingsView(single_model_df, console, "metric/eval/z_mean")
        view.render()  # should not crash
        data = view.get_data()
        # single model means no comparisons, rankings should have 1 entry
        assert "rankings" in data


# ============================================================================
# Quality Filter Tests
# ============================================================================


from boxing_gym.cli.quality_filter import apply_quality_filters
from boxing_gym.data_quality.config import Z_OUTLIER_THRESHOLD


class TestApplyQualityFilters:
    """Tests for apply_quality_filters() correctness."""

    def test_missing_z_col_returns_copy(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = apply_quality_filters(df, z_col="z_mean")
        assert len(result) == 3
        # verify it's a copy, not the same object
        assert result is not df

    def test_nan_z_dropped_by_default(self):
        df = pd.DataFrame({"z_mean": [1.0, float("nan"), 2.0]})
        result = apply_quality_filters(df, z_col="z_mean")
        assert len(result) == 2
        assert result["z_mean"].notna().all()

    def test_nan_z_kept_when_drop_nan_false(self):
        df = pd.DataFrame({"z_mean": [1.0, float("nan"), 2.0]})
        result = apply_quality_filters(df, z_col="z_mean", drop_nan=False)
        assert len(result) == 3

    def test_inf_z_dropped_by_default(self):
        df = pd.DataFrame({"z_mean": [1.0, float("inf"), float("-inf"), 2.0]})
        result = apply_quality_filters(df, z_col="z_mean")
        assert len(result) == 2

    def test_inf_z_kept_when_drop_inf_false(self):
        df = pd.DataFrame({"z_mean": [1.0, float("inf"), 2.0]})
        result = apply_quality_filters(df, z_col="z_mean", drop_inf=False)
        assert len(result) == 3

    def test_z_above_threshold_dropped(self):
        df = pd.DataFrame({"z_mean": [1.0, 150.0, -200.0, 2.0]})
        result = apply_quality_filters(df, z_col="z_mean", z_threshold=100)
        assert len(result) == 2
        assert (result["z_mean"].abs() <= 100).all()

    def test_z_at_threshold_kept(self):
        df = pd.DataFrame({"z_mean": [100.0, -100.0, 50.0]})
        result = apply_quality_filters(df, z_col="z_mean", z_threshold=100)
        assert len(result) == 3  # exactly at threshold is OK

    def test_default_threshold_matches_config(self):
        assert Z_OUTLIER_THRESHOLD == 100
        df = pd.DataFrame({"z_mean": [99.0, 100.0, 101.0]})
        result = apply_quality_filters(df, z_col="z_mean")
        # 99 and 100 kept, 101 dropped
        assert len(result) == 2

    def test_min_budget_filter(self):
        df = pd.DataFrame(
            {
                "z_mean": [1.0, 2.0, 3.0],
                "config/budget": [5, 10, 15],
            }
        )
        result = apply_quality_filters(df, z_col="z_mean", min_budget=10)
        assert len(result) == 2
        assert (result["config/budget"] >= 10).all()

    def test_min_budget_col_missing_ignored(self):
        df = pd.DataFrame({"z_mean": [1.0, 2.0, 3.0]})
        result = apply_quality_filters(df, z_col="z_mean", min_budget=10)
        # budget col missing, filter not applied
        assert len(result) == 3

    def test_combined_filters(self):
        df = pd.DataFrame(
            {
                "z_mean": [1.0, float("nan"), float("inf"), 150.0, 2.0],
                "config/budget": [5, 10, 10, 10, 15],
            }
        )
        result = apply_quality_filters(df, z_col="z_mean", z_threshold=100, min_budget=10)
        # row 0: z=1.0, budget=5 → dropped (budget)
        # row 1: nan → dropped
        # row 2: inf → dropped
        # row 3: z=150 → dropped (threshold)
        # row 4: z=2.0, budget=15 → kept
        assert len(result) == 1
        assert result["z_mean"].iloc[0] == 2.0

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"z_mean": []})
        result = apply_quality_filters(df, z_col="z_mean")
        assert len(result) == 0

    def test_returns_copy_original_unchanged(self):
        df = pd.DataFrame({"z_mean": [1.0, float("nan"), 2.0]})
        original_len = len(df)
        result = apply_quality_filters(df, z_col="z_mean")
        # original unchanged
        assert len(df) == original_len
        # result is filtered
        assert len(result) == 2

    def test_string_z_coerced(self):
        df = pd.DataFrame({"z_mean": ["1.0", "bad", "2.0"]})
        result = apply_quality_filters(df, z_col="z_mean")
        # "bad" becomes NaN and is dropped
        assert len(result) == 2
