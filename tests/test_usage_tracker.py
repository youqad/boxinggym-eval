"""Tests for UsageTrackerMixin."""

import pytest
from boxing_gym.agents.usage_tracker import UsageTrackerMixin


class ConcreteTracker(UsageTrackerMixin):
    """Concrete class for testing the mixin."""

    def __init__(self):
        self._init_usage_tracking()


class TestUsageTrackerInit:
    """Tests for mixin initialization."""

    def test_init_creates_usage_stats_dict(self):
        tracker = ConcreteTracker()
        assert hasattr(tracker, "_usage_stats")
        assert isinstance(tracker._usage_stats, dict)

    def test_init_sets_all_fields_to_zero(self):
        tracker = ConcreteTracker()
        assert tracker._usage_stats["prompt_tokens"] == 0
        assert tracker._usage_stats["completion_tokens"] == 0
        assert tracker._usage_stats["reasoning_tokens"] == 0
        assert tracker._usage_stats["total_tokens"] == 0
        assert tracker._usage_stats["total_cost_usd"] == 0.0
        assert tracker._usage_stats["call_count"] == 0
        assert tracker._usage_stats["latencies_ms"] == []
        assert tracker._usage_stats["retry_count"] == 0
        assert tracker._usage_stats["error_count"] == 0


class TestRecordUsage:
    """Tests for _record_usage method."""

    def test_record_usage_accumulates_tokens(self):
        tracker = ConcreteTracker()
        tracker._record_usage(prompt_tokens=100, completion_tokens=50)
        tracker._record_usage(prompt_tokens=200, completion_tokens=75)

        assert tracker._usage_stats["prompt_tokens"] == 300
        assert tracker._usage_stats["completion_tokens"] == 125
        assert tracker._usage_stats["total_tokens"] == 425

    def test_record_usage_tracks_reasoning_tokens(self):
        tracker = ConcreteTracker()
        tracker._record_usage(prompt_tokens=100, completion_tokens=50, reasoning_tokens=200)

        assert tracker._usage_stats["reasoning_tokens"] == 200
        assert tracker._usage_stats["total_tokens"] == 350  # 100+50+200

    def test_record_usage_accumulates_cost(self):
        tracker = ConcreteTracker()
        tracker._record_usage(cost_usd=0.05)
        tracker._record_usage(cost_usd=0.03)

        assert tracker._usage_stats["total_cost_usd"] == pytest.approx(0.08)

    def test_record_usage_increments_call_count(self):
        tracker = ConcreteTracker()
        tracker._record_usage()
        tracker._record_usage()
        tracker._record_usage()

        assert tracker._usage_stats["call_count"] == 3

    def test_record_usage_appends_latencies(self):
        tracker = ConcreteTracker()
        tracker._record_usage(latency_ms=100.5)
        tracker._record_usage(latency_ms=200.3)

        assert tracker._usage_stats["latencies_ms"] == [100.5, 200.3]


class TestRecordRetryAndError:
    """Tests for retry and error tracking."""

    def test_record_retry_increments_count(self):
        tracker = ConcreteTracker()
        tracker._record_retry()
        tracker._record_retry()
        tracker._record_retry()

        assert tracker._usage_stats["retry_count"] == 3

    def test_record_retry_with_count(self):
        tracker = ConcreteTracker()
        tracker._record_retry(5)
        tracker._record_retry(3)

        assert tracker._usage_stats["retry_count"] == 8

    def test_record_error_increments_count(self):
        tracker = ConcreteTracker()
        tracker._record_error()
        tracker._record_error()

        assert tracker._usage_stats["error_count"] == 2


class TestGetUsageStats:
    """Tests for get_usage_stats method."""

    def test_returns_all_fields(self):
        tracker = ConcreteTracker()
        tracker._record_usage(prompt_tokens=100, completion_tokens=50, cost_usd=0.01, latency_ms=150)

        stats = tracker.get_usage_stats()

        assert "prompt_tokens" in stats
        assert "completion_tokens" in stats
        assert "total_tokens" in stats
        assert "total_cost_usd" in stats
        assert "call_count" in stats
        assert "latency_mean_ms" in stats
        assert "latency_p50_ms" in stats
        assert "latency_p95_ms" in stats

    def test_calculates_latency_percentiles(self):
        tracker = ConcreteTracker()
        # add 25 latencies to get meaningful percentiles
        for i in range(1, 26):
            tracker._record_usage(latency_ms=float(i * 10))

        stats = tracker.get_usage_stats()

        assert stats["latency_min_ms"] == 10.0
        assert stats["latency_max_ms"] == 250.0
        assert stats["latency_mean_ms"] == 130.0  # mean of 10,20,...,250
        assert stats["latency_p50_ms"] == 130.0  # median
        assert stats["latency_p95_ms"] == 240.0  # 95th percentile

    def test_handles_empty_latencies(self):
        tracker = ConcreteTracker()
        stats = tracker.get_usage_stats()

        assert stats["latency_mean_ms"] == 0.0
        assert stats["latency_p50_ms"] == 0.0
        assert stats["latency_p95_ms"] == 0.0

    def test_does_not_modify_internal_state(self):
        tracker = ConcreteTracker()
        tracker._record_usage(latency_ms=100)

        # call twice
        stats1 = tracker.get_usage_stats()
        stats2 = tracker.get_usage_stats()

        # internal state should be preserved
        assert tracker._usage_stats["latencies_ms"] == [100]


class TestGetLatenciesMs:
    """Tests for get_latencies_ms method."""

    def test_returns_copy_of_latencies(self):
        tracker = ConcreteTracker()
        tracker._record_usage(latency_ms=100)
        tracker._record_usage(latency_ms=200)

        latencies = tracker.get_latencies_ms()

        assert latencies == [100, 200]

    def test_returns_independent_copy(self):
        tracker = ConcreteTracker()
        tracker._record_usage(latency_ms=100)

        latencies = tracker.get_latencies_ms()
        latencies.append(999)  # modify returned list

        # internal state should not change
        assert tracker._usage_stats["latencies_ms"] == [100]


class TestResetUsageStats:
    """Tests for reset_usage_stats method."""

    def test_resets_all_fields(self):
        tracker = ConcreteTracker()
        tracker._record_usage(
            prompt_tokens=100,
            completion_tokens=50,
            reasoning_tokens=25,
            cost_usd=0.05,
            latency_ms=150
        )
        tracker._record_retry()
        tracker._record_error()

        tracker.reset_usage_stats()

        assert tracker._usage_stats["prompt_tokens"] == 0
        assert tracker._usage_stats["completion_tokens"] == 0
        assert tracker._usage_stats["reasoning_tokens"] == 0
        assert tracker._usage_stats["total_tokens"] == 0
        assert tracker._usage_stats["total_cost_usd"] == 0.0
        assert tracker._usage_stats["call_count"] == 0
        assert tracker._usage_stats["latencies_ms"] == []
        assert tracker._usage_stats["retry_count"] == 0
        assert tracker._usage_stats["error_count"] == 0

    def test_allows_new_recording_after_reset(self):
        tracker = ConcreteTracker()
        tracker._record_usage(prompt_tokens=100)
        tracker.reset_usage_stats()
        tracker._record_usage(prompt_tokens=50)

        assert tracker._usage_stats["prompt_tokens"] == 50


class TestExtractTokenUsage:
    """extract_token_usage tests."""

    def test_none_returns_zeros(self):
        from boxing_gym.agents.usage_tracker import extract_token_usage
        result = extract_token_usage(None)
        assert result == {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}

    def test_openai_style_attrs(self):
        from boxing_gym.agents.usage_tracker import extract_token_usage

        class Usage:
            prompt_tokens = 100
            completion_tokens = 50
            reasoning_tokens = 25

        result = extract_token_usage(Usage())
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["reasoning_tokens"] == 25

    def test_anthropic_style_attrs(self):
        from boxing_gym.agents.usage_tracker import extract_token_usage

        class Usage:
            input_tokens = 100
            output_tokens = 50

        result = extract_token_usage(Usage())
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["reasoning_tokens"] == 0

    def test_dict_prompt_completion_keys(self):
        from boxing_gym.agents.usage_tracker import extract_token_usage
        result = extract_token_usage({"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 2})
        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 5
        assert result["reasoning_tokens"] == 2

    def test_dict_input_output_keys(self):
        from boxing_gym.agents.usage_tracker import extract_token_usage
        result = extract_token_usage({"input_tokens": 10, "output_tokens": 5})
        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 5

    def test_missing_reasoning_tokens_defaults_zero(self):
        from boxing_gym.agents.usage_tracker import extract_token_usage

        class Usage:
            prompt_tokens = 100
            completion_tokens = 50

        result = extract_token_usage(Usage())
        assert result["reasoning_tokens"] == 0

    def test_none_values_become_zero(self):
        from boxing_gym.agents.usage_tracker import extract_token_usage
        result = extract_token_usage({"prompt_tokens": None, "completion_tokens": None})
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
