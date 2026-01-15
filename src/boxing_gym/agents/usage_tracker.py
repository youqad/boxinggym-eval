"""Usage tracking mixin for LLM agents.

Tracks token usage, costs, latencies, and error counts for W&B logging.
Extracted from LMExperimenter to reduce class complexity.
"""

from typing import Dict, List


class UsageTrackerMixin:
    """Mixin providing usage statistics tracking for LLM agents.

    Tracks:
    - Token usage (prompt, completion, reasoning)
    - Costs (via LiteLLM or fallback pricing)
    - Latencies (per-call, with percentile stats)
    - Retry and error counts

    Usage:
        class MyAgent(UsageTrackerMixin):
            def __init__(self):
                self._init_usage_tracking()
    """

    def _init_usage_tracking(self) -> None:
        """Initialize usage tracking state. Call in subclass __init__."""
        self._usage_stats: Dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "call_count": 0,
            "latencies_ms": [],
            "retry_count": 0,
            "error_count": 0,
        }

    def _record_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        """Record usage from a single LLM call.

        Args:
            prompt_tokens: Input tokens for this call
            completion_tokens: Output tokens for this call
            reasoning_tokens: Reasoning tokens (for thinking models)
            cost_usd: Cost in USD for this call
            latency_ms: Latency in milliseconds
        """
        self._usage_stats["prompt_tokens"] += prompt_tokens
        self._usage_stats["completion_tokens"] += completion_tokens
        self._usage_stats["reasoning_tokens"] += reasoning_tokens
        self._usage_stats["total_tokens"] += prompt_tokens + completion_tokens + reasoning_tokens
        self._usage_stats["total_cost_usd"] += cost_usd
        self._usage_stats["call_count"] += 1
        self._usage_stats["latencies_ms"].append(latency_ms)

    def _record_retry(self, count: int = 1) -> None:
        """Record retry attempts.

        Args:
            count: Number of retries to record (default 1)
        """
        self._usage_stats["retry_count"] += count

    def _record_error(self) -> None:
        """Record an error (after max retries exhausted)."""
        self._usage_stats["error_count"] += 1

    def get_usage_stats(self) -> Dict:
        """Return accumulated usage stats for W&B logging.

        Returns dict with:
        - Token counts (prompt, completion, reasoning, total)
        - Cost in USD
        - Call count
        - Latency statistics (mean, p50, p95, min, max)
        - Retry and error counts
        """
        stats = dict(self._usage_stats)
        latencies = stats.pop("latencies_ms", [])

        if latencies:
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            stats["latency_mean_ms"] = sum(latencies) / n
            stats["latency_p50_ms"] = latencies_sorted[n // 2]
            stats["latency_p95_ms"] = latencies_sorted[int(n * 0.95)] if n >= 20 else latencies_sorted[-1]
            stats["latency_min_ms"] = latencies_sorted[0]
            stats["latency_max_ms"] = latencies_sorted[-1]
        else:
            stats["latency_mean_ms"] = 0.0
            stats["latency_p50_ms"] = 0.0
            stats["latency_p95_ms"] = 0.0
            stats["latency_min_ms"] = 0.0
            stats["latency_max_ms"] = 0.0

        return stats

    def get_latencies_ms(self) -> List[float]:
        """Return list of per-call latencies in milliseconds.

        Used by loop.py to calculate per-step latency.
        """
        return list(self._usage_stats.get("latencies_ms", []))

    def reset_usage_stats(self) -> None:
        """Reset usage statistics (useful between budget evaluations)."""
        self._init_usage_tracking()
