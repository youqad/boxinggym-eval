"""Local results loaders for TUI."""

from .jsonl_loader import (
    AgentStats,
    CallLogEntry,
    CallLogStats,
    aggregate_by_agent,
    aggregate_by_model,
    compute_stats,
    find_jsonl_files,
    find_latest_jsonl,
    load_jsonl,
    load_jsonl_to_df,
)
from .local_results import load_local_results

__all__ = [
    "load_local_results",
    "CallLogEntry",
    "CallLogStats",
    "AgentStats",
    "load_jsonl",
    "load_jsonl_to_df",
    "aggregate_by_agent",
    "aggregate_by_model",
    "compute_stats",
    "find_jsonl_files",
    "find_latest_jsonl",
]
