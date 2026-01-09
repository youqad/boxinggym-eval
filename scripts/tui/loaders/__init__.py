"""Local results loaders for TUI."""

from .local_results import load_local_results
from .jsonl_loader import (
    CallLogEntry,
    CallLogStats,
    AgentStats,
    load_jsonl,
    load_jsonl_to_df,
    aggregate_by_agent,
    aggregate_by_model,
    compute_stats,
    find_jsonl_files,
    find_latest_jsonl,
)

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
