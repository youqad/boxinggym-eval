"""JSONL call log loader utility."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class CallLogEntry:
    """Parsed call log entry."""

    agent: str
    model: str
    prompt: str
    response: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float
    has_reasoning: bool
    step_idx: Optional[int]
    call_type: str
    timestamp: str
    error: Optional[str] = None


def load_jsonl(path: Path) -> List[CallLogEntry]:
    """Load JSONL file into list of CallLogEntry objects.

    Skips malformed lines (e.g., from crash-truncated writes) to ensure
    crash-resilient reading even when crash-resilient writing occurred.
    """
    entries = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    entries.append(
                        CallLogEntry(
                            agent=data.get("agent", "unknown"),
                            model=data.get("model", "unknown"),
                            prompt=data.get("prompt", ""),
                            response=data.get("response", ""),
                            latency_ms=data.get("latency_ms", 0.0),
                            prompt_tokens=data.get("prompt_tokens", 0),
                            completion_tokens=data.get("completion_tokens", 0),
                            reasoning_tokens=data.get("reasoning_tokens", 0),
                            cost_usd=data.get("cost_usd", 0.0),
                            has_reasoning=data.get("has_reasoning", False),
                            step_idx=data.get("step_idx"),
                            call_type=data.get("call_type", "unknown"),
                            timestamp=data.get("timestamp", ""),
                            error=data.get("error"),
                        )
                    )
                except json.JSONDecodeError:
                    continue  # skip malformed lines
    return entries


def load_jsonl_to_df(path: Path) -> pd.DataFrame:
    """Load JSONL file directly to DataFrame."""
    entries = load_jsonl(path)
    if not entries:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "agent": e.agent,
                "model": e.model,
                "prompt": e.prompt,
                "response": e.response,
                "latency_ms": e.latency_ms,
                "prompt_tokens": e.prompt_tokens,
                "completion_tokens": e.completion_tokens,
                "reasoning_tokens": e.reasoning_tokens,
                "cost_usd": e.cost_usd,
                "has_reasoning": e.has_reasoning,
                "step_idx": e.step_idx,
                "call_type": e.call_type,
                "timestamp": e.timestamp,
                "error": e.error,
            }
            for e in entries
        ]
    )


def aggregate_by_agent(entries: List[CallLogEntry]) -> Dict[str, List[CallLogEntry]]:
    """Group entries by agent name."""
    result: Dict[str, List[CallLogEntry]] = {}
    for entry in entries:
        if entry.agent not in result:
            result[entry.agent] = []
        result[entry.agent].append(entry)
    return result


def aggregate_by_model(entries: List[CallLogEntry]) -> Dict[str, List[CallLogEntry]]:
    """Group entries by model."""
    result: Dict[str, List[CallLogEntry]] = {}
    for entry in entries:
        if entry.model not in result:
            result[entry.model] = []
        result[entry.model].append(entry)
    return result


@dataclass
class AgentStats:
    """Statistics for a single agent."""

    agent: str
    call_count: int
    total_cost: float
    avg_latency: float
    error_count: int
    total_tokens: int


@dataclass
class CallLogStats:
    """Summary statistics for call logs."""

    total_calls: int
    total_cost: float
    avg_latency: float
    error_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_reasoning_tokens: int
    agents: List[AgentStats] = field(default_factory=list)
    models: List[str] = field(default_factory=list)


def compute_stats(entries: List[CallLogEntry]) -> CallLogStats:
    """Compute summary statistics."""
    if not entries:
        return CallLogStats(
            total_calls=0,
            total_cost=0.0,
            avg_latency=0.0,
            error_count=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_reasoning_tokens=0,
        )

    total_cost = sum(e.cost_usd for e in entries)
    avg_latency = sum(e.latency_ms for e in entries) / len(entries)
    error_count = sum(1 for e in entries if e.error)
    total_prompt = sum(e.prompt_tokens for e in entries)
    total_completion = sum(e.completion_tokens for e in entries)
    total_reasoning = sum(e.reasoning_tokens for e in entries)

    by_agent = aggregate_by_agent(entries)
    agent_stats = []
    for agent, agent_entries in sorted(by_agent.items()):
        agent_stats.append(
            AgentStats(
                agent=agent,
                call_count=len(agent_entries),
                total_cost=sum(e.cost_usd for e in agent_entries),
                avg_latency=sum(e.latency_ms for e in agent_entries) / len(agent_entries),
                error_count=sum(1 for e in agent_entries if e.error),
                total_tokens=sum(
                    e.prompt_tokens + e.completion_tokens + e.reasoning_tokens
                    for e in agent_entries
                ),
            )
        )

    models = sorted(set(e.model for e in entries))

    return CallLogStats(
        total_calls=len(entries),
        total_cost=total_cost,
        avg_latency=avg_latency,
        error_count=error_count,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_reasoning_tokens=total_reasoning,
        agents=agent_stats,
        models=models,
    )


def find_jsonl_files(base_paths: Optional[List[Path]] = None) -> List[Path]:
    """Find all JSONL call log files across multiple directories.

    Searches both outputs/ and results/ by default, sorted by mtime (newest first).
    """
    if base_paths is None:
        base_paths = [Path("outputs"), Path("results")]

    files = []
    for base in base_paths:
        if base.exists():
            files.extend(base.rglob("llm_calls_*.jsonl"))

    # sort by modification time, newest first
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def find_latest_jsonl(base_paths: Optional[List[Path]] = None) -> Optional[Path]:
    """Find the most recent JSONL call log file."""
    files = find_jsonl_files(base_paths)
    return files[0] if files else None
