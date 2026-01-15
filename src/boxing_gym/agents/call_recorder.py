"""Crash-resilient LLM call recorder that writes per-call to disk.

This module provides a shared recorder for both sync (LMExperimenter) and async
(AsyncLiteLLM) LLM calls. Calls are buffered and synced periodically for performance.

Usage:
    from boxing_gym.agents.call_recorder import get_call_recorder, CallRecord

    # at experiment start
    recorder = get_call_recorder(run_id="my-run-123")

    # after each LLM call
    recorder.record(CallRecord(
        agent="scientist",
        model="gpt-4o",
        prompt="...",
        response="...",
        ...
    ))

    # at experiment end (for WandB upload)
    filepath = recorder.get_filepath()
    calls = recorder.get_all_calls()
"""

import atexit
import json
import logging
import os
import re
import tempfile
import time
import threading
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# sync to disk every N writes (balance speed vs crash resilience)
SYNC_INTERVAL = 10

# max length for prompt/response fields to prevent huge files
MAX_LOG_LENGTH = 50000

# thread lock for safe concurrent writes
_LOCK = threading.Lock()

# global singleton per run_id
_RECORDERS: Dict[str, "CallRecorder"] = {}


@dataclass
class CallRecord:
    """single LLM call record."""
    agent: str  # "scientist", "naive", "ppl", "box_loop"
    model: str
    prompt: str
    response: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    has_reasoning: bool = False
    step_idx: Optional[int] = None  # experiment step (if applicable)
    call_type: str = "chat"  # "chat", "ppl", "box_loop"
    timestamp: Optional[float] = None
    call_idx: Optional[int] = None
    call_uuid: str = field(default_factory=lambda: uuid.uuid4().hex)  # stable ID for correlation
    error: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class CallRecorder:
    """disk-backed JSONL recorder with buffered writes for performance."""

    def __init__(self, run_id: str, output_dir: str = "results"):
        # sanitize run_id to prevent path traversal attacks
        safe_run_id = re.sub(r'[^\w\-]', '_', run_id)
        self.run_id = safe_run_id

        # sanitize output_dir to prevent path traversal
        output_path = Path(output_dir).resolve()
        allowed_roots = [Path.cwd().resolve(), Path(tempfile.gettempdir()).resolve()]
        # use is_relative_to to prevent prefix bypass (e.g., /project vs /project-evil)
        is_safe = any(output_path.is_relative_to(root) for root in allowed_roots)
        if not is_safe:
            raise ValueError(f"output_dir must be within project root or temp directory: {output_dir}")

        self.output_dir = output_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / f"llm_calls_{safe_run_id}.jsonl"
        self._call_count = 0
        self._writes_since_sync = 0

        # count existing lines if resuming
        if self.filepath.exists():
            with open(self.filepath, "r") as f:
                self._call_count = sum(1 for _ in f)

        # keep file handle open for performance (line-buffered)
        self._file_handle = open(self.filepath, "a", buffering=1)

        # register cleanup on process exit
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """ensure file is flushed and closed on exit."""
        try:
            if hasattr(self, '_file_handle') and self._file_handle and not self._file_handle.closed:
                self._file_handle.flush()
                os.fsync(self._file_handle.fileno())
                self._file_handle.close()
        except Exception:
            pass  # best effort cleanup

    def close(self) -> None:
        """Close the recorder file handle and flush buffers."""
        self._cleanup()

    def record(self, call: CallRecord) -> None:
        """append call to JSONL file with periodic sync."""
        with _LOCK:
            idx = self._call_count
            call.call_idx = idx
            self._file_handle.write(json.dumps(asdict(call), default=str) + "\n")
            self._writes_since_sync += 1

            # sync to disk every SYNC_INTERVAL writes
            if self._writes_since_sync >= SYNC_INTERVAL:
                self._file_handle.flush()
                try:
                    os.fsync(self._file_handle.fileno())
                except OSError:
                    pass  # flush completed, fsync failure is non-fatal
                self._writes_since_sync = 0

            self._call_count = idx + 1

    def record_dict(self, data: Dict[str, Any]) -> None:
        """convenience method for dict-based recording."""
        # extract known fields, pass rest as-is
        # use `or ""` to handle explicit None values
        call = CallRecord(
            agent=data.get("agent", "unknown"),
            model=data.get("model", "unknown"),
            prompt=(data.get("prompt") or "")[:MAX_LOG_LENGTH],
            response=(data.get("response") or "")[:MAX_LOG_LENGTH],
            latency_ms=data.get("latency_ms", 0.0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            has_reasoning=data.get("has_reasoning", False),
            step_idx=data.get("step_idx"),
            call_type=data.get("call_type", "chat"),
            timestamp=data.get("timestamp"),
            error=data.get("error"),
        )
        self.record(call)

    def get_filepath(self) -> Path:
        """return path to JSONL file."""
        return self.filepath

    def get_all_calls(self) -> List[Dict]:
        """read all calls from disk (for end-of-run processing)."""
        calls = []
        if self.filepath.exists():
            with open(self.filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            calls.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        return calls

    def get_call_count(self) -> int:
        """return total number of recorded calls."""
        return self._call_count


def get_call_recorder(run_id: Optional[str] = None, output_dir: str = "results") -> CallRecorder:
    """get or create a CallRecorder for the given run_id.

    if run_id is None, generates a timestamp-based ID (milliseconds for uniqueness).
    """
    global _RECORDERS

    with _LOCK:
        # generate run_id INSIDE lock to prevent race conditions on auto-generated IDs
        if run_id is None:
            run_id = f"run_{int(time.time() * 1000)}"  # milliseconds for uniqueness

        if run_id not in _RECORDERS:
            _RECORDERS[run_id] = CallRecorder(run_id, output_dir)
        return _RECORDERS[run_id]


def clear_recorder(run_id: str) -> None:
    """remove recorder from cache (e.g., at end of run)."""
    global _RECORDERS
    with _LOCK:
        recorder = _RECORDERS.pop(run_id, None)
        if recorder is not None:
            recorder.close()
