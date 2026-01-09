"""Tests for CallRecorder crash-resilient LLM call logging."""

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from boxing_gym.agents.call_recorder import (
    CallRecorder,
    CallRecord,
    get_call_recorder,
    clear_recorder,
    MAX_LOG_LENGTH,
)


class TestCallRecord:
    """Tests for CallRecord dataclass."""

    def test_default_timestamp(self):
        """Test that timestamp is auto-generated if not provided."""
        before = time.time()
        record = CallRecord(
            agent="test",
            model="gpt-4o",
            prompt="test prompt",
            response="test response",
            latency_ms=100.0,
        )
        after = time.time()

        assert before <= record.timestamp <= after

    def test_explicit_timestamp(self):
        """Test that explicit timestamp is preserved."""
        explicit_ts = 1234567890.0
        record = CallRecord(
            agent="test",
            model="gpt-4o",
            prompt="test prompt",
            response="test response",
            latency_ms=100.0,
            timestamp=explicit_ts,
        )
        assert record.timestamp == explicit_ts

    def test_default_values(self):
        """Test default field values."""
        record = CallRecord(
            agent="test",
            model="gpt-4o",
            prompt="test",
            response="test",
            latency_ms=0.0,
        )
        assert record.prompt_tokens == 0
        assert record.completion_tokens == 0
        assert record.reasoning_tokens == 0
        assert record.cost_usd == 0.0
        assert record.has_reasoning is False
        assert record.call_type == "chat"
        assert record.error is None


class TestCallRecorder:
    """Tests for CallRecorder class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def recorder(self, temp_dir):
        """Create a CallRecorder for testing."""
        recorder = CallRecorder(run_id="test-run", output_dir=temp_dir)
        yield recorder
        recorder._cleanup()

    def test_creates_output_file(self, temp_dir):
        """Test that recorder creates JSONL file."""
        recorder = CallRecorder(run_id="create-test", output_dir=temp_dir)
        assert recorder.filepath.exists()
        assert recorder.filepath.suffix == ".jsonl"
        recorder._cleanup()

    def test_sanitizes_run_id(self, temp_dir):
        """Test that dangerous characters in run_id are sanitized."""
        dangerous_id = "../../../etc/passwd"
        recorder = CallRecorder(run_id=dangerous_id, output_dir=temp_dir)
        # run_id should be sanitized to safe characters (each . and / becomes _)
        assert ".." not in str(recorder.filepath)
        assert "/" not in recorder.run_id
        assert "." not in recorder.run_id
        recorder._cleanup()

    def test_rejects_path_traversal_output_dir(self, temp_dir):
        """Test that output_dir path traversal is rejected."""
        with pytest.raises(ValueError, match="must be within"):
            CallRecorder(run_id="test", output_dir="/etc/dangerous")

    def test_record_writes_jsonl(self, recorder, temp_dir):
        """Test that record() writes valid JSONL."""
        call = CallRecord(
            agent="scientist",
            model="gpt-4o",
            prompt="What is 2+2?",
            response="4",
            latency_ms=150.5,
            prompt_tokens=10,
            completion_tokens=1,
        )
        recorder.record(call)

        # force flush
        recorder._file_handle.flush()

        # verify JSONL content
        with open(recorder.filepath) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["agent"] == "scientist"
            assert data["model"] == "gpt-4o"
            assert data["prompt"] == "What is 2+2?"
            assert data["response"] == "4"
            assert data["latency_ms"] == 150.5
            assert data["call_idx"] == 0

    def test_record_dict_convenience(self, recorder):
        """Test record_dict() convenience method."""
        recorder.record_dict({
            "agent": "naive",
            "model": "claude-3-opus",
            "prompt": "test prompt",
            "response": "test response",
            "latency_ms": 200.0,
            "cost_usd": 0.05,
        })

        calls = recorder.get_all_calls()
        assert len(calls) == 1
        assert calls[0]["agent"] == "naive"
        assert calls[0]["cost_usd"] == 0.05

    def test_record_dict_handles_none_values(self, recorder):
        """Test that record_dict handles None for prompt/response."""
        recorder.record_dict({
            "agent": "test",
            "model": "test",
            "prompt": None,
            "response": None,
            "latency_ms": 0.0,
        })

        calls = recorder.get_all_calls()
        assert len(calls) == 1
        assert calls[0]["prompt"] == ""
        assert calls[0]["response"] == ""

    def test_truncates_long_content(self, recorder):
        """Test that long prompt/response are truncated."""
        long_text = "x" * (MAX_LOG_LENGTH + 1000)
        recorder.record_dict({
            "agent": "test",
            "model": "test",
            "prompt": long_text,
            "response": long_text,
            "latency_ms": 0.0,
        })

        calls = recorder.get_all_calls()
        assert len(calls[0]["prompt"]) == MAX_LOG_LENGTH
        assert len(calls[0]["response"]) == MAX_LOG_LENGTH

    def test_get_all_calls_returns_all(self, recorder):
        """Test that get_all_calls() returns all recorded calls."""
        for i in range(5):
            recorder.record_dict({
                "agent": f"agent_{i}",
                "model": "test",
                "prompt": f"prompt_{i}",
                "response": f"response_{i}",
                "latency_ms": float(i),
            })

        calls = recorder.get_all_calls()
        assert len(calls) == 5
        assert calls[0]["agent"] == "agent_0"
        assert calls[4]["agent"] == "agent_4"

    def test_get_call_count(self, recorder):
        """Test get_call_count() tracking."""
        assert recorder.get_call_count() == 0

        for i in range(3):
            recorder.record_dict({
                "agent": "test",
                "model": "test",
                "prompt": "x",
                "response": "y",
                "latency_ms": 0.0,
            })
            assert recorder.get_call_count() == i + 1

    def test_resumes_existing_file(self, temp_dir):
        """Test that recorder resumes count from existing file."""
        # create initial recorder and write some calls
        recorder1 = CallRecorder(run_id="resume-test", output_dir=temp_dir)
        for _ in range(3):
            recorder1.record_dict({
                "agent": "test", "model": "test",
                "prompt": "x", "response": "y", "latency_ms": 0.0,
            })
        recorder1._cleanup()

        # create new recorder for same run_id
        recorder2 = CallRecorder(run_id="resume-test", output_dir=temp_dir)
        assert recorder2.get_call_count() == 3

        # new writes should have correct call_idx
        recorder2.record_dict({
            "agent": "test", "model": "test",
            "prompt": "x", "response": "y", "latency_ms": 0.0,
        })
        calls = recorder2.get_all_calls()
        assert calls[-1]["call_idx"] == 3
        recorder2._cleanup()


class TestCallRecorderThreadSafety:
    """Tests for thread-safe concurrent writes."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_concurrent_writes_are_safe(self, temp_dir):
        """Test that concurrent writes from multiple threads are thread-safe."""
        recorder = CallRecorder(run_id="concurrent-test", output_dir=temp_dir)
        num_threads = 10
        writes_per_thread = 50
        total_writes = num_threads * writes_per_thread
        errors = []

        def writer(thread_id):
            try:
                for i in range(writes_per_thread):
                    recorder.record_dict({
                        "agent": f"thread_{thread_id}",
                        "model": "test",
                        "prompt": f"prompt_{thread_id}_{i}",
                        "response": f"response_{thread_id}_{i}",
                        "latency_ms": float(i),
                    })
            except Exception as e:
                errors.append(e)

        # run concurrent writers
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        recorder._cleanup()

        # verify no errors
        assert len(errors) == 0

        # verify all writes completed
        calls = recorder.get_all_calls()
        assert len(calls) == total_writes

        # verify unique call_idx values
        call_indices = [c["call_idx"] for c in calls]
        assert len(set(call_indices)) == total_writes
        assert sorted(call_indices) == list(range(total_writes))

    def test_concurrent_get_call_recorder_same_id(self, temp_dir):
        """Test that concurrent get_call_recorder calls for same ID return same instance."""
        recorders = []
        errors = []

        def get_recorder():
            try:
                r = get_call_recorder(run_id="singleton-test", output_dir=temp_dir)
                recorders.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_recorder) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        clear_recorder("singleton-test")

        assert len(errors) == 0
        # all should be the same instance
        assert len(set(id(r) for r in recorders)) == 1


class TestGetCallRecorder:
    """Tests for get_call_recorder factory function."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_auto_generates_run_id(self, temp_dir):
        """Test that run_id is auto-generated if not provided."""
        recorder = get_call_recorder(run_id=None, output_dir=temp_dir)
        assert recorder.run_id.startswith("run_")
        clear_recorder(recorder.run_id)

    def test_returns_same_instance_for_same_id(self, temp_dir):
        """Test singleton behavior for same run_id."""
        r1 = get_call_recorder(run_id="singleton", output_dir=temp_dir)
        r2 = get_call_recorder(run_id="singleton", output_dir=temp_dir)
        assert r1 is r2
        clear_recorder("singleton")

    def test_returns_different_instance_for_different_id(self, temp_dir):
        """Test different instances for different run_ids."""
        r1 = get_call_recorder(run_id="run-a", output_dir=temp_dir)
        r2 = get_call_recorder(run_id="run-b", output_dir=temp_dir)
        assert r1 is not r2
        clear_recorder("run-a")
        clear_recorder("run-b")


class TestClearRecorder:
    """Tests for clear_recorder function."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_removes_from_cache(self, temp_dir):
        """Test that clear_recorder removes from cache."""
        r1 = get_call_recorder(run_id="clear-test", output_dir=temp_dir)
        clear_recorder("clear-test")
        r2 = get_call_recorder(run_id="clear-test", output_dir=temp_dir)
        # should be a new instance
        assert r1 is not r2
        clear_recorder("clear-test")

    def test_clear_nonexistent_is_safe(self):
        """Test that clearing nonexistent run_id doesn't raise."""
        clear_recorder("nonexistent-run-id")  # should not raise
