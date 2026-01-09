from boxing_gym.experiment.step_logger import StepLogger


def test_log_llm_usage_step_tracks_deltas_and_cumulative():
    events = []

    def _capture(step, metrics):
        events.append(metrics)

    logger = StepLogger(log_callback=_capture)

    usage1 = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "reasoning_tokens": 0,
        "total_tokens": 15,
        "total_cost_usd": 0.01,
        "call_count": 1,
        "retry_count": 0,
        "error_count": 0,
    }
    logger.log_llm_usage_step(step_idx=0, usage_stats=usage1, agent_prefix="llm")
    m1 = events[-1]
    assert m1["step/idx"] == 0
    assert m1["llm/prompt_tokens_step"] == 10
    assert m1["llm/completion_tokens_step"] == 5
    assert m1["llm/total_tokens_step"] == 15
    assert m1["cumulative/llm_prompt_tokens"] == 10
    assert m1["cumulative/llm_completion_tokens"] == 5
    assert m1["cumulative/llm_reasoning_tokens"] == 0

    usage2 = {
        "prompt_tokens": 30,
        "completion_tokens": 7,
        "reasoning_tokens": 0,
        "total_tokens": 37,
        "total_cost_usd": 0.03,
        "call_count": 3,
        "retry_count": 1,
        "error_count": 0,
    }
    logger.log_llm_usage_step(step_idx=1, usage_stats=usage2, agent_prefix="llm")
    m2 = events[-1]
    assert m2["step/idx"] == 1
    assert m2["llm/prompt_tokens_step"] == 20
    assert m2["llm/completion_tokens_step"] == 2
    assert m2["llm/total_tokens_step"] == 22
    assert m2["cumulative/llm_prompt_tokens"] == 30
    assert m2["cumulative/llm_completion_tokens"] == 7

    # Simulate a reset (counters drop); deltas should fall back to current values.
    usage3 = {
        "prompt_tokens": 5,
        "completion_tokens": 1,
        "reasoning_tokens": 0,
        "total_tokens": 6,
        "total_cost_usd": 0.005,
        "call_count": 1,
        "retry_count": 0,
        "error_count": 0,
    }
    logger.log_llm_usage_step(step_idx=2, usage_stats=usage3, agent_prefix="llm")
    m3 = events[-1]
    assert m3["step/idx"] == 2
    assert m3["llm/prompt_tokens_step"] == 5
    assert m3["llm/completion_tokens_step"] == 1
    assert m3["llm/total_tokens_step"] == 6
    assert m3["cumulative/llm_prompt_tokens"] == 35
    assert m3["cumulative/llm_completion_tokens"] == 8


def test_step_offset_applies_to_logged_metrics():
    events = []

    def _capture(step, metrics):
        events.append(metrics)

    logger = StepLogger(log_callback=_capture)
    logger.set_step_offset(3, reset_timing=True)

    logger.log_step(step_idx=0, success=True)
    m1 = events[-1]
    assert m1["step/idx"] == 3

    logger.log_step(step_idx=1, success=False, retry_count=1)
    m2 = events[-1]
    assert m2["step/idx"] == 4
