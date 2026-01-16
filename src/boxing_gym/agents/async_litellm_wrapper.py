"""Async wrapper for LiteLLM supporting multiple model providers."""

from typing import List, Dict, Any, Optional
import os
import time
import uuid
import litellm

import logging

from boxing_gym.agents.litellm_utils import (
    is_responses_api_model,
    extract_text,
    messages_to_input_text,
    MIN_SAFE_MAX_TOKENS,
)

logger = logging.getLogger(__name__)
from boxing_gym.agents.call_recorder import CallRecorder
from boxing_gym.agents.usage_tracker import extract_token_usage

try:
    from boxing_gym.agents.agent import _model_profile, _extract_usage_dict
except ImportError:
    _model_profile = None  # fallback if circular import
    _extract_usage_dict = None  # fallback if circular import


def _default_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
    }


class AsyncLiteLLM:
    """Async wrapper for LiteLLM's unified model API.

    Supports any LiteLLM-compatible model string (OpenAI, Anthropic, DeepSeek, etc).
    See https://docs.litellm.ai/docs/providers for supported providers.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        max_tokens: int = 4096,
        num_retries: int = 3,
        timeout: int = 180,
        is_reasoning: bool | None = None,
        **litellm_kwargs
    ):
        self.model_name = model_name

        # robust coercion: handle strings, floats, booleans
        if max_tokens is None or isinstance(max_tokens, bool):
            max_tokens = 4096
        else:
            try:
                max_tokens = int(float(max_tokens))  # handles "8192.0" strings
            except (ValueError, TypeError):
                max_tokens = 4096
        # clamp to safe minimum
        if max_tokens < MIN_SAFE_MAX_TOKENS:
            logger.warning(
                f"[AsyncLiteLLM:{model_name}] TRUNCATION RISK: max_tokens={max_tokens} clamped to {MIN_SAFE_MAX_TOKENS}."
            )
            max_tokens = MIN_SAFE_MAX_TOKENS

        # reasoning models need headroom for hidden thinking tokens
        REASONING_MIN_TOKENS = 32768

        if is_reasoning is not None:
            # config-driven: enforce floor for reasoning, but clamp to model cap
            self.is_reasoning = is_reasoning
            target = int(max_tokens)
            if is_reasoning:
                floor = REASONING_MIN_TOKENS
                if _model_profile is not None:
                    cap = _model_profile(model_name).api_max_tokens
                else:
                    # fallback cap estimation if profile unavailable
                    cap = self._estimate_model_cap(model_name)
                floor = min(floor, cap)
                self.max_tokens = max(target, floor)
            else:
                self.max_tokens = target
        elif _model_profile is not None:
            # no config flag, use hardcoded profiles
            profile = _model_profile(model_name)
            self.is_reasoning = profile.is_reasoning
            if profile.is_reasoning:
                # reasoning models need headroom; use floor but respect user if higher
                floor = min(REASONING_MIN_TOKENS, profile.api_max_tokens)
                self.max_tokens = max(int(max_tokens), floor)
            else:
                self.max_tokens = int(max_tokens)
        else:
            # last resort: pattern match on model name
            model_lower = (model_name or "").lower()
            inflated = int(max_tokens)
            if "speciale" in model_lower or "deepseek-reasoner" in model_lower or "deepseek-r1" in model_lower:
                inflated = max(inflated, 65536)
            elif "gpt-5" in model_lower:
                inflated = max(inflated, 131072)
            elif "minimax" in model_lower:
                inflated = max(inflated, 196608)
            self.max_tokens = inflated
            self.is_reasoning = False
        self.num_retries = num_retries
        self.timeout = timeout
        self.litellm_kwargs = litellm_kwargs

        litellm.drop_params = True
        litellm.set_verbose = False
        if hasattr(litellm, "suppress_debug_info"):
            litellm.suppress_debug_info = True
        if hasattr(litellm, "turn_off_message_logging"):
            litellm.turn_off_message_logging = True

        # optional Weave tracing for async LiteLLM calls.
        try:
            import weave  # type: ignore
        except Exception:  # pragma: no cover
            weave = None
        self._weave = weave

        # disk-backed call recorder for crash-resilient logging
        self._call_recorder: Optional[CallRecorder] = None
        self._agent_name: str = "ppl"  # default for async PPL calls
        self._step_idx: Optional[int] = None

    @staticmethod
    def _estimate_model_cap(model_name: str) -> int:
        """Fallback cap estimation when _model_profile unavailable."""
        name = (model_name or "").lower()
        if "gpt-5" in name:
            return 131072
        if "minimax" in name:
            return 196608
        if "glm-4" in name or "glm4" in name:
            return 131072
        if "deepseek-r" in name or "reasoner" in name:
            return 32768
        if "kimi" in name or "moonshot" in name:
            return 32768
        # conservative default
        return 32768

    def _maybe_weave_op(self, fn):
        """Conditionally wrap a function with weave.op for tracing.

        We log a sanitized payload instead of raw LiteLLM responses to avoid
        serialization errors inside Weave.
        """
        if self._weave is None:
            return fn
        project = os.environ.get("WEAVE_PROJECT") or os.environ.get("WANDB_PROJECT")
        disabled = os.environ.get("WEAVE_DISABLED", "0").lower() in ("1", "true", "yes")
        if not project or disabled:
            return fn

        async def wrapped(*args, **kwargs):
            result = await fn(*args, **kwargs)
            try:
                payload = {
                    "model": self.model_name or "unknown",
                    "prompt": None,
                    "output": extract_text(result),
                    "usage": _default_usage(),
                }
                if "messages" in kwargs:
                    payload["prompt"] = messages_to_input_text(kwargs.get("messages") or [])
                elif "input" in kwargs:
                    payload["prompt"] = kwargs.get("input")
                elif args:
                    payload["prompt"] = args[0]
                if _extract_usage_dict is not None:
                    usage = _extract_usage_dict(result)
                    if usage is not None:
                        payload["usage"] = usage
                try:
                    op = self._weave.op(lambda x: x)
                    op(payload)
                except Exception:
                    pass
            except Exception:
                pass
            return result

        return wrapped

    def set_call_recorder(
        self,
        recorder: CallRecorder,
        agent_name: str = "ppl",
        step_idx: Optional[int] = None
    ) -> None:
        """Attach a disk-backed call recorder for crash-resilient logging."""
        self._call_recorder = recorder
        self._agent_name = agent_name
        self._step_idx = step_idx

    def set_step_idx(self, step_idx: int) -> None:
        """Update the current experiment step index."""
        self._step_idx = step_idx

    @property
    def llm_type(self) -> str:
        if "gpt" in self.model_name.lower():
            return "OpenAI"
        elif "claude" in self.model_name.lower():
            return "Claude"
        elif "deepseek" in self.model_name.lower():
            return "DeepSeek"
        elif "gemini" in self.model_name.lower():
            return "Gemini"
        else:
            return self.model_name

    def _is_responses_api_model(self) -> bool:
        return is_responses_api_model(self.model_name)

    def _should_use_fake_response(self) -> bool:
        """Check if fake LLM mode is enabled via environment."""
        flag = os.getenv("BOXINGGYM_FAKE_LLM", "")
        return bool(flag) and flag.lower() not in ("0", "false", "no")

    def _make_fake_response(self, messages: List[Dict[str, Any]]) -> Any:
        """Create fake response for smoke tests and record it."""
        class _FakeMsg:
            def __init__(self, content: str):
                self.content = content
                self.reasoning_content = None

        class _FakeChoice:
            def __init__(self, content: str):
                self.message = _FakeMsg(content)

        class _FakeResp:
            def __init__(self, model: str, content: str):
                self.model = model
                self.choices = [_FakeChoice(content)]
                self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

        fake_content = "mock response"
        if self._call_recorder is not None:
            prompt_text = messages_to_input_text(messages)
            self._call_recorder.record_dict({
                "agent": self._agent_name,
                "model": self.model_name,
                "prompt": prompt_text[:50000] if prompt_text else "",
                "response": fake_content,
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
                "has_reasoning": False,
                "step_idx": self._step_idx,
                "call_type": "fake",
                "call_uuid": uuid.uuid4().hex,
                "error": None,
            })
        return _FakeResp(self.model_name, fake_content)

    async def _execute_responses_api_call(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Any:
        """Execute call via Responses API (GPT-5 series)."""
        # convert chat messages into single text prompt
        prompt_lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content:
                prompt_lines.append(f"{role}: {content}")
        input_text = "\n".join(prompt_lines)

        # filter params not supported by Responses API
        responses_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["max_tokens", "max_output_tokens", "messages", "temperature"]
        }
        max_output_tokens = kwargs.get("max_output_tokens")
        if max_output_tokens is None:
            max_output_tokens = kwargs.get("max_tokens") or self.max_tokens

        async def _call():
            return await litellm.aresponses(
                model=self.model_name,
                input=input_text,
                max_output_tokens=max_output_tokens,
                num_retries=self.num_retries,
                timeout=self.timeout,
                **self.litellm_kwargs,
                **responses_kwargs,
            )

        return await self._maybe_weave_op(_call)()

    async def _execute_completion_call(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Any:
        """Execute call via standard Chat Completions API."""
        call_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "num_retries": self.num_retries,
            "timeout": self.timeout,
            **self.litellm_kwargs,
            **kwargs
        }

        async def _call():
            return await litellm.acompletion(**call_params)

        return await self._maybe_weave_op(_call)()

    def _record_call(
        self,
        response: Any,
        prompt_text: str,
        call_start: float,
        error_msg: Optional[str] = None
    ) -> None:
        """Record API call to disk. Safe even if response is None."""
        if self._call_recorder is None:
            return

        call_end = time.time()
        latency_ms = (call_end - call_start) * 1000

        response_text = ""
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}

        if response is not None:
            response_text = extract_text(response) or ""
            usage = getattr(response, "usage", None)
            if usage is None and hasattr(response, "__dict__"):
                usage = response.__dict__.get("usage")
            if usage is None and isinstance(response, dict):
                usage = response.get("usage")
            tokens = extract_token_usage(usage)

        self._call_recorder.record_dict({
            "agent": self._agent_name,
            "model": self.model_name,
            "prompt": prompt_text,
            "response": response_text,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": tokens["prompt_tokens"],
            "completion_tokens": tokens["completion_tokens"],
            "reasoning_tokens": tokens["reasoning_tokens"],
            "cost_usd": 0.0,
            "has_reasoning": tokens["reasoning_tokens"] > 0,
            "step_idx": self._step_idx,
            "call_type": "ppl",
            "call_uuid": uuid.uuid4().hex,
            "timestamp": call_end,
            "error": error_msg,
        })

    async def __call__(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Any:
        """Make async completion call via LiteLLM.

        Automatically routes to correct endpoint:
        - GPT-5 series → aresponses() with /v1/responses
        - All others → acompletion() with /v1/chat/completions

        LiteLLM handles provider routing and API key lookup automatically.
        Pass api_base/api_key via constructor kwargs for custom endpoints.
        """
        if self._should_use_fake_response():
            return self._make_fake_response(messages)

        prompt_text = messages_to_input_text(messages)
        call_start = time.time()
        response = None
        error_msg = None

        try:
            if self._is_responses_api_model():
                response = await self._execute_responses_api_call(messages, **kwargs)
            else:
                response = await self._execute_completion_call(messages, **kwargs)
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            self._record_call(response, prompt_text, call_start, error_msg)

        return response
