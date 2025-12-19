"""Async wrapper for LiteLLM supporting multiple model providers."""

from typing import List, Dict, Any
import os
import litellm

from boxing_gym.agents.litellm_utils import (
    is_responses_api_model,
    extract_text,
    messages_to_input_text,
)

try:
    from boxing_gym.agents.agent import _model_profile
except ImportError:
    _model_profile = None  # fallback if circular import


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
        **litellm_kwargs
    ):
        self.model_name = model_name
        # use centralized _model_profile for consistent token limits
        if _model_profile is not None:
            profile = _model_profile(model_name)
            self.max_tokens = max(int(max_tokens), profile.api_max_tokens) if profile.is_reasoning else int(max_tokens)
        else:
            # fallback: basic inflation for known reasoning models
            model_lower = (model_name or "").lower()
            inflated = int(max_tokens)
            if "speciale" in model_lower or "deepseek-reasoner" in model_lower or "deepseek-r1" in model_lower:
                inflated = max(inflated, 65536)
            elif "gpt-5" in model_lower:
                inflated = max(inflated, 131072)
            elif "minimax" in model_lower:
                inflated = max(inflated, 196608)
            self.max_tokens = inflated
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
                    "model": self.model_name,
                    "prompt": None,
                    "output": extract_text(result),
                }
                if "messages" in kwargs:
                    payload["prompt"] = messages_to_input_text(kwargs.get("messages") or [])
                elif "input" in kwargs:
                    payload["prompt"] = kwargs.get("input")
                elif args:
                    payload["prompt"] = args[0]
                try:
                    op = self._weave.op(lambda x: x)
                    op(payload)
                except Exception:
                    pass
            except Exception:
                pass
            return result

        return wrapped

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
        # optional offline mode for smoke tests (skips real API calls).
        flag = os.getenv("BOXINGGYM_FAKE_LLM", "")
        if bool(flag) and flag.lower() not in ("0", "false", "no"):
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

            return _FakeResp(self.model_name, "mock response")

        if self._is_responses_api_model():
            # convert chat messages into single text prompt for Responses API
            prompt_lines = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if content:  # skip empty messages
                    prompt_lines.append(f"{role}: {content}")
            input_text = "\n".join(prompt_lines)

            responses_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ["max_tokens", "messages", "temperature"]
            }

            async def _responses_call():
                return await litellm.aresponses(
                    model=self.model_name,
                    input=input_text,  # Responses API uses 'input' not 'messages'
                    max_output_tokens=kwargs.get("max_tokens") or self.max_tokens,
                    num_retries=self.num_retries,
                    timeout=self.timeout,
                    **self.litellm_kwargs,  # includes api_base/api_key if passed to constructor
                    **responses_kwargs,
                )

            return await self._maybe_weave_op(_responses_call)()

        # standard models use chat/completions, LiteLLM handles provider routing
        call_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "num_retries": self.num_retries,
            "timeout": self.timeout,
            **self.litellm_kwargs,  # includes api_base/api_key if passed to constructor
            **kwargs
        }

        async def _completion_call():
            return await litellm.acompletion(**call_params)

        response = await self._maybe_weave_op(_completion_call)()
        return response
