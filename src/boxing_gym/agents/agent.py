import copy
import hashlib
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass

import litellm

from boxing_gym.agents.call_recorder import CallRecorder
from boxing_gym.agents.litellm_utils import (
    extract_text,
    is_responses_api_model,
    messages_to_input_text,
)
from boxing_gym.agents.pricing import MODEL_REGISTRY, get_litellm_pricing_dict
from boxing_gym.agents.tag_parser import TagParser
from boxing_gym.agents.usage_tracker import UsageTrackerMixin, extract_token_usage

logger = logging.getLogger(__name__)

# register unified pricing with LiteLLM
try:
    litellm.register_model(model_cost=get_litellm_pricing_dict())
except Exception as e:
    logger.debug(f"Failed to register custom model pricing: {e}")

# pre-compiled regex patterns for performance (avoid re-compiling on each call)
_OBSERVE_RE = re.compile(r"<observe>(.*?)</observe>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_HAS_DIGIT_RE = re.compile(r"[0-9]")
_WORD_NUMBERS_RE = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|half|quarter|third)\b", re.IGNORECASE
)
_REASONING_MODEL_RE = re.compile(r"(^|/|\s)o\d")


# model profiles for token limits
@dataclass(frozen=True)
class ModelProfile:
    """Token limits and reasoning flag."""

    api_max_tokens: int
    is_reasoning: bool
    min_viable_tokens: int | None = (
        None  # Minimum for reasoning to work; defaults to api_max_tokens
    )

    @property
    def effective_min(self) -> int:
        return self.min_viable_tokens if self.min_viable_tokens is not None else self.api_max_tokens


def _model_profile(model_name: str) -> ModelProfile:
    """Token limits + reasoning flag for a model."""
    name = (model_name or "").lower()

    if "speciale" in name or "v3.2_speciale" in name:
        return ModelProfile(api_max_tokens=131072, is_reasoning=True)

    if "deepseek-reasoner" in name or "deepseek-r1" in name:
        return ModelProfile(api_max_tokens=32768, is_reasoning=True)

    if "deepseek-chat" in name or "deepseek-v3" in name:
        return ModelProfile(api_max_tokens=4096, is_reasoning=False)

    if "gpt-5" in name:
        return ModelProfile(api_max_tokens=131072, is_reasoning=True)

    if _REASONING_MODEL_RE.search(name):
        return ModelProfile(api_max_tokens=100000, is_reasoning=True)

    if ("kimi" in name or "moonshot" in name) and ("thinking" in name or "kimi-for-coding" in name):
        return ModelProfile(api_max_tokens=32768, is_reasoning=True)

    if "minimax" in name:
        return ModelProfile(api_max_tokens=196608, is_reasoning=True)

    if "glm-4" in name or "glm4" in name:
        return ModelProfile(api_max_tokens=131072, is_reasoning=True)  # GLM-4.7 has thinking mode

    if "kimi" in name or "moonshot" in name:
        return ModelProfile(api_max_tokens=32768, is_reasoning=False)

    if "gpt-4o" in name:
        return ModelProfile(api_max_tokens=16384, is_reasoning=False)

    if is_responses_api_model(model_name):
        return ModelProfile(api_max_tokens=16384, is_reasoning=False)

    return ModelProfile(api_max_tokens=16384, is_reasoning=False)


def _compute_effective_max_tokens(
    model_name: str,
    user_max_tokens: int | None,
    allow_truncation: bool = False,
) -> tuple[int, bool]:
    """Compute effective max_tokens with reasoning guardrails."""
    profile = _model_profile(model_name)

    logger.debug(
        f"[{model_name}] Profile: api_max={profile.api_max_tokens:,}, "
        f"is_reasoning={profile.is_reasoning}, user_max={user_max_tokens}"
    )

    if user_max_tokens is None:
        return (profile.api_max_tokens, False)

    mt = int(user_max_tokens)

    if mt <= 0:
        logger.warning(
            f"[{model_name}] Invalid max_tokens={mt}; using API default {profile.api_max_tokens}."
        )
        return (profile.api_max_tokens, True)

    if not profile.is_reasoning:
        return (mt, False)

    if mt >= profile.effective_min:
        return (mt, False)

    if allow_truncation:
        logger.warning(
            f"[{model_name}] Low max_tokens={mt} for a reasoning model. "
            f"Recommended min: {profile.effective_min}. "
            f"Reasoning output may be truncated."
        )
        return (mt, False)

    logger.warning(
        f"[{model_name}] Overriding max_tokens={mt} → {profile.api_max_tokens} "
        f"(reasoning needs more room). "
        f"Set allow_truncation=True to keep the lower limit."
    )
    return (profile.api_max_tokens, True)


# Optional Weave tracing for LLM calls.
# enabled when WEAVE_PROJECT or WANDB_PROJECT is set (and WEAVE_DISABLED is not).
try:
    import weave  # type: ignore
except Exception:  # pragma: no cover
    weave = None


_WEAVE_ECHO_OP = None


def _extract_usage_dict(resp) -> dict | None:
    """Extract usage dict from response. Returns None if no usage found."""
    usage = getattr(resp, "usage", None)
    if usage is None and hasattr(resp, "__dict__"):
        usage = resp.__dict__.get("usage")
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if not usage:
        return None
    tokens = extract_token_usage(usage)
    # return None if all zeros (preserves original behavior)
    if not any(tokens.values()):
        return None
    return tokens


_SECRET_PATTERNS = re.compile(
    r'(api[_-]?key|bearer|token|secret|password|authorization|credential|private[_-]?key|access[_-]?key|client[_-]?secret|signing[_-]?key)["\']?\s*[:=]\s*["\']?[^\s"\',}{]+',
    re.IGNORECASE,
)


def _sanitize_error(error: str) -> str:
    """Remove potential secrets from error strings before logging."""
    if not error:
        return error
    sanitized = _SECRET_PATTERNS.sub(r"\1=<REDACTED>", error)
    sanitized = re.sub(r"\b[A-Za-z0-9+/]{40,}={0,2}\b", "<REDACTED_TOKEN>", sanitized)  # base64
    sanitized = re.sub(r"\b[0-9a-fA-F]{32,}\b", "<REDACTED_HEX>", sanitized)  # hex tokens
    return sanitized


# suffix dedup: prompts have observations at start, stable template at end
_TEMPLATE_MAX_LENGTH: dict[str, int] = {}
_TEMPLATE_LOCK = threading.Lock()  # guards _TEMPLATE_MAX_LENGTH and _WEAVE_ECHO_OP
_TEMPLATE_MAX_ENTRIES = 1000
SUFFIX_CHARS = 200


def _get_template_key(prompt: str, model: str = "unknown") -> str:
    """Hash model + suffix of prompt for deduplication."""
    text = prompt or ""
    suffix = text[-SUFFIX_CHARS:] if len(text) > SUFFIX_CHARS else text
    raw_key = f"{model}::{suffix}"
    return hashlib.sha256(raw_key.encode()).hexdigest()[:16]


def _should_trace_to_weave(payload: dict) -> tuple[bool, str, int]:
    """Returns (should_trace, template_key, prompt_len). Caller holds lock for dedup."""
    # always trace errors/retries (bypass dedup)
    if payload.get("is_retry") or payload.get("retrying_after_parse_fail") or payload.get("error"):
        return True, "", 0

    prompt = payload.get("prompt") or ""
    model = str(payload.get("model") or "unknown")
    template_key = _get_template_key(prompt, model)
    prompt_len = len(prompt)

    return True, template_key, prompt_len  # caller decides with lock held


def _weave_log_payload(payload: dict) -> None:
    """Log LLM call to Weave if it passes dedup filter."""
    global _WEAVE_ECHO_OP
    if weave is None:
        return

    should_trace, template_key, prompt_len = _should_trace_to_weave(payload)
    if not should_trace:
        return

    if template_key:
        with _TEMPLATE_LOCK:
            max_seen = _TEMPLATE_MAX_LENGTH.get(template_key, 0)
            if prompt_len <= max_seen:
                return  # shorter than what we already traced
            # FIFO eviction
            while len(_TEMPLATE_MAX_LENGTH) >= _TEMPLATE_MAX_ENTRIES:
                oldest = next(iter(_TEMPLATE_MAX_LENGTH))
                del _TEMPLATE_MAX_LENGTH[oldest]
            _TEMPLATE_MAX_LENGTH[template_key] = prompt_len

    try:
        with _TEMPLATE_LOCK:
            if _WEAVE_ECHO_OP is None:

                def _llm_call(x):
                    return x

                _WEAVE_ECHO_OP = weave.op(name="llm_call")(_llm_call)
            local_op = _WEAVE_ECHO_OP

        payload = payload.copy()
        if template_key:
            payload["template_key"] = template_key

        local_op(payload)
    except Exception as e:
        logger.debug(f"Weave tracing failed: {e}")


def _build_weave_payload(
    weave_context: dict | None,
    args: tuple,
    kwargs: dict,
    latency_ms: float,
    result=None,
    error: str | None = None,
) -> dict:
    """Build payload for Weave logging. Shared by success and error paths."""
    payload = dict(weave_context or {})

    # model (from context or kwargs); ensure non-empty for downstream summary code
    if "model" not in payload or payload.get("model") in (None, ""):
        payload["model"] = kwargs.get("model") or "unknown"

    # prompt (from context, kwargs, or positional args)
    prompt = payload.get("prompt")
    if prompt is None:
        if "messages" in kwargs:
            prompt = messages_to_input_text(kwargs.get("messages") or [])
        elif "input" in kwargs:
            prompt = kwargs.get("input")
        elif args:
            prompt = args[0]
    payload["prompt"] = prompt

    # output and usage (success only)
    if error:
        payload["error"] = _sanitize_error(error)
        payload["output"] = None
    else:
        payload["output"] = extract_text(result)
        usage = _extract_usage_dict(result)
        # Weave expects a mapping for usage; avoid None to prevent summary errors.
        payload["usage"] = (
            usage
            if usage is not None
            else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
            }
        )

    # metadata
    payload.setdefault("latency_ms", round(latency_ms, 1))
    if kwargs.get("custom_llm_provider") and "provider" not in payload:
        payload["provider"] = kwargs["custom_llm_provider"]
    if kwargs.get("api_base") and "is_custom_endpoint" not in payload:
        payload["is_custom_endpoint"] = True

    return payload


def _maybe_weave_op(fn, weave_context: dict | None = None):
    """Wrap function with Weave tracing. Logs sanitized payload on success or error."""
    if weave is None:
        return fn
    project = os.environ.get("WEAVE_PROJECT") or os.environ.get("WANDB_PROJECT")
    disabled = os.environ.get("WEAVE_DISABLED", "0").lower() in ("1", "true", "yes")
    if not project or disabled:
        return fn

    def wrapped(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            try:
                payload = _build_weave_payload(
                    weave_context, args, kwargs, latency_ms, error=str(e)
                )
                _weave_log_payload(payload)
            except Exception:
                pass  # don't mask the original error
            raise
        # success
        latency_ms = (time.perf_counter() - start_time) * 1000
        try:
            payload = _build_weave_payload(weave_context, args, kwargs, latency_ms, result=result)
            _weave_log_payload(payload)
        except Exception as e:
            logger.debug(f"Weave payload logging failed: {e}")
        return result

    return wrapped


class LMExperimenter(UsageTrackerMixin):
    """LLM-powered experimenter agent for BoxingGym environments.

    Inherits from UsageTrackerMixin for token/cost/latency tracking.
    """

    def __init__(
        self,
        model_name,
        temperature=0.0,
        max_tokens: int | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        enable_thinking: bool = False,
        allow_truncation: bool = False,
        custom_llm_provider: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens, self._max_tokens_overridden = _compute_effective_max_tokens(
            model_name, max_tokens, allow_truncation
        )
        self.api_base = api_base
        self.api_key = api_key
        self.enable_thinking = enable_thinking  # ex: DeepSeek V3.2 thinking via extra_body
        self.allow_truncation = allow_truncation
        self.custom_llm_provider = custom_llm_provider  # custom endpoints (GLM, MiniMax, Kimi)
        self.extra_headers = extra_headers  # Kimi requires User-Agent header
        self.system = None
        self.messages = []
        self.all_messages = []
        # offline mode for smoke tests
        self.fake_mode = self._fake_mode_enabled()
        # LiteLLM global flags.
        litellm.drop_params = True
        litellm.modify_params = True  # param conversion for custom endpoints
        litellm.set_verbose = False

        # usage tracking via mixin
        self._init_usage_tracking()

        # per-call history for WandB logging (independent of Weave)
        self._call_history: list[dict] = []
        # optional disk-backed recorder for crash resilience
        self._call_recorder: CallRecorder | None = None
        self._agent_name: str = "unknown"  # set via set_call_recorder()

    def set_system_message(self, message):
        self.all_messages.append(f"role:system, message:{message}")
        self.system = message
        # content parts format for compatibility.
        self.messages.append({"role": "system", "content": [{"type": "text", "text": message}]})

    def add_message(self, message, role="user"):
        self.all_messages.append(f"role:{role}, message:{message}")
        self.messages.append({"role": role, "content": [{"type": "text", "text": message}]})

    def prompt_llm(self, request_prompt, _weave_context: dict | None = None):
        # Add user message and call LiteLLM.
        self.add_message(request_prompt)
        if self.fake_mode:
            call_start = time.time()
            fake_response = self._fake_response(request_prompt)
            call_end = time.time()
            self._add_assistant_message(fake_response, None)

            # record fake calls for consistency in smoke tests
            call_uuid = uuid.uuid4().hex
            call_record = {
                "call_idx": len(self._call_history),
                "call_uuid": call_uuid,
                "timestamp": call_end,
                "model": self.model_name,
                "prompt": request_prompt[:10000] if request_prompt else "",
                "response": fake_response[:10000] if fake_response else "",
                "latency_ms": round((call_end - call_start) * 1000, 1),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
                "has_reasoning": False,
            }
            self._call_history.append(call_record)

            if self._call_recorder is not None:
                disk_record = dict(call_record)
                disk_record["agent"] = self._agent_name
                disk_record["call_type"] = "fake"
                disk_record["prompt"] = request_prompt or ""
                disk_record["response"] = fake_response or ""
                self._call_recorder.record_dict(disk_record)

            # record usage for consistency with real LLM path
            self._record_usage(
                prompt_tokens=0,
                completion_tokens=0,
                reasoning_tokens=0,
                cost_usd=0.0,
                latency_ms=round((call_end - call_start) * 1000, 1),
            )

            return fake_response
        base = (
            self.api_base or os.environ.get("LITELLM_API_BASE") or os.environ.get("OPENAI_API_BASE")
        )
        key = self.api_key or os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

        # ensure model is in weave context for both responses and completion paths
        if _weave_context is None:
            _weave_context = {}
        _weave_context.setdefault("model", self.model_name)

        call_start = time.time()

        if self._is_responses_api_model(self.model_name):
            # Responses-only models: flatten messages into text
            prompt_lines = []
            for m in self.messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict) and "text" in c:
                            parts.append(str(c["text"]))
                        else:
                            parts.append(str(c))
                    content = " ".join(parts)
                prompt_lines.append(f"{role}: {content}")
            input_text = "\n".join(prompt_lines)

            def _responses_call(input_text_local: str):
                return litellm.responses(
                    model=self.model_name,
                    input=input_text_local,
                    api_base=base,
                    api_key=key,
                    max_output_tokens=self.max_tokens,
                    num_retries=3,
                    timeout=180,
                    # some Responses-only models reject temperature; drop it to avoid 400s
                )

            resp = _maybe_weave_op(_responses_call, _weave_context)(input_text)
        else:
            call_params = {
                "model": self.model_name,
                "messages": self.messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "num_retries": 3,
                "timeout": 180,
            }
            # provider overrides if supplied (or via env for local MLX/vLLM)
            if base:
                call_params["api_base"] = base
            # determine bedrock bearer token auth mode (api_key injected only at call time to prevent leaks)
            is_bedrock_bearer = (
                self.model_name.startswith("bedrock/")
                and bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK"))
                and not bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
            )
            # NOTE: api_key is NOT added to call_params here to prevent leakage via exception traces
            # It's injected at call time inside _completion_call()
            # custom LLM provider for non-native endpoints (GLM, MiniMax, Kimi)
            if self.custom_llm_provider:
                call_params["custom_llm_provider"] = self.custom_llm_provider
            # extra headers (Kimi requires User-Agent identifying coding agent)
            if self.extra_headers:
                call_params["extra_headers"] = self.extra_headers
            # thinking mode support via extra_body (DeepSeek V3.2, Kimi K2)
            should_enable_thinking = (
                self.enable_thinking
                or "deepseek-reasoner" in self.model_name.lower()
                or "speciale" in self.model_name.lower()
                or "kimi-k2-thinking" in self.model_name.lower()
            )
            # ex: DeepSeek requires extra_body for thinking mode
            # but Kimi K2 thinking is native (no extra_body needed)
            if should_enable_thinking and self._is_deepseek_model(self.model_name):
                call_params["extra_body"] = {"thinking": {"type": "enabled"}}

            # Bedrock bearer token auth: pass dummy creds to satisfy boto3
            if self.model_name.startswith("bedrock/"):
                has_bearer = bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK"))
                has_std_creds = bool(
                    os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")
                )
                if has_bearer and not has_std_creds:
                    call_params["aws_access_key_id"] = "DUMMY"
                    call_params["aws_secret_access_key"] = "DUMMY"
                region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
                if region:
                    call_params["aws_region_name"] = region

            # call_params is already safe (no api_key) - use directly for tracing

            def _completion_call(**params):
                # inject secrets only at call time to prevent leakage via exception traces
                call_kwargs = dict(params)
                if key and not is_bedrock_bearer:
                    call_kwargs["api_key"] = key
                if base:
                    call_kwargs["api_base"] = base
                return litellm.completion(**call_kwargs)

            resp = _maybe_weave_op(_completion_call, _weave_context)(**call_params)

        content = None
        reasoning_content = None

        # Responses API objects expose `.output`
        if content is None and hasattr(resp, "output"):
            try:
                for item in getattr(resp, "output", []) or []:
                    item_type = getattr(item, "type", None) or (
                        isinstance(item, dict) and item.get("type")
                    )
                    if item_type == "message":
                        msg_content = getattr(item, "content", None) or (
                            item.get("content") if isinstance(item, dict) else None
                        )
                        if msg_content:
                            for c in msg_content:
                                text = getattr(c, "text", None)
                                if text is None and isinstance(c, dict):
                                    text = c.get("text")
                                if text:
                                    content = text
                                    break
                    if content:
                        break
            except Exception as e:
                logger.debug(f"Responses API parsing fallback: {e}")

        if content is None:
            try:
                msg = resp.choices[0].message
                content = msg.content
                reasoning_content = getattr(msg, "reasoning_content", None)
                if reasoning_content is None and hasattr(msg, "__dict__"):
                    reasoning_content = msg.__dict__.get("reasoning_content")
            except Exception as e:
                logger.debug(f"Standard message parsing failed, trying dict access: {e}")
                try:
                    content = resp["choices"][0]["message"]["content"]
                    reasoning_content = resp["choices"][0]["message"].get("reasoning_content")
                except Exception as e2:
                    logger.debug(f"Dict access also failed, using empty content: {e2}")
                    content = ""

        # usage metrics
        call_end = time.time()
        latency_ms = (call_end - call_start) * 1000

        # token usage from response
        usage = getattr(resp, "usage", None)
        if usage is None and hasattr(resp, "__dict__"):
            usage = resp.__dict__.get("usage")
        if usage is None and isinstance(resp, dict):
            usage = resp.get("usage")

        tokens = extract_token_usage(usage)
        prompt_tokens = tokens["prompt_tokens"]
        completion_tokens = tokens["completion_tokens"]
        reasoning_tokens = tokens["reasoning_tokens"]
        cost = 0.0

        if usage:
            # cost using LiteLLM's built-in cost tracking
            try:
                cost = litellm.completion_cost(completion_response=resp) or 0.0
            except Exception as e:
                logger.debug(f"LiteLLM cost calculation failed, using fallback: {e}")

            # fallback to unified pricing registry if LiteLLM didn't return cost
            if not cost:
                pricing = MODEL_REGISTRY.get(self.model_name)
                if pricing and (prompt_tokens or completion_tokens or reasoning_tokens):
                    cost = prompt_tokens * pricing.get("input_cost_per_token", 0) + (
                        completion_tokens + reasoning_tokens
                    ) * pricing.get("output_cost_per_token", 0)

        # record usage via mixin helper
        self._record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

        # some reasoning models return the answer in reasoning_content.
        full_response = content or reasoning_content or ""

        # record call in history for WandB logging (independent of Weave)
        call_uuid = uuid.uuid4().hex
        call_record = {
            "call_idx": len(self._call_history),
            "call_uuid": call_uuid,
            "timestamp": call_end,
            "model": self.model_name,
            "prompt": request_prompt[:10000] if request_prompt else "",  # truncate for storage
            "response": full_response[:10000] if full_response else "",  # truncate for storage
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cost_usd": cost or 0.0,
            "has_reasoning": bool(reasoning_content),
        }
        self._call_history.append(call_record)

        # immediately write to disk for crash resilience
        if self._call_recorder is not None:
            disk_record = dict(call_record)
            disk_record["agent"] = self._agent_name
            disk_record["call_type"] = "chat"
            # use full text for disk (not truncated like in-memory)
            disk_record["prompt"] = request_prompt or ""
            disk_record["response"] = full_response or ""
            self._call_recorder.record_dict(disk_record)

        self._add_assistant_message(full_response, reasoning_content)
        return full_response

    def _add_assistant_message(self, content: str, reasoning_content: str | None = None):
        """Add assistant message with optional reasoning_content."""
        self.all_messages.append(f"role:assistant, message:{content}")
        msg = {"role": "assistant", "content": [{"type": "text", "text": content}]}
        # include reasoning_content for thinking models
        if reasoning_content and self._is_thinking_model(self.model_name):
            msg["reasoning_content"] = reasoning_content
        self.messages.append(msg)

    @staticmethod
    def _is_deepseek_model(model_name: str) -> bool:
        """True for DeepSeek models (thinking support)."""
        name = (model_name or "").lower()
        return "deepseek" in name

    @staticmethod
    def _is_kimi_model(model_name: str) -> bool:
        """True for Kimi/Moonshot models (thinking support)."""
        name = (model_name or "").lower()
        return "kimi" in name or "moonshot" in name or "kimi-for-coding" in name

    @staticmethod
    def _is_thinking_model(model_name: str) -> bool:
        """True if model supports reasoning_content."""
        name = (model_name or "").lower()
        return "deepseek" in name or "kimi" in name or "moonshot" in name

    @staticmethod
    def _fake_mode_enabled() -> bool:
        flag = os.getenv("BOXINGGYM_FAKE_LLM", "")
        return bool(flag) and flag.lower() not in ("0", "false", "no")

    @staticmethod
    def _default_max_tokens(model_name: str) -> int:
        """Return API max output tokens (via _model_profile)."""
        return _model_profile(model_name).api_max_tokens

    @staticmethod
    def _is_responses_api_model(model_name: str) -> bool:
        return is_responses_api_model(model_name)

    def _fake_response(self, request_prompt: str) -> str:
        """
        Deterministic, parseable response for offline smoke tests.
        """
        lower_prompt = request_prompt.lower()
        is_answer = "<answer" in lower_prompt or "answer>" in lower_prompt
        is_observe = (
            "<observe" in lower_prompt
            or "observe>" in lower_prompt
            or "observation" in lower_prompt
        )
        tag = (
            "answer" if (is_answer and not is_observe) else ("observe" if is_observe else "answer")
        )
        content = self._fake_content_for_prompt(lower_prompt)
        return f"<thought>offline smoke test</thought>\n<{tag}>{content}</{tag}>"

    def _fake_content_for_prompt(self, lower_prompt: str) -> str:
        """
        Payloads that satisfy validation across environments without real LLM calls
        """
        hint = os.getenv("BOXINGGYM_FAKE_LLM", "").lower()
        if "dugong" in lower_prompt or "sea cow" in lower_prompt or hint.startswith("dugong"):
            return "2.0"  # age within [0,5]
        if (
            "flow rate" in lower_prompt
            or "surfactant" in lower_prompt
            or "droplet" in lower_prompt
            or hint.startswith("microfluidics")
        ):
            return "concentration: 0.8, flow_rate: 45, surfactant: 0.018"
        if "lotka" in lower_prompt or "predator" in lower_prompt:
            return "0.5, 0.4"
        if "peregrine" in lower_prompt:
            return "0.5"
        return "0.5"

    def parse_response(self, response, is_observation):
        regex = _OBSERVE_RE if is_observation else _ANSWER_RE
        match = regex.search(response)
        if match:
            return match.group(1).strip()
        else:
            return None

    def parse_response_v2(self, response: str, is_observation: bool) -> str | None:
        """Robust parser using TagParser with fallback to legacy regex."""
        tag = "observe" if is_observation else "answer"
        parser = TagParser(tag_name=tag, validation_mode="moderate")
        parsed = parser.parse(response)
        if parsed is not None:
            return parsed
        # fallback to the original simple regex logic
        return self.parse_response(response, is_observation)

    def _diagnose_parse_failure_v2(self, response: str, is_observation: bool) -> str:
        """Human-readable reason why parsing failed (used in tests and retry prompts)."""
        tag = "observe" if is_observation else "answer"
        parser = TagParser(tag_name=tag, validation_mode="moderate")
        return parser.diagnose_failure(response)

    def prompt_llm_and_parse(self, request_prompt, is_observation, max_tries=4):
        used_retries = 0
        retrying_after_parse_fail = False
        for i in range(max_tries):
            weave_ctx = {
                "is_retry": i > 0,
                "retrying_after_parse_fail": retrying_after_parse_fail,
                "attempt": i + 1,
            }
            full_response = self.prompt_llm(request_prompt, _weave_context=weave_ctx)
            # use robust tag parsing (v2) with fallback to legacy regex.
            response = self.parse_response_v2(full_response, is_observation)
            if response is not None:
                # check if response has numbers (digits or common word-form numbers)
                # relaxed pattern: digits, decimals, or word numbers like "zero", "one", "half"
                has_digits = bool(_HAS_DIGIT_RE.search(response))
                has_word_numbers = bool(_WORD_NUMBERS_RE.search(response))
                if not has_digits and not has_word_numbers:
                    response = None
            if response is None or ("done" in str(response).lower()):
                # structured feedback about what went wrong, then retry
                retrying_after_parse_fail = True
                diag = self._diagnose_parse_failure_v2(full_response, is_observation)
                if is_observation:
                    request_prompt = (
                        f"{diag} "
                        "Please stick to the specified format and respond using <observe> tags. "
                        "Continue making observations even if you think you have an accurate estimate. "
                        "Your previous response was not valid."
                    )
                else:
                    request_prompt = (
                        f"{diag} "
                        "Please stick to the specified format and respond using <answer> tags. "
                        "Make assumptions and provide your best guess. "
                        "Your previous response was not valid."
                    )
                used_retries += 1
            else:
                break
        # track retries (extra attempts beyond the first) for W&B usage stats
        if used_retries > 0:
            self._record_retry(used_retries)
        if used_retries == max_tries:
            self._record_error()
            raise ValueError("Failed to get valid response after retries")
        return response, used_retries

    def generate_predictions(self, request_prompt):
        request_prompt += "\nAnswer in the following format:\n<answer>your answer</answer>."
        prediction, used_retries = self.prompt_llm_and_parse(request_prompt, False)
        self.messages = self.messages[: -2 * (used_retries + 1)]  # Remove the last 2 messages
        return prediction

    def generate_actions(self, experiment_results=None):
        if experiment_results is None:
            follow_up_prompt = "Think about where to observe next. Articulate your strategy for choosing measurements in <thought>.\nProvide a new measurement point in the format:\n<thought>your thought</thought>\n<observe> your observation</observe>\nMake an observation now."
        else:
            follow_up_prompt = (
                "Result: "
                + str(experiment_results)
                + "\nThink about where to observe next. Articulate your strategy for choosing measurements in <thought>.\nProvide a new measurement point in the format:\nThought: <thought>\n<observe> your observation (remember the type of inputs accepted)</observe>"
            )
        observe, used_retries = self.prompt_llm_and_parse(follow_up_prompt, True)
        return observe

    def print_log(self):
        for entry in self.messages:
            print("Message Type:", type(entry).__name__)
            if isinstance(entry, dict):
                print("Content:", entry.get("content"))
            else:
                print("Content:", getattr(entry, "content", entry))
            print("------")

    # get_usage_stats(), get_latencies_ms(), reset_usage_stats() inherited from UsageTrackerMixin

    def get_call_history(self) -> list[dict]:
        """Return per-call history for WandB logging (independent of Weave)."""
        return list(self._call_history)

    def clone_for_eval(self, share_call_recorder: bool = True) -> "LMExperimenter":
        """Create a thread-safe clone for evaluation.

        Copies prompt context while isolating mutable message state.
        """
        clone = LMExperimenter(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_base=self.api_base,
            api_key=self.api_key,
            enable_thinking=self.enable_thinking,
            allow_truncation=self.allow_truncation,
            custom_llm_provider=self.custom_llm_provider,
            extra_headers=self.extra_headers,
        )
        clone.system = self.system
        # deep copy required: messages have nested content lists that must be isolated
        clone.messages = copy.deepcopy(self.messages)
        clone.all_messages = list(self.all_messages)
        clone.fake_mode = self.fake_mode
        if share_call_recorder and self._call_recorder is not None:
            clone.set_call_recorder(self._call_recorder, agent_name=self._agent_name)
        return clone

    def set_call_recorder(self, recorder: CallRecorder, agent_name: str = "scientist") -> None:
        """Attach a disk-backed call recorder for crash-resilient logging."""
        self._call_recorder = recorder
        self._agent_name = agent_name
