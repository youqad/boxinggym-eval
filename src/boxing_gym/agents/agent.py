import re
import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import litellm
from boxing_gym.agents.tag_parser import TagParser
from boxing_gym.agents.litellm_utils import (
    is_responses_api_model,
    extract_text,
    messages_to_input_text,
)

logger = logging.getLogger(__name__)

_CUSTOM_MODEL_PRICING = {
    # MiniMax M2 (204K context)
    "openai/MiniMax-M2": {
        "input_cost_per_token": 0.0000003,   # $0.30 / 1M tokens
        "output_cost_per_token": 0.0000012,  # $1.20 / 1M tokens
        "max_tokens": 204800,
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "MiniMax-M2": {
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000012,
        "max_tokens": 204800,
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "minimax/MiniMax-M2": {
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000012,
        "max_tokens": 204800,
        "litellm_provider": "minimax",
        "mode": "chat",
    },
    # DeepSeek V3.2 (cache miss pricing)
    "deepseek/deepseek-chat": {
        "input_cost_per_token": 0.00000028,  # $0.28 / 1M tokens (cache miss)
        "output_cost_per_token": 0.00000042, # $0.42 / 1M tokens
        "max_tokens": 128000,
        "litellm_provider": "deepseek",
        "mode": "chat",
    },
    "deepseek/deepseek-reasoner": {
        "input_cost_per_token": 0.00000028,  # $0.28 / 1M tokens (cache miss)
        "output_cost_per_token": 0.00000042, # $0.42 / 1M tokens
        "max_tokens": 128000,
        "litellm_provider": "deepseek",
        "mode": "chat",
    },
    # GLM-4.6 (Z.AI / Zhipu AI)
    "anthropic/glm-4.6": {
        "input_cost_per_token": 0.0000004,   # $0.40 / 1M tokens
        "output_cost_per_token": 0.00000175, # $1.75 / 1M tokens
        "max_tokens": 200000,
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "glm-4.6": {
        "input_cost_per_token": 0.0000004,   # $0.40 / 1M tokens
        "output_cost_per_token": 0.00000175, # $1.75 / 1M tokens
        "max_tokens": 200000,
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
}

try:
    litellm.register_model(model_cost=_CUSTOM_MODEL_PRICING)
except Exception:
    pass

# model profiles for token limits
@dataclass(frozen=True)
class ModelProfile:
    """Token limits and reasoning flag."""
    api_max_tokens: int
    is_reasoning: bool
    min_viable_tokens: Optional[int] = None  # Minimum for reasoning to work; defaults to api_max_tokens

    @property
    def effective_min(self) -> int:
        return self.min_viable_tokens if self.min_viable_tokens is not None else self.api_max_tokens


def _model_profile(model_name: str) -> ModelProfile:
    """Token limits + reasoning flag for a model."""
    name = (model_name or "").lower()

    if "speciale" in name or "v3.2_speciale" in name:
        return ModelProfile(api_max_tokens=131072, is_reasoning=True)

    if "deepseek-reasoner" in name or "deepseek-r1" in name:
        return ModelProfile(api_max_tokens=8192, is_reasoning=True)

    if "deepseek-chat" in name or "deepseek-v3" in name:
        return ModelProfile(api_max_tokens=4096, is_reasoning=False)

    if "gpt-5" in name:
        return ModelProfile(api_max_tokens=131072, is_reasoning=True)

    if re.search(r"(^|/|\s)o\d", name):
        return ModelProfile(api_max_tokens=100000, is_reasoning=True)

    if ("kimi" in name or "moonshot" in name) and ("thinking" in name or "kimi-for-coding" in name):
        return ModelProfile(api_max_tokens=32768, is_reasoning=True)

    if "minimax" in name:
        return ModelProfile(api_max_tokens=196608, is_reasoning=True)

    if "glm-4" in name or "glm4" in name:
        return ModelProfile(api_max_tokens=131072, is_reasoning=False)

    if "kimi" in name or "moonshot" in name:
        return ModelProfile(api_max_tokens=32768, is_reasoning=False)

    if "gpt-4o" in name:
        return ModelProfile(api_max_tokens=16384, is_reasoning=False)

    if is_responses_api_model(model_name):
        return ModelProfile(api_max_tokens=16384, is_reasoning=False)

    return ModelProfile(api_max_tokens=16384, is_reasoning=False)


def _compute_effective_max_tokens(
    model_name: str,
    user_max_tokens: Optional[int],
    allow_truncation: bool = False,
) -> Tuple[int, bool]:
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


def _extract_usage_dict(resp) -> Optional[dict]:
    usage = getattr(resp, "usage", None)
    if usage is None and hasattr(resp, "__dict__"):
        usage = resp.__dict__.get("usage")
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if not usage:
        return None
    try:
        if hasattr(usage, "prompt_tokens"):
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "reasoning_tokens": getattr(usage, "reasoning_tokens", 0) or 0,
            }
        if isinstance(usage, dict):
            return {
                "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0,
                "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0,
                "reasoning_tokens": usage.get("reasoning_tokens", 0) or 0,
            }
    except Exception:
        return None
    return None


def _weave_log_payload(payload: dict) -> None:
    global _WEAVE_ECHO_OP
    if weave is None:
        return
    try:
        if _WEAVE_ECHO_OP is None:
            def _echo(x):
                return x
            _WEAVE_ECHO_OP = weave.op(_echo)
        _WEAVE_ECHO_OP(payload)
    except Exception:
        pass


def _maybe_weave_op(fn, weave_context: Optional[dict] = None):
    # conditionally wrap a function with weave.op for tracing.
    # we log a sanitized payload (prompt + text + usage), not raw responses.

    if weave is None:
        return fn
    project = os.environ.get("WEAVE_PROJECT") or os.environ.get("WANDB_PROJECT")
    disabled = os.environ.get("WEAVE_DISABLED", "0").lower() in ("1", "true", "yes")
    if not project or disabled:
        return fn

    def wrapped(*args, **kwargs):
        result = fn(*args, **kwargs)
        try:
            payload = dict(weave_context or {})
            if "model" not in payload:
                payload["model"] = kwargs.get("model")
            prompt = payload.get("prompt")
            if prompt is None:
                if "messages" in kwargs:
                    prompt = messages_to_input_text(kwargs.get("messages") or [])
                elif "input" in kwargs:
                    prompt = kwargs.get("input")
                elif args:
                    prompt = args[0]
            payload["prompt"] = prompt
            payload["output"] = extract_text(result)
            payload["usage"] = _extract_usage_dict(result)
            _weave_log_payload(payload)
        except Exception:
            pass
        return result

    return wrapped

class LMExperimenter:
    def __init__(
        self,
        model_name,
        temperature=0.0,
        max_tokens: Optional[int] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_thinking: bool = False,
        allow_truncation: bool = False,
        custom_llm_provider: Optional[str] = None,
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
        self.system = None
        self.messages = []
        self.all_messages = []
        # track reasoning_content for thinking models.
        self._last_reasoning_content = None
        # offline mode for smoke tests
        self.fake_mode = self._fake_mode_enabled()
        # LiteLLM global flags.
        litellm.drop_params = True
        litellm.modify_params = True  # param conversion for custom endpoints
        litellm.set_verbose = False

        # tracking for W&B metrics.
        self._usage_stats = {
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

    def set_system_message(self, message):
        self.all_messages.append(f"role:system, message:{message}")
        self.system = message
        # content parts format for compatibility.
        self.messages.append({"role": "system", "content": [{"type": "text", "text": message}]})
        
    def add_message(self, message, role='user'):
        self.all_messages.append(f"role:{role}, message:{message}")
        self.messages.append(
            {
                "role": role,
                "content": [
                    {"type": "text", "text": message}
                ]
            }
        )

    def prompt_llm(self, request_prompt):
        # Add user message and call LiteLLM.
        self.add_message(request_prompt)
        if self.fake_mode:
            fake_response = self._fake_response(request_prompt)
            self._last_reasoning_content = None
            self._add_assistant_message(fake_response, None)
            return fake_response
        base = self.api_base or os.environ.get("LITELLM_API_BASE") or os.environ.get("OPENAI_API_BASE")
        key = self.api_key or os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

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

            resp = _maybe_weave_op(_responses_call)(input_text)
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
            # don't pass api_key for bedrock models using bearer token auth (LiteLLM must fall back to AWS_BEARER_TOKEN_BEDROCK)
            is_bedrock_bearer = (
                self.model_name.startswith("bedrock/") and
                bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK")) and
                not bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
            )
            if key and not is_bedrock_bearer:
                call_params["api_key"] = key
            # custom LLM provider for non-native endpoints (GLM, MiniMax, Kimi)
            if self.custom_llm_provider:
                call_params["custom_llm_provider"] = self.custom_llm_provider
            # thinking mode support via extra_body (DeepSeek V3.2, Kimi K2)
            should_enable_thinking = (
                self.enable_thinking or
                "deepseek-reasoner" in self.model_name.lower() or
                "speciale" in self.model_name.lower() or
                "kimi-k2-thinking" in self.model_name.lower()
            )
            # ex: DeepSeek requires extra_body for thinking mode
            # but Kimi K2 thinking is native (no extra_body needed)
            if should_enable_thinking and self._is_deepseek_model(self.model_name):
                call_params["extra_body"] = {"thinking": {"type": "enabled"}}

            # Bedrock bearer token auth: pass dummy creds to satisfy boto3
            if self.model_name.startswith("bedrock/"):
                has_bearer = bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK"))
                has_std_creds = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
                if has_bearer and not has_std_creds:
                    call_params["aws_access_key_id"] = "DUMMY"
                    call_params["aws_secret_access_key"] = "DUMMY"
                region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
                if region:
                    call_params["aws_region_name"] = region

            # safe params dict for tracing (avoid logging secrets)
            call_params_safe = dict(call_params)
            if "api_key" in call_params_safe:
                call_params_safe["api_key"] = "<redacted>"

            def _completion_call(**params_safe):
                params = dict(params_safe)
                # re-inject real secrets before calling LiteLLM.
                if key and not is_bedrock_bearer:
                    params["api_key"] = key
                if base:
                    params["api_base"] = base
                return litellm.completion(**params)

            resp = _maybe_weave_op(_completion_call)(**call_params_safe)

        content = None
        reasoning_content = None

        # Responses API objects expose `.output`
        if content is None and hasattr(resp, "output"):
            try:
                for item in getattr(resp, "output", []) or []:
                    item_type = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
                    if item_type == "message":
                        msg_content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
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
            except Exception:
                pass

        if content is None:
            try:
                msg = resp.choices[0].message
                content = msg.content
                reasoning_content = getattr(msg, "reasoning_content", None)
                if reasoning_content is None and hasattr(msg, "__dict__"):
                    reasoning_content = msg.__dict__.get("reasoning_content")
            except Exception:
                try:
                    content = resp["choices"][0]["message"]["content"]
                    reasoning_content = resp["choices"][0]["message"].get("reasoning_content")
                except Exception:
                    content = ""

        # store reasoning_content for multi-turn conversations.
        self._last_reasoning_content = reasoning_content

        # usage metrics
        call_end = time.time()
        latency_ms = (call_end - call_start) * 1000
        self._usage_stats["latencies_ms"].append(latency_ms)
        self._usage_stats["call_count"] += 1

        # token usage from response
        usage = getattr(resp, "usage", None)
        if usage is None and hasattr(resp, "__dict__"):
            usage = resp.__dict__.get("usage")
        if usage is None and isinstance(resp, dict):
            usage = resp.get("usage")

        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or (usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0)
            completion_tokens = getattr(usage, "completion_tokens", 0) or (usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0)
            reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or (usage.get("reasoning_tokens", 0) if isinstance(usage, dict) else 0)

            self._usage_stats["prompt_tokens"] += prompt_tokens
            self._usage_stats["completion_tokens"] += completion_tokens
            self._usage_stats["reasoning_tokens"] += reasoning_tokens
            # include reasoning_tokens in total (separate from completion_tokens)
            self._usage_stats["total_tokens"] += prompt_tokens + completion_tokens + reasoning_tokens

            # cost using LiteLLM's built-in cost tracking.
            try:
                cost = litellm.completion_cost(completion_response=resp)
                if cost:
                    self._usage_stats["total_cost_usd"] += cost
            except Exception:
                pass

        # some reasoning models return the answer in reasoning_content.
        full_response = content or reasoning_content or ""
        self._add_assistant_message(full_response, reasoning_content)
        return full_response

    def _add_assistant_message(self, content: str, reasoning_content: Optional[str] = None):
        """Add assistant message with optional reasoning_content."""
        self.all_messages.append(f"role:assistant, message:{content}")
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": content}]
        }
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
        return (
            "deepseek" in name or
            "kimi" in name or
            "moonshot" in name
        )

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
        is_observe = "<observe" in lower_prompt or "observe>" in lower_prompt or "observation" in lower_prompt
        tag = "answer" if (is_answer and not is_observe) else ("observe" if is_observe else "answer")
        content = self._fake_content_for_prompt(lower_prompt)
        return f"<thought>offline smoke test</thought>\n<{tag}>{content}</{tag}>"

    def _fake_content_for_prompt(self, lower_prompt: str) -> str:
        """
        Payloads that satisfy validation across environments without real LLM calls
        """
        hint = os.getenv("BOXINGGYM_FAKE_LLM", "").lower()
        if "dugong" in lower_prompt or "sea cow" in lower_prompt or hint.startswith("dugong"):
            return "2.0"  # age within [0,5]
        if "flow rate" in lower_prompt or "surfactant" in lower_prompt or "droplet" in lower_prompt or hint.startswith("microfluidics"):
            return "concentration: 0.8, flow_rate: 45, surfactant: 0.018"
        if "lotka" in lower_prompt or "predator" in lower_prompt:
            return "0.5, 0.4"
        if "peregrine" in lower_prompt:
            return "0.5"
        return "0.5"

    def parse_response(self, response, is_observation):
        if is_observation:
            pattern = r'<observe>(.*?)</observe>'   
        else:
            pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None

    def parse_response_v2(self, response: str, is_observation: bool) -> Optional[str]:
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
        for i in range(max_tries):
            full_response = self.prompt_llm(request_prompt)
            # print(full_response)
            # use robust tag parsing (v2) with fallback to legacy regex.
            response = self.parse_response_v2(full_response, is_observation)
            # print(f"parsed response: {response}")
            if response is not None:
                # check if response has numbers
                numbers = re.findall(r'[0-9]+', response)
                if len(numbers) == 0:
                    response = None
            if response is None or ("done" in str(response).lower()):
                # structured feedback about what went wrong, then retry
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
        try:
            if used_retries > 0:
                self._usage_stats["retry_count"] += used_retries
        except Exception:
            pass
        if used_retries == max_tries:
            try:
                self._usage_stats["error_count"] += 1
            except Exception:
                pass
            raise ValueError("Failed to get valid response after retries")
        return response, used_retries
    
    def generate_predictions(self, request_prompt):
        request_prompt += "\nAnswer in the following format:\n<answer>your answer</answer>."
        prediction, used_retries = self.prompt_llm_and_parse(request_prompt, False)
        self.messages = self.messages[:-2 * (used_retries+1)]  # Remove the last 2 messages
        # reset stale reasoning_content after truncation.
        self._last_reasoning_content = None
        return prediction
    
    def generate_actions(self, experiment_results=None):
        if experiment_results is None:
            follow_up_prompt = "Think about where to observe next. Articulate your strategy for choosing measurements in <thought>.\nProvide a new measurement point in the format:\n<thought>your thought</thought>\n<observe> your observation</observe>\nMake an observation now."
        else:
            follow_up_prompt = "Result: " + str(experiment_results) + "\nThink about where to observe next. Articulate your strategy for choosing measurements in <thought>.\nProvide a new measurement point in the format:\nThought: <thought>\n<observe> your observation (remember the type of inputs accepted)</observe>"
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

    def get_usage_stats(self) -> dict:
        """Return accumulated usage stats for W&B logging."""
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

    def reset_usage_stats(self):
        """Reset usage statistics (useful between budget evaluations)."""
        self._usage_stats = {
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
