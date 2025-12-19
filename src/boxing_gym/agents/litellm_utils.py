"""LiteLLM helpers.

Small pieces of LiteLLM/OpenAI Responses-API glue used across agents and environments.
"""

from __future__ import annotations

from typing import Any, Optional, List, Dict
import os
import re

import litellm


# safe global defaults for LiteLLM
litellm.drop_params = True
litellm.set_verbose = False
if hasattr(litellm, "suppress_debug_info"):
    litellm.suppress_debug_info = True
if hasattr(litellm, "turn_off_message_logging"):
    litellm.turn_off_message_logging = True


def is_responses_api_model(model_name: str) -> bool:
    """Return True if the model must be called via /v1/responses."""
    name = (model_name or "").lower()
    if "gpt-5" in name:
        return True
    # any OpenAI o‑series reasoning model (o1, o3, o4-mini, etc.)
    if re.search(r"(^|/|\s)o\d", name):
        return True
    if "gpt-4.1" in name:
        return True
    return False


def _fake_llm_enabled() -> bool:
    flag = os.getenv("BOXINGGYM_FAKE_LLM", "")
    return bool(flag) and flag.lower() not in ("0", "false", "no")


def messages_to_input_text(messages: List[Dict[str, Any]]) -> str:
    """Flatten chat-style messages into a single text prompt.

    The Responses API does not accept structured role/message arrays for
    all providers, so we collapse to a simple transcript.
    """
    lines: List[str] = []
    for m in messages or []:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            parts: List[str] = []
            for c in content:
                if isinstance(c, dict):
                    if "text" in c:
                        parts.append(str(c["text"]))
                    elif "value" in c:
                        parts.append(str(c["value"]))
                    else:
                        parts.append(str(c))
                else:
                    parts.append(str(c))
            content = " ".join(parts)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def extract_text(resp: Any) -> str:
    """Extract plain text from either ChatCompletions or Responses schema."""
    # ChatCompletions-style
    try:
        if hasattr(resp, "choices") and getattr(resp, "choices"):
            msg = resp.choices[0].message
            content = getattr(msg, "content", None)
            if content:
                return content
    except Exception:
        pass

    # Responses-style
    output = getattr(resp, "output", None)
    if output:
        for item in output:
            item_type = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            if item_type == "message":
                msg_content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
                if msg_content:
                    for c in msg_content:
                        text = getattr(c, "text", None)
                        if text is None and isinstance(c, dict):
                            text = c.get("text")
                        if text:
                            return text

    # dict fallback (rare)
    try:
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices:
                return (choices[0].get("message") or {}).get("content") or ""
    except Exception:
        pass

    return ""


def call_llm_messages_sync(
    model_name: str,
    messages: List[Dict[str, Any]],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    num_retries: int = 3,
    timeout: int = 180,
    **kwargs,
) -> Any:
    """Call LiteLLM synchronously from a messages list.

    Returns the raw LiteLLM response object (ChatCompletions or Responses).
    """
    if _fake_llm_enabled():
        return {"choices": [{"message": {"content": "mock response"}}]}

    # avoid double-passing core parameters that we set below
    reserved_keys = {"messages", "max_tokens", "temperature", "input", "max_output_tokens"}
    safe_kwargs = {k: v for k, v in kwargs.items() if k not in reserved_keys}

    # Bedrock bearer token auth: pass dummy creds to satisfy boto3
    # and DON'T pass api_key (let LiteLLM use AWS_BEARER_TOKEN_BEDROCK env var)
    is_bedrock_bearer = False
    if model_name.startswith("bedrock/"):
        has_bearer = bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK"))
        has_std_creds = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
        if has_bearer and not has_std_creds:
            safe_kwargs["aws_access_key_id"] = "DUMMY"
            safe_kwargs["aws_secret_access_key"] = "DUMMY"
            is_bedrock_bearer = True
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        if region:
            safe_kwargs["aws_region_name"] = region

    # for bedrock bearer auth, don't pass api_key, let LiteLLM use env var
    effective_api_key = None if is_bedrock_bearer else api_key

    if is_responses_api_model(model_name):
        input_text = messages_to_input_text(messages)
        return litellm.responses(
            model=model_name,
            input=input_text,
            api_base=api_base,
            api_key=effective_api_key,
            max_output_tokens=max_tokens,
            num_retries=num_retries,
            timeout=timeout,
            **safe_kwargs,
        )

    return litellm.completion(
        model=model_name,
        messages=messages,
        api_base=api_base,
        api_key=effective_api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        num_retries=num_retries,
        timeout=timeout,
        **safe_kwargs,
    )


def call_llm_sync(
    model_name: str,
    system_text: str,
    user_text: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    num_retries: int = 3,
    timeout: int = 180,
) -> str:
    """Convenience wrapper for simple system+user prompts."""
    if _fake_llm_enabled():
        return "The player feels happy because they won."
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    resp = call_llm_messages_sync(
        model_name=model_name,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        num_retries=num_retries,
        timeout=timeout,
    )
    return extract_text(resp)
