"""Model configuration for LiteLLM providers.

Usage:
    from boxing_gym.agents.model_config import get_model_config

    config = get_model_config("deepseek/deepseek-chat")
    agent = LMExperimenter(
        model_name=config["model_name"],
        api_base=config["api_base"],
        api_key=config["api_key"]
    )
"""

import os
from typing import Dict, Any

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # OpenAI models (default, no special config needed)
    "gpt-4o": {"api_base": None, "api_key_var": "OPENAI_API_KEY"},
    "gpt-4o-mini": {"api_base": None, "api_key_var": "OPENAI_API_KEY"},
    "gpt-4": {"api_base": None, "api_key_var": "OPENAI_API_KEY"},
    "gpt-3.5": {"api_base": None, "api_key_var": "OPENAI_API_KEY"},
    "gpt-5": {"api_base": None, "api_key_var": "OPENAI_API_KEY"},
    "o1": {"api_base": None, "api_key_var": "OPENAI_API_KEY"},
    "o3": {"api_base": None, "api_key_var": "OPENAI_API_KEY"},

    # DeepSeek models
    "deepseek/deepseek-chat": {
        "input_cost_per_token": 0.00000028,  # $0.28 / 1M tokens (cache miss)
        "output_cost_per_token": 0.00000042, # $0.42 / 1M tokens
        "max_tokens": 8192,                  # API limit for output
        "litellm_provider": "deepseek",
        "mode": "chat",
        "api_base": "https://api.deepseek.com/v1",
        "api_key_var": "DEEPSEEK_API_KEY",
    },
    "deepseek/deepseek-reasoner": {
        "api_base": "https://api.deepseek.com/v1",
        "api_key_var": "DEEPSEEK_API_KEY",
    },

    # Kimi/Moonshot models (uses Anthropic-compatible API format)
    "anthropic/kimi-for-coding": {
        "api_base": "https://api.kimi.com/coding/",
        "api_key_var": "MOONSHOT_API_KEY",
    },
    "moonshot/kimi-k2": {
        "api_base": "https://api.kimi.com/coding/",
        "api_key_var": "MOONSHOT_API_KEY",
    },

    # MiniMax models (uses OpenAI-compatible API format)
    "openai/MiniMax-M2": {
        "api_base": "https://api.minimax.io/v1",
        "api_key_var": "MINIMAX_API_KEY",
    },

    # GLM/ZhipuAI models (uses Z.AI Anthropic-compatible endpoint)
    "anthropic/glm-4.6": {
        "api_base": "https://api.z.ai/api/anthropic",
        "api_key_var": "ZHIPUAI_API_KEY",
    },

    # Anthropic Claude models
    "claude-3": {"api_base": None, "api_key_var": "ANTHROPIC_API_KEY"},
    "claude-3.5": {"api_base": None, "api_key_var": "ANTHROPIC_API_KEY"},
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get API configuration for a model.

    Args:
        model_name: The LiteLLM model name (e.g., "deepseek/deepseek-chat")

    Returns:
        Dict with keys:
            - model_name: The model name (same as input)
            - api_base: API endpoint URL or None for default
            - api_key: The actual API key value (from env var) or None
            - api_key_var: Name of the env var used (for debugging)
    """
    if model_name in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[model_name]
        api_key = os.environ.get(cfg["api_key_var"]) if cfg.get("api_key_var") else None
        return {
            "model_name": model_name,
            "api_base": cfg.get("api_base"),
            "api_key": api_key,
            "api_key_var": cfg.get("api_key_var"),
        }

    model_lower = model_name.lower()

    # DeepSeek prefix match
    if model_lower.startswith("deepseek/") or "deepseek" in model_lower:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        return {
            "model_name": model_name,
            "api_base": "https://api.deepseek.com/v1",
            "api_key": api_key,
            "api_key_var": "DEEPSEEK_API_KEY",
        }

    # Kimi/Moonshot prefix match
    if "kimi" in model_lower or "moonshot" in model_lower:
        api_key = os.environ.get("MOONSHOT_API_KEY")
        return {
            "model_name": model_name,
            "api_base": "https://api.kimi.com/coding/",
            "api_key": api_key,
            "api_key_var": "MOONSHOT_API_KEY",
        }

    # MiniMax prefix match
    if "minimax" in model_lower:
        api_key = os.environ.get("MINIMAX_API_KEY")
        return {
            "model_name": model_name,
            "api_base": "https://api.minimax.io/v1",
            "api_key": api_key,
            "api_key_var": "MINIMAX_API_KEY",
        }

    # GLM/ZhipuAI prefix match
    if "glm" in model_lower or "zhipu" in model_lower:
        api_key = os.environ.get("ZHIPUAI_API_KEY")
        return {
            "model_name": model_name,
            "api_base": "https://api.z.ai/api/anthropic",
            "api_key": api_key,
            "api_key_var": "ZHIPUAI_API_KEY",
        }

    # Claude prefix match
    if "claude" in model_lower:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        return {
            "model_name": model_name,
            "api_base": None,
            "api_key": api_key,
            "api_key_var": "ANTHROPIC_API_KEY",
        }

    # unknown provider: let LiteLLM handle routing and API key lookup
    return {
        "model_name": model_name,
        "api_base": None,
        "api_key": None,
        "api_key_var": None,
    }


def create_experimenter(model_name: str, temperature: float = 0.0, **kwargs):
    """Create an LMExperimenter with API configuration.

    Convenience function that handles model-specific API config.

    Args:
        model_name: The LiteLLM model name
        temperature: Model temperature (default 0.0)
        **kwargs: Additional kwargs passed to LMExperimenter

    Returns:
        Configured LMExperimenter instance
    """
    from boxing_gym.agents.agent import LMExperimenter

    config = get_model_config(model_name)
    return LMExperimenter(
        model_name=config["model_name"],
        temperature=temperature,
        api_base=config["api_base"],
        api_key=config["api_key"],
        **kwargs
    )
