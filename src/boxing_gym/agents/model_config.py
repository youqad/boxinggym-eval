"""Model configuration for LiteLLM providers.

This module provides API configuration lookup for models. The actual pricing
and model data is stored in pricing.py (single source of truth).

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
from typing import Any

# import from single source of truth
from boxing_gym.agents.pricing import MODEL_REGISTRY


def get_model_config(model_name: str) -> dict[str, Any]:
    """Get API configuration for a model.

    First checks MODEL_REGISTRY for exact match, then falls back to
    provider-based heuristics for unknown models.

    Args:
        model_name: The LiteLLM model name (e.g., "deepseek/deepseek-chat")

    Returns:
        Dict with keys:
            - model_name: The model name (same as input)
            - api_base: API endpoint URL or None for default
            - api_key: The actual API key value (from env var) or None
            - api_key_var: Name of the env var used (for debugging)
    """
    # check registry first (covers most cases)
    if model_name in MODEL_REGISTRY:
        cfg = MODEL_REGISTRY[model_name]
        api_key_var = cfg.get("api_key_var")
        api_key = os.environ.get(api_key_var) if api_key_var else None
        return {
            "model_name": model_name,
            "api_base": cfg.get("api_base"),
            "api_key": api_key,
            "api_key_var": api_key_var,
        }

    # fallback: provider detection for unknown models
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

    # Together.AI prefix match
    if model_lower.startswith("together_ai/") or "together" in model_lower:
        api_key = os.environ.get("TOGETHER_API_KEY")
        return {
            "model_name": model_name,
            "api_base": None,  # LiteLLM handles Together.AI routing
            "api_key": api_key,
            "api_key_var": "TOGETHER_API_KEY",
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

    # OpenAI/default: let LiteLLM handle routing
    api_key = os.environ.get("OPENAI_API_KEY")
    return {
        "model_name": model_name,
        "api_base": None,
        "api_key": api_key,
        "api_key_var": "OPENAI_API_KEY" if api_key else None,
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
        **kwargs,
    )
