"""Unified model pricing and configuration registry.

This is the single source of truth for all model pricing, token limits, and API configuration.
Previously this data was scattered across agent.py, model_config.py, and base_agent.py.

Usage:
    from boxing_gym.agents.pricing import (
        get_model_pricing,
        get_input_cost,
        get_output_cost,
        get_api_config,
        MODEL_REGISTRY,
    )
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Unified model registry with all pricing and configuration data
# Schema: {
#   "input_cost_per_token": float,   # USD per token
#   "output_cost_per_token": float,  # USD per token
#   "max_tokens": int,               # Max output tokens (API limit)
#   "context_window": int,           # Max context window (optional)
#   "api_base": str | None,          # Custom API base URL
#   "api_key_var": str,              # Environment variable for API key
#   "litellm_provider": str,         # LiteLLM provider name
#   "mode": str,                     # "chat" | "completion"
# }

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # OpenAI Models
    # =========================================================================
    "gpt-4": {
        "input_cost_per_token": 3e-05,      # $30/1M tokens
        "output_cost_per_token": 6e-05,     # $60/1M tokens
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-4o": {
        "input_cost_per_token": 2.5e-06,    # $2.50/1M tokens
        "output_cost_per_token": 1e-05,     # $10/1M tokens
        "max_tokens": 16384,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-4o-mini": {
        "input_cost_per_token": 1.5e-07,    # $0.15/1M tokens
        "output_cost_per_token": 6e-07,     # $0.60/1M tokens
        "max_tokens": 16384,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-4o-2024-05-13": {
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 1e-05,
        "max_tokens": 16384,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-4o-2024-08-06": {
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 1e-05,
        "max_tokens": 16384,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-4-turbo": {
        "input_cost_per_token": 1e-05,      # $10/1M tokens
        "output_cost_per_token": 3e-05,     # $30/1M tokens
        "max_tokens": 4096,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-4-turbo-2024-04-09": {
        "input_cost_per_token": 1e-05,
        "output_cost_per_token": 3e-05,
        "max_tokens": 4096,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-4-1106-preview": {
        "input_cost_per_token": 1e-05,
        "output_cost_per_token": 3e-05,
        "max_tokens": 4096,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "openai/gpt-4-1106-preview": {
        "input_cost_per_token": 1e-05,
        "output_cost_per_token": 3e-05,
        "max_tokens": 4096,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-4-1106-vision-preview": {
        "input_cost_per_token": 1e-05,
        "output_cost_per_token": 3e-05,
        "max_tokens": 4096,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-4.1": {
        "input_cost_per_token": 2e-06,      # $2/1M tokens
        "output_cost_per_token": 8e-06,     # $8/1M tokens
        "max_tokens": 32768,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-3.5-turbo": {
        "input_cost_per_token": 5e-07,      # $0.50/1M tokens
        "output_cost_per_token": 1.5e-06,   # $1.50/1M tokens
        "max_tokens": 4096,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-3.5-turbo-0125": {
        "input_cost_per_token": 5e-07,
        "output_cost_per_token": 1.5e-06,
        "max_tokens": 4096,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-5": {
        "input_cost_per_token": 1.25e-06,   # $1.25/1M tokens
        "output_cost_per_token": 1e-05,     # $10/1M tokens
        "max_tokens": 131072,  # reasoning model needs headroom
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-5-mini": {
        "input_cost_per_token": 2.5e-07,    # $0.25/1M tokens
        "output_cost_per_token": 2e-06,     # $2/1M tokens
        "max_tokens": 32768,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-5-nano": {
        "input_cost_per_token": 5e-08,      # $0.05/1M tokens
        "output_cost_per_token": 4e-07,     # $0.40/1M tokens
        "max_tokens": 16384,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-5.1": {
        "input_cost_per_token": 1.25e-06,   # $1.25/1M tokens
        "output_cost_per_token": 1e-05,     # $10/1M tokens
        "max_tokens": 32768,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-5.1-mini": {
        "input_cost_per_token": 2.5e-07,    # $0.25/1M tokens
        "output_cost_per_token": 2e-06,     # $2/1M tokens
        "max_tokens": 32768,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-5.1-codex-mini": {
        "input_cost_per_token": 5e-07,      # $0.50/1M tokens
        "output_cost_per_token": 2e-06,     # $2/1M tokens
        "max_tokens": 32768,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-5.2": {
        "input_cost_per_token": 1.75e-06,   # $1.75/1M tokens
        "output_cost_per_token": 1.4e-05,   # $14/1M tokens
        "max_tokens": 32768,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "openai/gpt-5.2": {
        "input_cost_per_token": 1.75e-06,
        "output_cost_per_token": 1.4e-05,
        "max_tokens": 32768,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "o1": {
        "input_cost_per_token": 1.5e-05,    # $15/1M tokens
        "output_cost_per_token": 6e-05,     # $60/1M tokens
        "max_tokens": 32768,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "o1-mini": {
        "input_cost_per_token": 3e-06,      # $3/1M tokens
        "output_cost_per_token": 1.2e-05,   # $12/1M tokens
        "max_tokens": 65536,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "o3": {
        "input_cost_per_token": 1.5e-05,    # $15/1M tokens (estimated)
        "output_cost_per_token": 6e-05,     # $60/1M tokens (estimated)
        "max_tokens": 65536,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "o3-mini": {
        "input_cost_per_token": 1.1e-06,    # $1.10/1M tokens
        "output_cost_per_token": 4.4e-06,   # $4.40/1M tokens
        "max_tokens": 65536,
        "api_base": None,
        "api_key_var": "OPENAI_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },

    # =========================================================================
    # DeepSeek Models
    # =========================================================================
    "deepseek/deepseek-chat": {
        "input_cost_per_token": 2.8e-07,    # $0.28/1M tokens (cache miss)
        "output_cost_per_token": 4.2e-07,   # $0.42/1M tokens
        "max_tokens": 8192,                 # API output limit
        "context_window": 128000,           # Full context window
        "api_base": "https://api.deepseek.com/v1",
        "api_key_var": "DEEPSEEK_API_KEY",
        "litellm_provider": "deepseek",
        "mode": "chat",
    },
    "deepseek-chat": {
        "input_cost_per_token": 2.8e-07,
        "output_cost_per_token": 4.2e-07,
        "max_tokens": 8192,
        "context_window": 128000,
        "api_base": "https://api.deepseek.com/v1",
        "api_key_var": "DEEPSEEK_API_KEY",
        "litellm_provider": "deepseek",
        "mode": "chat",
    },
    "deepseek/deepseek-reasoner": {
        "input_cost_per_token": 5.5e-07,    # $0.55/1M tokens
        "output_cost_per_token": 2.19e-06,  # $2.19/1M tokens
        "max_tokens": 32768,  # reasoning model needs headroom
        "context_window": 128000,
        "api_base": "https://api.deepseek.com/v1",
        "api_key_var": "DEEPSEEK_API_KEY",
        "litellm_provider": "deepseek",
        "mode": "chat",
    },
    "deepseek-reasoner": {
        "input_cost_per_token": 5.5e-07,
        "output_cost_per_token": 2.19e-06,
        "max_tokens": 32768,  # reasoning model needs headroom
        "context_window": 128000,
        "api_base": "https://api.deepseek.com/v1",
        "api_key_var": "DEEPSEEK_API_KEY",
        "litellm_provider": "deepseek",
        "mode": "chat",
    },

    # =========================================================================
    # Kimi/Moonshot Models
    # =========================================================================
    "moonshot/kimi-k2": {
        "input_cost_per_token": 6e-07,      # $0.60/1M tokens
        "output_cost_per_token": 2.5e-06,   # $2.50/1M tokens
        "max_tokens": 32768,
        "api_base": "https://api.kimi.com/coding/",
        "api_key_var": "MOONSHOT_API_KEY",
        "litellm_provider": "moonshot",
        "mode": "chat",
    },
    "kimi-k2": {
        "input_cost_per_token": 6e-07,
        "output_cost_per_token": 2.5e-06,
        "max_tokens": 32768,
        "api_base": "https://api.kimi.com/coding/",
        "api_key_var": "MOONSHOT_API_KEY",
        "litellm_provider": "moonshot",
        "mode": "chat",
    },
    "anthropic/kimi-for-coding": {
        "input_cost_per_token": 6e-07,
        "output_cost_per_token": 2.5e-06,
        "max_tokens": 32768,
        "api_base": "https://api.kimi.com/coding/",
        "api_key_var": "MOONSHOT_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "kimi-for-coding": {
        "input_cost_per_token": 1.5e-07,    # $0.15/1M tokens (estimate)
        "output_cost_per_token": 6e-07,     # $0.60/1M tokens (estimate)
        "max_tokens": 32768,
        "api_base": "https://api.kimi.com/coding/",
        "api_key_var": "MOONSHOT_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "moonshot/kimi-for-coding": {
        "input_cost_per_token": 1.5e-07,
        "output_cost_per_token": 6e-07,
        "max_tokens": 32768,
        "api_base": "https://api.kimi.com/coding/",
        "api_key_var": "MOONSHOT_API_KEY",
        "litellm_provider": "moonshot",
        "mode": "chat",
    },
    "openrouter/moonshotai/kimi-k2": {
        "input_cost_per_token": 4.56e-07,   # $0.456/1M tokens (OpenRouter)
        "output_cost_per_token": 2.5e-06,
        "max_tokens": 32768,
        "api_base": None,
        "api_key_var": "OPENROUTER_API_KEY",
        "litellm_provider": "openrouter",
        "mode": "chat",
    },
    "moonshotai/kimi-k2": {
        "input_cost_per_token": 4.56e-07,
        "output_cost_per_token": 2.5e-06,
        "max_tokens": 32768,
        "api_base": None,
        "api_key_var": "OPENROUTER_API_KEY",
        "litellm_provider": "openrouter",
        "mode": "chat",
    },

    # =========================================================================
    # MiniMax Models
    # =========================================================================
    "openai/MiniMax-M2.1": {
        "input_cost_per_token": 1.2e-07,    # $0.12/1M tokens
        "output_cost_per_token": 3e-07,     # $0.30/1M tokens
        "max_tokens": 204800,
        "api_base": "https://api.minimax.io/v1",
        "api_key_var": "MINIMAX_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "MiniMax-M2.1": {
        "input_cost_per_token": 1.2e-07,
        "output_cost_per_token": 3e-07,
        "max_tokens": 204800,
        "api_base": "https://api.minimax.io/v1",
        "api_key_var": "MINIMAX_API_KEY",
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "minimax/MiniMax-M2.1": {
        "input_cost_per_token": 1.2e-07,
        "output_cost_per_token": 3e-07,
        "max_tokens": 204800,
        "api_base": "https://api.minimax.io/v1",
        "api_key_var": "MINIMAX_API_KEY",
        "litellm_provider": "minimax",
        "mode": "chat",
    },
    "anthropic/minimax-m2.1": {
        "input_cost_per_token": 1.2e-07,
        "output_cost_per_token": 3e-07,
        "max_tokens": 204800,
        "api_base": "https://api.minimax.io/anthropic",
        "api_key_var": "MINIMAX_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },

    # =========================================================================
    # GLM/ZhipuAI Models
    # =========================================================================
    "anthropic/glm-4.7": {
        "input_cost_per_token": 4.5e-07,    # $0.45/1M tokens
        "output_cost_per_token": 1.9e-06,   # $1.90/1M tokens
        "max_tokens": 200000,
        "api_base": "https://api.z.ai/api/anthropic",
        "api_key_var": "ZHIPUAI_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "glm-4.7": {
        "input_cost_per_token": 4.5e-07,
        "output_cost_per_token": 1.9e-06,
        "max_tokens": 200000,
        "api_base": "https://api.z.ai/api/anthropic",
        "api_key_var": "ZHIPUAI_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "zai/glm-4.7": {
        "input_cost_per_token": 4e-07,      # $0.40/1M tokens
        "output_cost_per_token": 1.75e-06,  # $1.75/1M tokens
        "max_tokens": 200000,
        "api_base": "https://api.z.ai/api/anthropic",
        "api_key_var": "ZHIPUAI_API_KEY",
        "litellm_provider": "zai",
        "mode": "chat",
    },

    # =========================================================================
    # Anthropic Claude Models
    # =========================================================================
    "claude-3-5-sonnet": {
        "input_cost_per_token": 3e-06,      # $3/1M tokens
        "output_cost_per_token": 1.5e-05,   # $15/1M tokens
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "ANTHROPIC_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-3-5-sonnet-20241022": {
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "ANTHROPIC_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-3-opus": {
        "input_cost_per_token": 1.5e-05,    # $15/1M tokens
        "output_cost_per_token": 7.5e-05,   # $75/1M tokens
        "max_tokens": 4096,
        "api_base": None,
        "api_key_var": "ANTHROPIC_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-3-haiku": {
        "input_cost_per_token": 2.5e-07,    # $0.25/1M tokens
        "output_cost_per_token": 1.25e-06,  # $1.25/1M tokens
        "max_tokens": 4096,
        "api_base": None,
        "api_key_var": "ANTHROPIC_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-3-5-haiku": {
        "input_cost_per_token": 8e-07,      # $0.80/1M tokens
        "output_cost_per_token": 4e-06,     # $4/1M tokens
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "ANTHROPIC_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-3-5-haiku-20241022": {
        "input_cost_per_token": 8e-07,
        "output_cost_per_token": 4e-06,
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "ANTHROPIC_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-haiku-4-5": {
        "input_cost_per_token": 1e-06,      # $1/1M tokens
        "output_cost_per_token": 5e-06,     # $5/1M tokens
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "ANTHROPIC_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-sonnet-4": {
        "input_cost_per_token": 3e-06,      # $3/1M tokens
        "output_cost_per_token": 1.5e-05,   # $15/1M tokens
        "max_tokens": 16384,
        "api_base": None,
        "api_key_var": "ANTHROPIC_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-opus-4": {
        "input_cost_per_token": 1.5e-05,    # $15/1M tokens
        "output_cost_per_token": 7.5e-05,   # $75/1M tokens
        "max_tokens": 16384,
        "api_base": None,
        "api_key_var": "ANTHROPIC_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-opus-4-5": {
        "input_cost_per_token": 1.5e-05,    # $15/1M tokens
        "output_cost_per_token": 7.5e-05,   # $75/1M tokens
        "max_tokens": 16384,
        "api_base": None,
        "api_key_var": "ANTHROPIC_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-3": {
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "ANTHROPIC_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-3.5": {
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "ANTHROPIC_API_KEY",
        "litellm_provider": "anthropic",
        "mode": "chat",
    },

    # =========================================================================
    # Google Gemini Models
    # =========================================================================
    "gemini-2.0-flash": {
        "input_cost_per_token": 1e-07,      # $0.10/1M tokens
        "output_cost_per_token": 4e-07,     # $0.40/1M tokens
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "GOOGLE_API_KEY",
        "litellm_provider": "google",
        "mode": "chat",
    },
    "gemini-2.0-flash-lite": {
        "input_cost_per_token": 7.5e-08,    # $0.075/1M tokens
        "output_cost_per_token": 3e-07,     # $0.30/1M tokens
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "GOOGLE_API_KEY",
        "litellm_provider": "google",
        "mode": "chat",
    },
    "gemini-2.5-pro": {
        "input_cost_per_token": 1.25e-06,   # $1.25/1M tokens
        "output_cost_per_token": 1e-05,     # $10/1M tokens
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "GOOGLE_API_KEY",
        "litellm_provider": "google",
        "mode": "chat",
    },
    "gemini-2.5-flash": {
        "input_cost_per_token": 1.5e-07,    # $0.15/1M tokens
        "output_cost_per_token": 6e-07,     # $0.60/1M tokens
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "GOOGLE_API_KEY",
        "litellm_provider": "google",
        "mode": "chat",
    },
    "gemini-3.0-pro": {
        "input_cost_per_token": 1.25e-06,   # $1.25/1M tokens (estimated)
        "output_cost_per_token": 1e-05,     # $10/1M tokens (estimated)
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "GOOGLE_API_KEY",
        "litellm_provider": "google",
        "mode": "chat",
    },

    # =========================================================================
    # Qwen Cloud API Models
    # =========================================================================
    "qwen-max": {
        "input_cost_per_token": 1.6e-06,    # $1.60/1M tokens
        "output_cost_per_token": 6.4e-06,   # $6.40/1M tokens
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "DASHSCOPE_API_KEY",
        "litellm_provider": "dashscope",
        "mode": "chat",
    },
    "qwen2.5-max": {
        "input_cost_per_token": 1.6e-06,
        "output_cost_per_token": 6.4e-06,
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "DASHSCOPE_API_KEY",
        "litellm_provider": "dashscope",
        "mode": "chat",
    },
    "qwen-plus": {
        "input_cost_per_token": 4.2e-07,    # $0.42/1M tokens
        "output_cost_per_token": 1.26e-06,  # $1.26/1M tokens
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "DASHSCOPE_API_KEY",
        "litellm_provider": "dashscope",
        "mode": "chat",
    },
    "qwen-turbo": {
        "input_cost_per_token": 5.25e-08,   # $0.0525/1M tokens
        "output_cost_per_token": 2.1e-07,   # $0.21/1M tokens
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "DASHSCOPE_API_KEY",
        "litellm_provider": "dashscope",
        "mode": "chat",
    },
    "qwen3-235b-a22b": {
        "input_cost_per_token": 2.415e-07,  # $0.2415/1M tokens
        "output_cost_per_token": 2.415e-06, # $2.415/1M tokens
        "max_tokens": 8192,
        "api_base": None,
        "api_key_var": "DASHSCOPE_API_KEY",
        "litellm_provider": "dashscope",
        "mode": "chat",
    },

    # =========================================================================
    # Local/Ollama Models (free)
    # =========================================================================
    "ollama/gpt-oss:20b": {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "max_tokens": 8192,
        "api_base": "http://localhost:11434",
        "api_key_var": None,
        "litellm_provider": "ollama",
        "mode": "chat",
    },
    "ollama/qwen3-coder:30b": {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "max_tokens": 8192,
        "api_base": "http://localhost:11434",
        "api_key_var": None,
        "litellm_provider": "ollama",
        "mode": "chat",
    },
    "ollama/qwen3:4b-instruct": {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "max_tokens": 8192,
        "api_base": "http://localhost:11434",
        "api_key_var": None,
        "litellm_provider": "ollama",
        "mode": "chat",
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_model_pricing(model_name: str) -> Optional[Dict[str, Any]]:
    """Get full pricing and config for a model.

    Args:
        model_name: Model identifier (e.g., "gpt-4o", "deepseek/deepseek-chat")

    Returns:
        Dict with pricing and config, or None if not found
    """
    return MODEL_REGISTRY.get(model_name)


def get_input_cost(model_name: str, default: float = 0.0) -> float:
    """Get input cost per token for a model.

    Args:
        model_name: Model identifier
        default: Default value if model not found

    Returns:
        Cost per input token in USD
    """
    config = MODEL_REGISTRY.get(model_name)
    if config:
        return config.get("input_cost_per_token", default)
    return default


def get_output_cost(model_name: str, default: float = 0.0) -> float:
    """Get output cost per token for a model.

    Args:
        model_name: Model identifier
        default: Default value if model not found

    Returns:
        Cost per output token in USD
    """
    config = MODEL_REGISTRY.get(model_name)
    if config:
        return config.get("output_cost_per_token", default)
    return default


def get_max_tokens(model_name: str, default: int = 4096) -> int:
    """Get max output tokens for a model.

    Args:
        model_name: Model identifier
        default: Default value if model not found

    Returns:
        Maximum output tokens
    """
    config = MODEL_REGISTRY.get(model_name)
    if config:
        return config.get("max_tokens", default)
    return default


def get_api_config(model_name: str) -> Dict[str, Any]:
    """Get API configuration for a model.

    Args:
        model_name: Model identifier

    Returns:
        Dict with api_base, api_key (resolved from env), and other config
    """
    config = MODEL_REGISTRY.get(model_name, {})

    api_key = None
    api_key_var = config.get("api_key_var")
    if api_key_var:
        api_key = os.environ.get(api_key_var)

    return {
        "model_name": model_name,
        "api_base": config.get("api_base"),
        "api_key": api_key,
        "litellm_provider": config.get("litellm_provider"),
        "max_tokens": config.get("max_tokens", 4096),
    }


def calculate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int
) -> float:
    """Calculate total cost for a request.

    Args:
        model_name: Model identifier
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Total cost in USD
    """
    input_cost = get_input_cost(model_name) * prompt_tokens
    output_cost = get_output_cost(model_name) * completion_tokens
    return input_cost + output_cost


def get_litellm_pricing_dict() -> Dict[str, Dict[str, Any]]:
    """Get pricing dict in LiteLLM format for registration.

    Returns:
        Dict suitable for litellm.register_model(model_cost=...)
    """
    litellm_pricing = {}
    for model_name, config in MODEL_REGISTRY.items():
        litellm_pricing[model_name] = {
            "input_cost_per_token": config.get("input_cost_per_token", 0),
            "output_cost_per_token": config.get("output_cost_per_token", 0),
            "max_tokens": config.get("max_tokens", 4096),
            "litellm_provider": config.get("litellm_provider", "openai"),
            "mode": config.get("mode", "chat"),
        }
    return litellm_pricing


# For backwards compatibility with existing code
MODEL_COST_PER_INPUT = {k: v["input_cost_per_token"] for k, v in MODEL_REGISTRY.items()}
MODEL_COST_PER_OUTPUT = {k: v["output_cost_per_token"] for k, v in MODEL_REGISTRY.items()}
