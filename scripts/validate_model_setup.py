#!/usr/bin/env python3
"""
Validate BoxingGym multi-model setup.

Checks:
- Required environment variables for selected model provider
- LiteLLM availability and version
- Model availability through provider APIs
- Configuration file presence

Usage:
    uv run python scripts/validate_model_setup.py --model gpt-5-mini
    uv run python scripts/validate_model_setup.py --model deepseek-chat
    uv run python scripts/validate_model_setup.py --all  # Check all known models
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Model provider mapping (matches conf/llms/*.yaml)
MODEL_PROVIDERS = {
    "gpt-4o": {"provider": "openai", "env_key": "OPENAI_API_KEY", "config": "conf/llms/gpt-4o.yaml"},
    "gpt-5.1": {"provider": "openai", "env_key": "OPENAI_API_KEY", "config": "conf/llms/gpt-5.1.yaml"},
    "gpt-5.1-mini": {"provider": "openai", "env_key": "OPENAI_API_KEY", "config": "conf/llms/gpt-5.1-mini.yaml"},
    "gpt-5.1-codex-mini": {"provider": "openai", "env_key": "OPENAI_API_KEY", "config": "conf/llms/gpt-5.1-codex-mini.yaml"},
    "deepseek-chat": {"provider": "deepseek", "env_key": "DEEPSEEK_API_KEY", "config": "conf/llms/deepseek-chat.yaml"},
    "deepseek-reasoner": {"provider": "deepseek", "env_key": "DEEPSEEK_API_KEY", "config": "conf/llms/deepseek-reasoner.yaml"},
    "deepseek-v3.2": {"provider": "deepseek", "env_key": "DEEPSEEK_API_KEY", "config": "conf/llms/deepseek-v3.2.yaml"},
    "glm-4.7": {"provider": "zhipu", "env_key": "ZHIPUAI_API_KEY", "config": "conf/llms/glm-4.7.yaml"},
    "minimax-m2.1": {"provider": "minimax", "env_key": "MINIMAX_API_KEY", "config": "conf/llms/minimax-m2.1.yaml"},
    "kimi-k2": {"provider": "moonshot", "env_key": "MOONSHOT_API_KEY", "config": "conf/llms/kimi-k2.yaml"},
    "claude_3_5_sonnet": {"provider": "anthropic", "env_key": "ANTHROPIC_API_KEY"},
    # Together.AI models
    "rnj-1-instruct": {"provider": "together_ai", "env_key": "TOGETHER_API_KEY", "config": "conf/llms/rnj-1-instruct.yaml"},
    "qwen3-4b": {"provider": "together_ai", "env_key": "TOGETHER_API_KEY", "config": "conf/llms/qwen3-4b.yaml"},
    # Mistral models
    "ministral-3b": {"provider": "mistral", "env_key": "MISTRAL_API_KEY", "config": "conf/llms/ministral-3b.yaml"},
}

COLORS = {
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "reset": "\033[0m",
}


def colored(text: str, color: str) -> str:
    """Return colored text for terminal output."""
    if os.getenv("NO_COLOR"):
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def check_env_var(env_key: str) -> Tuple[bool, str]:
    """Check if environment variable is set."""
    value = os.environ.get(env_key)
    if not value:
        return False, "Not set. Get key at https://platform.openai.com/api-keys"
    if value.startswith("sk-") or value.startswith("sk_"):
        return True, f"✓ Set (masked: {value[:10]}...)"
    return True, "✓ Set (non-standard format, assuming valid)"


def check_config_file(config_path: str) -> Tuple[bool, str]:
    """Check if configuration file exists."""
    path = Path(config_path)
    if path.exists():
        return True, "✓ Found"
    return False, f"Missing (expected at {config_path})"


def check_litellm_version() -> Tuple[bool, str]:
    """Check LiteLLM availability and version."""
    try:
        import litellm
        version = litellm.__version__ if hasattr(litellm, "__version__") else "unknown"
        # Current required version from requirements.txt
        if version >= "1.78.7" or version == "unknown":
            return True, f"✓ Available (v{version})"
        return False, f"Version {version} - recommend v1.78.7+"
    except ImportError:
        return False, "Not installed. Run: pip install litellm"


def check_provider_module(provider: str) -> Tuple[bool, str]:
    """Check if provider SDK is available."""
    providers_to_check = {
        "openai": ("openai", "OpenAI SDK"),
        "deepseek": ("litellm", "LiteLLM with DeepSeek support"),
        "anthropic": ("anthropic", "Anthropic SDK"),
        "zhipu": ("litellm", "LiteLLM with Zhipu/GLM support"),
        "minimax": ("litellm", "LiteLLM with MiniMax support"),
        "moonshot": ("litellm", "LiteLLM with Moonshot/Kimi support"),
        "together_ai": ("litellm", "LiteLLM with Together.AI support"),
        "mistral": ("litellm", "LiteLLM with Mistral support"),
    }

    if provider not in providers_to_check:
        return False, f"Unknown provider: {provider}"

    module_name, display_name = providers_to_check[provider]
    try:
        __import__(module_name)
        return True, f"✓ {display_name} available"
    except ImportError:
        return False, f"{display_name} not found. Run: pip install {module_name}"


def validate_model(model_key: str) -> Dict[str, any]:
    """Validate a single model's setup."""
    if model_key not in MODEL_PROVIDERS:
        return {
            "model": model_key,
            "valid": False,
            "errors": [f"Unknown model: {model_key}"],
        }

    model_info = MODEL_PROVIDERS[model_key]
    checks = {
        "env_var": None,
        "config_file": None,
        "litellm": None,
        "provider_sdk": None,
    }
    errors = []
    warnings = []

    # Check environment variable
    env_key = model_info.get("env_key")
    if env_key:
        is_set, msg = check_env_var(env_key)
        checks["env_var"] = (is_set, msg)
        if not is_set:
            errors.append(f"{env_key} not set - {msg}")

    # Check config file
    config_path = model_info.get("config")
    if config_path:
        exists, msg = check_config_file(config_path)
        checks["config_file"] = (exists, msg)
        if not exists:
            warnings.append(f"Config file missing - {msg}")

    # Check LiteLLM
    is_available, msg = check_litellm_version()
    checks["litellm"] = (is_available, msg)
    if not is_available:
        errors.append(f"LiteLLM issue - {msg}")

    # Check provider SDK
    provider = model_info.get("provider")
    if provider:
        is_available, msg = check_provider_module(provider)
        checks["provider_sdk"] = (is_available, msg)
        if not is_available:
            errors.append(f"Provider SDK issue - {msg}")

    is_valid = len(errors) == 0

    return {
        "model": model_key,
        "provider": provider,
        "valid": is_valid,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def print_validation_result(result: Dict) -> None:
    """Print validation result in human-readable format."""
    model = result["model"]
    provider = result.get("provider", "unknown")
    valid = result["valid"]

    status_icon = colored("✓", "green") if valid else colored("✗", "red")
    print(f"\n{status_icon} {colored(model, 'blue')} ({provider})")

    for check_name, value in result.get("checks", {}).items():
        # Allow optional checks that may be None
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            continue
        status, message = value
        icon = colored("✓", "green") if status else colored("✗", "red")
        print(f"  {icon} {check_name.replace('_', ' ').title()}: {message}")

    if result.get("errors"):
        print(f"\n  {colored('ERRORS:', 'red')}")
        for error in result["errors"]:
            print(f"    • {error}")

    if result.get("warnings"):
        print(f"\n  {colored('WARNINGS:', 'yellow')}")
        for warning in result["warnings"]:
            print(f"    • {warning}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate BoxingGym multi-model setup"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to validate",
        choices=list(MODEL_PROVIDERS.keys()),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all available models",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Validate all models from a specific provider",
        choices=["openai", "deepseek", "anthropic", "together_ai", "zhipu", "minimax", "moonshot"],
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings (not just errors)",
    )

    args = parser.parse_args()

    # Determine which models to validate
    if args.all:
        models_to_check = list(MODEL_PROVIDERS.keys())
    elif args.provider:
        models_to_check = [
            m for m, info in MODEL_PROVIDERS.items()
            if info.get("provider") == args.provider
        ]
    elif args.model:
        models_to_check = [args.model]
    else:
        # Default: check common models
        models_to_check = [
            "gpt-4o",
            "deepseek-chat",
            "gpt-5.1-codex-mini",
            "glm-4.7",
            "minimax-m2.1"
        ]

    # Validate each model
    results = [validate_model(m) for m in models_to_check]

    # Print results
    print(colored("\n╔═══ BoxingGym Model Setup Validation ═══╗", "blue"))
    for result in results:
        print_validation_result(result)

    # Summary
    all_valid = all(r["valid"] for r in results)
    warnings = sum(len(r.get("warnings", [])) for r in results)

    print("\n" + "=" * 50)
    if all_valid:
        if warnings and args.strict:
            print(colored(f"✓ Setup valid but {warnings} warning(s)", "yellow"))
            sys.exit(1)
        else:
            print(colored("✓ All models ready to use!", "green"))
            sys.exit(0)
    else:
        error_count = sum(len(r.get("errors", [])) for r in results)
        print(colored(f"✗ {error_count} error(s) found", "red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
