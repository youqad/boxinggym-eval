#!/usr/bin/env python3
"""
Minimal smoke test to verify LLM integration for GPT-5-mini and DeepSeek Chat.

Usage:
  uv run python scripts/smoke_llm_calls.py --model gpt-5-mini
  uv run python scripts/smoke_llm_calls.py --model deepseek/deepseek-chat

Reads API keys from environment. Make sure OPENAI_API_KEY / DEEPSEEK_API_KEY are set.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boxing_gym.agents.agent import LMExperimenter
from boxing_gym.agents.model_config import get_model_config


def run_smoke(model: str) -> int:
    # Get model-specific API configuration
    config = get_model_config(model)

    # Check if required API key is set
    if config["api_key_var"] and not config["api_key"]:
        print(f"Missing {config['api_key_var']}")
        return 2

    # Let the agent use model-appropriate defaults
    # (LMExperimenter auto-detects and protects reasoning models from truncation)
    agent = LMExperimenter(
        model_name=config["model_name"],
        temperature=0.0,
        api_base=config["api_base"],
        api_key=config["api_key"]
    )

    # Optional system message
    agent.set_system_message("You are a careful scientist. Always follow the requested XML tag format.")

    # Simple format-check prompt
    prompt = "Respond exactly with <answer>42</answer>"
    try:
        output = agent.prompt_llm(prompt)
        print("Model:", model)
        print("Output:", output)
        # Minimal success heuristic
        if "<answer>42</answer>" in output.replace(" ", ""):
            print("✅ Smoke test passed")
            return 0
        print("⚠️  Response did not contain expected tags/value")
        return 1
    except Exception as e:
        print("❌ Error during smoke call:", e)
        return 1


def main():
    parser = argparse.ArgumentParser(description="LLM integration smoke test")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-5-mini, deepseek/deepseek-chat)")
    args = parser.parse_args()
    sys.exit(run_smoke(args.model))


if __name__ == "__main__":
    main()

