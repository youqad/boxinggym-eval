#!/usr/bin/env python3
"""
Minimal Qwen3-4B MLX inference helper (uv-friendly).

Usage (examples):
  uv run python scripts/qwen3_infer_mlx.py \
    --model mlx-community/Qwen3-4B-Instruct-2507-8bit \
    --system "You are a helpful assistant." \
    --user "Give me three bullets on droplet microfluidics."

Notes:
- Applies the tokenizer's chat_template if present.
- For adapter inference (LoRA), prefer the CLI wrapper:
    uv run -m mlx_lm.generate --model <base> --adapter-path ./adapters/.. --prompt "..."
"""

import argparse
from typing import List, Dict

from mlx_lm import load, generate


def build_messages(system: str, user: str) -> List[Dict[str, str]]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3 MLX inference (uv)")
    parser.add_argument("--model", type=str, required=False,
                        default="mlx-community/Qwen3-4B-Instruct-2507-8bit",
                        help="HF model id (MLX/quantized recommended)")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.",
                        help="System message")
    parser.add_argument("--user", type=str, required=True, help="User prompt text")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args()

    model, tokenizer = load(args.model)

    prompt = args.user
    # Prefer chat template if available (Qwen chat models include one)
    if getattr(tokenizer, "chat_template", None):
        messages = build_messages(args.system, args.user)
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        verbose=False,
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
