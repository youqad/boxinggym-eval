#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper for Qwen3 inference with uv + mlx_lm.
# Example:
#   ./scripts/qwen3_infer_uv.sh "mlx-community/Qwen3-4B-Instruct-2507-8bit" \
#     "Write a one-paragraph abstract about single-cell microfluidics."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. See https://github.com/astral-sh/uv" >&2
  exit 1
fi

MODEL=${1:-mlx-community/Qwen3-4B-Instruct-2507-8bit}
PROMPT=${2:-"Hello from MLX Qwen3."}

uv run -m mlx_lm.generate \
  --model "${MODEL}" \
  --prompt "${PROMPT}" \
  --max-tokens 256 --temp 0.2 --top-p 0.9
