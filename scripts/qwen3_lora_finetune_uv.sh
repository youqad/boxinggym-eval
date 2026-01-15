#!/usr/bin/env bash
set -euo pipefail

# LoRA fine-tuning for Qwen3-4B with MLX using uv.
# Minimal defaults designed for 16–32 GB RAM Macs.
# Example usage:
#   ./scripts/qwen3_lora_finetune_uv.sh \
#     Qwen/Qwen3-4B-Instruct-2507 \
#     ./data/my_chat_dataset \
#     ./adapters/qwen3_4b_lora

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. See https://github.com/astral-sh/uv" >&2
  exit 1
fi

MODEL=${1:-Qwen/Qwen3-4B-Instruct-2507}
DATA=${2:-./data/sample_chat}
OUT_ADAPTER=${3:-./adapters/qwen3_4b_lora}

mkdir -p "$(dirname "${OUT_ADAPTER}")"

uv run -m mlx_lm.lora \
  --model "${MODEL}" \
  --load-in-8bit \
  --train \
  --data "${DATA}" \
  --batch-size 2 --accum-steps 8 \
  --lr 2e-4 --epochs 3 \
  --lora-r 16 --lora-alpha 32 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save-adapter "${OUT_ADAPTER}"
