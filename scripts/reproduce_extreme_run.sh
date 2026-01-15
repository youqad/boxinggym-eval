#!/bin/bash
# Reproduce an extreme z-score run
# Set RUN_ID env var or pass as argument

set -e

RUN_ID="${1:-${RUN_ID:-YOUR_RUN_ID}}"
ENTITY="${WANDB_ENTITY:-your-entity}"
PROJECT="${WANDB_PROJECT:-boxing-gym}"

echo "================================================================================"
echo "REPRODUCING EXTREME Z-SCORE RUN: $RUN_ID"
echo "================================================================================"
echo ""
echo "Original run: https://wandb.ai/$ENTITY/$PROJECT/runs/$RUN_ID"
echo "Prediction: 67,219,139.70"
echo "Z-score: 4,273.94"
echo ""
echo "Configuration:"
echo "  - Seed: 2"
echo "  - Experiment: oed"
echo "  - Environment: peregrines_direct"
echo "  - Model: bedrock-qwen3-32b"
echo ""
echo "================================================================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Please create .env with AWS/Bedrock credentials"
    exit 1
fi

# Activate environment
if [ -d .venv ]; then
    source .venv/bin/activate
    echo "✅ Activated virtual environment"
else
    echo "⚠️  No .venv found, proceeding anyway"
fi

echo ""
echo "🚀 Starting reproduction run..."
echo "This will:"
echo "  1. Run the exact configuration that produced the extreme z-score"
echo "  2. Log to WandB with Weave tracing enabled"
echo "  3. Save results locally for inspection"
echo "  4. Show raw LLM responses in terminal"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

echo ""
echo "================================================================================"
echo "RUNNING EXPERIMENT"
echo "================================================================================"
echo ""

# Run the experiment
uv run python run_experiment.py \
  seeds=null \
  seed=2 \
  exp=oed \
  envs=peregrines_direct \
  llms=bedrock-qwen3-32b \
  +wandb=true

echo ""
echo "================================================================================"
echo "RUN COMPLETE"
echo "================================================================================"
echo ""
echo "📊 Check results:"
echo "  1. Local results: results/peregrines/peregrines_direct_bedrock-qwen3-32b_*.json"
echo "  2. WandB run page (check terminal output for URL)"
echo "  3. Weave traces (click Weave tab on WandB run page)"
echo ""
echo "🔍 Compare with original:"
echo "  - Original z-score: 4,273.94"
echo "  - Original prediction: 67,219,139.70"
echo "  - If reproduced → systematic model failure"
echo "  - If different → random sampling artifact"
echo ""
echo "💡 To analyze the local results:"
echo "  jq '.data.results' results/peregrines/peregrines_direct_bedrock-qwen3-32b_*.json"
echo ""
