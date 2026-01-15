#!/bin/bash
# Run DeepSeek-R1 experiments on the same environments tested with deepseek-chat

SEED=1

echo "Running DeepSeek-R1 experiments with seed=$SEED"
echo "Model: deepseek/deepseek-reasoner with max_tokens=4096 for reasoning"
echo ""

# Note: use_ppl=false to avoid GPT-4o dependency for probabilistic programming
# We're testing pure DeepSeek-R1 performance, not the PPL-augmented approach

# Hyperbolic Temporal Discounting - Discovery
echo "=== Hyperbolic Temporal Discounting (Discovery) ==="
uv run python run_experiment.py seeds=null seed=$SEED llms=deepseek-reasoner include_prior=true use_ppl=false exp=discovery envs=hyperbolic_direct_naive

# Dugongs - Discovery
echo "=== Dugongs Growth Model (Discovery) ==="
uv run python run_experiment.py seeds=null seed=$SEED llms=deepseek-reasoner include_prior=true use_ppl=false exp=discovery envs=dugongs_direct_naive

# Peregrines - Discovery
echo "=== Peregrines Population Dynamics (Discovery) ==="
uv run python run_experiment.py seeds=null seed=$SEED llms=deepseek-reasoner include_prior=true use_ppl=false exp=discovery envs=peregrines_direct_naive

# Lotka-Volterra - Discovery
echo "=== Lotka-Volterra Predator-Prey (Discovery) ==="
uv run python run_experiment.py seeds=null seed=$SEED llms=deepseek-reasoner include_prior=true use_ppl=false exp=discovery envs=lotka_volterra_direct_naive

echo ""
echo "All DeepSeek-R1 experiments completed!"
echo "Results saved to results/ directory with 'deepseek-reasoner' in filename"
