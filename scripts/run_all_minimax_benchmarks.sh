#!/bin/bash
# Complete MiniMax M2.1 Benchmark Suite
# Runs ALL priorities to maximize free API usage before Nov 7 deadline

set -e  # Exit on error

echo "🥊 MiniMax M2.1 COMPLETE Benchmark Suite"
echo "======================================="
echo "⏰ Started: $(date)"
echo "💰 Status: 100% FREE until Nov 7, 2025"
echo ""

# Ensure logs directory
mkdir -p logs/minimax

# ===================================================================
# P1 COMPLETION: Missing budgets (0, 5) for seeds 1,2,3
# ===================================================================
echo "🎯 P1 COMPLETION: Missing Dugongs budgets"
echo "==========================================="

for seed in 1 2 3; do
    for budget in 0 5; do
        echo "Running P1: seed=$seed, budget=$budget..."
        uv run python run_experiment.py \
          seeds=null \
          seed=$seed \
          llms=minimax-m2.1 \
          envs=dugongs_direct \
          "exp.num_experiments=[$budget]" \
          exp.experiment_type=oed \
          include_prior=true \
          >> logs/minimax/p1_completion.log 2>&1
        echo "✅ P1 Complete: seed=$seed, budget=$budget"
    done
done

echo "✅ P1 COMPLETION DONE"
echo ""

# ===================================================================
# P2: Core Comparison (Peregrines, IRT with all budgets)
# ===================================================================
echo "🎯 P2: Core Comparison"
echo "======================"

for env in peregrines irt; do
    for seed in 1 2 3; do
        for budget in 0 5 10; do
            echo "Running P2: env=$env, seed=$seed, budget=$budget..."
            uv run python run_experiment.py \
              seeds=null \
              seed=$seed \
              llms=minimax-m2.1 \
              envs=${env}_direct \
              "exp.num_experiments=[$budget]" \
              exp.experiment_type=oed \
              include_prior=true \
              >> logs/minimax/p2_${env}.log 2>&1
            echo "✅ P2 Complete: $env, seed=$seed, budget=$budget"
        done
    done
done

echo "✅ P2 COMPLETE"
echo ""

# ===================================================================
# P3: Reasoning Tasks
# ===================================================================
echo "🎯 P3: Reasoning Tasks"
echo "======================"

for env in death_process hyperbolic morals; do
    for seed in 1 2 3; do
        for budget in 0 5 10; do
            echo "Running P3: env=$env, seed=$seed, budget=$budget..."
            uv run python run_experiment.py \
              seeds=null \
              seed=$seed \
              llms=minimax-m2.1 \
              envs=${env}_direct \
              "exp.num_experiments=[$budget]" \
              exp.experiment_type=oed \
              include_prior=true \
              >> logs/minimax/p3_${env}.log 2>&1
            echo "✅ P3 Complete: $env, seed=$seed, budget=$budget"
        done
    done
done

echo "✅ P3 COMPLETE"
echo ""

# ===================================================================
# P4: Discovery Mode
# ===================================================================
echo "🎯 P4: Discovery Mode"
echo "====================="

for env in dugongs hyperbolic irt; do
    for seed in 1 2; do
        for budget in 0 10; do
            echo "Running P4: env=$env discovery, seed=$seed, budget=$budget..."
            uv run python run_experiment.py \
              seeds=null \
              seed=$seed \
              llms=minimax-m2.1 \
              envs=${env}_direct_naive \
              "exp.num_experiments=[$budget]" \
              exp.experiment_type=discovery \
              include_prior=true \
              >> logs/minimax/p4_${env}_discovery.log 2>&1
            echo "✅ P4 Complete: $env discovery, seed=$seed, budget=$budget"
        done
    done
done

echo "✅ P4 COMPLETE"
echo ""

# ===================================================================
# Summary
# ===================================================================
echo "=========================================="
echo "🎉 MiniMax M2.1 COMPLETE Benchmark Suite DONE!"
echo "=========================================="
echo "⏰ Finished: $(date)"
echo ""
echo "📊 Results: $(find results -name '*MiniMax-M2.1*' | wc -l) files"
echo "📁 Location: results/"
echo ""
echo "Next: Run 'make aggregate' and 'make plot' for analysis"
