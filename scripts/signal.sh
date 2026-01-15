#!/usr/bin/env bash
set -euo pipefail

SEEDS=(1 2 3 4 5)
for seed in "${SEEDS[@]}"; do
    # OED
    python run_experiment.py seeds=null seed=$seed llms=openai include_prior=true exp=oed envs=location_finding_direct
    python run_experiment.py seeds=null seed=$seed llms=openai include_prior=false exp=oed envs=location_finding_direct
    python run_experiment.py seeds=null seed=$seed llms=openai include_prior=true exp=oed envs=location_finding_source
    # Discovery
    python run_experiment.py seeds=null seed=$seed llms=openai include_prior=true exp=discovery envs=location_finding_direct_naive
    python run_experiment.py seeds=null seed=$seed llms=openai include_prior=false exp=discovery envs=location_finding_direct_naive
done
