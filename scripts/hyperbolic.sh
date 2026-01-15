SEEDS=(1 2 3 4 5)
for seed in "${SEEDS[@]}"; do
    # OED
    uv run python run_experiment.py seeds=null seed=$seed llms=gpt-4o include_prior=true use_ppl=false exp=oed envs=hyperbolic_direct
    uv run python run_experiment.py seeds=null seed=$seed llms=gpt-4o include_prior=false use_ppl=false exp=oed envs=hyperbolic_direct
    uv run python run_experiment.py seeds=null seed=$seed llms=gpt-4o include_prior=true use_ppl=false exp=oed envs=hyperbolic_discount
    # Discovery
    uv run python run_experiment.py seeds=null seed=$seed llms=gpt-4o include_prior=true use_ppl=false exp=discovery envs=hyperbolic_direct_naive
    uv run python run_experiment.py seeds=null seed=$seed llms=gpt-4o include_prior=false use_ppl=false exp=discovery envs=hyperbolic_direct_naive
done
