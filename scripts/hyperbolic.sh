SEEDS=(1) #2 3 4 5)
for seed in "${SEEDS[@]}"; do
    # OED
    python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=hyperbolic_direct 
    # python run_experiment.py seed=$seed llms=qwen include_prior=false exp=oed envs=hyperbolic_direct 
    # python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=hyperbolic_discout 
    # # Discovery
    # python run_experiment.py seed=$seed llms=claude include_prior=true exp=discovery envs=hyperbolic_direct_discovery
    # python run_experiment.py seed=$seed llms=claude include_prior=false exp=discovery envs=hyperbolic_direct_discovery
done
