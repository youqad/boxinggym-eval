SEEDS=(1) #2 3)
for seed in "${SEEDS[@]}"; do
    # OED
    python run_experiment.py seed=$seed llms=qwen include_prior=true exp=oed envs=location_finding_direct 
    # python run_experiment.py seed=$seed llms=claude include_prior=false exp=oed envs=location_finding_direct 
    # python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=location_finding_discout # not sure what this is doing, throws an error
    # Discovery
    # python run_experiment.py seed=$seed llms=claude include_prior=true exp=discovery envs=location_finding_direct_discovery
    # python run_experiment.py seed=$seed llms=claude include_prior=false exp=discovery envs=location_finding_direct_discovery
done