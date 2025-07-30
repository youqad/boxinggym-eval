SEEDS=(1) #2 3 4 5)
for seed in "${SEEDS[@]}"; do
    # OED
    python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=irt_direct 
    python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=survival_direct 
    python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=peregrines_direct 
    python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=emotion_direct 
    python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=morals_direct 
    python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=lotka_volterra_direct 
    python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=death_process_direct
    python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=hyperbolic_direct
    python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=location_finding_direct
    python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=dugongs_direct
done

    python run_experiment.py seed=1 llms=deepseek include_prior=true exp=oed envs=irt_direct
    python run_experiment.py seed=1 llms=openthinker include_prior=true exp=oed envs=irt_direct 
 
