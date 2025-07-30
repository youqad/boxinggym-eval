SEEDS=(1) #2 3 4 5)
for seed in "${SEEDS[@]}"; do 
    python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=emotion_direct 
    # python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=morals_direct
    #python run_experiment.py seed=$seed llms=claude include_prior=true exp=oed envs=dugongs_direct
done 