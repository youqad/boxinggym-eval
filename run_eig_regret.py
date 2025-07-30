import tqdm
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import os
import glob
from pathlib import Path

import src.boxing_gym.envs.location_finding as location_finding
import src.boxing_gym.envs.hyperbolic_temporal_discount as hyperbolic_temporal_discount
import src.boxing_gym.envs.death_process as death_process
import src.boxing_gym.envs.irt as irt
import src.boxing_gym.envs.survival_analysis as survival_analysis
import src.boxing_gym.envs.peregrines as peregrines
import src.boxing_gym.envs.dugongs as dugongs
import src.boxing_gym.envs.lotka_volterra as lotka_volterra
import src.boxing_gym.envs.moral_machines as moral_machines
import src.boxing_gym.envs.emotion as emotion


@hydra.main(version_base=None, config_path="conf", config_name="config_eig")
def main(config: DictConfig):
    seed = config.seed
    print(f"Seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    model_name = config.llms.model_name
    temperature = config.llms.temperature
    max_tokens = config.llms.max_tokens
    num_experiments = config.exp.num_experiments
    env_params = config.envs.env_params
    experiment_type = config.exp.experiment_type
    include_prior = config.include_prior
    num_evals = config.envs.num_evals
    env_name = config.envs.env_name
    goal_name = config.envs.goal_name
    num_random = config.num_random
    redo = config.redo
    box = config.box

    print(f"Running EIG regret calculation for:")
    print(f"  Environment: {env_name}")
    print(f"  Goal: {goal_name}")
    print(f"  Model: {model_name}")
    print(f"  Experiment type: {experiment_type}")
    print(f"  Include prior: {include_prior}")
    print(f"  Number of random samples: {num_random}")
    print(f"  BoxLoop: {box}")

    # This is important: extract the actual environment name from env_name
    # For example, extract "hyperbolic_temporal_discount" from "hyperbolic_direct"
    if "_direct" in env_name:
        actual_env_name = env_name.replace("_direct", "")
        # Special case handling for hyperbolic
        if actual_env_name == "hyperbolic":
            actual_env_name = "hyperbolic_temporal_discount"
    else:
        actual_env_name = env_name
        
    print(f"  Actual environment name: {actual_env_name}")

    nametoenv = {    
        "location_finding": location_finding.Signal,
        "hyperbolic_temporal_discount": hyperbolic_temporal_discount.TemporalDiscount,
        "death_process": death_process.DeathProcess,
        "irt": irt.IRT,
        "survival": survival_analysis.SurvivalAnalysis,
        "dugongs": dugongs.Dugongs,
        "peregrines": peregrines.Peregrines,
        "morals": moral_machines.MoralMachine,
        "emotion": emotion.EmotionFromOutcome,
        "lotka_volterra": lotka_volterra.LotkaVolterra
    }
    
    nameenvtogoal = {
        ("hyperbolic_temporal_discount", "direct"): hyperbolic_temporal_discount.DirectGoal,
        ("hyperbolic_temporal_discount", "discount"): hyperbolic_temporal_discount.DiscountGoal,
        ("hyperbolic_temporal_discount", "direct_discovery"): hyperbolic_temporal_discount.DirectGoalNaive,
        ("location_finding", "direct"): location_finding.DirectGoal,
        ("location_finding", "source"): location_finding.SourceGoal,
        ("location_finding", "direct_discovery"): location_finding.DirectGoalNaive,
        ("death_process", "direct"): death_process.DirectDeath,
        ("death_process", "direct_discovery"): death_process.DirectDeathNaive,
        ("death_process", "infection"): death_process.InfectionRate,
        ("irt", "direct"): irt.DirectCorrectness,
        ("irt", "direct_discovery"): irt.DirectCorrectnessNaive,
        ("irt", "best_student"): irt.BestStudent,
        ("irt", "difficult_question"): irt.DifficultQuestion,
        ("irt", "discriminate_question"): irt.DiscriminatingQuestion,
        ("survival", "direct"): survival_analysis.DirectGoal,
        ("survival", "direct_discovery"): survival_analysis.DirectGoalNaive,
        ("dugongs", "direct"): dugongs.DirectGoal,
        ("dugongs", "direct_discovery"): dugongs.DirectGoalNaive,
        ("peregrines", "direct"): peregrines.DirectGoal,
        ("peregrines", "direct_discovery"): peregrines.DirectGoalNaive,
        ("emotion", "direct"): emotion.DirectEmotionPrediction,
        ("emotion", "direct_discovery"): emotion.DirectEmotionNaive,
        ("morals", "direct"): moral_machines.DirectPrediction,
        ("morals", "direct_discovery"): moral_machines.DirectPredictionNaive,
        ("lotka_volterra", "direct"): lotka_volterra.DirectGoal,
        ("lotka_volterra", "direct_discovery"): lotka_volterra.DirectGoalNaive,
    }

    # Create environment and goal instances
    try:
        env = nametoenv[actual_env_name](**env_params)
        goal = nameenvtogoal[(actual_env_name, goal_name)](env)
    except KeyError as e:
        print(f"ERROR: Could not create environment or goal. Key error: {e}")
        print(f"Available environment keys: {list(nametoenv.keys())}")
        return

    # Determine the model name and setup result directory
    model_suffix = "-boxloop" if box else ""
    model_name_full = f"{model_name}{model_suffix}"
    res_dir = f"results/{actual_env_name}"
    
    # Make sure the results directory exists
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    
    # Handle different filename formats based on experiment type
    if experiment_type == "discovery":
        # For discovery: direct_discovery_{model}_{prior}_{seed}.json
        res_filename = f"{goal_name}_discovery_{model_name_full}_{include_prior}_{seed}.json"
    else:
        # For oed: direct_{model}_oed_{prior}_{seed}.json
        res_filename = f"{goal_name}_{model_name_full}_{experiment_type}_{include_prior}_{seed}.json"
    
    # Full path to the result file
    result_file_path = os.path.join(res_dir, res_filename)
    
    print(f"Looking for results file: {result_file_path}")
    
    # If file not found, try possible alternatives
    if not os.path.exists(result_file_path):
        print(f"File not found, searching for alternatives...")
        
        # Search pattern for possible matching files
        search_pattern = os.path.join(res_dir, f"{goal_name}*{model_name_full}*{include_prior}_{seed}.json")
        alternative_files = glob.glob(search_pattern)
        
        if alternative_files:
            result_file_path = alternative_files[0]
            print(f"Found alternative file: {result_file_path}")
        else:
            print(f"Error: No matching results files found for pattern: {search_pattern}")
            return
    
    # Load the results
    try:
        with open(result_file_path, "r") as f:
            data = json.load(f)
        print(f"Successfully loaded results file")
    except Exception as e:
        print(f"Error loading results file: {e}")
        return
    
    # Extract queries and successes
    try:
        queries = data['data']['queries']
        successes = data['data']['successes']
        
        # Check if EIGs are already calculated
        if not redo and 'eigs' in data['data'] and data['data']['eigs']:
            eigs = data['data']['eigs']
            print(f"Using existing EIGs from results file")
        else:
            eigs = []
            print(f"Will calculate EIGs for each query")
    except KeyError as e:
        print(f"Error accessing required data in results file: {e}")
        print(f"Data structure: {list(data.keys())}")
        if 'data' in data:
            print(f"Data substructure: {list(data['data'].keys())}")
        return

    # Initialize result lists
    eigs_regret = []
    eigs_max = []
    inputs_max = []
    
    # Process each query
    print(f"Processing {len(queries)} queries...")
    for i, query in tqdm.tqdm(enumerate(queries)):
        if i >= len(successes) or not successes[i]:
            print(f"Skipping query {i} (unsuccessful or out of range)")
            continue
            
        # Sample random inputs and calculate their EIGs
        try:
            eigs_random = []
            inputs = []
            for n in range(num_random):
                random_input = env.sample_random_input()
                eig = goal.expected_information_gain(random_input)
                inputs.append(random_input)
                eigs_random.append(eig)
            
            # Calculate EIG for the actual query if needed
            if redo or not eigs or i >= len(eigs):
                input_query = env.validate_input(query)
                eig = goal.expected_information_gain(input_query)
                eigs.append(eig)
            else:
                eig = eigs[i]
            
            # Find the maximum EIG among random inputs
            max_eig = max(eigs_random)
            max_input = inputs[eigs_random.index(max_eig)]
            eigs_max.append(max_eig)
            inputs_max.append(max_input)
            
            # Calculate regret
            regret = max_eig - eig
            eigs_regret.append(regret)
            
            print(f"Query {i}: EIG = {eig:.4f}, Max EIG = {max_eig:.4f}, Regret = {regret:.4f}")
            
            # Make observation (execute the query)
            _ = goal.env.run_experiment(query)
        except Exception as e:
            print(f"Error processing query {i}: {e}")
            continue
    
    if not eigs_regret:
        print("No valid queries processed. Exiting.")
        return
    
    # Prepare results for JSON serialization
    eigs_regret = [float(x) for x in eigs_regret]
    eigs_max = [float(x) for x in eigs_max]
    eigs = [float(x) for x in eigs]
    
    # Convert inputs to serializable format
    inputs_max = [x.tolist() if isinstance(x, np.ndarray) else x for x in inputs_max]
    
    # Handle tuples containing numpy arrays
    new_inputs_max = []
    for x in inputs_max:
        if isinstance(x, tuple):
            new_x = [y.tolist() if isinstance(y, np.ndarray) else y for y in x]
            new_inputs_max.append(tuple(new_x))
        else:
            new_inputs_max.append(x)
    
    # Prepare final results
    store_dict = {
        "eigs_regret": eigs_regret,
        "eigs_max": eigs_max,
        "inputs_max": new_inputs_max,
        "eigs": eigs
    }
    
    # Calculate average regret
    avg_regret = sum(eigs_regret) / len(eigs_regret) if eigs_regret else 0
    print(f"Average regret: {avg_regret:.4f}")
    
    # Save results
    try:
        output_filename = os.path.basename(result_file_path)
        regret_file_path = os.path.join(res_dir, f"regret_{output_filename}")
        with open(regret_file_path, 'w') as f:
            json.dump(store_dict, f, indent=4)
        print(f"Results saved to: {regret_file_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()