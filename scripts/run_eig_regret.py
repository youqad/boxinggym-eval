import tqdm
import random
import numpy as np
import hydra
from omegaconf import DictConfig
import json
import os

from hydra.utils import get_original_cwd

import boxing_gym.envs.location_finding as location_finding
import boxing_gym.envs.hyperbolic_temporal_discount as hyperbolic_temporal_discount
import boxing_gym.envs.death_process as death_process
import boxing_gym.envs.irt as irt
import boxing_gym.envs.survival_analysis as survival_analysis
import boxing_gym.envs.peregrines as peregrines
import boxing_gym.envs.dugongs as dugongs
import boxing_gym.envs.lotka_volterra as lotka_volterra
import boxing_gym.envs.moral_machines as moral_machines
import boxing_gym.envs.emotion as emotion


@hydra.main(version_base=None, config_path="conf", config_name="config_eig")
def main(config: DictConfig):
    seed = config.seed
    print(f"seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    model_name = config.llms.model_name
    env_params = config.envs.env_params
    experiment_type = config.exp.experiment_type
    include_prior = config.include_prior
    env_name = config.envs.env_name
    goal_name = config.envs.goal_name
    num_random = config.num_random
    redo = config.redo
    box = config.box

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
        ("hyperbolic_temporal_discount", "direct_naive"): hyperbolic_temporal_discount.DirectGoalNaive,
        ("location_finding", "direct"): location_finding.DirectGoal,
        ("location_finding", "source"): location_finding.SourceGoal,
        ("location_finding", "direct_naive"): location_finding.DirectGoalNaive,
        ("death_process", "direct"): death_process.DirectDeath,
        ("death_process", "direct_naive"): death_process.DirectDeathNaive,
        ("death_process", "infection"): death_process.InfectionRate,
        ("irt", "direct"): irt.DirectCorrectness,
        ("irt", "direct_naive"): irt.DirectCorrectnessNaive,
        ("irt", "best_student"): irt.BestStudent,
        ("irt", "difficult_question"): irt.DifficultQuestion,
        ("irt", "discriminate_question"): irt.DiscriminatingQuestion,
        ("survival", "direct"): survival_analysis.DirectGoal,
        ("survival", "direct_naive"): survival_analysis.DirectGoalNaive,
        ("dugongs", "direct"): dugongs.DirectGoal,
        ("dugongs", "direct_naive"): dugongs.DirectGoalNaive,
        ("peregrines", "direct"): peregrines.DirectGoal,
        ("peregrines", "direct_naive"): peregrines.DirectGoalNaive,
        ("emotion", "direct"): emotion.DirectEmotionPrediction,
        ("emotion", "direct_naive"): emotion.DirectEmotionNaive,
        ("morals", "direct"): moral_machines.DirectPrediction,
        ("morals", "direct_naive"): moral_machines.DirectPredictionNaive,
        ("lotka_volterra", "direct"): lotka_volterra.DirectGoal,
        ("lotka_volterra", "direct_naive"): lotka_volterra.DirectGoalNaive,
    }

    env = nametoenv[env_name](**env_params)
    goal = nameenvtogoal[(env_name, goal_name)](env)

    save_model_name = str(model_name).replace("/", "_")
    if box:
        save_model_name = f"{save_model_name}-boxloop"
    prior_tag = "true" if bool(include_prior) else "false"
    res_dir = os.path.join(get_original_cwd(), "results", str(env_name))
    res_filename = None

    try:
        import glob

        pattern = os.path.join(
            res_dir,
            f"env={env_name}_goal={goal_name}_model={save_model_name}_exp={experiment_type}_prior={prior_tag}_seed={seed}_*.json",
        )
        matches = sorted(glob.glob(pattern))
        if matches:
            res_filename = matches[-1]
    except Exception:
        res_filename = None

    if not res_filename:
        legacy = os.path.join(res_dir, f"{goal_name}_{save_model_name}_{experiment_type}_{prior_tag}_{seed}.json")
        if os.path.exists(legacy):
            res_filename = legacy
        else:
            raise FileNotFoundError(
                f"Could not find results JSON under {res_dir} for env={env_name} goal={goal_name} model={save_model_name} exp={experiment_type} prior={prior_tag} seed={seed}"
            )

    with open(res_filename, "r") as f:
        data = json.load(f)
    

    queries = data['data']['queries']
    successes = data['data']['successes']
    if not redo:
        eigs = data['data']['eigs']
    else:
        eigs = []

    eigs_regret = []
    eigs_max = []
    inputs_max = []
    for i, o in tqdm.tqdm(enumerate(queries)):
        if not successes[i]:
            continue
        eigs_random = []
        inputs = []
        for n in range(num_random):
            random_input = env.sample_random_input()
            try:
                eig = goal.expected_information_gain(random_input)
            except NotImplementedError:
                print(f"EIG not implemented for env={env_name} goal={goal_name}; aborting.")
                return
            inputs.append(random_input)
            eigs_random.append(eig)
        if redo:
            input_query = env.validate_input(o)
            try:
                eig = goal.expected_information_gain(input_query)
            except NotImplementedError:
                print(f"EIG not implemented for env={env_name} goal={goal_name}; aborting.")
                return
            eigs.append(eig)
        else:
            eig = eigs[i]
        max_eig = max(eigs_random)
        max_input = inputs[eigs_random.index(max_eig)]
        eigs_max.append(max_eig)
        inputs_max.append(max_input)
        eigs_regret.append(max_eig - eig)
        print(f"eig: {eig}, max_eig: {max_eig}, regret: {max_eig - eig}")
        # make obs
        _ = goal.env.run_experiment(o)
    
    
    # convert all elements of list to float to be json serializable
    eigs_regret = [float(x) for x in eigs_regret]
    eigs_max = [float(x) for x in eigs_max]
    eigs = [float(x) for x in eigs]
    # convert inputs to list if they are np arrays or tuples
    inputs_max = [x.tolist() if isinstance(x, np.ndarray) else x for x in inputs_max]
    # check if inputs are tuples, and the elements are np arrays
    new_inputs_max = []
    for x in inputs_max:
        if isinstance(x, tuple):
            new_x = [y.tolist() if isinstance(y, np.ndarray) else y for y in x]
            new_inputs_max.append(tuple(new_x))
        else:
            new_inputs_max.append(x)
    
    store_dict = {
            "eigs_regret": eigs_regret,
            "eigs_max": eigs_max,
            "inputs_max": new_inputs_max,
            "eigs": eigs
    }
    out_path = os.path.join(res_dir, f"regret_{os.path.basename(res_filename)}")
    with open(out_path, 'w') as f:
        json.dump(store_dict, f, indent=4)

if __name__ == "__main__":
    main()
