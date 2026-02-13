import logging

logger = logging.getLogger(__name__)


def import_env_modules():
    envs = {}
    try:
        import boxing_gym.envs.location_finding as location_finding

        envs["location_finding"] = location_finding
    except ImportError:
        pass

    try:
        import boxing_gym.envs.hyperbolic_temporal_discount as hyperbolic_temporal_discount

        envs["hyperbolic_temporal_discount"] = hyperbolic_temporal_discount
    except ImportError:
        pass

    try:
        import boxing_gym.envs.death_process as death_process

        envs["death_process"] = death_process
    except ImportError:
        pass

    try:
        import boxing_gym.envs.irt as irt

        envs["irt"] = irt
    except ImportError:
        pass

    try:
        import boxing_gym.envs.survival_analysis as survival_analysis

        envs["survival_analysis"] = survival_analysis
    except ImportError:
        pass

    try:
        import boxing_gym.envs.peregrines as peregrines

        envs["peregrines"] = peregrines
    except ImportError:
        pass

    try:
        import boxing_gym.envs.dugongs as dugongs

        envs["dugongs"] = dugongs
    except ImportError:
        pass

    try:
        import boxing_gym.envs.lotka_volterra as lotka_volterra

        envs["lotka_volterra"] = lotka_volterra
    except ImportError:
        pass

    try:
        import boxing_gym.envs.moral_machines as moral_machines

        envs["moral_machines"] = moral_machines
    except ImportError:
        pass

    try:
        import boxing_gym.envs.emotion as emotion

        envs["emotion"] = emotion
    except ImportError:
        pass

    try:
        import boxing_gym.envs.microfluidics as microfluidics

        envs["microfluidics"] = microfluidics
    except ImportError:
        pass

    return envs


def get_environment_registry():
    """
    Returns:
        nametoenv: Dict[str, EnvClass]
        nameenvtogoal: Dict[Tuple[str, str], GoalClass]
    """
    ENV_MODULES = import_env_modules()

    nametoenv = {}
    nameenvtogoal = {}

    if "location_finding" in ENV_MODULES:
        mod = ENV_MODULES["location_finding"]
        nametoenv["location_finding"] = mod.Signal
        nameenvtogoal[("location_finding", "direct")] = mod.DirectGoal
        nameenvtogoal[("location_finding", "source")] = mod.SourceGoal
        nameenvtogoal[("location_finding", "direct_naive")] = mod.DirectGoalNaive

    if "hyperbolic_temporal_discount" in ENV_MODULES:
        mod = ENV_MODULES["hyperbolic_temporal_discount"]
        nametoenv["hyperbolic_temporal_discount"] = mod.TemporalDiscount
        nameenvtogoal[("hyperbolic_temporal_discount", "direct")] = mod.DirectGoal
        nameenvtogoal[("hyperbolic_temporal_discount", "discount")] = mod.DiscountGoal
        nameenvtogoal[("hyperbolic_temporal_discount", "direct_naive")] = mod.DirectGoalNaive

    if "death_process" in ENV_MODULES:
        mod = ENV_MODULES["death_process"]
        nametoenv["death_process"] = mod.DeathProcess
        nameenvtogoal[("death_process", "direct")] = mod.DirectDeath
        nameenvtogoal[("death_process", "direct_naive")] = mod.DirectDeathNaive
        nameenvtogoal[("death_process", "infection")] = mod.InfectionRate

    if "irt" in ENV_MODULES:
        mod = ENV_MODULES["irt"]
        nametoenv["irt"] = mod.IRT
        nameenvtogoal[("irt", "direct")] = mod.DirectCorrectness
        nameenvtogoal[("irt", "direct_naive")] = mod.DirectCorrectnessNaive
        nameenvtogoal[("irt", "best_student")] = mod.BestStudent
        nameenvtogoal[("irt", "difficult_question")] = mod.DifficultQuestion
        nameenvtogoal[("irt", "discriminate_question")] = mod.DiscriminatingQuestion

    if "survival_analysis" in ENV_MODULES:
        mod = ENV_MODULES["survival_analysis"]
        nametoenv["survival"] = mod.SurvivalAnalysis
        nameenvtogoal[("survival", "direct")] = mod.DirectGoal
        nameenvtogoal[("survival", "direct_naive")] = mod.DirectGoalNaive

    if "dugongs" in ENV_MODULES:
        mod = ENV_MODULES["dugongs"]
        nametoenv["dugongs"] = mod.Dugongs
        nameenvtogoal[("dugongs", "direct")] = mod.DirectGoal
        nameenvtogoal[("dugongs", "direct_naive")] = mod.DirectGoalNaive

    if "peregrines" in ENV_MODULES:
        mod = ENV_MODULES["peregrines"]
        nametoenv["peregrines"] = mod.Peregrines
        nameenvtogoal[("peregrines", "direct")] = mod.DirectGoal
        nameenvtogoal[("peregrines", "direct_naive")] = mod.DirectGoalNaive

    if "emotion" in ENV_MODULES:
        mod = ENV_MODULES["emotion"]
        nametoenv["emotion"] = mod.EmotionFromOutcome
        nameenvtogoal[("emotion", "direct")] = mod.DirectEmotionPrediction
        nameenvtogoal[("emotion", "direct_naive")] = mod.DirectEmotionNaive

    if "moral_machines" in ENV_MODULES:
        mod = ENV_MODULES["moral_machines"]
        nametoenv["morals"] = mod.MoralMachine
        nameenvtogoal[("morals", "direct")] = mod.DirectPrediction
        nameenvtogoal[("morals", "direct_naive")] = mod.DirectPredictionNaive

    if "lotka_volterra" in ENV_MODULES:
        mod = ENV_MODULES["lotka_volterra"]
        nametoenv["lotka_volterra"] = mod.LotkaVolterra
        nameenvtogoal[("lotka_volterra", "direct")] = mod.DirectGoal
        nameenvtogoal[("lotka_volterra", "direct_naive")] = mod.DirectGoalNaive

    if "microfluidics" in ENV_MODULES:
        mod = ENV_MODULES["microfluidics"]
        nametoenv["microfluidics"] = mod.SingleCellMicrofluidics
        # microfluidics goals should be defined if the module exists
        if hasattr(mod, "DirectGoal"):
            nameenvtogoal[("microfluidics", "direct")] = mod.DirectGoal
        if hasattr(mod, "DirectGoalNaive"):
            nameenvtogoal[("microfluidics", "direct_naive")] = mod.DirectGoalNaive

    return nametoenv, nameenvtogoal
