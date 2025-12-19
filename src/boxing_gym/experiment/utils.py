import importlib
import sys
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def _import_real_module(name: str, required_attr: str):
    """Import a module, retrying without repo-root shadowing if needed."""
    try:
        mod = importlib.import_module(name)
        if hasattr(mod, required_attr):
            return mod
    except Exception:
        mod = None

    # shadowing fallback: remove repo root / cwd from sys.path and retry.
    try:
        # filter out CWD to avoid local shadowing.
        old_sys_path = list(sys.path)

        filtered = []
        cwd = os.getcwd()
        for p in old_sys_path:
            # sys.path can contain "" which maps to cwd
            # we want to exclude the repo root if it's in the path and causing shadowing
            filtered.append(p)

        # keep the original behavior to avoid local shadowing.
        
        # if we are running from root, sys.path[0] is root.
        if "" in filtered:
            filtered.remove("")
        if "." in filtered:
            filtered.remove(".")
        if cwd in filtered:
            filtered.remove(cwd)
        
        sys.path = filtered
        sys.modules.pop(name, None)
        mod2 = importlib.import_module(name)
        if hasattr(mod2, required_attr):
            logger.warning(
                f"Recovered real '{name}' package after detecting shadowing by local folder."
            )
            return mod2
    except Exception:
        mod2 = None
    finally:
        try:
            sys.path = old_sys_path
        except Exception:
            pass

    if mod is not None and not hasattr(mod, required_attr):
        logger.warning(
            f"Imported a '{name}' module without {name}.{required_attr}; likely shadowed by local artifacts. "
            f"Disabling {name} integration for this run."
        )
    return None

def create_fallback_result(error_msg=""):
    """
    Create a minimal valid result tuple when evaluation fails.
    Returns the same first-4-element structure as evaluate():
    (evaluation_score, questions, gts, predictions)
    Note: ppl_evaluate() may append a 5th element (box_loop_stats) in PPL mode.
    
    Marks evaluation as failed with placeholder values to allow
    experiment to save partial results and continue.
    """
    fallback_score = {
        "evaluation_failed": True,
        "error_message": error_msg,
        "accuracy": None,
        "score": 0.0,
    }
    questions = []
    gts = []
    predictions = []
    
    return fallback_score, questions, gts, predictions

def compute_z_score(err_mean: float, norm_mu: float, norm_sigma: float) -> float:
    """Standardize raw error to Z-score for paper comparison.

    The paper reports standardized Z-scores, not raw error values.
    This function converts raw MSE/MAE to the paper's standardized metric.

    Note: The paper's reported std is computed as std(z_means) ACROSS multiple
    seeds during aggregation, not per-run. Per-run z_std is not meaningful
    for paper comparison and should be computed during aggregation.

    Args:
        err_mean: Raw error mean (MSE/MAE)
        norm_mu: Baseline mean from Goal class (naive predictor error)
        norm_sigma: Baseline std from Goal class

    Returns:
        z_mean, or None if inputs are invalid
    """
    # validate inputs
    if any(v is None for v in (err_mean, norm_mu, norm_sigma)):
        return None
    if norm_sigma <= 0:  # invalid: std dev must be positive
        return None
    z_mean = (err_mean - norm_mu) / norm_sigma
    return z_mean

def _format_emotion_prediction(vals):
    keys = [
        "Happiness",
        "Sadness",
        "Anger",
        "Surprise",
        "Fear",
        "Disgust",
        "Contentment",
        "Disappointment",
    ]
    out_lines = []
    for k, v in zip(keys, vals):
        try:
            fv = float(v)
        except Exception:
            fv = 5.0
        if not np.isfinite(fv):
            fv = 5.0
        # keep on 1-9 scale; clamp to a sane range.
        fv = float(np.clip(fv, 1.0, 9.0))
        out_lines.append(f"{k}: {fv:.2f}/9")
    return "\n".join(out_lines)

def _make_dummy_eval_input_for_env(env):
    """Create a single eval_input row suitable for construct_features(env, data=[...])."""
    if env is None:
        return None
    if not hasattr(env, "sample_random_input"):
        return None
    try:
        dummy = env.sample_random_input()
    except Exception:
        return None

    # construct_features expects each row to match what the env uses for eval_points[:-1]
    # which is often a tuple of features, but some envs use a single array wrapped in a tuple.
    try:
        if env.env_name in ["location_finding"]:
            # sample_random_input returns a vector; construct_features expects (vector,) row
            return (dummy,)
        if env.env_name in ["death_process", "lotka_volterra"]:
            # scalar x; represent as a one-tuple
            return (dummy,)
        # most envs already return tuples matching feature columns
        if isinstance(dummy, (list, tuple)):
            return tuple(dummy)
        return (dummy,)
    except Exception:
        return None

def _baseline_prediction_for_goal(goal):
    """Return a parseable baseline prediction string for the goal."""
    goal_mod = getattr(goal.__class__, "__module__", "")
    goal_cls = getattr(goal.__class__, "__name__", "")
    goal_full = f"{goal_mod}.{goal_cls}"
    env = getattr(goal, "env", None)

    if goal_full.endswith("emotion.DirectEmotionPrediction") or goal_full.endswith("emotion.DirectEmotionNaive"):
        # 8 emotions, Likert 1-9. Use neutral midpoint.
        vals = [5.0] * 8
        keys = [
            "Happiness",
            "Sadness",
            "Anger",
            "Surprise",
            "Fear",
            "Disgust",
            "Contentment",
            "Disappointment",
        ]
        return "\n".join([f"{k}: {v}/9" for k, v in zip(keys, vals)])

    if goal_full.endswith("lotka_volterra.DirectGoal") or goal_full.endswith("lotka_volterra.DirectGoalNaive"):
        return "[0, 0]"

    if goal_full.endswith("location_finding.SourceGoal"):
        # list-of-lists with shape (num_sources, dim)
        try:
            num_sources = int(getattr(env, "num_sources", 1))
            dim = int(getattr(env, "dim", 2))
        except Exception:
            num_sources, dim = 1, 2
        zeros = [[0.0 for _ in range(dim)] for _ in range(num_sources)]
        return str(zeros)

    # binary/categorical defaults
    if goal_full.endswith("moral_machines.DirectPrediction") or goal_full.endswith("moral_machines.DirectPredictionNaive"):
        return "1"

    if goal_full.endswith("hyperbolic_temporal_discount.DirectGoal") or goal_full.endswith(
        "hyperbolic_temporal_discount.DirectGoalNaive"
    ):
        return "0"
    if goal_full.endswith("survival_analysis.DirectGoal") or goal_full.endswith("survival_analysis.DirectGoalNaive"):
        return "0"
    if goal_full.endswith("irt.DirectCorrectness") or goal_full.endswith("irt.DirectCorrectnessNaive"):
        return "0"

    # parameter goals
    if goal_full.endswith("hyperbolic_temporal_discount.DiscountGoal"):
        # reasonable positive default; the evaluation is MSE on k
        return str(float(getattr(env, "k_mean", 0.1)) if env is not None else 0.1)
    if goal_full.endswith("death_process.InfectionRate"):
        # reasonable positive default
        return str(float(getattr(env, "mu", 0.1)) if env is not None else 0.1)

    return "0"
