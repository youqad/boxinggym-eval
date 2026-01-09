"""PPL (Probabilistic Programming Language) prediction and inference utilities.

WARNING - ARCHITECTURAL NOTE (2026-01):
    This module contains environment-specific prediction logic that dispatches
    based on goal class names (e.g., `goal_full.endswith("hyperbolic...")`).

    Goal classes in src/boxing_gym/envs/ ALSO contain probabilistic models
    (e.g., `expected_information_gain()` methods with full PyMC definitions).

    This creates potential for "split-brain" model definitions where the same
    environment has slightly different probabilistic assumptions in different
    code paths. Before modifying either this file OR goal class models:

    1. Verify which code path is active for your use case
    2. Ensure changes are reflected in both locations if needed
    3. Compare trace outputs on identical seeds to verify consistency

    Future work (deferred Phase 3.2): Unify model definitions into a single
    source of truth, likely by moving prediction logic into goal classes.
"""
import os
import tempfile
import uuid
import importlib
import importlib.util
import re
import logging
import warnings
import numpy as np
import pymc as pm

from boxing_gym.agents.box_loop_helper import construct_features
from boxing_gym.experiment.utils import (
    _baseline_prediction_for_goal,
    _format_emotion_prediction,
    _make_dummy_eval_input_for_env
)

logger = logging.getLogger(__name__)

def _suppress_ppl_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"pymc(\..*)?$")
    warnings.filterwarnings("ignore", message=r".*MutableData is deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=r".*Estimated shape parameter of Pareto distribution.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*point-wise LOO.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*point-wise WAIC.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*posterior variance of the log predictive densities exceeds.*", category=UserWarning)


def _quiet_ppl_loggers() -> None:
    for name in (
        "pymc",
        "arviz",
    ):
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False


def get_gen_model(gen_code):
    """Persist and load a generated PyMC program.

    This is used in prior-mode PPL evaluation. Hydra often changes the working
    directory (`hydra.job.chdir=true`), which can make `import src.*` brittle.
    We therefore load the module directly from its file path.
    """
    # write each generated program to a unique temp module to avoid collisions
    # across parallel sweeps/processes.
    tmp_root = os.path.join(tempfile.gettempdir(), "oed_llms", "stan")
    os.makedirs(tmp_root, exist_ok=True)
    unique_name = f"ppl_gen_model_{uuid.uuid4().hex}"
    module_path = os.path.join(tmp_root, f"{unique_name}.py")
    # compatibility shim for PPL-generated code
    shim = (
        "# --- BoxingGym PPL compatibility shim ---\n"
        "import pymc as pm\n"
        "import pytensor.tensor as pt\n"
        "if not hasattr(pm, 'MutableData') and hasattr(pm, 'Data'):\n"
        "    pm.MutableData = pm.Data\n"
        "if not hasattr(pm.math, 'square') and hasattr(pt, 'square'):\n"
        "    pm.math.square = pt.square\n"
        "if not hasattr(pm.math, 'expand_dims') and hasattr(pt, 'expand_dims'):\n"
        "    pm.math.expand_dims = pt.expand_dims\n"
        "# Legacy PyMC3 compatibility: pm.Bound(...) wrapper (removed in PyMC5)\n"
        "if not hasattr(pm, 'Bound') and hasattr(pm, 'TruncatedNormal'):\n"
        "    def _Bound(dist, lower=None, upper=None):\n"
        "        def _inner(name, *args, **kwargs):\n"
        "            try:\n"
        "                if dist is pm.Normal:\n"
        "                    mu = kwargs.pop('mu', 0.0)\n"
        "                    sigma = kwargs.pop('sigma', 1.0)\n"
        "                    return pm.TruncatedNormal(name, mu=mu, sigma=sigma, lower=lower, upper=upper, **kwargs)\n"
        "            except Exception:\n"
        "                pass\n"
        "            return dist(name, *args, **kwargs)\n"
        "        return _inner\n"
        "    pm.Bound = _Bound\n"
        "# --- End shim ---\n\n"
    )
    with open(module_path, "w") as file:
        file.write(shim + (gen_code or ""))

    importlib.invalidate_caches()

    # load directly from path to avoid sys.path / chdir issues.
    spec = importlib.util.spec_from_file_location(unique_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gen_model_fn = getattr(module, "gen_model", None)
    if gen_model_fn is None:
        raise AttributeError("Generated PPL module did not define gen_model")
    return gen_model_fn


def _preflight_check_observed_data_columns(gen_code: str, available_columns) -> None:
    """Fail fast if generated code indexes missing observed_data columns.

    This mirrors the preflight check used in BoxLoop scoring, but is also useful
    in prior-mode evaluation where `observed_data` is a single-row feature frame.
    """
    if not gen_code or not isinstance(gen_code, str):
        return
    try:
        cols = set(str(c) for c in list(available_columns))
    except Exception:
        cols = set()
    if not cols:
        return

    # strip docstrings/comments to reduce false positives.
    code_for_scan = re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', "", gen_code, flags=re.DOTALL)
    code_for_scan = re.sub(r"#.*", "", code_for_scan)
    referenced_cols = set(
        re.findall(r"observed_data\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", code_for_scan)
    )
    missing = referenced_cols - cols
    if missing:
        raise KeyError(
            f"Generated program refers to missing columns {sorted(missing)}. "
            f"Available columns: {sorted(cols)}"
        )

def _trace_posterior_var_names(trace):
    if trace is None:
        return []
    if hasattr(trace, "posterior") and getattr(trace, "posterior", None) is not None:
        try:
            return list(trace.posterior.data_vars.keys())
        except Exception:
            return []
    if hasattr(trace, "varnames"):
        try:
            return list(trace.varnames)
        except Exception:
            return []
    return []



def _goal_full_name(goal) -> str:
    goal_mod = getattr(goal.__class__, "__module__", "")
    goal_cls = getattr(goal.__class__, "__name__", "")
    return f"{goal_mod}.{goal_cls}"


def _pp_mean(arr):
    arr = np.asarray(arr)
    if arr.ndim <= 1:
        return float(arr.mean())
    return arr.mean(axis=tuple(range(arr.ndim - 1))).squeeze()


def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(k).lower())


def _alias_map_for_goal(goal_full: str) -> dict:
    if goal_full.endswith("hyperbolic_temporal_discount.DirectGoal") or goal_full.endswith(
        "hyperbolic_temporal_discount.DirectGoalNaive"
    ):
        return {
            "Immediate_Reward": ["ir", "iR", "immediate_reward"],
            "Delayed_Reward": ["dr", "dR", "delayed_reward"],
            "Delay_Days": ["days", "D", "delay_days"],
        }
    if goal_full.endswith("emotion.DirectEmotionPrediction") or goal_full.endswith("emotion.DirectEmotionNaive"):
        return {
            "Prize_1": ["prize_1", "prize1"],
            "Prize_2": ["prize_2", "prize2"],
            "Prize_3": ["prize_3", "prize3"],
            "Prob_1": ["prob_1", "prob1"],
            "Prob_2": ["prob_2", "prob2"],
            "Prob_3": ["prob_3", "prob3"],
            "win": ["Outcome", "outcome", "win"],
        }
    return {}


def _map_model_data(goal_full: str, model, data_dict: dict) -> dict:
    model_vars = set()
    try:
        model_vars = set(getattr(model, "named_vars", {}).keys())
    except Exception:
        model_vars = set()
    try:
        model_vars |= set(getattr(model, "data_vars", {}).keys())
    except Exception:
        pass

    norm_model = {}
    for mv in model_vars:
        nk = _norm_key(mv)
        if nk not in norm_model:
            norm_model[nk] = mv

    alias_map = _alias_map_for_goal(goal_full)

    mapped = {}
    used_model_vars = set()
    for col, values in data_dict.items():
        if col in model_vars and col not in used_model_vars:
            mapped[col] = values
            used_model_vars.add(col)
            continue
        norm = _norm_key(col)
        if norm in norm_model and norm_model[norm] not in used_model_vars:
            target = norm_model[norm]
            mapped[target] = values
            used_model_vars.add(target)
            continue
        if col in alias_map:
            for alt in alias_map[col]:
                if alt in model_vars and alt not in used_model_vars:
                    mapped[alt] = values
                    used_model_vars.add(alt)
                    break
                alt_norm = _norm_key(alt)
                if alt_norm in norm_model and norm_model[alt_norm] not in used_model_vars:
                    target = norm_model[alt_norm]
                    mapped[target] = values
                    used_model_vars.add(target)
                    break

    return mapped if mapped else data_dict


def _prepare_prior_predictive(goal_full, env, program_dict, eval_input, goal):
    if eval_input is None:
        dummy = _make_dummy_eval_input_for_env(env)
        if dummy is None:
            return None, eval_input
        eval_input = dummy
    str_prob_prog = program_dict["str_prob_prog"]
    prior_model = get_gen_model(str_prob_prog)
    observed_df = construct_features(env, data=[eval_input])
    _preflight_check_observed_data_columns(str_prob_prog, observed_df.columns)
    with warnings.catch_warnings():
        _suppress_ppl_warnings()
        _quiet_ppl_loggers()
        prior_out = prior_model(observed_df)

    model = None
    prior_predictive = None
    try:
        if isinstance(prior_out, (list, tuple)):
            if len(prior_out) == 2:
                model, prior_predictive = prior_out
            elif len(prior_out) >= 3:
                model, prior_predictive = prior_out[:2]
            else:
                raise ValueError("prior gen_model returned empty tuple")
        else:
            prior_predictive = prior_out
    except Exception as e:
        logger.error(f"Prior gen_model returned invalid structure: {e}")
        raise

    if not prior_predictive or not isinstance(prior_predictive, dict):
        raise ValueError("prior_predictive is empty or not a dict")

    return prior_predictive, eval_input


def _predict_param_goal_posterior(goal_full, env, trace, goal):
    if trace is None:
        return None

    if goal_full.endswith("hyperbolic_temporal_discount.DiscountGoal"):
        var_names = _trace_posterior_var_names(trace)
        if "k" in var_names:
            k_hat = float(_trace_get_samples(trace, "k").mean())
            if np.isfinite(k_hat) and k_hat > 0:
                return str(k_hat)
            return _baseline_prediction_for_goal(goal)
        if "log_k" in var_names:
            k_hat = float(np.exp(_trace_get_samples(trace, "log_k").mean()))
            if np.isfinite(k_hat) and k_hat > 0:
                return str(k_hat)
            return _baseline_prediction_for_goal(goal)
        for vn in var_names:
            if "k" in vn.lower() and "sigma" not in vn.lower():
                try:
                    samp = _trace_get_samples(trace, vn)
                    if np.asarray(samp).ndim == 1:
                        k_hat = float(np.asarray(samp).mean())
                        if np.isfinite(k_hat) and k_hat > 0:
                            return str(k_hat)
                except Exception:
                    continue
        return _baseline_prediction_for_goal(goal)

    if goal_full.endswith("death_process.InfectionRate"):
        var_names = _trace_posterior_var_names(trace)
        if "theta" in var_names:
            theta_hat = float(_trace_get_samples(trace, "theta").mean())
            if np.isfinite(theta_hat) and theta_hat > 0:
                return str(theta_hat)
            return _baseline_prediction_for_goal(goal)
        if "log_theta" in var_names:
            theta_hat = float(np.exp(_trace_get_samples(trace, "log_theta").mean()))
            if np.isfinite(theta_hat) and theta_hat > 0:
                return str(theta_hat)
            return _baseline_prediction_for_goal(goal)
        for vn in var_names:
            if "theta" in vn.lower() and "sigma" not in vn.lower():
                try:
                    samp = _trace_get_samples(trace, vn)
                    if np.asarray(samp).ndim == 1:
                        theta_hat = float(np.asarray(samp).mean())
                        if np.isfinite(theta_hat) and theta_hat > 0:
                            return str(theta_hat)
                except Exception:
                    continue
        return _baseline_prediction_for_goal(goal)

    if goal_full.endswith("location_finding.SourceGoal"):
        var_names = _trace_posterior_var_names(trace)
        target_shape = (int(getattr(env, "num_sources", 1)), int(getattr(env, "dim", 2)))
        if "theta" in var_names:
            theta_hat = np.asarray(_trace_get_samples(trace, "theta")).mean(axis=0)
            if theta_hat.shape == target_shape and np.all(np.isfinite(theta_hat)):
                return str(theta_hat.tolist())
        for vn in var_names:
            try:
                theta_hat = np.asarray(_trace_get_samples(trace, vn)).mean(axis=0)
            except Exception:
                continue
            if theta_hat.shape == target_shape and np.all(np.isfinite(theta_hat)):
                return str(theta_hat.tolist())
        return _baseline_prediction_for_goal(goal)

    if goal_full.endswith("irt.BestStudent"):
        var_names = _trace_posterior_var_names(trace)
        num_students = int(getattr(env, "num_students", 0))
        candidates = []
        for vn in var_names:
            try:
                mean_v = np.asarray(_trace_get_samples(trace, vn)).mean(axis=0)
            except Exception:
                continue
            if mean_v.shape == (num_students,):
                score = 0
                lvn = vn.lower()
                if "alpha" in lvn or "ability" in lvn or "student" in lvn:
                    score += 2
                candidates.append((score, vn, mean_v))
        if candidates:
            _, _, mean_v = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
            return str(int(np.argmax(mean_v)))
        return _baseline_prediction_for_goal(goal)

    if goal_full.endswith("irt.DifficultQuestion"):
        var_names = _trace_posterior_var_names(trace)
        num_questions = int(getattr(env, "num_questions", 0))
        candidates = []
        for vn in var_names:
            try:
                mean_v = np.asarray(_trace_get_samples(trace, vn)).mean(axis=0)
            except Exception:
                continue
            if mean_v.shape == (num_questions,):
                score = 0
                lvn = vn.lower()
                if "beta" in lvn or "difficulty" in lvn or "question" in lvn:
                    score += 2
                candidates.append((score, vn, mean_v))
        if candidates:
            _, _, mean_v = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
            return str(int(np.argmax(mean_v)))
        return _baseline_prediction_for_goal(goal)

    if goal_full.endswith("irt.DiscriminatingQuestion"):
        var_names = _trace_posterior_var_names(trace)
        num_questions = int(getattr(env, "num_questions", 0))
        candidates = []
        for vn in var_names:
            try:
                mean_v = np.asarray(_trace_get_samples(trace, vn)).mean(axis=0)
            except Exception:
                continue
            if mean_v.shape == (num_questions,):
                score = 0
                lvn = vn.lower()
                if "gamma" in lvn or "disc" in lvn or "discrimin" in lvn:
                    score += 2
                candidates.append((score, vn, mean_v))
        if candidates:
            _, _, mean_v = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
            return str(int(np.argmax(mean_v)))
        return _baseline_prediction_for_goal(goal)

    return None


def _predict_param_goal_prior(goal_full, env, prior_predictive, goal):
    if goal_full.endswith("hyperbolic_temporal_discount.DiscountGoal"):
        try:
            if "k" in prior_predictive:
                k_hat = float(_prior_get_samples(prior_predictive, "k").mean())
                if np.isfinite(k_hat) and k_hat > 0:
                    return str(k_hat)
            if "log_k" in prior_predictive:
                k_hat = float(np.exp(_prior_get_samples(prior_predictive, "log_k").mean()))
                if np.isfinite(k_hat) and k_hat > 0:
                    return str(k_hat)
            for vn in list(prior_predictive.keys()):
                lvn = str(vn).lower()
                if "k" in lvn and "sigma" not in lvn and "log" not in lvn:
                    try:
                        samp = _prior_get_samples(prior_predictive, vn)
                    except Exception:
                        continue
                    if np.asarray(samp).ndim == 1:
                        k_hat = float(np.asarray(samp).mean())
                        if np.isfinite(k_hat) and k_hat > 0:
                            return str(k_hat)
        except Exception:
            pass
        return _baseline_prediction_for_goal(goal)

    if goal_full.endswith("death_process.InfectionRate"):
        try:
            if "theta" in prior_predictive:
                theta_hat = float(_prior_get_samples(prior_predictive, "theta").mean())
                if np.isfinite(theta_hat) and theta_hat > 0:
                    return str(theta_hat)
            if "log_theta" in prior_predictive:
                theta_hat = float(np.exp(_prior_get_samples(prior_predictive, "log_theta").mean()))
                if np.isfinite(theta_hat) and theta_hat > 0:
                    return str(theta_hat)
            for vn in list(prior_predictive.keys()):
                lvn = str(vn).lower()
                if "theta" in lvn and "sigma" not in lvn and "log" not in lvn:
                    try:
                        samp = _prior_get_samples(prior_predictive, vn)
                    except Exception:
                        continue
                    if np.asarray(samp).ndim == 1:
                        theta_hat = float(np.asarray(samp).mean())
                        if np.isfinite(theta_hat) and theta_hat > 0:
                            return str(theta_hat)
        except Exception:
            pass
        return _baseline_prediction_for_goal(goal)

    if goal_full.endswith("location_finding.SourceGoal"):
        try:
            target_shape = (int(getattr(env, "num_sources", 1)), int(getattr(env, "dim", 2)))
        except Exception:
            target_shape = (1, 2)
        for vn in list(prior_predictive.keys()):
            try:
                samples = _prior_get_samples(prior_predictive, vn)
            except Exception:
                continue
            try:
                mean_v = np.asarray(samples).mean(axis=0)
            except Exception:
                continue
            if mean_v.shape == target_shape:
                return str(mean_v.tolist())
        return _baseline_prediction_for_goal(goal)

    if goal_full.endswith("irt.BestStudent") or goal_full.endswith("irt.DifficultQuestion") or goal_full.endswith("irt.DiscriminatingQuestion"):
        return _baseline_prediction_for_goal(goal)

    return None


def _predict_direct_prior(goal_full, prior_predictive, goal):
    if goal_full.endswith("moral_machines.DirectPrediction") or goal_full.endswith("moral_machines.DirectPredictionNaive"):
        if "y_obs" in prior_predictive:
            pp = prior_predictive["y_obs"]
        elif "choice" in prior_predictive:
            pp = prior_predictive["choice"]
        else:
            return _baseline_prediction_for_goal(goal)
        arr = np.asarray(pp)
        p1 = float(arr.mean())
        if not np.isfinite(p1):
            return _baseline_prediction_for_goal(goal)
        if p1 > 1.0:
            return str(2 if p1 >= 1.5 else 1)
        return str(2 if p1 >= 0.5 else 1)
    if goal_full.endswith("hyperbolic_temporal_discount.DirectGoal") or goal_full.endswith(
        "hyperbolic_temporal_discount.DirectGoalNaive"
    ):
        if "y_obs" not in prior_predictive:
            return _baseline_prediction_for_goal(goal)
        arr = np.asarray(prior_predictive["y_obs"])
        p1 = float(arr.mean())
        if not np.isfinite(p1):
            return _baseline_prediction_for_goal(goal)
        return str(1 if p1 >= 0.5 else 0)
    if goal_full.endswith("survival_analysis.DirectGoal") or goal_full.endswith("survival_analysis.DirectGoalNaive"):
        if "y_obs" not in prior_predictive:
            return _baseline_prediction_for_goal(goal)
        arr = np.asarray(prior_predictive["y_obs"])
        p1 = float(arr.mean())
        if not np.isfinite(p1):
            return _baseline_prediction_for_goal(goal)
        return str(1 if p1 >= 0.5 else 0)
    if goal_full.endswith("irt.DirectCorrectness") or goal_full.endswith("irt.DirectCorrectnessNaive"):
        if "y_obs" not in prior_predictive:
            return _baseline_prediction_for_goal(goal)
        arr = np.asarray(prior_predictive["y_obs"])
        p1 = float(arr.mean())
        if not np.isfinite(p1):
            return _baseline_prediction_for_goal(goal)
        return str(1 if p1 >= 0.5 else 0)
    if goal_full.endswith("lotka_volterra.DirectGoal") or goal_full.endswith("lotka_volterra.DirectGoalNaive"):
        if "prey" in prior_predictive and "predator" in prior_predictive:
            prey_mu = float(np.asarray(prior_predictive["prey"]).mean())
            pred_mu = float(np.asarray(prior_predictive["predator"]).mean())
            if not (np.isfinite(prey_mu) and np.isfinite(pred_mu)):
                return _baseline_prediction_for_goal(goal)
            return str([prey_mu, pred_mu])
        if "y_obs" in prior_predictive:
            arr = np.asarray(prior_predictive["y_obs"])
            mean = arr.mean(axis=tuple(range(arr.ndim - 1))).squeeze() if arr.ndim >= 2 else float(arr.mean())
            vec = np.asarray(mean).reshape(-1)
            if vec.size >= 2:
                a, b = float(vec[0]), float(vec[1])
                if np.isfinite(a) and np.isfinite(b):
                    return str([a, b])
        return _baseline_prediction_for_goal(goal)
    if goal_full.endswith("emotion.DirectEmotionPrediction") or goal_full.endswith("emotion.DirectEmotionNaive"):
        vals = []
        for k in ["happiness", "sadness", "anger", "surprise", "fear", "disgust", "contentment", "disappointment"]:
            kk = k if k in prior_predictive else f"{k}i" if f"{k}i" in prior_predictive else None
            if kk is None:
                vals.append(5.0)
            else:
                vals.append(float(np.asarray(prior_predictive[kk]).mean()))
        if any(v != 5.0 for v in vals):
            return _format_emotion_prediction(vals)
        if "y_obs" in prior_predictive:
            arr = np.asarray(prior_predictive["y_obs"])
            mean = arr.mean(axis=tuple(range(arr.ndim - 1))).squeeze() if arr.ndim >= 2 else float(arr.mean())
            vec = np.asarray(mean).reshape(-1)
            if vec.size >= 8:
                return _format_emotion_prediction(vec[:8])
        return _baseline_prediction_for_goal(goal)

    if "y_obs" in prior_predictive:
        arr = np.asarray(prior_predictive["y_obs"])
        mu = arr.mean(axis=tuple(range(arr.ndim - 1))).squeeze() if arr.ndim >= 2 else float(arr.mean())
        val = float(np.asarray(mu).squeeze())
        if np.isfinite(val):
            return str(val)
        return _baseline_prediction_for_goal(goal)

    return _baseline_prediction_for_goal(goal)


def _run_posterior_predictive(goal_full, model, trace, eval_input, env):
    if eval_input is None:
        return None
    features_df = construct_features(env, data=[eval_input])
    data_dict = {col: np.asarray(features_df[col].to_numpy()) for col in features_df.columns}
    mapped = _map_model_data(goal_full, model, data_dict)

    with warnings.catch_warnings():
        _suppress_ppl_warnings()
        _quiet_ppl_loggers()
        with model:
            pm.set_data(mapped)
            obs_vars = _infer_observed_var_names(trace, model=model)
            return pm.sample_posterior_predictive(trace, var_names=obs_vars, return_inferencedata=False)


def _predict_direct_posterior(goal_full, post_pred, goal):
    if post_pred is None or not isinstance(post_pred, dict):
        return None

    if goal_full.endswith("moral_machines.DirectPrediction") or goal_full.endswith("moral_machines.DirectPredictionNaive"):
        key = "y_obs" if "y_obs" in post_pred else list(post_pred.keys())[0]
        p1 = float(np.asarray(_pp_mean(post_pred[key])).squeeze())
        if not np.isfinite(p1):
            return _baseline_prediction_for_goal(goal)
        if p1 > 1.0:
            return str(2 if p1 >= 1.5 else 1)
        return str(2 if p1 >= 0.5 else 1)

    if goal_full.endswith("hyperbolic_temporal_discount.DirectGoal") or goal_full.endswith(
        "hyperbolic_temporal_discount.DirectGoalNaive"
    ):
        key = "y_obs" if "y_obs" in post_pred else list(post_pred.keys())[0]
        p1 = float(np.asarray(_pp_mean(post_pred[key])).squeeze())
        if not np.isfinite(p1):
            return _baseline_prediction_for_goal(goal)
        return str(1 if p1 >= 0.5 else 0)

    if goal_full.endswith("survival_analysis.DirectGoal") or goal_full.endswith("survival_analysis.DirectGoalNaive"):
        key = "y_obs" if "y_obs" in post_pred else list(post_pred.keys())[0]
        p1 = float(np.asarray(_pp_mean(post_pred[key])).squeeze())
        if not np.isfinite(p1):
            return _baseline_prediction_for_goal(goal)
        return str(1 if p1 >= 0.5 else 0)

    if goal_full.endswith("irt.DirectCorrectness") or goal_full.endswith("irt.DirectCorrectnessNaive"):
        key = "y_obs" if "y_obs" in post_pred else list(post_pred.keys())[0]
        p1 = float(np.asarray(_pp_mean(post_pred[key])).squeeze())
        if not np.isfinite(p1):
            return _baseline_prediction_for_goal(goal)
        return str(1 if p1 >= 0.5 else 0)

    if goal_full.endswith("lotka_volterra.DirectGoal") or goal_full.endswith("lotka_volterra.DirectGoalNaive"):
        if "y_obs" in post_pred:
            vec = np.asarray(_pp_mean(post_pred["y_obs"])).reshape(-1)
            if vec.size >= 2:
                a, b = float(vec[0]), float(vec[1])
                if np.isfinite(a) and np.isfinite(b):
                    return str([a, b])
        prey_key = None
        pred_key = None
        for k in post_pred.keys():
            lk = k.lower()
            if prey_key is None and "prey" in lk:
                prey_key = k
            if pred_key is None and "pred" in lk:
                pred_key = k
        if prey_key and pred_key:
            a = float(_pp_mean(post_pred[prey_key]))
            b = float(_pp_mean(post_pred[pred_key]))
            if np.isfinite(a) and np.isfinite(b):
                return str([a, b])
        return _baseline_prediction_for_goal(goal)

    if goal_full.endswith("emotion.DirectEmotionPrediction") or goal_full.endswith("emotion.DirectEmotionNaive"):
        if "y_obs" in post_pred:
            vec = np.asarray(_pp_mean(post_pred["y_obs"])).reshape(-1)
            if vec.size >= 8:
                return _format_emotion_prediction(vec[:8])
        vals = []
        for k in ["happiness", "sadness", "anger", "surprise", "fear", "disgust", "contentment", "disappointment"]:
            kk = k if k in post_pred else f"{k}i" if f"{k}i" in post_pred else None
            if kk is None:
                vals.append(5.0)
            else:
                vals.append(float(np.asarray(_pp_mean(post_pred[kk])).squeeze()))
        return _format_emotion_prediction(vals)

    if goal_full.endswith("death_process.DirectDeath") or goal_full.endswith("death_process.DirectDeathNaive"):
        key = "y_obs" if "y_obs" in post_pred else list(post_pred.keys())[0]
        mu = float(np.asarray(_pp_mean(post_pred[key])).squeeze())
        if not np.isfinite(mu):
            return _baseline_prediction_for_goal(goal)
        return str(int(round(mu)))

    key = "y_obs" if "y_obs" in post_pred else list(post_pred.keys())[0]
    mu = _pp_mean(post_pred[key])
    val = float(np.asarray(mu).squeeze())
    if np.isfinite(val):
        return str(val)
    return _baseline_prediction_for_goal(goal)


def _trace_observed_var_names(trace):
    if trace is None:
        return []
    if hasattr(trace, "observed_data") and getattr(trace, "observed_data", None) is not None:
        try:
            return list(trace.observed_data.data_vars.keys())
        except Exception:
            return []
    return []


def _trace_get_samples(trace, var_name):
    """Return posterior samples as a numpy array, flattened over chains/draws."""
    if trace is None:
        raise ValueError("trace is None")

    if hasattr(trace, "posterior") and getattr(trace, "posterior", None) is not None:
        arr = np.asarray(trace.posterior[var_name].values)
        # expected shape: (chain, draw, *event_shape) OR (draw, *event_shape)
        if arr.ndim >= 2:
            flat = arr.reshape(-1, *arr.shape[2:]) if arr.ndim >= 3 else arr.reshape(-1, *arr.shape[1:])
            return flat
        return arr

    # PyMC MultiTrace / legacy
    if hasattr(trace, "get_values"):
        return np.asarray(trace.get_values(var_name, combine=True))

    # last resort, try dict-like access.
    try:
        return np.asarray(trace[var_name])
    except Exception as exc:
        raise KeyError(f"Could not extract '{var_name}' from trace") from exc


def _prior_get_samples(prior_predictive, var_name):
    if not prior_predictive or var_name not in prior_predictive:
        raise KeyError(var_name)
    arr = np.asarray(prior_predictive[var_name])
    if arr.ndim >= 2:
        return arr.reshape(-1, *arr.shape[1:])
    return arr.reshape(-1)


def _infer_observed_var_names(trace, model=None, fallback=None):
    obs = _trace_observed_var_names(trace)
    if obs:
        return obs
    # InferenceData may omit observed_data in some configurations; fall back to
    # the PyMC model's observed RVs when available.
    if model is not None:
        try:
            obs_rvs = getattr(model, "observed_RVs", None) or []
            names = []
            for rv in obs_rvs:
                n = getattr(rv, "name", None)
                if n:
                    names.append(str(n))
            if names:
                return names
        except Exception:
            pass
    if fallback:
        return list(fallback)
    return ["y_obs"]


# NOTE: get_ppl_prediction does NOT use @weave_op because it receives complex
# objects (Goal, program_dict with PyMC traces) that Weave cannot serialize.
def get_ppl_prediction(goal, program_dict, eval_input, prior_mode):
    """Generate a prediction string for the current goal using the top PPL program.

    - For direct predictive goals: uses (prior|posterior) predictive sampling.
    - For parameter-identification goals: extracts parameter estimates from trace.
    """
    goal_mod = getattr(goal.__class__, "__module__", "")
    goal_cls = getattr(goal.__class__, "__name__", "")
    goal_full = f"{goal_mod}.{goal_cls}"
    env = goal.env

    # Parameter / ID goals
    if not prior_mode:
        trace = program_dict.get("trace")
        pred = _predict_param_goal_posterior(goal_full, env, trace, goal)
        if pred is not None:
            return pred

    # Predictive goals
    if prior_mode:
        prior_predictive, eval_input = _prepare_prior_predictive(
            goal_full, env, program_dict, eval_input, goal
        )
        if prior_predictive is None:
            return _baseline_prediction_for_goal(goal)

        pred = _predict_param_goal_prior(goal_full, env, prior_predictive, goal)
        if pred is not None:
            return pred

        pred = _predict_direct_prior(goal_full, prior_predictive, goal)
        if pred is not None:
            return pred

        return _baseline_prediction_for_goal(goal)

    # posterior predictive (non-prior mode)
    if eval_input is None:
        # no input features available; fall back to parameter baseline.
        return _baseline_prediction_for_goal(goal)
    model = program_dict["model"]
    trace = program_dict["trace"]
    post_pred = _run_posterior_predictive(goal_full, model, trace, eval_input, env)
    pred = _predict_direct_posterior(goal_full, post_pred, goal)
    if pred is not None:
        return pred
    return _baseline_prediction_for_goal(goal)

def augment_scientist_with_ppl(scientist,
                               proposed_programs_all,
                               critic_info_all, critic_mode=False):
    assert len(proposed_programs_all[-1]) > 0

    program_dict = proposed_programs_all[-1][0]
    str_prob_prog = program_dict['str_prob_prog']

    prompt_msg = f"""
To help guide your experimentation, your brilliant colleague has proposed the following program for the data.
Use this program to guide your experimentation.
Program: {str_prob_prog} \n
\n
"""
    if critic_mode:
        assert len(critic_info_all[-1]) > 0
        str_hypotheses = critic_info_all[-1][0]['str_hypotheses']
        synthesis = critic_info_all[-1][0]['synthesis']
        prompt_msg += f"""
Here is criticism of the previous model:
{str_hypotheses} \n
{synthesis} \n
"""

    system_message = scientist.system
    system_message += f"\n {prompt_msg}"
    print(f"system_message: {system_message}")
    scientist.messages[0]['content'] = [{"type": "text", "text": system_message}]
