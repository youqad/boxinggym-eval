import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import pymc as pm  # type: ignore
except Exception:
    pm = None

try:
    import arviz as az  # type: ignore
except Exception:
    az = None

pytestmark = pytest.mark.skipif(
    pm is None or az is None, reason="pymc/arviz not installed (BoxLM/PPL tests skipped)"
)

if pm is not None and az is not None:
    from boxing_gym.agents.box_loop_helper import construct_features, pymc_evaluate
    from boxing_gym.envs.dugongs import Dugongs, DirectGoal as DugongsDirectGoal
    from boxing_gym.envs.emotion import DirectEmotionPrediction, EmotionFromOutcome
    from boxing_gym.envs.hyperbolic_temporal_discount import DirectGoal as TDDirectGoal
    from boxing_gym.envs.hyperbolic_temporal_discount import TemporalDiscount
    from boxing_gym.envs.lotka_volterra import DirectGoal as LVDirectGoal
    from boxing_gym.envs.lotka_volterra import LotkaVolterra
    from boxing_gym.envs.hyperbolic_temporal_discount import DiscountGoal
    from boxing_gym.envs.death_process import DeathProcess, InfectionRate
    from boxing_gym.experiment.ppl import get_ppl_prediction


def test_construct_features_emotion_flattens_win_index():
    env = EmotionFromOutcome()
    env.include_prior = True

    prizes = np.array([10.0, 20.0, 30.0])
    probs = np.array([0.2, 0.3, 0.5])
    win_idx = 1

    df = construct_features(env, data=[(prizes, probs, win_idx)])
    assert list(df.columns) == env.get_ordered_features()
    assert float(df.iloc[0]["win"]) == float(prizes[win_idx])


def test_construct_dataframe_handles_empty_observations():
    # Some runs (especially with strict validation) can end up with no successful
    # observations at a non-zero budget; construct_dataframe should return an
    # empty DF instead of raising IndexError.
    from boxing_gym.agents.box_loop_helper import construct_dataframe

    env = EmotionFromOutcome()
    env.include_prior = True
    df = construct_dataframe(env)
    assert list(df.columns) == env.get_ordered_column_names()
    assert df.empty


def test_emotion_validate_input_accepts_numeric_list_and_keyed_formats():
    env = EmotionFromOutcome()

    env.include_prior = True
    prizes, probs, win = env.validate_input("[30, 50, 70, 0.33, 0.33, 0.34, 1]")
    assert prizes.shape == (3,)
    assert probs.shape == (3,)
    assert float(win) == float(prizes[1])

    # Non-prior keyword variant should also work.
    env.include_prior = False
    prizes2, probs2, win2 = env.validate_input("x: [10, 20, 30], p: [0.2, 0.3, 0.5], outcome: 2")
    assert prizes2.shape == (3,)
    assert probs2.shape == (3,)
    assert float(win2) == float(prizes2[2])


def test_lotka_volterra_get_df_and_features_are_consistent():
    env = LotkaVolterra()
    env.include_prior = True

    # Need at least one observation to build env.df
    _, ok = env.run_experiment("0.5")
    assert ok
    env.get_df()

    assert env.df.shape[1] == 3
    assert env.get_ordered_features() == ["t"]


def test_get_ppl_prediction_binary_goal_returns_discrete_label():
    env = TemporalDiscount()
    env.include_prior = True
    goal = TDDirectGoal(env)

    # Tiny training dataset
    rows = []
    rng = np.random.default_rng(0)
    for _ in range(6):
        ir = int(rng.integers(1, 50))
        dr = int(rng.integers(ir + 1, 60))
        days = int(rng.integers(1, 30))
        y = int(env.step(ir, dr, days))
        rows.append((ir, dr, days, y))
    obs = np.array(rows)
    ir_obs, dr_obs, days_obs, y_obs = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]

    with pm.Model() as model:
        ir = pm.Data("ir", ir_obs, dims="obs_id")
        dr = pm.Data("dr", dr_obs, dims="obs_id")
        days = pm.Data("days", days_obs, dims="obs_id")

        beta = pm.Normal("beta", 0.0, 0.1, shape=3)
        intercept = pm.Normal("intercept", 0.0, 0.1)
        logits = intercept + beta[0] * ir + beta[1] * dr + beta[2] * days
        p = pm.Deterministic("p", pm.math.sigmoid(logits), dims="obs_id")
        pm.Bernoulli("y_obs", p=p, observed=y_obs, dims="obs_id")

        trace = pm.sample(
            20,
            tune=20,
            chains=1,
            cores=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    program_dict = {"model": model, "trace": trace}

    # Generate an eval input (fills goal.eval_points)
    goal.get_goal_eval_question(include_prior=True)
    eval_input = goal.eval_points[0][:-1]

    pred = get_ppl_prediction(goal, program_dict, eval_input, prior_mode=False)
    assert pred in {"0", "1"}


def test_get_ppl_prediction_emotion_format_is_parseable():
    env = EmotionFromOutcome()
    env.include_prior = True
    goal = DirectEmotionPrediction(env)

    # Small training dataset: 7 features -> 8 outputs
    rng = np.random.default_rng(1)
    X_rows = []
    Y_rows = []
    emotion_keys = [
        "happiness",
        "sadness",
        "anger",
        "surprise",
        "fear",
        "disgust",
        "contentment",
        "disappointment",
    ]
    for _ in range(5):
        prizes = rng.uniform(0, 100, 3)
        probs = rng.dirichlet(np.ones(3))
        win_idx = int(rng.integers(0, 3))
        win = float(prizes[win_idx])
        _, emotions = env.step(prizes, probs, win)
        emo_vec = [emotions[k] for k in emotion_keys]

        X_rows.append([*map(float, prizes), *map(float, probs), float(win)])
        Y_rows.append(list(map(float, emo_vec)))

    X = np.asarray(X_rows, dtype=float)
    Y = np.asarray(Y_rows, dtype=float)

    with pm.Model() as model:
        pm.Data("prize_1", X[:, 0], dims="obs_id")
        pm.Data("prize_2", X[:, 1], dims="obs_id")
        pm.Data("prize_3", X[:, 2], dims="obs_id")
        pm.Data("prob_1", X[:, 3], dims="obs_id")
        pm.Data("prob_2", X[:, 4], dims="obs_id")
        pm.Data("prob_3", X[:, 5], dims="obs_id")
        pm.Data("win", X[:, 6], dims="obs_id")

        alpha = pm.Normal("alpha", mu=5.0, sigma=2.0, shape=8)
        sigma = pm.HalfNormal("sigma", sigma=1.0, shape=8)

        pm.Normal("happiness", mu=alpha[0], sigma=sigma[0], observed=Y[:, 0], dims="obs_id")
        pm.Normal("sadness", mu=alpha[1], sigma=sigma[1], observed=Y[:, 1], dims="obs_id")
        pm.Normal("anger", mu=alpha[2], sigma=sigma[2], observed=Y[:, 2], dims="obs_id")
        pm.Normal("surprise", mu=alpha[3], sigma=sigma[3], observed=Y[:, 3], dims="obs_id")
        pm.Normal("fear", mu=alpha[4], sigma=sigma[4], observed=Y[:, 4], dims="obs_id")
        pm.Normal("disgust", mu=alpha[5], sigma=sigma[5], observed=Y[:, 5], dims="obs_id")
        pm.Normal("contentment", mu=alpha[6], sigma=sigma[6], observed=Y[:, 6], dims="obs_id")
        pm.Normal("disappointment", mu=alpha[7], sigma=sigma[7], observed=Y[:, 7], dims="obs_id")

        trace = pm.sample(
            10,
            tune=10,
            chains=1,
            cores=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    program_dict = {"model": model, "trace": trace}

    # Use the goal's own eval mechanism (provides gt dict)
    _, gt = goal.get_goal_eval_question(include_prior=True)
    eval_input = goal.eval_points[0][:-1]

    pred = get_ppl_prediction(goal, program_dict, eval_input, prior_mode=False)

    # Must be parseable by the goal's strict regex
    mean_err, std_err = goal.evaluate_predictions([pred], [gt])
    assert np.isfinite(mean_err)
    assert np.isfinite(std_err)


def test_pymc_evaluate_supports_multiple_loglikelihood_vars():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6,))
    Y = rng.normal(size=(6,))
    Z = rng.normal(size=(6,))

    with pm.Model() as model:
        x = pm.Data("x", X, dims="obs_id")
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        pm.Normal("y_obs", a * x, 1, observed=Y, dims="obs_id")
        pm.Normal("z_obs", b * x, 1, observed=Z, dims="obs_id")
        trace = pm.sample(
            10,
            tune=10,
            chains=1,
            cores=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    res = pymc_evaluate(trace)
    assert "loo" in res and "waic" in res
    assert np.isfinite(float(res["loo"]))
    assert np.isfinite(float(res["waic"]))


def test_pymc_evaluate_sums_event_dims_for_multivariate_observation():
    rng = np.random.default_rng(0)
    n = 6
    y = rng.normal(size=(n, 2))

    # Intentionally omit dims so PyMC uses auto-generated dim names like
    # y_obs_dim_0/y_obs_dim_1; pymc_evaluate should still compute a joint score
    # per row observation (n), not per element (n*2).
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1, shape=2)
        sigma = pm.HalfNormal("sigma", 1, shape=2)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, shape=y.shape)
        trace = pm.sample(
            10,
            tune=10,
            chains=1,
            cores=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    # Ensure we actually created an event dimension (so this test is meaningful).
    ll = trace.log_likelihood["y_obs"]
    non_sample_dims = [d for d in ll.dims if d not in {"chain", "draw"}]
    assert len(non_sample_dims) == 2

    res = pymc_evaluate(trace)
    assert np.isfinite(float(res["loo"]))
    assert np.isfinite(float(res["waic"]))


def test_pymc_evaluate_handles_mismatched_obs_dim_names_across_vars():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6,))
    Y = rng.normal(size=(6,))
    Z = rng.normal(size=(6,))

    # y_obs uses dims="obs_id" while z_obs omits dims; this produces different
    # observation dimension names in the log_likelihood group. pymc_evaluate
    # should normalize dims and avoid xarray broadcasting bugs.
    with pm.Model() as model:
        x = pm.Data("x", X, dims="obs_id")
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        pm.Normal("y_obs", a * x, 1, observed=Y, dims="obs_id")
        pm.Normal("z_obs", b * x, 1, observed=Z, shape=Z.shape)
        trace = pm.sample(
            10,
            tune=10,
            chains=1,
            cores=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    res = pymc_evaluate(trace)
    assert np.isfinite(float(res["loo"]))
    assert np.isfinite(float(res["waic"]))


def test_pymc_evaluate_prefers_obs_id_dim_when_not_first():
    rng = np.random.default_rng(0)
    n = 6
    y = rng.normal(size=(2, n))

    # Intentionally set dims so the observation dimension is not first.
    # Our scorer should still identify obs_id as the observation axis and sum
    # over the event dimension.
    with pm.Model(coords={"event": np.arange(2), "obs_id": np.arange(n)}) as model:
        mu = pm.Normal("mu", 0, 1, shape=(2,))
        sigma = pm.HalfNormal("sigma", 1, shape=(2,))
        pm.Normal("y_obs", mu=mu[:, None], sigma=sigma[:, None], observed=y, dims=("event", "obs_id"))
        trace = pm.sample(
            10,
            tune=10,
            chains=1,
            cores=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    res = pymc_evaluate(trace)

    # Compute the expected joint score by summing out the event dim while keeping obs_id.
    da = trace.log_likelihood["y_obs"].sum(dim=["event"])
    trace_expected = trace.copy()
    trace_expected.log_likelihood["__joint__"] = da.rename({"obs_id": "__obs__"}).assign_coords({"__obs__": np.arange(n)})
    loo_expected = az.loo(trace_expected, var_name="__joint__").elpd_loo
    waic_expected = az.waic(trace_expected, var_name="__joint__").elpd_waic

    assert np.isfinite(float(res["loo"]))
    assert np.isfinite(float(res["waic"]))
    assert np.isclose(float(res["loo"]), float(loo_expected))
    assert np.isclose(float(res["waic"]), float(waic_expected))


def test_box_loop_get_ppcs_handles_vector_observation():
    from boxing_gym.agents.box_loop_experiment import BoxLoop_Experiment

    env = LotkaVolterra()
    env.include_prior = True
    for t in ["0.2", "0.4", "0.6"]:
        _, ok = env.run_experiment(t)
        assert ok
    env.get_df()

    logger = logging.getLogger("test_box_loop_ppcs")
    log_dir = str(Path(tempfile.gettempdir()) / "boxing_gym_tests")
    exp = BoxLoop_Experiment(
        dataset=env,
        corrupt=False,
        logger=logger,
        log_dir=log_dir,
        language_synthesize=False,
        prior_mode=False,
    )

    code = """
import numpy as np
import pymc as pm

def gen_model(observed_data):
    t = np.asarray(observed_data["t"])
    prey = np.asarray(observed_data["prey"])
    predator = np.asarray(observed_data["predator"])
    y = np.stack([prey, predator], axis=1)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(314)
    with pm.Model() as model:
        pm.Data("t", t, dims="obs_id")
        mu = pm.Normal("mu", mu=0.0, sigma=50.0, shape=2)
        sigma = pm.HalfNormal("sigma", sigma=50.0, shape=2)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, shape=y.shape)

        trace = pm.sample(
            20,
            tune=20,
            chains=1,
            cores=1,
            random_seed=rng1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        posterior_predictive = pm.sample_posterior_predictive(
            trace, random_seed=rng2, return_inferencedata=False
        )
        return model, posterior_predictive, trace
"""

    ppcs = exp.get_ppcs(code, env.df, stats_fn_list=["mean", "std"], logger=logger)
    assert ppcs is not None

    df_stats, ppc_stats_str, raw_stats_str, model, posterior_predictive, trace = ppcs
    assert df_stats is not None and not df_stats.empty
    assert isinstance(posterior_predictive, dict)
    assert "y_obs" in posterior_predictive


def test_prior_mode_parameter_goal_uses_program_prior_when_possible():
    # DiscountGoal: eval_input None in prior-mode; we should still be able to
    # extract a finite k from prior samples if the program defines it.
    env = TemporalDiscount()
    env.include_prior = True
    goal = DiscountGoal(env)

    code = """
import numpy as np
import pymc as pm

def gen_model(observed_data):
    # Features are available but not needed for k itself.
    rng2 = np.random.default_rng(314)
    with pm.Model() as model:
        k = pm.LogNormal('k', mu=np.log(0.2), sigma=0.1)
        prior_predictive = pm.sample_prior_predictive(samples=200, random_seed=rng2, return_inferencedata=False)
        return model, prior_predictive, None
"""
    program_dict = {"str_prob_prog": code}
    pred = get_ppl_prediction(goal, program_dict, eval_input=None, prior_mode=True)
    k_hat = float(pred)
    assert np.isfinite(k_hat)
    assert k_hat > 0


def test_posterior_mode_parameter_goal_extracts_k():
    env = TemporalDiscount()
    env.include_prior = True
    goal = DiscountGoal(env)

    obs = np.asarray([0.2])
    with pm.Model() as model:
        k = pm.LogNormal("k", mu=np.log(0.2), sigma=0.1)
        pm.Normal("y_obs", mu=k, sigma=0.1, observed=obs)
        trace = pm.sample(
            20,
            tune=20,
            chains=1,
            cores=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    program_dict = {"model": model, "trace": trace}
    pred = get_ppl_prediction(goal, program_dict, eval_input=None, prior_mode=False)
    k_hat = float(pred)
    assert np.isfinite(k_hat)
    assert k_hat > 0


def test_prior_mode_infection_goal_uses_program_prior_when_possible():
    env = DeathProcess()
    env.include_prior = True
    goal = InfectionRate(env)

    code = """
import numpy as np
import pymc as pm

def gen_model(observed_data):
    rng2 = np.random.default_rng(314)
    with pm.Model() as model:
        theta = pm.LogNormal('theta', mu=np.log(0.3), sigma=0.1)
        prior_predictive = pm.sample_prior_predictive(samples=200, random_seed=rng2, return_inferencedata=False)
        return model, prior_predictive, None
"""
    program_dict = {"str_prob_prog": code}
    pred = get_ppl_prediction(goal, program_dict, eval_input=None, prior_mode=True)
    theta_hat = float(pred)
    assert np.isfinite(theta_hat)
    assert theta_hat > 0


def test_prior_mode_does_not_use_arbitrary_single_key_as_observation():
    # For generic regression-style goals (like dugongs), prior-mode should not
    # treat a lone parameter draw (e.g., sigma) as the observation.
    env = Dugongs()
    env.include_prior = True
    goal = DugongsDirectGoal(env)

    # Generate one eval input to ensure construct_features has valid input.
    goal.get_goal_eval_question(include_prior=True)
    eval_input = goal.eval_points[0][:-1]

    code = """
import numpy as np

def gen_model(observed_data):
    # Return only a parameter-like key (no y_obs). This should NOT be used as an outcome.
    prior_predictive = {'sigma': np.asarray([1.23])}
    return None, prior_predictive, None
"""
    program_dict = {"str_prob_prog": code}
    pred = get_ppl_prediction(goal, program_dict, eval_input=eval_input, prior_mode=True)
    assert pred == "0"


def test_prior_mode_emotion_prediction_is_parseable():
    env = EmotionFromOutcome()
    env.include_prior = True
    goal = DirectEmotionPrediction(env)

    # Use the goal's own eval mechanism (provides gt dict)
    _, gt = goal.get_goal_eval_question(include_prior=True)
    eval_input = goal.eval_points[0][:-1]

    code = """
import numpy as np

def gen_model(observed_data):
    rng = np.random.default_rng(0)
    # samples x 8 emotions, in the right 1-9-ish range
    y = rng.uniform(1.0, 9.0, size=(200, 8))
    return None, {'y_obs': y}, None
"""
    program_dict = {"str_prob_prog": code}
    pred = get_ppl_prediction(goal, program_dict, eval_input=eval_input, prior_mode=True)
    mean_err, std_err = goal.evaluate_predictions([pred], [gt])
    assert np.isfinite(mean_err)
    assert np.isfinite(std_err)


def test_prior_mode_lotka_prediction_is_parseable():
    env = LotkaVolterra()
    env.include_prior = True
    goal = LVDirectGoal(env)

    _, gt = goal.get_goal_eval_question(include_prior=True)
    eval_input = goal.eval_points[0][:-1]

    code = """
import numpy as np

def gen_model(observed_data):
    rng = np.random.default_rng(0)
    y = rng.normal(loc=10.0, scale=2.0, size=(200, 2))
    return None, {'y_obs': y}, None
"""
    program_dict = {"str_prob_prog": code}
    pred = get_ppl_prediction(goal, program_dict, eval_input=eval_input, prior_mode=True)
    mean_err, std_err = goal.evaluate_predictions([pred], [gt])
    assert np.isfinite(mean_err)
    assert np.isfinite(std_err)
