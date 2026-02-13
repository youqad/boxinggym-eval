import importlib
import logging
import os
import re
import tempfile
import threading
import traceback
import uuid
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
import pymc as pm

from boxing_gym.agents.box_loop_helper import pymc_evaluate
from boxing_gym.agents.box_loop_prompts import (
    get_stan_system_prompt,
    get_stan_system_prompt_prior,
    get_stan_user_prompt,
    get_stan_user_prompt_prior,
)


def _quiet_ppl_loggers() -> None:
    for name in (
        "pymc",
        "arviz",
    ):
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False


_PPL_LOGGER_NAMES = (
    "pymc",
    "pymc.stats",
    "pymc.stats.convergence",
    "pymc.sampling",
    "pymc.backends",
    "arviz",
)

# lock for thread-safe logger modifications
_PPL_LOGGER_LOCK = threading.Lock()


class _PPLLogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        msg = record.getMessage()
        if msg:
            self.messages.append(msg)


def _dedupe_messages(messages, limit=5):
    seen = set()
    ordered = []
    for msg in messages:
        text = str(msg).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    if limit is None:
        return ordered
    return ordered[-limit:]


def _diagnostics_from_trace(trace, rhat_warn=1.01, ess_warn=100):
    if trace is None:
        return []
    try:
        import arviz as az
    except Exception:
        return []

    diagnostics = []
    try:
        div = trace.sample_stats.get("diverging", None)
        if div is not None:
            n_div = int(np.sum(div))
            if n_div > 0:
                diagnostics.append(f"{n_div} divergences after tuning")
    except Exception:
        pass

    try:
        rhat = az.rhat(trace)
        max_rhat = float(np.nanmax(rhat.to_array().values))
        if np.isfinite(max_rhat) and max_rhat > rhat_warn:
            diagnostics.append(f"max rhat={max_rhat:.3f} (target <= {rhat_warn})")
    except Exception:
        pass

    try:
        ess = az.ess(trace, method="bulk")
        min_ess = float(np.nanmin(ess.to_array().values))
        if np.isfinite(min_ess) and min_ess < ess_warn:
            diagnostics.append(f"min ESS={min_ess:.1f} (target >= {ess_warn})")
    except Exception:
        pass

    return diagnostics


def _trace_stats(trace):
    stats = {
        "n_divergences": None,
        "max_rhat": None,
        "min_ess_bulk": None,
    }
    if trace is None:
        return stats
    try:
        import arviz as az
    except Exception:
        return stats

    try:
        div = trace.sample_stats.get("diverging", None)
        if div is not None:
            stats["n_divergences"] = int(np.sum(div))
    except Exception:
        pass

    try:
        rhat = az.rhat(trace)
        stats["max_rhat"] = float(np.nanmax(rhat.to_array().values))
    except Exception:
        pass

    try:
        ess = az.ess(trace, method="bulk")
        stats["min_ess_bulk"] = float(np.nanmin(ess.to_array().values))
    except Exception:
        pass

    return stats


@contextmanager
def _ppl_logger_settings(level: int = logging.WARNING, propagate: bool = False):
    with _PPL_LOGGER_LOCK:
        loggers = [logging.getLogger(name) for name in _PPL_LOGGER_NAMES]
        previous = [(logger, logger.level, logger.propagate) for logger in loggers]
        for logger in loggers:
            logger.setLevel(level)
            logger.propagate = propagate
        try:
            yield
        finally:
            for logger, prev_level, prev_propagate in previous:
                logger.setLevel(prev_level)
                logger.propagate = prev_propagate


class BoxLoop_Experiment:
    def __init__(
        self,
        dataset,
        corrupt,
        logger,
        log_dir,
        language_synthesize,
        prior_mode=False,
        diagnostics_cfg=None,
    ) -> None:
        self.corrupt = corrupt
        self.dataset = dataset
        self.prior_mode = prior_mode

        if not prior_mode:
            self.observed_data = self.dataset.df

        self.stats_fn_list = ["mean", "std", "median"]
        self.language_synthesize = language_synthesize
        self.checkpoint_dir = "/tmp/oed_llms/stan/"
        self.logger = logger
        self.log_dir = log_dir
        self.recent_program_failures = []
        self.recent_program_diagnostics = []
        self.diagnostics_cfg = diagnostics_cfg or {}
        self.rhat_warn = float(self.diagnostics_cfg.get("rhat_warn", 1.01))
        self.ess_warn = float(self.diagnostics_cfg.get("ess_warn", 100))
        self.max_diag_messages = int(self.diagnostics_cfg.get("max_messages", 5))
        self.include_logger_warnings = bool(
            self.diagnostics_cfg.get("include_logger_warnings", True)
        )

    def get_posterior_predictive(self, str_prob_prog, observed_data):
        """str_prob_prog -> posterior predictive samples"""
        self.logger.debug("entering get_posterior_predictive")

        if observed_data is not None:
            code_for_scan = re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', "", str_prob_prog, flags=re.DOTALL)
            code_for_scan = re.sub(r"#.*", "", code_for_scan)
            referenced_cols = set(
                re.findall(r"observed_data\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", code_for_scan)
            )
            if referenced_cols:
                missing_cols = referenced_cols - set(observed_data.columns)
                if missing_cols:
                    raise KeyError(
                        f"Generated program refers to missing columns {sorted(missing_cols)}. "
                        f"Available columns: {list(observed_data.columns)}"
                    )

        gen_model = self.get_gen_model(str_prob_prog)
        result = [None, None, None]

        def worker():
            try:
                with warnings.catch_warnings(record=True) as warn_records:
                    warnings.simplefilter("always")
                    log_capture = _PPLLogCapture()
                    loggers = [logging.getLogger(name) for name in _PPL_LOGGER_NAMES]
                    if self.include_logger_warnings:
                        for logger in loggers:
                            logger.addHandler(log_capture)
                    try:
                        with StringIO() as _out_buf, StringIO() as _err_buf:
                            with _ppl_logger_settings(level=logging.WARNING, propagate=False):
                                with redirect_stdout(_out_buf), redirect_stderr(_err_buf):
                                    out = gen_model(observed_data)
                    finally:
                        if self.include_logger_warnings:
                            for logger in loggers:
                                logger.removeHandler(log_capture)
                model = None
                posterior_predictive = None
                trace = None
                if isinstance(out, (list, tuple)):
                    if len(out) == 2:
                        model, posterior_predictive = out
                    elif len(out) >= 3:
                        model, posterior_predictive, trace = out[:3]
                    else:
                        raise ValueError("gen_model returned empty tuple")
                else:
                    posterior_predictive = out
                result[:] = [model, posterior_predictive, trace]
                diagnostics = []
                if self.include_logger_warnings:
                    diagnostics.extend(_dedupe_messages(log_capture.messages, limit=None))
                    diagnostics.extend(
                        _dedupe_messages((str(w.message) for w in warn_records), limit=None)
                    )
                diagnostics.extend(
                    _diagnostics_from_trace(
                        trace,
                        rhat_warn=self.rhat_warn,
                        ess_warn=self.ess_warn,
                    )
                )
                diagnostics = _dedupe_messages(diagnostics, limit=self.max_diag_messages)
                if diagnostics:
                    self.recent_program_diagnostics.extend(diagnostics)
                    self.recent_program_diagnostics = self.recent_program_diagnostics[
                        -self.max_diag_messages :
                    ]
            except TimeoutError:
                self.logger.warning("PYMC program timed out")
            except Exception:
                self.logger.info(f"failed program: {str_prob_prog}")
                with StringIO() as buf:
                    traceback.print_exc(file=buf)
                    tb_str = buf.getvalue()
                self.logger.info(f"traceback: {tb_str}")
                try:
                    self.recent_program_failures.append(str(tb_str).splitlines()[-1])
                except Exception:
                    pass

        TIMEOUT_TIME = 60 * 8

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=TIMEOUT_TIME)

        if thread.is_alive():
            self.logger.warning("PYMC program timed out (thread join)")
            return None, None, None

        self.logger.debug("get_posterior_predictive done")
        return result[0], result[1], result[2]

    def get_gen_model(self, gen_code):
        tmp_root = os.path.join(tempfile.gettempdir(), "oed_llms", "stan")
        os.makedirs(tmp_root, exist_ok=True)
        unique_name = f"ppl_gen_model_{uuid.uuid4().hex}"
        module_path = os.path.join(tmp_root, f"{unique_name}.py")
        # Compatibility shim for newer PyMC versions + common helpers.
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
        spec = importlib.util.spec_from_file_location(unique_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        gen_model_fn = getattr(module, "gen_model", None)
        if gen_model_fn is None:
            raise AttributeError("Generated PPL module did not define gen_model")
        return gen_model_fn

    def get_prior_samples(self, str_prob_prog):
        try:
            model, posterior_predictive, trace = self.get_posterior_predictive(
                str_prob_prog=str_prob_prog, observed_data=None
            )
            return "", "", "", model, posterior_predictive, trace
        except Exception as e:
            with StringIO() as buf:
                traceback.print_exc(file=buf)
                tb_str = buf.getvalue()
            self.logger.error(f"get_prior_samples failed: {e}\n{tb_str}")
            return None

    def score_programs(self, program, logger, llm_response):
        try:
            ppc_results = self.get_ppcs(
                str_prob_prog=program,
                observed_data=self.observed_data,
                stats_fn_list=self.stats_fn_list,
                logger=logger,
            )

            if ppc_results is None:
                logger.warning(f"Program returned None; skipping: {program[:100]}...")
                return None

            (
                df_posterior_stats,
                ppc_stats_str,
                raw_stats_str,
                model,
                posterior_predictive,
                trace,
            ) = ppc_results

            summary_stats_str = ppc_stats_str
            if trace is not None and model is not None and not hasattr(trace, "log_likelihood"):
                # Compute log_likelihood so ArviZ loo/waic works.
                try:
                    trace = pm.compute_log_likelihood(trace, model=model, progressbar=False)
                except Exception as exc:
                    logger.warning(f"Could not compute log_likelihood for trace: {exc}")

            diag_messages = []
            stats = _trace_stats(trace)
            with warnings.catch_warnings(record=True) as warn_records:
                warnings.simplefilter("always")
                res = pymc_evaluate(trace)
                try:
                    diag_messages.extend(
                        _diagnostics_from_trace(
                            trace,
                            rhat_warn=self.rhat_warn,
                            ess_warn=self.ess_warn,
                        )
                    )
                except Exception:
                    pass
                stats = _trace_stats(trace)

            warn_messages = _dedupe_messages((str(w.message) for w in warn_records), limit=None)
            diag_messages.extend(warn_messages)
            diag_messages = _dedupe_messages(diag_messages, limit=self.max_diag_messages)
            if diag_messages:
                self.recent_program_diagnostics.extend(diag_messages)
                self.recent_program_diagnostics = self.recent_program_diagnostics[
                    -self.max_diag_messages :
                ]

            logger.info(f"Program {res['loo']}: \n {program}")
            return {
                "loo": res["loo"],
                "waic": res["waic"],
                "summary_stats": summary_stats_str,
                # "posterior_predictive": posterior_predictive,
                "summary_stats_df": df_posterior_stats,
                "str_prob_prog": program,
                "trace": trace,
                "model": model,
                "full_llm_response": llm_response,
                "posterior_predictive": posterior_predictive,
                "diagnostics": diag_messages,
                "n_divergences": stats.get("n_divergences"),
                "max_rhat": stats.get("max_rhat"),
                "min_ess_bulk": stats.get("min_ess_bulk"),
            }

        except Exception as e:
            err_msg = str(e)
            if isinstance(e, KeyError) and "Generated program refers to missing columns" in err_msg:
                cleaned = err_msg.strip("'")
                self.recent_program_failures.append(cleaned)
                if len(self.recent_program_failures) > 3:
                    self.recent_program_failures = self.recent_program_failures[-3:]
            with StringIO() as buf:
                traceback.print_exc(file=buf)
                tb_str = buf.getvalue()
            logger.error(f"score_programs failed: {e}")
            logger.info(f"Full traceback:\n{tb_str}")
            return None

    def get_ppcs(self, str_prob_prog, observed_data, stats_fn_list, logger):
        try:
            model, posterior_predictive, trace = self.get_posterior_predictive(
                str_prob_prog=str_prob_prog, observed_data=observed_data
            )

            if not posterior_predictive or not isinstance(posterior_predictive, dict):
                raise ValueError("posterior_predictive is empty or not a dict")

            def get_observation_index(i):
                return observed_data.index[i].split("True Observation")[-1]

            def round_to_n_significant(x, n=2):
                if not np.isfinite(x) or x == 0:
                    return x
                else:
                    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))

            n_obs = int(observed_data.shape[0]) if observed_data is not None else None

            def ppc_to_sample_matrix(arr, key_name):
                """Convert a posterior predictive array into a 2D sample matrix.

                Returns (matrix, column_names) where:
                  - matrix shape: (n_samples, n_columns)
                  - each column corresponds to one observation (and, if multi-output,
                    one flattened output component).
                """
                if n_obs is None:
                    a = np.asarray(arr)
                    if a.size == 0:
                        return None, None
                    return a.reshape(-1, 1), [f"{key_name} Model Sampled (no_obs_axis)"]

                a = np.asarray(arr)
                if a.size == 0:
                    return None, None

                obs_axes = [i for i, s in enumerate(a.shape) if s == n_obs]
                if not obs_axes:
                    raise ValueError(
                        f"Posterior predictive '{key_name}' has shape {a.shape} but no axis matches "
                        f"n_obs={n_obs}. This program cannot be scored against the dataset."
                    )

                if a.shape[-1] == n_obs:
                    obs_axis = a.ndim - 1
                elif a.ndim >= 2 and a.shape[-2] == n_obs:
                    obs_axis = a.ndim - 2
                else:
                    obs_axis = obs_axes[-1]

                sample_axes = list(range(0, obs_axis))
                event_axes = list(range(obs_axis + 1, a.ndim))
                perm = sample_axes + [obs_axis] + event_axes
                a_perm = np.transpose(a, axes=perm)

                sample_nd = len(sample_axes)
                sample_size = int(np.prod(a_perm.shape[:sample_nd])) if sample_nd > 0 else 1
                event_size = int(np.prod(a_perm.shape[sample_nd + 1 :])) if event_axes else 1

                a_3 = a_perm.reshape(sample_size, n_obs, event_size)
                mat = a_3.reshape(sample_size, n_obs * event_size)

                cols = []
                for i in range(n_obs):
                    obs_idx = get_observation_index(i)
                    if event_size == 1:
                        cols.append(f"{key_name} Model Sampled Observation {obs_idx}")
                    else:
                        for e in range(event_size):
                            cols.append(f"{key_name}[{e}] Model Sampled Observation {obs_idx}")
                return mat, cols

            keys = (
                ["y_obs"] if "y_obs" in posterior_predictive else list(posterior_predictive.keys())
            )
            if not keys:
                raise ValueError("posterior_predictive contains no variables")

            matrices = []
            col_names = []
            for k in keys:
                mat, cols = ppc_to_sample_matrix(posterior_predictive[k], k)
                if mat is None or cols is None:
                    continue
                matrices.append(mat)
                col_names.extend(cols)

            if not matrices:
                raise ValueError(
                    "No usable posterior predictive variables could be converted to PPC stats"
                )

            min_n = min(m.shape[0] for m in matrices)
            matrices = [m[:min_n] for m in matrices]

            posterior_predictive_mat = (
                np.concatenate(matrices, axis=1) if len(matrices) > 1 else matrices[0]
            )

            df_posterior_data = pd.DataFrame(posterior_predictive_mat, columns=col_names)

            df_posterior_stats = df_posterior_data.agg(stats_fn_list).T
            df_true_stats = observed_data

            df_posterior_stats_round = df_posterior_stats.map(
                lambda x: round_to_n_significant(x) if np.issubdtype(type(x), np.number) else x
            )
            df_true_stats_round = df_true_stats.map(
                lambda x: round_to_n_significant(x) if np.issubdtype(type(x), np.number) else x
            )

            ppc_stats_str = df_posterior_stats_round.to_string()
            raw_stats_str = df_true_stats_round.to_string()

            return (
                df_posterior_stats,
                ppc_stats_str,
                raw_stats_str,
                model,
                posterior_predictive,
                trace,
            )

        except Exception as e:
            with StringIO() as buf:
                traceback.print_exc(file=buf)
                tb_str = buf.getvalue()
            logger.warning(f"get_ppcs failed: {e}")
            logger.info(f"Full traceback:\n{tb_str}")
            return None

    def filter_fn(self, results):
        # TODO: filter out divergent runs
        return results

    def sort_fn(self, results):
        if self.prior_mode:
            return results

        results = sorted(results, key=lambda x: -x["loo"])
        return results

    def get_user_message(self, mode, str_hypotheses, synthesis, vision_only=False):
        assert mode in ["proposal", "critic"]

        if self.prior_mode:
            return get_stan_user_prompt_prior(
                mode=mode,
                str_hypotheses=str_hypotheses,
                vision_only=vision_only,
                synthesis=synthesis,
            )
        else:
            return get_stan_user_prompt(
                mode=mode,
                str_hypotheses=str_hypotheses,
                vision_only=vision_only,
                synthesis=synthesis,
            )

    def get_system_message(
        self,
        mode,
        prior_mode=False,
        vision_only=False,
        critic_strategy=None,
        prev_synthesis=None,
        prev_str_hypotheses=None,
    ):
        assert mode in ["proposal", "critic"]

        if self.prior_mode:
            dataset_description = self.dataset.get_description()
            df_str = ""
            column_description = self.dataset.describe_data_columns()
            try:
                available_cols = list(self.dataset.df.columns)
                column_description += (
                    f"\n\nAVAILABLE COLUMN NAMES (use exactly these, no others): {available_cols}\n"
                )
            except Exception:
                pass
            if mode == "proposal" and self.recent_program_failures:
                failures_block = (
                    "\nRECENT EXECUTION FAILURES TO AVOID:\n"
                    + "\n".join(f"* {m}" for m in self.recent_program_failures)
                    + "\n"
                )
                column_description += failures_block
                self.recent_program_failures = []
            if mode == "proposal" and self.recent_program_diagnostics:
                diag_block = (
                    "\nRECENT SAMPLING DIAGNOSTICS TO ADDRESS:\n"
                    + "\n".join(f"* {m}" for m in self.recent_program_diagnostics)
                    + "\n"
                )
                column_description += diag_block
                self.recent_program_diagnostics = []
            system_str = get_stan_system_prompt_prior(
                mode=mode,
                dataset_description=dataset_description,
                df_str=df_str,
                expert_context="",
                vision_only=vision_only,
                column_description=column_description,
                critic_strategy=critic_strategy,
                prev_str_hypotheses=prev_str_hypotheses,
                prev_synthesis=prev_synthesis,
            )

        else:
            dataset_description = self.dataset.get_description()
            df_str = self.dataset.df.to_string(index=False)
            column_description = self.dataset.describe_data_columns()
            try:
                available_cols = list(self.dataset.df.columns)
                column_description += (
                    f"\n\nAVAILABLE COLUMN NAMES (use exactly these, no others): {available_cols}\n"
                )
            except Exception:
                pass
            if mode == "proposal" and self.recent_program_failures:
                failures_block = (
                    "\nRECENT EXECUTION FAILURES TO AVOID:\n"
                    + "\n".join(f"* {m}" for m in self.recent_program_failures)
                    + "\n"
                )
                column_description += failures_block
                self.recent_program_failures = []
            if mode == "proposal" and self.recent_program_diagnostics:
                diag_block = (
                    "\nRECENT SAMPLING DIAGNOSTICS TO ADDRESS:\n"
                    + "\n".join(f"* {m}" for m in self.recent_program_diagnostics)
                    + "\n"
                )
                column_description += diag_block
                self.recent_program_diagnostics = []
            system_str = get_stan_system_prompt(
                mode=mode,
                dataset_description=dataset_description,
                df_str=df_str,
                expert_context="",
                vision_only=vision_only,
                column_description=column_description,
                critic_strategy=critic_strategy,
                prev_str_hypotheses=prev_str_hypotheses,
                prev_synthesis=prev_synthesis,
            )
        return system_str

    def evaluate(self, programs_all, logger, cfg, critic_info, proposal_agent, critic_agent):
        if self.prior_mode:
            return programs_all

        programs_all = sorted(programs_all, key=lambda x: -x["loo"])

        for i, r in enumerate(programs_all[:3]):
            logger.info(f"top {i} program {r['loo']}: \n {r['str_prob_prog']} \n")

        trace_dict = {}
        if len(programs_all) > 0:
            trace_dict["LLM"] = programs_all[0]["trace"]

        return programs_all
