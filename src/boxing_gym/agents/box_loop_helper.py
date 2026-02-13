import re

import arviz as az
import numpy as np
import pandas as pd


def construct_dataframe(env):
    # Get the data from the environment
    data = env.get_data()

    if env.env_name in ["location_finding"]:
        data = [(x, y, value) for ((x, y), value) in data]

    # Construct a DataFrame from the data
    column_names = env.get_ordered_column_names()
    if not data:
        # No successful observations yet; return an empty frame with the
        # correct schema so downstream code can fall back gracefully.
        df = pd.DataFrame(columns=column_names)
        df.index = []
        return df

    if len(column_names) != len(data[0]):
        raise ValueError(
            f"Data row length {len(data[0])} does not match columns {len(column_names)} "
            f"for env '{env.env_name}'. Columns: {column_names}"
        )
    df = pd.DataFrame(data, columns=column_names)
    df.index = [f"True Observation {i}" for i in range(len(df))]
    return df


def construct_features(env, data):
    if env.env_name in ["location_finding"]:
        data = [(arr[0][0], arr[0][1]) for arr in data]

    if env.env_name in ["emotion"]:
        # data entries are tuples like: (prizes[3], probs[3], win)
        # where win may be either an index (0/1/2) or the realized prize value.
        flattened = []
        for prizes, probs, win in data:
            prizes_list = list(prizes)
            probs_list = list(probs)
            win_val = win
            try:
                # If win looks like an index, convert to the realized prize value.
                if isinstance(win, (int, np.integer)) and 0 <= int(win) < len(prizes_list):
                    win_val = prizes_list[int(win)]
            except Exception:
                pass
            flattened.append(
                [
                    float(prizes_list[0]),
                    float(prizes_list[1]),
                    float(prizes_list[2]),
                    float(probs_list[0]),
                    float(probs_list[1]),
                    float(probs_list[2]),
                    float(win_val),
                ]
            )
        data = flattened

    if env.env_name in ["moral"]:
        group1 = data[0][0]
        group2 = data[0][1]
        data_tuple = []
        intervention = data[0][2]
        row = []
        for attribute in [
            "count",
            "gender",
            "age",
            "social_status",
            "fitness",
            "species",
        ]:
            attribute_diff = env.calculate_attr_diff(group1, group2, attribute)
            row.append(attribute_diff)

        if intervention == "swerve":
            intervention_encoded = 1
        else:
            intervention_encoded = 0

        data_tuple.append(row + [intervention_encoded])
        data = [row + [intervention_encoded]]

    # Construct a DataFrame from the data
    column_names = env.get_ordered_features()
    assert len(column_names) == len(data[0])
    df = pd.DataFrame(data, columns=column_names)
    df.index = [f"True Observation {i}" for i in range(len(df))]
    return df


def pymc_evaluate(trace):
    """
    trace: arviz.data.inference_data.InferenceData
    """

    # For multi-output environments, compute joint PSIS-LOO/WAIC by:
    # 1) Sum out event/output dims per likelihood RV → (chain, draw, obs)
    # 2) Sum per-RV pointwise log-likelihoods into single joint array
    # 3) Run az.loo/az.waic on synthetic joint variable
    # Handles mismatched observation dimension names (y_obs_dim_0 vs obs_id).
    try:
        ll_vars = (
            list(trace.log_likelihood.data_vars.keys()) if hasattr(trace, "log_likelihood") else []
        )
    except Exception:
        ll_vars = []

    if ll_vars:
        sample_dims = {"chain", "draw"}
        joint = None
        n_obs = None

        for v in ll_vars:
            da = trace.log_likelihood[v]

            # Identify non-sample dims. We treat the *first* such dim as the
            # observation axis and sum out the remaining event dims.
            non_sample_dims = [d for d in da.dims if d not in sample_dims]
            if not non_sample_dims:
                # Scalar log_likelihood (rare); treat as a single-observation.
                non_sample_dims = []
                da = da.expand_dims({"__obs__": 1})
            else:
                # Prefer the conventional observation dimension name if present.
                # This guards against cases where event dims appear before obs dims.
                obs_dim = "obs_id" if "obs_id" in non_sample_dims else non_sample_dims[0]
                event_dims = [d for d in non_sample_dims if d != obs_dim]
                if event_dims:
                    da = da.sum(dim=event_dims)

                # Normalize obs dimension name to avoid xarray broadcasting
                # when multiple likelihood RVs use different dim names.
                da = da.assign_coords({obs_dim: np.arange(da.sizes[obs_dim])})
                da = da.rename({obs_dim: "__obs__"})

            # Validate consistent observation count across likelihood RVs.
            if n_obs is None:
                n_obs = int(da.sizes.get("__obs__", 1))
            else:
                if int(da.sizes.get("__obs__", 1)) != n_obs:
                    raise ValueError(
                        f"Cannot compute joint LOO/WAIC: log_likelihood vars have different "
                        f"observation counts (expected {n_obs}, got {int(da.sizes.get('__obs__', 1))} for '{v}')."
                    )

            joint = da if joint is None else (joint + da)

        trace_joint = trace.copy()
        trace_joint.log_likelihood["__joint__"] = joint

        loo = az.loo(trace_joint, var_name="__joint__")
        waic = az.waic(trace_joint, var_name="__joint__")
        return {"loo": loo.elpd_loo, "waic": waic.elpd_waic}

    # Fallback: single log_likelihood or legacy traces.
    loo = az.loo(trace)
    waic = az.waic(trace)
    return {"loo": loo.elpd_loo, "waic": waic.elpd_waic}


def extract_python_code(code_string):
    """
    Extract python code from an LLM response.

    This extractor is tolerant:
    - Prefer the last ```python```/```py``` block.
    - Fall back to any fenced block that looks like a PyMC gen_model.
    - Finally, fall back to locating a raw `def gen_model(...)` snippet.
    """

    if not code_string or not isinstance(code_string, str):
        raise Exception("No code found :(")

    python_patterns = [
        r"```python\s*\n(.*?)```",
        r"```py\s*\n(.*?)```",
    ]
    for pat in python_patterns:
        matches = re.findall(pat, code_string, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    any_blocks = re.findall(r"```(?:[a-zA-Z0-9_\-]+)?\s*\n(.*?)```", code_string, re.DOTALL)
    candidates = []
    for block in any_blocks:
        b = block or ""
        if "def gen_model" in b or "import pymc" in b or "pm.Model" in b:
            candidates.append(b)
    if candidates:
        return candidates[-1].strip()

    m = re.search(
        r"(import\s+pymc.*?def\s+gen_model\s*\(.*)",
        code_string,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    m = re.search(r"(def\s+gen_model\s*\(.*)", code_string, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # as a last resort, return the final fenced block even if it doesn't
    # explicitly look like python.
    if any_blocks:
        return any_blocks[-1].strip()

    raise Exception("No code found :(")


def extract_python_from_llm(llm_response):
    llm_message = llm_response["choices"][0]["message"]["content"]
    code = extract_python_code(llm_message)
    return code


def extract_text_within_markers(text, marker):
    """Extracts and returns all text enclosed within specified markers in a given text."""
    pattern = rf"{marker}\n([\s\S]*?)\n```"
    matches = re.findall(pattern, text)
    return matches
