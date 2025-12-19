import logging
import tqdm
import arviz as az
from typing import Optional, TYPE_CHECKING

from boxing_gym.experiment.utils import (
    _baseline_prediction_for_goal
)
from boxing_gym.experiment.ppl import (
    get_ppl_prediction
)
# try importing run_box_loop, handle if missing
try:
    from boxing_gym.agents.model_search import run_box_loop
except ImportError:
    run_box_loop = None

if TYPE_CHECKING:
    from boxing_gym.experiment.step_logger import StepLogger

logger = logging.getLogger(__name__)


# NOTE: these functions intentionally do not use @weave_op decorators because
# they receive complex objects (Goal, Agent, PyMC traces) that Weave cannot
# serialize, causing "Invalid Client ID digest" errors. LLM calls are still
# traced via Weave's auto-patching of LiteLLM.

def evaluate(final_results, goal, scientist, num_evals, include_prior):
    predictions, gts, questions = [], [], []
    print(f"running {num_evals} evals")
    if hasattr(goal, "eval_pointer"):
        goal.eval_pointer = 0 # reset pointer, some goals have a static eval set
    
    for _ in tqdm.tqdm(range(num_evals)):
        question, gt = goal.get_goal_eval_question(include_prior)
        question = final_results + '\n' + question
        try:
            prediction = scientist.generate_predictions(question)
        except ValueError as exc:
            print(f"LLM prediction failed: {exc}")
            prediction = "0"
        gts.append(gt)
        questions.append(question)
        predictions.append(prediction)
        print(f"prediction: {prediction}, gt: {gt}")
    return goal.evaluate_predictions(predictions, gts), questions, gts, predictions


def ppl_evaluate(final_results, goal, scientist, num_evals, include_prior, proposed_programs_all, critic_info_all, prior_mode=False, critic_mode=False, llm_model=None):

    if not prior_mode:
        goal.env.get_df()

    if len(critic_info_all) > 0:
        if len(critic_info_all[-1]) > 0:
            prev_str_hypotheses = critic_info_all[-1][0]['str_hypotheses']
            prev_synthesis = critic_info_all[-1][0]['synthesis']
        else:
            prev_str_hypotheses = None
            prev_synthesis = None

    else:
        prev_str_hypotheses = None
        prev_synthesis = None

    if run_box_loop is None:
         raise ImportError("run_box_loop not available. Check boxing_gym.agents.model_search")

    # use scientist's LLM for Box's Loop
    model_name = llm_model or getattr(scientist, 'model_name', None)

    proposed_programs, critic_info, box_loop_stats = run_box_loop(
        env=goal.env,
        prior_mode=prior_mode,
        critic_mode=critic_mode,
        prev_synthesis=prev_synthesis,
        prev_str_hypotheses=prev_str_hypotheses,
        warm_start_examples=proposed_programs_all[-1],
        llm_model=model_name,
    )

    proposed_programs_all.append(proposed_programs)
    critic_info_all.append(critic_info)

    if not proposed_programs_all[-1]:
        logger.warning("Box's loop returned no programs; using goal-specific baseline predictions.")
        predictions, gts, questions = [], [], []
        if hasattr(goal, "eval_pointer"):
            try:
                goal.eval_pointer = 0
            except Exception:
                pass
        for _ in tqdm.tqdm(range(num_evals)):
            q_text, gt = goal.get_goal_eval_question(include_prior)
            gts.append(gt)
            questions.append(q_text)
            predictions.append(_baseline_prediction_for_goal(goal))
        return goal.evaluate_predictions(predictions, gts), questions, gts, predictions, box_loop_stats

    program_dict = proposed_programs_all[-1][0]

    predictions, gts, questions = [], [], []
    print(f"running {num_evals} evals")
    if hasattr(goal, "eval_pointer"):
        try:
            goal.eval_pointer = 0  # reset pointer, some goals have a static eval set
        except Exception:
            pass
    for _ in tqdm.tqdm(range(num_evals)):
        q_text, gt = goal.get_goal_eval_question(include_prior)

        # if this goal uses eval_points, the just-generated point (or selected point)
        # will live at eval_points[eval_pointer-1] and contain (features..., gt).
        eval_input = None
        if hasattr(goal, "eval_points") and hasattr(goal, "eval_pointer"):
            try:
                ep = goal.eval_points[goal.eval_pointer - 1]
                if isinstance(ep, (list, tuple)) and len(ep) >= 2:
                    eval_input = ep[:-1]
            except Exception:
                eval_input = None

        try:
            prediction = get_ppl_prediction(goal, program_dict, eval_input, prior_mode)
        except Exception as e:
            logger.warning(f"PPL prediction failed, using baseline: {e}")
            prediction = _baseline_prediction_for_goal(goal)

        gts.append(gt)
        questions.append(q_text)
        predictions.append(prediction)
        print(f"prediction: {prediction}, gt: {gt}")
    

    # return box_loop_stats as an optional 5th element for downstream logging (W&B)
    return goal.evaluate_predictions(predictions, gts), questions, gts, predictions, box_loop_stats


def evaluate_naive_explanation(
        final_results,
        goal,
        scientist,
        naive_agent,
        num_evals,
        include_prior,
        com_limit,
        use_ppl=False,
        step_logger: Optional["StepLogger"] = None,
        norm_mu: Optional[float] = None,
        norm_sigma: Optional[float] = None,
        llm_model=None,
):
    if use_ppl:
        goal.env.get_df()
        if run_box_loop is None:
             raise ImportError("run_box_loop not available. Check boxing_gym.agents.model_search")

        # use scientist's LLM for Box's Loop
        model_name = llm_model or getattr(scientist, 'model_name', None)

        proposed_programs, _, _ = run_box_loop(
            env=goal.env,
            warm_start_examples=None,
            llm_model=model_name,
        )
        if not proposed_programs:
            logger.warning(
                "Box's loop returned no valid programs for naive explanation; "
                "falling back to non-PPL comm prompt."
            )
            str_prob_prog = None
            request_prompt = goal.get_comm_prompt(
                com_limit=com_limit,
                include_prior=include_prior,
            )
        else:
            str_prob_prog = proposed_programs[0].get('str_prob_prog')
            trace = proposed_programs[0].get('trace')
            params_summary_str = ""
            try:
                if trace is not None:
                    params_summary_str = az.summary(trace)['mean'].to_string()
            except Exception as e:
                logger.warning(f"Failed to summarize PPL trace for comm prompt: {e}")
                params_summary_str = ""
            request_prompt = goal.get_comm_prompt(
                com_limit=com_limit, 
                include_prior=include_prior, 
                use_ppl=use_ppl, 
                str_prob_prog=str_prob_prog,
                params_summary_str=params_summary_str,
            )
    else:
        str_prob_prog = None
        request_prompt = goal.get_comm_prompt(com_limit=com_limit, include_prior=include_prior)

    print(f"request prompt: {request_prompt}")
    explanation = scientist.prompt_llm(request_prompt)
    print(f"explanation: {explanation}")
    naive_system_message = goal.get_naive_system_message(include_prior)
    naive_system_message += explanation
    print(f"naive_system_message: {naive_system_message}")
    naive_agent.set_system_message(naive_system_message)

    # evaluate naive agent with scientist's explanation
    naive_result = evaluate(final_results, goal, naive_agent, num_evals, include_prior)

    # log communication metrics if step_logger is available
    if step_logger is not None:
        try:
            naive_eval_score = naive_result[0] if naive_result and len(naive_result) > 0 else None

            # z-scores if normalization parameters available
            naive_z_mean = None
            if naive_eval_score is not None and norm_mu is not None and norm_sigma is not None and norm_sigma > 0:
                naive_mean_val = None
                if isinstance(naive_eval_score, dict):
                    naive_mean_val = naive_eval_score.get("mse", naive_eval_score.get("accuracy", naive_eval_score.get("score")))
                elif isinstance(naive_eval_score, (list, tuple)) and len(naive_eval_score) >= 1:
                    naive_mean_val = naive_eval_score[0]
                if naive_mean_val is not None:
                    naive_z_mean = (float(naive_mean_val) - norm_mu) / norm_sigma

            # for scientist z_mean, we'd need to evaluate scientist separately
            # for now, we log what we have
            if naive_z_mean is not None:
                step_logger.log_communication(
                    scientist_z_mean=0.0,  # TODO:placeholder, need separate scientist eval
                    naive_z_mean=naive_z_mean,
                    explanation=explanation,
                )
        except Exception as e:
            logger.debug(f"Step logger communication log failed: {e}")

    return naive_result, explanation
