import logging
import traceback
import tqdm
from typing import Optional, TYPE_CHECKING, Tuple
try:
    from omegaconf import ListConfig
except Exception:
    ListConfig = ()
from boxing_gym.experiment.utils import create_fallback_result
from boxing_gym.experiment.ppl import augment_scientist_with_ppl
from boxing_gym.experiment.evaluation import (
    evaluate,
    evaluate_naive_explanation,
    ppl_evaluate
)

if TYPE_CHECKING:
    from boxing_gym.experiment.step_logger import StepLogger

logger = logging.getLogger(__name__)

MAX_TRIES = 3


def _normalize_budgets(num_experiments):
    """Normalize budget schedule into a sorted list of unique ints."""
    if num_experiments is None:
        return []
    if isinstance(num_experiments, (list, tuple, set, ListConfig)):
        budgets = []
        for b in num_experiments:
            try:
                budgets.append(int(b))
            except Exception:
                continue
        return sorted(set(budgets))
    try:
        return [int(num_experiments)]
    except Exception:
        return []


def _get_step_latency_ms(scientist, prev_latency_count: int) -> Optional[float]:
    """Get total LLM latency for this step by summing new latencies since prev_latency_count."""
    try:
        if hasattr(scientist, "_usage_stats"):
            latencies = scientist._usage_stats.get("latencies_ms", [])
            if len(latencies) > prev_latency_count:
                new_latencies = latencies[prev_latency_count:]
                return sum(new_latencies)
    except Exception:
        pass
    return None


def _get_latency_count(scientist) -> int:
    """Get current count of latency measurements for tracking per-step latency."""
    try:
        if hasattr(scientist, "_usage_stats"):
            return len(scientist._usage_stats.get("latencies_ms", []))
    except Exception:
        pass
    return 0


def _compute_z_stats(
    eval_score,
    norm_mu: Optional[float],
    norm_sigma: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    if eval_score is None or norm_mu is None or norm_sigma is None or norm_sigma <= 0:
        return None, None

    mean_val = None
    std_val = None
    if isinstance(eval_score, dict):
        mean_val = eval_score.get("mse", eval_score.get("accuracy", eval_score.get("score")))
        std_val = eval_score.get("std_mse", eval_score.get("std", eval_score.get("std_accuracy")))
    elif isinstance(eval_score, (list, tuple)) and len(eval_score) >= 1:
        mean_val = eval_score[0]
        if len(eval_score) >= 2:
            std_val = eval_score[1]

    if mean_val is None:
        return None, None

    z_mean = None
    z_std = None
    try:
        z_mean = (float(mean_val) - norm_mu) / norm_sigma
        if std_val is not None:
            z_std = float(std_val) / norm_sigma
        elif z_mean is not None:
            z_std = 0.0
    except Exception:
        return None, None
    return z_mean, z_std


def _log_budget_evaluation(step_logger, budget, result, norm_mu, norm_sigma, is_prior_only: bool):
    if step_logger is None:
        return
    try:
        eval_score = result[0] if result and len(result) > 0 else None
        z_mean, z_std = _compute_z_stats(eval_score, norm_mu, norm_sigma)

        box_loop_stats = None
        try:
            if isinstance(result, (list, tuple)) and len(result) > 4:
                box_loop_stats = result[4]
        except Exception:
            pass

        step_logger.log_evaluation(
            budget=budget,
            eval_score=eval_score,
            z_mean=z_mean,
            z_std=z_std,
            is_prior_only=is_prior_only,
            box_loop_stats=box_loop_stats,
        )
    except Exception as e:
        logger.debug(f"Step logger evaluation log failed: {e}")


def _run_evaluation(
    final_results,
    goal,
    scientist,
    naive_agent,
    num_evals,
    include_prior,
    com_limit,
    use_ppl,
    step_logger,
    norm_mu,
    norm_sigma,
    llm_model,
    proposed_programs_all,
    critic_info_all,
    prior_mode,
    critic_mode,
    log_prefix,
):
    explanation = None
    if use_ppl:
        if naive_agent is not None:
            try:
                result, explanation = evaluate_naive_explanation(
                    final_results,
                    goal,
                    scientist,
                    naive_agent,
                    num_evals,
                    include_prior,
                    com_limit,
                    use_ppl=use_ppl,
                    step_logger=step_logger,
                    norm_mu=norm_mu,
                    norm_sigma=norm_sigma,
                    llm_model=llm_model,
                )
            except Exception as e:
                logger.warning(f"{log_prefix} failed: {str(e)}")
                logger.debug(traceback.format_exc())
                result = create_fallback_result(str(e))
        else:
            try:
                result = ppl_evaluate(
                    final_results,
                    goal,
                    scientist,
                    num_evals,
                    include_prior,
                    proposed_programs_all,
                    critic_info_all,
                    prior_mode=prior_mode,
                    critic_mode=critic_mode,
                    llm_model=llm_model,
                )
                if proposed_programs_all and proposed_programs_all[-1]:
                    augment_scientist_with_ppl(scientist, proposed_programs_all, critic_info_all, critic_mode=critic_mode)
                else:
                    logger.warning("Skipping PPL augmentation: Box's loop returned no programs.")
            except Exception as e:
                logger.warning(f"{log_prefix} failed: {str(e)}")
                logger.debug(traceback.format_exc())
                result = create_fallback_result(str(e))
    else:
        if naive_agent is not None:
            try:
                result, explanation = evaluate_naive_explanation(
                    final_results,
                    goal,
                    scientist,
                    naive_agent,
                    num_evals,
                    include_prior,
                    com_limit,
                    step_logger=step_logger,
                    norm_mu=norm_mu,
                    norm_sigma=norm_sigma,
                )
            except Exception as e:
                logger.warning(f"{log_prefix} failed: {str(e)}")
                logger.debug(traceback.format_exc())
                result = create_fallback_result(str(e))
        else:
            try:
                result = evaluate(final_results, goal, scientist, num_evals, include_prior)
            except Exception as e:
                logger.warning(f"{log_prefix} failed: {str(e)}")
                logger.debug(traceback.format_exc())
                result = create_fallback_result(str(e))

    return result, explanation


# No @weave_op here; it receives Goal/Agent objects that Weave can't serialize.
# LiteLLM calls are still traced via Weave auto-patching.
def iterative_experiment(
        goal,
        scientist,
        num_experiments,
        num_evals,
        include_prior,
        naive_agent=None,
        com_limit=None,
        check_eig=False,
        use_ppl=False,
        step_logger: Optional["StepLogger"] = None,
        norm_mu: Optional[float] = None,
        norm_sigma: Optional[float] = None,
):
    results = []
    queries = []
    observations = []
    successes = []
    explanations = []
    eigs = []
    proposed_programs_all = [[]]
    critic_info_all = []

    # extract scientist's model name for Box's Loop
    llm_model = getattr(scientist, 'model_name', None)

    budgets = _normalize_budgets(num_experiments)
    budget_set = set(budgets)

    if 0 in budget_set:
        final_results = "You cannot make observations now. Make assumptions and provide your best guess to the following query."

        result, explanation = _run_evaluation(
            final_results=final_results,
            goal=goal,
            scientist=scientist,
            naive_agent=naive_agent,
            num_evals=num_evals,
            include_prior=include_prior,
            com_limit=com_limit,
            use_ppl=use_ppl,
            step_logger=step_logger,
            norm_mu=norm_mu,
            norm_sigma=norm_sigma,
            llm_model=llm_model,
            proposed_programs_all=proposed_programs_all,
            critic_info_all=critic_info_all,
            prior_mode=True,
            critic_mode=False,
            log_prefix="Evaluation (prior)",
        )
        if explanation is not None:
            explanations.append(explanation)

        results.append(result)
        
        _log_budget_evaluation(step_logger, budget=0, result=result, norm_mu=norm_mu, norm_sigma=norm_sigma, is_prior_only=True)
        
    observation = None
    # use last budget entry as the loop cap
    max_iters = budgets[-1] if budgets else 0
    
    for i in tqdm.tqdm(range(max_iters)):
        success = False
        # track latency: note count before step
        prev_latency_count = _get_latency_count(scientist)

        try:
            observe = scientist.generate_actions(observation)
        except ValueError as exc:
            print(f"LLM action generation failed: {exc}")
            observe = "0"
        queries.append(observe)
        observation, success = goal.env.run_experiment(observe)
        observations.append(observation)
        successes.append(success)
        tries = 1
        while not success and tries < MAX_TRIES:
            observe, _ = scientist.prompt_llm_and_parse(observation, True)
            queries.append(observe)
            observation, success = goal.env.run_experiment(observe)
            observations.append(observation)
            successes.append(success)
            if not success:
                tries += 1

        retry_count = tries - 1

        # get step latency (sum of all LLM calls this step)
        step_latency_ms = _get_step_latency_ms(scientist, prev_latency_count)

        eig_value = None
        optimal_eig = None
        if success and check_eig:
            query_point = goal.env.validate_input(observe)
            eig_value = goal.expected_information_gain(query_point)
            eigs.append(eig_value)

            # try to get optimal EIG for regret tracking (if environment supports it)
            try:
                if hasattr(goal, "get_optimal_eig"):
                    optimal_eig = goal.get_optimal_eig()
            except Exception:
                pass  # optimal EIG not available for this environment
        
        # log per-step metrics for live progress tracking
        if step_logger is not None:
            try:
                step_logger.log_step(
                    step_idx=i,
                    success=success,
                    retry_count=retry_count,
                    eig=eig_value,
                    optimal_eig=optimal_eig,
                    observation=observation,
                    query=observe,
                    latency_ms=step_latency_ms,
                )
                # log LLM usage per step for live cost tracking
                if hasattr(scientist, "get_usage_stats"):
                    step_logger.log_llm_usage_step(
                        step_idx=i,
                        usage_stats=scientist.get_usage_stats(),
                        agent_prefix="llm",
                    )
            except Exception as e:
                logger.debug(f"Step logger log_step failed: {e}")

        if i + 1 in budget_set:
            final_results = f"The final result is {observation}."
            result, explanation = _run_evaluation(
                final_results=final_results,
                goal=goal,
                scientist=scientist,
                naive_agent=naive_agent,
                num_evals=num_evals,
                include_prior=include_prior,
                com_limit=com_limit,
                use_ppl=use_ppl,
                step_logger=step_logger,
                norm_mu=norm_mu,
                norm_sigma=norm_sigma,
                llm_model=llm_model,
                proposed_programs_all=proposed_programs_all,
                critic_info_all=critic_info_all,
                prior_mode=False,
                critic_mode=True,
                log_prefix=f"Evaluation (iteration {i+1})",
            )
            if explanation is not None:
                explanations.append(explanation)
            results.append(result)
            
            _log_budget_evaluation(step_logger, budget=i + 1, result=result, norm_mu=norm_mu, norm_sigma=norm_sigma, is_prior_only=False)


    # log exit status (experiment completed successfully)
    if step_logger is not None:
        try:
            from boxing_gym.experiment.step_logger import ExitStatus
            step_logger.log_exit_status(
                ExitStatus.COMPLETED,
                reason=f"Completed {max_iters} experiments"
            )
        except Exception as e:
            logger.debug(f"Step logger exit status log failed: {e}")

    return results, queries, observations, successes, explanations, eigs, proposed_programs_all
