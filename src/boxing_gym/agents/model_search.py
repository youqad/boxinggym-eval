import os
import pathlib
import traceback
import logging
import warnings
import re
from typing import List
from io import StringIO
from itertools import chain
import time

from tqdm import tqdm
from omegaconf import OmegaConf
from hydra import compose

from boxing_gym.agents.box_loop_helper import extract_python_code, extract_text_within_markers
from boxing_gym.agents.async_litellm_wrapper import AsyncLiteLLM
from boxing_gym.agents.llm import LLMAgent, StanProposalLLMAgent, StanCriticLLMAgent
from boxing_gym.agents.box_loop_experiment import BoxLoop_Experiment

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

def extract_code_helper(response, logger):
  try:
    # extract the ```python ``` block
    code_extract = extract_python_code(response)
    return code_extract, response
  except Exception:
    with StringIO() as buf:
        traceback.print_exc(file=buf)
        tb_str = buf.getvalue()
    logger.error(tb_str)
    logger.info(f"unable to extract code, response: {response}")
    return None

def select_exemplars(num_exemplars, curr_programs, programs, cfg, experiment):

  if cfg.incontext_heuristic == "recent":
    return curr_programs[:num_exemplars]
  else: 
    raise Exception(f"ERROR: {cfg.critic_exemplar_heuristic} is not a valid heuristic")

def get_critic_exemplars(curr_programs, programs, cfg, experiment):

    if cfg.critic_exemplar_heuristic == "recent":
      programs_to_evaluate = curr_programs.copy()
      return programs_to_evaluate
    else: 
      raise Exception(f"ERROR: {cfg.critic_exemplar_heuristic} is not a valid heuristic")
  
def extract_code_from_proposals(responses: List, logger):
  # extract code from LLM response
  programs = [extract_code_helper(r, logger) for r in responses]
  # sometimes code not extracted properly
  programs = [item for item in programs if item is not None]
  return programs

def propose(
    agent: LLMAgent,
    experiment,
    system_message: str, 
    exemplar_list: List,
    user_messages: List[str],
    temperature: float, 
    logger,
    num_proposals: int,
    log_dir,
):
  '''
    data_image (str): encoded image of the dataset
    temperature (float): temperature to pass to agent
    experiment_config (dict): contains datasets
  '''
  responses = agent.batch_prompt(
          expertise=system_message, 
          messages=user_messages, 
          data_image=None,
          incontext_info=exemplar_list,
          temperature=temperature,
          n=num_proposals,
  )

  if not responses:
    logger.warning("No proposals received; returning empty list.")
    programs = []
    results = []
    proposal_stats = {
        "num_proposals_requested": num_proposals,
        "num_responses_received": 0,
        "num_programs_extracted": 0,
        "extraction_rate": 0.0,
        "num_programs_scored": 0,
        "scoring_rate": 0.0,
        "num_filtered_for_none": 0,
        "num_filtered_for_divergence": 0,
        "num_final_programs": 0,
        "final_success_rate": 0.0,
    }
    return results, proposal_stats

  if not agent.batch_mode:
    if not responses[0] or len(responses[0]) != num_proposals:
      logger.warning(
          f"Unexpected proposal shape: expected {num_proposals}, got {len(responses[0]) if responses else 0}"
      )
    responses = list(chain.from_iterable(responses)) if responses else []

  if agent.batch_mode:
    if len(responses) != len(user_messages):
      logger.warning(
          f"Unexpected batch count: expected {len(user_messages)}, got {len(responses)}"
      )

  for r in responses:
      logger.info(f"unscored program: {r}")

  programs = extract_code_from_proposals(responses, logger)

  # TODO: can i actually log this info somewhere to a dict?
  extraction_rate = len(programs) / len(responses) if len(responses) > 0 else 0
  logger.info(f"program extraction rate: {extraction_rate}")

  if experiment.prior_mode:
    results = [
      {
        "full_llm_response": llm_response, 
        "str_prob_prog": program, 
        "trace": "", 
        "loo": "", 
        "waic": "", 
        "round":"" 
      }
      for program, llm_response in programs
    ]
  else:
    results = [
      experiment.score_programs(program=program, logger=logger, llm_response=llm_response) 
      for program, llm_response in programs
    ]

  assert len(results) == len(programs)

  # filter out failed runs 
  pre_filter = len(results)
  results = [item for item in results if item is not None]
  
  # we always filter out nones, but may also want to filter out other things depending on context
  after_none = len(results)
  results = experiment.filter_fn(results)
  after_divergence = len(results)

  scoring_rate = after_none / num_proposals if num_proposals > 0 else 0
  logger.info(f"Percent successfuly scored programs: {scoring_rate}")

  results = experiment.sort_fn(results)

  proposal_stats = {
      "num_proposals_requested": num_proposals,
      "num_responses_received": len(responses),
      "num_programs_extracted": len(programs),
      "extraction_rate": extraction_rate,
      "num_programs_scored": after_none,
      "scoring_rate": scoring_rate,
      "num_filtered_for_none": pre_filter - after_none,
      "num_filtered_for_divergence": after_none - after_divergence,
      "num_final_programs": len(results),
      "final_success_rate": len(results) / num_proposals if num_proposals > 0 else 0,
  }
  try:
    if hasattr(agent, "get_usage_stats"):
      u = agent.get_usage_stats()
      for k in [
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
        "total_cost_usd",
        "call_count",
        "retry_count",
        "error_count",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
      ]:
        proposal_stats[f"proposal_{k}"] = u.get(k, 0)
  except Exception:
    pass

  return results, proposal_stats

def initialize_experiment(cfg, logger, log_dir, env, prior_mode):
  return BoxLoop_Experiment(
    dataset=env,
    logger=logger, 
    corrupt=cfg.corrupt,
    log_dir=log_dir,
    language_synthesize=cfg.language_synthesize,
    prior_mode=prior_mode,
    diagnostics_cfg=cfg.get("diagnostics", None),
  )

def get_llms(cfg):
  llm_cfg = getattr(cfg, 'llms', None) or cfg
  model_name = getattr(cfg, 'llm', None) or getattr(llm_cfg, 'model_name', None)

  experiment2tokens = {"ODE": 2000, "GPs": 1500, "box_loop": 4096, "Other_ODE": 1500}
  max_tokens = getattr(llm_cfg, 'max_tokens', None) or experiment2tokens.get(cfg.experiment, 2048)

  kwargs = {}
  api_base = getattr(llm_cfg, 'api_base', None)
  api_key = getattr(llm_cfg, 'api_key', None)
  custom_llm_provider = getattr(llm_cfg, 'custom_llm_provider', None)
  if api_base:
      kwargs['api_base'] = api_base
  if api_key:
      kwargs['api_key'] = api_key
  if custom_llm_provider:
      kwargs['custom_llm_provider'] = custom_llm_provider

  return AsyncLiteLLM(model_name=model_name, max_tokens=max_tokens, **kwargs)

def get_proposal_agent(cfg, warm_start_examples, logger):

  llm = get_llms(cfg)

  if cfg.warm_start:
    experiment_to_agent = {
      "box_loop": StanProposalLLMAgent(
        llm, 
        model_id=cfg.llm, 
        batch_mode=False,
        warm_start_examples=warm_start_examples,
        logger=logger,
        use_vision=cfg.use_vision,
        ),
    }
    assert cfg.experiment in experiment_to_agent
    return experiment_to_agent[cfg.experiment]
  
  else:
    experiment_to_agent = {
      "box_loop": StanProposalLLMAgent(llm, vision_only=cfg.vision_only, model_id=cfg.llm, use_vision=cfg.use_vision, batch_mode=False, logger=logger),
    }

    assert cfg.experiment in experiment_to_agent
    return experiment_to_agent[cfg.experiment]

def get_critic_agent(cfg, logger):
  llm = get_llms(cfg)

  experiment_to_agent = {
      "box_loop": StanCriticLLMAgent(llm, 
                                 logger=logger, 
                                 vision_only=cfg.vision_only ,model_id=cfg.llm, use_vision=cfg.use_vision, batch_mode=False),
  }

  assert cfg.experiment in experiment_to_agent
  return experiment_to_agent[cfg.experiment]

def validate_exemplars(cfg, round_idx, exemplar_list):
  # if using the recent heuristic, make sure the exemplars are all from previous round
  if (len(exemplar_list) > 0) and (cfg.incontext_heuristic == "recent"):
    for r in exemplar_list:
      assert r['round'] == round_idx-1

def critic(
    agent: LLMAgent, 
    system_message: str, 
    incontext_info: List,
    user_messages: List[str],
    temperature: float, 
    experiment,
    logger,
    cfg, 
):

  responses = agent.batch_prompt(
          expertise=system_message, 
          messages=user_messages, 
          data_image=None,
          incontext_info=incontext_info,
          temperature=temperature
  )

  if not responses:
    logger.warning("No responses received from critic agent; skipping criticism.")
    return "", ""
  if isinstance(responses[0], list):
    llm_response = responses[0][0] if responses[0] else ""
  else:
    llm_response = responses[0]

  logger.info(f"Critic response: {llm_response}")

  try:
    def _extract_section(text, label):
      parts = extract_text_within_markers(text, label)
      if parts:
        return parts[0].strip()
      fenced = re.search(rf"```\\s*{re.escape(label)}\\s*\\n([\\s\\S]*?)```", text, re.IGNORECASE)
      if fenced:
        return fenced.group(1).strip()
      loose = re.search(rf"{re.escape(label)}\\s*:\\s*([\\s\\S]*?)(?:\\n\\n|$)", text, re.IGNORECASE)
      if loose:
        return loose.group(1).strip()
      return ""

    str_hypotheses = _extract_section(llm_response, "Hypotheses")
    synthesis = _extract_section(llm_response, "Synthesis")
    if not str_hypotheses or not synthesis:
      logger.warning("Critic response missing Hypotheses/Synthesis blocks.")
    logger.info(f"hypotheses: \n {str_hypotheses} \n")
    logger.info(f"synthesis: \n {synthesis} \n")
  except Exception as e:
    with StringIO() as buf:
        traceback.print_exc(file=buf)
        tb_str = buf.getvalue()
    logger.error(f"unable to extract hypotheses: {e}")
    logger.error(tb_str)
    return "", "" 
  
  return str_hypotheses, synthesis

def get_warm_start_example(experiment):

  results = [{"loo": experiment.expert_loo}]
  return results

def get_log_dir(cfg, env):
  import uuid
  from datetime import datetime

  os.makedirs("./logs", exist_ok=True)

  model_name = str(cfg.llm).replace("/", "_").replace(":", "_")
  wandb_run_id = os.getenv("WANDB_RUN_ID", "local")
  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  unique_suffix = uuid.uuid4().hex[:6]

  log_dir = f"./logs/{env.env_name}_{model_name}_{wandb_run_id}_{timestamp}_{unique_suffix}/"
  log_file = f"{env.env_name}_{model_name}_{wandb_run_id}_{timestamp}.log"

  os.makedirs(log_dir, exist_ok=True)
  return log_dir, log_file

def validate_critic_state(cfg, critic_system_message, programs_to_evaluate, curr_programs, programs_all):
  if cfg.critic_exemplar_heuristic == "state_space":
    assert "Here are the hypotheses from the previous round" in critic_system_message
    assert "Here are the synthesis from the previous round" in critic_system_message

def _find_llm_config(model_name: str) -> str | None:
  """Scan conf/llms/*.yaml to find config with matching model_name."""
  conf_dir = pathlib.Path(__file__).parent.parent.parent.parent / "conf" / "llms"
  for yaml_file in conf_dir.glob("*.yaml"):
    try:
      cfg = OmegaConf.load(yaml_file)
      if getattr(cfg, 'model_name', None) == model_name:
        return yaml_file.stem  # filename without extension
    except Exception:
      continue
  return None


# no @weave_op here; it handles PyMC objects/traces. LiteLLM calls are still traced.
def run_box_loop(env, warm_start_examples, prior_mode=False, critic_mode=False, prev_str_hypotheses="", prev_synthesis="", llm_model=None):

  _suppress_ppl_warnings()
  _quiet_ppl_loggers()

  target_llm = llm_model or os.getenv("BOX_LOOP_LLM")
  overrides = []
  if target_llm:
    llm_config = _find_llm_config(target_llm)
    if llm_config:
      overrides.append(f"llms={llm_config}")
    else:
      warnings.warn(
        f"No config for '{target_llm}' in conf/llms/ - using LiteLLM defaults",
        UserWarning,
        stacklevel=2
      )

  cfg = compose(config_name="box_loop", overrides=overrides)

  if target_llm:
    cfg.llm = target_llm
  elif not llm_model and not os.getenv("BOX_LOOP_LLM"):
    warnings.warn(
      "run_box_loop() using default LLM - pass llm_model for fair evaluation",
      UserWarning,
      stacklevel=2
    )

  log_dir, log_file = get_log_dir(cfg, env)

  start = time.time()
  
  # reset handlers so each run writes to its own log file.
  logging.basicConfig(filename=os.path.join(log_dir, log_file), level=logging.INFO, force=True)
  logger = logging.getLogger(__name__)

  args_str = ', '.join(f'{key}={value}' for key, value in vars(cfg).items())
  logger.info(f'Running with these args: {args_str}')

  # returns a class with all the datasets and state you need to run our algo and compute metrics
  experiment = initialize_experiment(
    cfg=cfg, 
    logger=logger, 
    log_dir=log_dir, 
    env=env,
    prior_mode=prior_mode,
  )

  proposal_agent = get_proposal_agent(cfg, warm_start_examples=warm_start_examples, logger=logger)
  critic_agent = get_critic_agent(cfg, logger=logger)
  
  if not os.path.exists(experiment.checkpoint_dir):
    print(f"making path: {experiment.checkpoint_dir}")
    pathlib.Path(experiment.checkpoint_dir).mkdir(parents=True, exist_ok=True)
  else:
    print(f"path: {experiment.checkpoint_dir} already exists")

  temperature_sched = [cfg.proposal_temperature] * cfg.num_rounds
  num_candidates = [cfg.num_proposals] * cfg.num_rounds

  assert len(num_candidates) == cfg.num_rounds
  assert len(temperature_sched) == cfg.num_rounds

  exemplar_list = []
  programs_all = []
  exemplars_all = []
  critic_info = []
  round_statistics = []

  str_hypotheses = ""
  synthesis = ""

  assert cfg.num_to_consider >= cfg.num_proposals

  print("Entering model_search loop")

  for round_idx in tqdm(range(cfg.num_rounds)):

    proposal_system_message = experiment.get_system_message(
      mode="proposal",
      vision_only=cfg.vision_only,
      prev_str_hypotheses=prev_str_hypotheses,
      prev_synthesis=prev_synthesis,
    )

    proposal_user_message = experiment.get_user_message(
      mode="proposal", 
      synthesis=synthesis, 
      str_hypotheses=str_hypotheses,
      vision_only=cfg.vision_only,
    )

    logger.info(f"proposal_system_message: {proposal_system_message}")
    print(f"proposal_system_message: {proposal_system_message}")
    logger.info(f"proposal_user_message: {proposal_user_message}")

    validate_exemplars(cfg=cfg, round_idx=round_idx, exemplar_list=exemplar_list)

    curr_programs, proposal_stats = propose(
        agent=proposal_agent,
        experiment=experiment,
        system_message=proposal_system_message, 
        exemplar_list=exemplar_list,
        user_messages=[proposal_user_message],
        temperature=temperature_sched[round_idx], 
        logger=logger,
        num_proposals=num_candidates[round_idx],
        log_dir=log_dir,
    )

    try:
      proposal_stats["proposal_total_cost_usd"] = float(getattr(proposal_agent, "total_inference_cost", 0.0))
      proposal_stats["critic_total_cost_usd"] = float(getattr(critic_agent, "total_inference_cost", 0.0))
    except Exception:
      pass

    proposal_stats['round_idx'] = round_idx
    round_statistics.append(proposal_stats)


    for p in curr_programs:
      p['round'] = round_idx
      programs_all.append(p)

    exemplar_list = select_exemplars(
      num_exemplars=cfg.num_exemplars, 
      curr_programs=curr_programs, 
      programs=programs_all,
      cfg=cfg,
      experiment=experiment,
      )
    
    for exemplar in exemplar_list:
      exemplars_all.append((round_idx, exemplar))


    if critic_mode: 

      if round_idx == 0:
        assert len(synthesis) == 0
        assert len(str_hypotheses) == 0
      
      
      if cfg.critic_exemplar_heuristic in ["recent"]:
        critic_system_message = experiment.get_system_message(
          mode="critic", 
          critic_strategy=cfg.critic_exemplar_heuristic,  
          vision_only=cfg.vision_only,
          prev_str_hypotheses=prev_str_hypotheses,
          prev_synthesis=prev_str_hypotheses,
          )
        critic_user_message = experiment.get_user_message(
          mode="critic", 
          vision_only=cfg.vision_only, 
          synthesis=None, 
          str_hypotheses=None)
      
      else:
        critic_system_message = experiment.get_system_message(
          mode="critic", 
          critic_strategy=cfg.critic_exemplar_heuristic,  
          vision_only=cfg.vision_only,
          str_hypotheses=str_hypotheses,
          synthesis=synthesis,
          )

        critic_user_message = experiment.get_user_message(
          mode="critic", 
          vision_only=cfg.vision_only, 
          synthesis=synthesis, 
          str_hypotheses=str_hypotheses)

      logger.info(f"critic_system_message: {critic_system_message}")
      logger.info(f"critic_user_message: {critic_user_message}")
      logger.info(f"str_hypotheses: {str_hypotheses}")
      logger.info(f"synthesis: {synthesis}")
      
      programs_to_evaluate = get_critic_exemplars(
        curr_programs=curr_programs, programs=programs_all, cfg=cfg, experiment=experiment)

      validate_critic_state(cfg=cfg, 
                            critic_system_message=critic_system_message, 
                            programs_to_evaluate=programs_to_evaluate, 
                            curr_programs=curr_programs, 
                            programs_all=programs_all)

      str_hypotheses, synthesis = critic(
          agent=critic_agent, 
          system_message=critic_system_message, 
          experiment=experiment,
          incontext_info=programs_to_evaluate, # summarize best proposals across history
          user_messages=[critic_user_message] * 1,
          logger=logger,
          temperature=cfg.critic_temperature,
          cfg=cfg,
      )
      try:
        if round_statistics:
          round_statistics[-1]["critic_total_cost_usd"] = float(getattr(critic_agent, "total_inference_cost", 0.0))
          if hasattr(critic_agent, "get_usage_stats"):
            cu = critic_agent.get_usage_stats()
            for k in [
              "prompt_tokens",
              "completion_tokens",
              "reasoning_tokens",
              "total_tokens",
              "total_cost_usd",
              "call_count",
              "retry_count",
              "error_count",
              "latency_mean_ms",
              "latency_p50_ms",
              "latency_p95_ms",
            ]:
              round_statistics[-1][f"critic_{k}"] = cu.get(k, 0)
      except Exception:
        pass
      critic_info.append(({"round_idx": round_idx, "str_hypotheses": str_hypotheses, "synthesis": synthesis}))

  end = time.time()

  programs_all = experiment.evaluate(programs_all=programs_all, 
                      logger=logger, 
                      cfg=cfg, 
                      critic_info=critic_info,
                      proposal_agent=proposal_agent,
                      critic_agent=critic_agent,
                      )

  end = time.time()
  time_taken = (end-start)/60
  logger.info(f"time taken: {time_taken}")
  return programs_all, critic_info, round_statistics

if __name__ == "__main__":
  run_box_loop()