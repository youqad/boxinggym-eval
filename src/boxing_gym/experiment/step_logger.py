"""Step-by-step WandB logger for live progress tracking.
Tracks EIG regret, token growth, exit status, communication
metrics, and latency statistics.
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExitStatus(Enum):
    """Categorization of why experiment runs terminate."""
    COMPLETED = "completed"           # all experiments ran successfully
    BUDGET_EXHAUSTED = "budget_exhausted"  # hit experiment budget limit
    COST_LIMIT = "cost_limit"         # hit cost limit
    MAX_RETRIES = "max_retries"       # too many failed experiments
    TIMEOUT = "timeout"               # wall clock timeout
    ERROR = "error"                   # unrecoverable error


@dataclass
class StepLogger:
    """Logger for per-step and cumulative metrics during experiment runs.
    
    Args:
        wandb_module: The wandb module (or None if disabled)
        wandb_run: Active wandb run (or None if disabled)
        start_time: Experiment start time for wall clock tracking
        log_callback: Optional callback for custom logging (receives step_idx, metrics dict)
    """
    wandb_module: Any = None
    wandb_run: Any = None
    start_time: float = field(default_factory=time.time)
    log_callback: Optional[Callable[[int, Dict], None]] = None
    
    # cumulative tracking
    _total_queries: int = field(default=0)
    _total_observations: int = field(default=0)
    _total_successes: int = field(default=0)
    _total_retries: int = field(default=0)
    _total_eig: float = field(default=0.0)
    _eig_count: int = field(default=0)
    _step_times: List[float] = field(default_factory=list)
    _last_step_time: float = field(default=0.0)

    # per-budget evaluation tracking (for cumulative eval metrics)
    _eval_means: List[float] = field(default_factory=list)
    _eval_z_means: List[float] = field(default_factory=list)
    _eval_z_stds: List[float] = field(default_factory=list)

    # EIG regret tracking (gap from optimal)
    _total_eig_regret: float = field(default=0.0)
    _eig_values: List[float] = field(default_factory=list)
    _optimal_eig_values: List[float] = field(default_factory=list)

    # token growth tracking (cumulative by type)
    _cumulative_prompt_tokens: int = field(default=0)
    _cumulative_completion_tokens: int = field(default=0)
    _cumulative_reasoning_tokens: int = field(default=0)
    _last_usage_totals: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # latency tracking (per-call timing)
    _latencies_ms: List[float] = field(default_factory=list)

    # exit status tracking
    _exit_status: Optional[ExitStatus] = field(default=None)
    _exit_reason: Optional[str] = field(default=None)

    # communication metrics (discovery mode)
    _scientist_z_mean: Optional[float] = field(default=None)
    _naive_z_mean: Optional[float] = field(default=None)
    _explanation_lengths: List[int] = field(default_factory=list)

    # evaluation metric direction (True if lower is better)
    _eval_lower_is_better: Optional[bool] = field(default=None)

    # last step index for aligning one-off logs
    _last_step_idx: Optional[int] = field(default=None)

    # step index offset for multi-seed or resumed runs
    _step_offset: int = field(default=0)
    
    # track if metrics have been defined
    _metrics_defined: bool = field(default=False)

    @property
    def enabled(self) -> bool:
        return self.wandb_module is not None and self.wandb_run is not None

    def _define_metrics(self) -> None:
        """Define WandB metrics with separate x-axes to avoid step conflicts."""
        if self._metrics_defined or not self.enabled:
            return
        try:
            # step metrics use step/idx as x-axis
            self.wandb_module.define_metric("step/*", step_metric="step/idx")
            self.wandb_module.define_metric("cumulative/*", step_metric="step/idx")
            self.wandb_module.define_metric("llm/*", step_metric="step/idx")

            # eval metrics use eval/budget as x-axis (independent of step)
            self.wandb_module.define_metric("eval/*", step_metric="eval/budget")

            # exit and comm metrics logged once per run, using step/idx for consistency
            self.wandb_module.define_metric("exit/*", step_metric="step/idx")
            self.wandb_module.define_metric("comm/*", step_metric="step/idx")
            self.wandb_module.define_metric("ppl/*", step_metric="eval/budget")

            self._metrics_defined = True
        except Exception as e:
            logger.debug(f"WandB define_metric failed: {e}")

    def _log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        """Internal logging to wandb and optional callback."""
        if self.enabled:
            self._define_metrics()
            try:
                if step is not None:
                    self.wandb_module.log(metrics, step=step, commit=commit)
                else:
                    self.wandb_module.log(metrics, commit=commit)
            except Exception as e:
                logger.debug(f"WandB log failed: {e}")
        
        if self.log_callback is not None:
            try:
                self.log_callback(step, metrics)
            except Exception as e:
                logger.debug(f"Log callback failed: {e}")

    def set_step_offset(self, offset: int, reset_timing: bool = True) -> None:
        """Set a global step offset (useful for multi-seed runs).

        Args:
            offset: Number of steps already completed before this segment.
            reset_timing: If True, resets last-step timing so the next
                step duration doesn't include time between segments.
        """
        try:
            self._step_offset = max(0, int(offset))
        except Exception:
            self._step_offset = 0
        if reset_timing:
            self._last_step_time = 0.0
    
    def log_step(
        self,
        step_idx: int,
        success: bool,
        retry_count: int = 0,
        eig: Optional[float] = None,
        optimal_eig: Optional[float] = None,
        observation: Optional[Any] = None,
        query: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Log per-step metrics for live progress tracking.
        Args:
            step_idx: Current step index (0-based)
            success: Whether this step's experiment succeeded
            retry_count: Number of retries needed for this step
            eig: Expected information gain (if computed)
            optimal_eig: Optimal EIG for this step (for regret tracking)
            observation: The observation received (for size tracking)
            query: The query made (for length tracking)
            latency_ms: LLM call latency in milliseconds
        """
        now = time.time()
        global_step_idx = step_idx + self._step_offset
        self._last_step_idx = global_step_idx
        step_duration = now - self._last_step_time if self._last_step_time > 0 else 0
        self._last_step_time = now
        self._step_times.append(step_duration)
        
        # update cumulative counters
        self._total_queries += 1 + retry_count  # Original query + retries
        self._total_observations += 1 + retry_count
        if success:
            self._total_successes += 1
        self._total_retries += retry_count
        
        if eig is not None:
            self._total_eig += eig
            self._eig_count += 1
            self._eig_values.append(eig)

            # EIG regret tracking
            if optimal_eig is not None:
                regret = max(0.0, optimal_eig - eig)
                self._total_eig_regret += regret
                self._optimal_eig_values.append(optimal_eig)

        # latency tracking
        if latency_ms is not None:
            self._latencies_ms.append(latency_ms)
        
        # calculate running stats
        # success rate: successes / steps attempted (excluding retries from denominator)
        total_steps = self._total_queries - self._total_retries
        cumulative_success_rate = self._total_successes / total_steps if total_steps > 0 else 0.0
        avg_retries = self._total_retries / (global_step_idx + 1) if global_step_idx >= 0 else 0.0
        wall_time = now - self.start_time
        
        # build metrics payload
        # per-step (instantaneous) metrics
        metrics = {
            "step/idx": global_step_idx,
            "step/success": 1 if success else 0,
            "step/retry_count": retry_count,
            "step/duration_sec": step_duration,
            "step/wall_time_sec": wall_time,
        }
        
        if eig is not None:
            metrics["step/eig"] = eig
            # EIG regret (gap from optimal)
            if optimal_eig is not None:
                metrics["step/optimal_eig"] = optimal_eig
                metrics["step/eig_regret"] = max(0.0, optimal_eig - eig)

        # latency metrics
        if latency_ms is not None:
            metrics["step/latency_ms"] = latency_ms

        # query/observation size metrics (useful for debugging)
        if query is not None:
            metrics["step/query_length"] = len(str(query))
        if observation is not None:
            metrics["step/observation_length"] = len(str(observation))
        
        # cumulative (running) metrics
        metrics.update({
            "cumulative/success_rate": cumulative_success_rate,
            "cumulative/total_queries": self._total_queries,
            "cumulative/total_observations": self._total_observations,
            "cumulative/total_successes": self._total_successes,
            "cumulative/total_retries": self._total_retries,
            "cumulative/avg_retries_per_step": avg_retries,
        })
        
        if self._eig_count > 0:
            metrics["cumulative/eig_mean"] = self._total_eig / self._eig_count
            metrics["cumulative/eig_sum"] = self._total_eig
            metrics["cumulative/eig_count"] = self._eig_count
            # EIG regret cumulative
            if self._optimal_eig_values:
                metrics["cumulative/eig_regret_sum"] = self._total_eig_regret
                metrics["cumulative/eig_regret_mean"] = self._total_eig_regret / len(self._optimal_eig_values)

        # latency cumulative statistics
        if self._latencies_ms:
            latencies = np.array(self._latencies_ms)
            metrics["cumulative/latency_ms_mean"] = float(np.mean(latencies))
            metrics["cumulative/latency_ms_p50"] = float(np.percentile(latencies, 50))
            metrics["cumulative/latency_ms_p95"] = float(np.percentile(latencies, 95))
            metrics["cumulative/latency_ms_std"] = float(np.std(latencies))
        
        # throughput metrics
        if wall_time > 0:
            metrics["cumulative/steps_per_minute"] = (global_step_idx + 1) / (wall_time / 60.0)
            metrics["cumulative/queries_per_minute"] = self._total_queries / (wall_time / 60.0)
        
        # average step duration (excluding first step which may include setup)
        if len(self._step_times) > 1:
            metrics["cumulative/avg_step_duration_sec"] = sum(self._step_times[1:]) / len(self._step_times[1:])
        
        self._log(metrics)  # Uses step/idx from payload as x-axis per define_metric
    
    def log_evaluation(
        self,
        budget: int,
        eval_score: Any,
        z_mean: Optional[float] = None,
        z_std: Optional[float] = None,
        is_prior_only: bool = False,
        box_loop_stats: Optional[List[Dict]] = None,
    ) -> None:
        """Log evaluation metrics at budget checkpoints.
        
        Logs both per-budget metrics and cumulative evaluation statistics.
        
        Args:
            budget: The experiment budget at this checkpoint
            eval_score: Evaluation score (dict with metrics or tuple (mean, std))
            z_mean: Z-score normalized mean (for paper comparison)
            is_prior_only: Whether this is a prior-only evaluation (budget=0)
            box_loop_stats: Optional PPL/Box loop statistics
        """
        metrics = {"eval/budget": budget}
        
        # parse evaluation score
        mean_val, std_val = None, None
        mean_key = None
        if isinstance(eval_score, dict):
            for k, v in eval_score.items():
                if isinstance(v, (int, float)) or v is None:
                    metrics[f"eval/{k}"] = v
            if "mse" in eval_score:
                mean_key = "mse"
                mean_val = eval_score.get("mse")
            elif "accuracy" in eval_score:
                mean_key = "accuracy"
                mean_val = eval_score.get("accuracy")
            elif "score" in eval_score:
                mean_key = "score"
                mean_val = eval_score.get("score")
            std_val = eval_score.get("std_mse", eval_score.get("std", eval_score.get("std_accuracy")))
        elif isinstance(eval_score, (list, tuple)) and len(eval_score) >= 2:
            mean_val, std_val = eval_score[0], eval_score[1]
            metrics["eval/mse"] = float(mean_val) if mean_val is not None else None
            metrics["eval/std_mse"] = float(std_val) if std_val is not None else None
        
        # generic names for sweep compatibility
        metrics["eval/mean"] = float(mean_val) if mean_val is not None else None
        metrics["eval/std"] = float(std_val) if std_val is not None else None
        
        if z_mean is not None:
            metrics["eval/z_mean"] = z_mean
            self._eval_z_means.append(z_mean)
        if z_std is not None:
            metrics["eval/z_std"] = z_std
            self._eval_z_stds.append(z_std)
        
        if mean_val is not None:
            self._eval_means.append(float(mean_val))

        if self._eval_lower_is_better is None and mean_key is not None:
            key = mean_key.lower()
            if key in {"mse", "rmse", "mae", "loss"}:
                self._eval_lower_is_better = True
            elif key in {"accuracy", "acc", "score", "auc"}:
                self._eval_lower_is_better = False
        
        # cumulative evaluation metrics
        if self._eval_means:
            metrics["cumulative/eval_mean_avg"] = sum(self._eval_means) / len(self._eval_means)
            lower_is_better = self._eval_lower_is_better
            if lower_is_better is None:
                lower_is_better = True
            if lower_is_better:
                metrics["cumulative/eval_mean_best"] = min(self._eval_means)
                metrics["cumulative/eval_mean_worst"] = max(self._eval_means)
            else:
                metrics["cumulative/eval_mean_best"] = max(self._eval_means)
                metrics["cumulative/eval_mean_worst"] = min(self._eval_means)
            metrics["cumulative/num_evaluations"] = len(self._eval_means)
        
        if self._eval_z_means:
            metrics["cumulative/z_mean_avg"] = sum(self._eval_z_means) / len(self._eval_z_means)
            metrics["cumulative/z_mean_best"] = min(self._eval_z_means)
            metrics["cumulative/z_mean_latest"] = self._eval_z_means[-1]
        if self._eval_z_stds:
            metrics["cumulative/z_std_latest"] = self._eval_z_stds[-1]
        
        metrics["eval/is_prior_only"] = 1 if is_prior_only else 0
        metrics["eval/wall_time_sec"] = time.time() - self.start_time
        
        # PPL/Box loop stats
        if isinstance(box_loop_stats, list) and box_loop_stats:
            metrics["ppl/num_rounds"] = len(box_loop_stats)
            numeric_keys = set()
            for s in box_loop_stats:
                if not isinstance(s, dict):
                    continue
                for k, v in s.items():
                    if k == "round_idx":
                        continue
                    if isinstance(v, (int, float)):
                        numeric_keys.add(k)
            for k in numeric_keys:
                vals = [
                    float(s[k]) for s in box_loop_stats
                    if isinstance(s, dict) and isinstance(s.get(k), (int, float))
                ]
                if vals:
                    metrics[f"ppl/{k}_mean"] = sum(vals) / len(vals)
                    metrics[f"ppl/{k}_last"] = vals[-1]
        
        self._log(metrics)  # uses eval/budget from payload as x-axis per define_metric
    
    def log_llm_usage_step(
        self,
        step_idx: int,
        usage_stats: Dict[str, Any],
        agent_prefix: str = "llm",
    ) -> None:
        """Log per-step LLM usage for live cost/token tracking.

        Args:
            step_idx: Current step index
            usage_stats: Usage stats from agent.get_usage_stats()
            agent_prefix: Prefix for metric names (e.g., "llm", "naive_llm")
        """
        if not usage_stats:
            return

        metrics = {"step/idx": step_idx}

        # usage_stats are cumulative; compute deltas for per-step accounting
        last_totals = self._last_usage_totals.setdefault(agent_prefix, {})
        delta_keys = [
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
            "total_cost_usd",
            "call_count",
            "retry_count",
            "error_count",
        ]
        deltas = {}
        for k in delta_keys:
            current = usage_stats.get(k, 0) or 0
            prev = last_totals.get(k, 0) or 0
            delta = current - prev
            if delta < 0:
                delta = current
            last_totals[k] = current
            deltas[k] = delta

        # update cumulative counters for token growth tracking
        self._cumulative_prompt_tokens += int(deltas.get("prompt_tokens", 0) or 0)
        self._cumulative_completion_tokens += int(deltas.get("completion_tokens", 0) or 0)
        self._cumulative_reasoning_tokens += int(deltas.get("reasoning_tokens", 0) or 0)

        for k in [
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
            "total_cost_usd",
            "call_count",
            "retry_count",
            "error_count",
        ]:
            val = usage_stats.get(k)
            if val is not None:
                metrics[f"{agent_prefix}/{k}"] = val
            delta_val = deltas.get(k)
            if delta_val is not None:
                metrics[f"{agent_prefix}/{k}_step"] = delta_val

        # per-step cost
        if "total_cost_usd" in usage_stats:
            metrics[f"cumulative/{agent_prefix}_cost_usd"] = usage_stats["total_cost_usd"]
        if "total_tokens" in usage_stats:
            metrics[f"cumulative/{agent_prefix}_tokens"] = usage_stats["total_tokens"]

        # token growth tracking
        metrics[f"cumulative/{agent_prefix}_prompt_tokens"] = self._cumulative_prompt_tokens
        metrics[f"cumulative/{agent_prefix}_completion_tokens"] = self._cumulative_completion_tokens
        metrics[f"cumulative/{agent_prefix}_reasoning_tokens"] = self._cumulative_reasoning_tokens

        # token breakdown ratios
        total_cumulative = (
            self._cumulative_prompt_tokens
            + self._cumulative_completion_tokens
            + self._cumulative_reasoning_tokens
        )
        if total_cumulative > 0:
            metrics[f"cumulative/{agent_prefix}_prompt_fraction"] = (
                self._cumulative_prompt_tokens / total_cumulative
            )
            metrics[f"cumulative/{agent_prefix}_completion_fraction"] = (
                self._cumulative_completion_tokens / total_cumulative
            )
            if self._cumulative_reasoning_tokens > 0:
                metrics[f"cumulative/{agent_prefix}_reasoning_fraction"] = (
                    self._cumulative_reasoning_tokens / total_cumulative
                )
                # reasoning efficiency: ratio of reasoning to completion tokens
                if self._cumulative_completion_tokens > 0:
                    metrics[f"cumulative/{agent_prefix}_reasoning_efficiency"] = (
                        self._cumulative_reasoning_tokens / self._cumulative_completion_tokens
                    )
                metrics[f"{agent_prefix}/is_thinking_model"] = 1

        if metrics:
            self._log(metrics)  # uses step/idx from payload as x-axis per define_metric
    
    def get_cumulative_stats(self) -> Dict[str, Any]:
        """Get all cumulative statistics for final summary."""
        wall_time = time.time() - self.start_time
        stats = {
            "total_queries": self._total_queries,
            "total_observations": self._total_observations,
            "total_successes": self._total_successes,
            "total_retries": self._total_retries,
            "wall_time_sec": wall_time,
        }

        # success rate: successful steps / total steps attempted (excluding retries)
        total_steps = self._total_queries - self._total_retries
        if total_steps > 0:
            stats["success_rate"] = self._total_successes / total_steps
        elif self._total_queries > 0:
            # edge case: all queries were retries
            stats["success_rate"] = 0.0

        if self._eig_count > 0:
            stats["eig_mean"] = self._total_eig / self._eig_count
            stats["eig_sum"] = self._total_eig
            stats["eig_count"] = self._eig_count

        # EIG regret statistics
        if self._optimal_eig_values:
            stats["eig_regret_sum"] = self._total_eig_regret
            stats["eig_regret_mean"] = self._total_eig_regret / len(self._optimal_eig_values)

        if self._eval_means:
            stats["eval_mean_avg"] = sum(self._eval_means) / len(self._eval_means)
            lower_is_better = self._eval_lower_is_better
            if lower_is_better is None:
                lower_is_better = True
            if lower_is_better:
                stats["eval_mean_best"] = min(self._eval_means)
            else:
                stats["eval_mean_best"] = max(self._eval_means)
            stats["num_evaluations"] = len(self._eval_means)

        if self._eval_z_means:
            stats["z_mean_avg"] = sum(self._eval_z_means) / len(self._eval_z_means)
            stats["z_mean_best"] = min(self._eval_z_means)
            stats["z_mean_final"] = self._eval_z_means[-1]
        if self._eval_z_stds:
            stats["z_std_final"] = self._eval_z_stds[-1]

        # latency statistics
        if self._latencies_ms:
            latencies = np.array(self._latencies_ms)
            stats["latency_ms_mean"] = float(np.mean(latencies))
            stats["latency_ms_p50"] = float(np.percentile(latencies, 50))
            stats["latency_ms_p95"] = float(np.percentile(latencies, 95))
            stats["latency_ms_std"] = float(np.std(latencies))

        # token growth statistics
        stats["cumulative_prompt_tokens"] = self._cumulative_prompt_tokens
        stats["cumulative_completion_tokens"] = self._cumulative_completion_tokens
        stats["cumulative_reasoning_tokens"] = self._cumulative_reasoning_tokens

        # exit status
        if self._exit_status is not None:
            stats["exit_status"] = self._exit_status.value
            if self._exit_reason:
                stats["exit_reason"] = self._exit_reason

        # communication metrics (discovery mode)
        if self._scientist_z_mean is not None:
            stats["scientist_z_mean"] = self._scientist_z_mean
        if self._naive_z_mean is not None:
            stats["naive_z_mean"] = self._naive_z_mean
        if self._scientist_z_mean is not None and self._naive_z_mean is not None:
            stats["comm_transfer_gap"] = self._scientist_z_mean - self._naive_z_mean
        if self._explanation_lengths:
            stats["avg_explanation_length"] = sum(self._explanation_lengths) / len(
                self._explanation_lengths
            )

        return stats

    def log_exit_status(
        self,
        status: ExitStatus,
        reason: Optional[str] = None,
    ) -> None:
        """Log why the experiment run terminated.

        Args:
            status: The exit status category
            reason: Optional human-readable reason for termination
        """
        self._exit_status = status
        self._exit_reason = reason

        metrics = {
            "exit/status": status.value,
            "exit/wall_time_sec": time.time() - self.start_time,
        }
        if self._last_step_idx is not None:
            metrics["step/idx"] = self._last_step_idx

        # log exit category flags for easy WandB filtering
        for s in ExitStatus:
            metrics[f"exit/{s.value}"] = 1 if s == status else 0

        if reason:
            # WandB doesn't handle string metrics well in charts, but useful in tables
            metrics["exit/reason"] = reason

        self._log(metrics)  # single event, no step association needed

    def log_communication(
        self,
        scientist_z_mean: float,
        naive_z_mean: float,
        explanation: Optional[str] = None,
        accuracy: Optional[float] = None,
    ) -> None:
        """Log communication metrics for discovery mode experiments.

        Tracks how well the scientist agent can communicate findings to a naive agent.

        Args:
            scientist_z_mean: Z-score of scientist's predictions
            naive_z_mean: Z-score of naive agent's predictions after explanation
            explanation: The explanation provided by the scientist
            accuracy: Optional communication accuracy score
        """
        self._scientist_z_mean = scientist_z_mean
        self._naive_z_mean = naive_z_mean

        if explanation is not None:
            self._explanation_lengths.append(len(explanation))

        transfer_gap = scientist_z_mean - naive_z_mean

        metrics = {
            "comm/scientist_z_mean": scientist_z_mean,
            "comm/naive_z_mean": naive_z_mean,
            "comm/transfer_gap": transfer_gap,
        }
        if self._last_step_idx is not None:
            metrics["step/idx"] = self._last_step_idx

        if accuracy is not None:
            metrics["comm/accuracy"] = accuracy

        if explanation is not None:
            metrics["comm/explanation_length"] = len(explanation)

        if self._explanation_lengths:
            metrics["cumulative/avg_explanation_length"] = sum(
                self._explanation_lengths
            ) / len(self._explanation_lengths)

        self._log(metrics)  # single event, no step association needed
