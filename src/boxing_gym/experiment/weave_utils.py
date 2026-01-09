"""Weave helpers for boxing-gym.

Conditional @weave_op: uses weave.op() when enabled, no-op otherwise.

Tracing strategy:
- Weave auto-patching is DISABLED by default (implicitly_patch_integrations=False)
  to ensure stability with custom LLM providers (DeepSeek, MiniMax, Kimi, GLM).
- LiteLLM calls are traced manually via _weave_log_payload() in agent.py,
  which logs sanitized payloads (prompt, output, usage) as "llm_call" ops.
- Set WEAVE_IMPLICITLY_PATCH_INTEGRATIONS=true to enable auto-patching (use with caution).

Use @weave_op only on functions with serializable inputs/outputs. Goal/Agent/PyMC
traces break serialization ("Invalid Client ID digest").

Usage:
    from boxing_gym.experiment.weave_utils import weave_op

    @weave_op()
    def my_function(text: str, count: int) -> str:
        ...

Avoid @weave_op on (due to serialization issues):
- evaluate(), ppl_evaluate(), evaluate_naive_explanation()
- iterative_experiment()
- run_box_loop()
- get_ppl_prediction()
"""

import os
from typing import Callable, Optional, TypeVar, Any

try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    weave = None
    WEAVE_AVAILABLE = False

F = TypeVar("F", bound=Callable[..., Any])


def _is_weave_enabled() -> bool:
    """True if Weave tracing is enabled."""
    if not WEAVE_AVAILABLE or weave is None:
        return False
    disabled = os.environ.get("WEAVE_DISABLED", "0").lower() in ("1", "true", "yes")
    if disabled:
        return False
    project = os.environ.get("WEAVE_PROJECT") or os.environ.get("WANDB_PROJECT")
    return bool(project)


def weave_op(
    name: Optional[str] = None,
    call_display_name: Optional[str] = None,
) -> Callable[[F], F]:
    """Conditional weave.op decorator."""
    def decorator(fn: F) -> F:
        if not _is_weave_enabled():
            return fn

        op_kwargs = {}
        if name:
            op_kwargs["name"] = name
        if call_display_name:
            op_kwargs["call_display_name"] = call_display_name

        try:
            wrapped = weave.op(**op_kwargs)(fn) if op_kwargs else weave.op()(fn)
            return wrapped
        except Exception:
            return fn

    return decorator


def weave_op_method(
    name: Optional[str] = None,
) -> Callable[[F], F]:
    """Conditional weave.op decorator for methods."""
    return weave_op(name=name)


def get_weave():
    """Return weave module if available, else None."""
    return weave if WEAVE_AVAILABLE else None
