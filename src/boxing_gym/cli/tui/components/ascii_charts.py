"""ASCII chart utilities for TUI visualization."""

import math


def horizontal_bar(value: float, max_value: float, width: int = 10) -> str:
    """Return bar like '████░░░░░░'."""
    if max_value <= 0 or not math.isfinite(max_value) or not math.isfinite(value):
        return "░" * width

    filled = max(0, min(int((value / max_value) * width), width))
    empty = width - filled
    return "█" * filled + "░" * empty


def z_color(z: float, low: float = -0.3, high: float = 0.3) -> str:
    """Return Rich color name (green/yellow/red) based on z thresholds."""
    if z < low:
        return "green"
    elif z > high:
        return "red"
    return "yellow"


def colored_z(z: float, low: float = -0.3, high: float = 0.3) -> str:
    """Return z value with Rich color markup."""
    if math.isnan(z):
        return "[dim]NaN[/dim]"
    color = z_color(z, low, high)
    return f"[{color}]{z:+.3f}[/{color}]"


def trend_indicator(start: float, end: float, threshold: float = 0.1) -> str:
    """Return trend arrow (lower is better): ▼▼/▼ = improving, ▲/▲▲ = worsening."""
    diff = end - start

    if diff < -threshold * 2:
        return "[green]▼▼[/green]"  # big improvement
    elif diff < -threshold:
        return "[green]▼[/green]"  # slight improvement
    elif diff > threshold * 2:
        return "[red]▲▲[/red]"  # big worsening
    elif diff > threshold:
        return "[red]▲[/red]"  # slight worsening
    return "[dim]─[/dim]"  # stable


def short_model_name(name: str, max_len: int = 12) -> str:
    """Strip provider prefix and truncate for column headers."""
    if not name:
        return "???"
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    is_box = name.endswith(" (box)")
    if is_box:
        name = name[:-6]
    name = name.replace("deepseek-", "ds-")
    name = name.replace("codex-mini", "codex")
    name = name.replace("qwen3-32b-v1:0", "qwen3-32b")
    if is_box:
        if len(name) > max_len - 1:
            return name[: max_len - 2] + "…†"
        return name + "†"
    if len(name) > max_len:
        return name[: max_len - 1] + "…"
    return name


def sparkline(values: list, width: int = 10) -> str:
    """Return sparkline using block chars (▁▂▃▄▅▆▇█), sampled to width."""
    if not values:
        return "─" * width

    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if range_val == 0:
        return "▄" * len(values)

    blocks = " ▁▂▃▄▅▆▇█"
    result = []
    for v in values:
        idx = int((v - min_val) / range_val * 8)
        idx = min(idx, 8)
        result.append(blocks[idx])

    return "".join(result)
