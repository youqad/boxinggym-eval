"""ASCII chart utilities for TUI visualization."""

from typing import Optional


def horizontal_bar(value: float, max_value: float, width: int = 10) -> str:
    """Render a horizontal bar chart.

    Args:
        value: Current value
        max_value: Maximum value for scaling
        width: Total width in characters

    Returns:
        String like "████░░░░░░"
    """
    if max_value <= 0:
        return "░" * width

    filled = min(int((value / max_value) * width), width)
    empty = width - filled
    return "█" * filled + "░" * empty


def z_color(z: float, low: float = -0.3, high: float = 0.3) -> str:
    """Get Rich color name based on z_mean value.

    Args:
        z: The z_mean value
        low: Threshold for green (good)
        high: Threshold for red (poor)

    Returns:
        Rich color name: "green", "yellow", or "red"
    """
    if z < low:
        return "green"
    elif z > high:
        return "red"
    return "yellow"


def colored_z(z: float, low: float = -0.3, high: float = 0.3) -> str:
    """Return z value with Rich color markup.

    Args:
        z: The z_mean value
        low: Threshold for green (good)
        high: Threshold for red (poor)

    Returns:
        Rich markup string like "[green]-0.450[/green]"
    """
    color = z_color(z, low, high)
    return f"[{color}]{z:+.3f}[/{color}]"


def trend_indicator(start: float, end: float, threshold: float = 0.1) -> str:
    """Return trend indicator comparing start and end values.

    Lower z_mean is better, so:
    - ▼▼ = significant improvement (decrease)
    - ▼ = slight improvement
    - ─ = no change
    - ▲ = slight worsening
    - ▲▲ = significant worsening

    Args:
        start: Starting value
        end: Ending value
        threshold: Change threshold for single arrow

    Returns:
        Trend indicator string
    """
    diff = end - start

    if diff < -threshold * 2:
        return "[green]▼▼[/green]"  # big improvement
    elif diff < -threshold:
        return "[green]▼[/green]"   # slight improvement
    elif diff > threshold * 2:
        return "[red]▲▲[/red]"      # big worsening
    elif diff > threshold:
        return "[red]▲[/red]"       # slight worsening
    return "[dim]─[/dim]"           # stable


def sparkline(values: list, width: int = 10) -> str:
    """Render a simple sparkline from a list of values.

    Args:
        values: List of numeric values
        width: Target width (values will be sampled if longer)

    Returns:
        Sparkline string using block characters
    """
    if not values:
        return "─" * width

    # sample values if too many
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
