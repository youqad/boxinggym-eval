"""Menu rendering utilities for TUI."""

from typing import List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def render_menu(
    console: Console,
    title: str,
    subtitle: str,
    options: List[Tuple[str, str]],
) -> None:
    """Render a numbered menu with options.

    Args:
        console: Rich console for output
        title: Main title
        subtitle: Subtitle (e.g., sweep info)
        options: List of (key, label) tuples, where key is the menu number
    """
    text = Text()
    text.append(f"{title}\n", style="bold cyan")
    text.append(f"{subtitle}\n\n", style="dim")

    for key, label in options:
        if key == "0":
            text.append("\n")
        text.append(f"  [{key}] ", style="bold yellow")
        text.append(f"{label}\n", style="white")

    panel = Panel(
        text,
        border_style="cyan",
        expand=True,
        padding=(1, 2),
    )
    console.print(panel)


def clear_screen(console: Console) -> None:
    """Clear the terminal screen."""
    console.clear()


def press_enter_to_continue(console: Console) -> None:
    """Wait for user to press Enter."""
    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()
