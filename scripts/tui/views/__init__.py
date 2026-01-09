"""Base view class for TUI views."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd
from rich.console import Console


__all__ = ["BaseView"]


class BaseView(ABC):
    """Abstract base class for TUI views."""

    def __init__(self, df: pd.DataFrame, console: Console, metric: str = "metric/eval/z_mean"):
        self.df = df
        self.console = console
        self.metric = metric

    @property
    @abstractmethod
    def title(self) -> str:
        """View title for the menu."""
        pass

    @abstractmethod
    def render(self) -> None:
        """Render the view to the console."""
        pass

    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """Return structured data for machine-readable output.

        Returns:
            Dictionary with view-specific data structure suitable for
            JSON serialization or CSV export.
        """
        pass
