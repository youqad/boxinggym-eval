"""TUI view base class."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from rich.console import Console

__all__ = ["BaseView"]


class BaseView(ABC):
    def __init__(self, df: pd.DataFrame, console: Console, metric: str = "metric/eval/z_mean"):
        self.df = df
        self.console = console
        self.metric = metric

    @property
    @abstractmethod
    def title(self) -> str:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def get_data(self) -> dict[str, Any]:
        """Return structured data for JSON/CSV export."""
        pass

    def get_csv_rows(self) -> list[list[str]]:
        """Override to provide CSV rows. First row should be headers."""
        return []

    def to_plotly(self) -> Optional["plotly.graph_objects.Figure"]:  # noqa: F821
        """Override to provide Plotly figure. Returns None by default."""
        return None
