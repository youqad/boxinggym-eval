"""Parameter Importance view for TUI."""

from typing import Optional

import numpy as np
import pandas as pd
from rich.table import Table

from ..components.ascii_charts import horizontal_bar
from . import BaseView


class ParameterImportanceView(BaseView):
    def __init__(self, df: pd.DataFrame, console, metric: str, include_seed: bool = False):
        super().__init__(df, console, metric)
        self.include_seed = include_seed

    @property
    def title(self) -> str:
        return "Parameter Importance"

    def _compute_importance(self) -> pd.DataFrame:
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            self.console.print("[yellow]sklearn not available for importance analysis[/yellow]")
            return pd.DataFrame()

        if self.metric not in self.df.columns:
            return pd.DataFrame()

        # filter config columns
        # 'seed' excluded by default: it determines env instance/difficulty via np.random.seed().
        # With fixed normalization, seed acts as dataset-ID. Including it causes RF
        # to overfit to instance difficulties (test-set leakage) rather than learning
        # generalizable hyperparameter effects. See deliberation in commit history.
        # Use --include-seed flag to enable for diagnostic purposes.
        # note: avoid bare "_" which would exclude include_prior, use_ppl, etc.
        noise_patterns = [
            "results",
            "ppl/",
            "ppl.",
            "hydra",
            "filename",
            "system_prompt",
            "wandb",
            "_wandb",
            "_runtime",
            "_step",
            "_timestamp",
        ]
        if not self.include_seed:
            noise_patterns.append("seed")

        config_cols = [
            c
            for c in self.df.columns
            if c.startswith("config/") and not any(noise in c.lower() for noise in noise_patterns)
        ]

        if not config_cols:
            return pd.DataFrame()

        X_data = self.df[config_cols].copy()
        y = self.df[self.metric].values

        valid_mask = ~np.isnan(y)
        X_data = X_data[valid_mask]
        y = y[valid_mask]

        if len(y) < 10:
            return pd.DataFrame()

        X_encoded = pd.DataFrame()
        for col in X_data.columns:
            if X_data[col].dtype == object or X_data[col].dtype.name == "category":
                le = LabelEncoder()
                values = X_data[col].fillna("__NULL__").astype(str)
                X_encoded[col] = le.fit_transform(values)
            else:
                X_encoded[col] = X_data[col].fillna(0)

        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_encoded, y)

        perm_result = permutation_importance(
            rf, X_encoded, y, n_repeats=10, random_state=42, n_jobs=-1
        )

        importance = pd.DataFrame(
            {
                "parameter": config_cols,
                "importance": perm_result.importances_mean,
                "std": perm_result.importances_std,
            }
        ).sort_values("importance", ascending=False)

        correlations = []
        for col in config_cols:
            try:
                if X_encoded[col].nunique() > 1 and len(y) > 1:
                    corr = np.corrcoef(X_encoded[col].values.astype(float), y.astype(float))[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0
            except Exception:
                corr = 0.0
            correlations.append(corr)

        importance["correlation"] = correlations
        return importance

    def render(self) -> None:
        importance = self._compute_importance()

        if importance.empty:
            self.console.print("[yellow]Not enough data for parameter importance analysis[/yellow]")
            return

        max_imp = importance["importance"].max()

        table = Table(
            title="Top Parameters by Permutation Importance",
            border_style="cyan",
            header_style="bold magenta",
        )
        table.add_column("Parameter", style="green", width=30)
        table.add_column("Importance", justify="right", width=16)
        table.add_column("Bar", width=12)
        table.add_column("Correlation", justify="right", width=12)

        for _, row in importance.head(15).iterrows():
            param = row["parameter"].replace("config/", "")
            imp = row["importance"]
            std = row["std"]
            corr = row["correlation"]

            bar = horizontal_bar(imp, max_imp, width=10)

            if corr > 0.2:
                corr_str = f"[green]{corr:+.3f}[/green]"
            elif corr < -0.2:
                corr_str = f"[red]{corr:+.3f}[/red]"
            else:
                corr_str = f"[yellow]{corr:+.3f}[/yellow]"

            table.add_row(param[:30], f"{imp:.3f} ± {std:.3f}", bar, corr_str)

        self.console.print(table)

        # interpretation note
        self.console.print(
            "\n[dim]Method: Permutation importance (model-agnostic, less biased than MDI)[/dim]"
        )
        self.console.print(
            "[dim]Note: Positive correlation = higher value increases z_mean (worse)[/dim]"
        )
        self.console.print(
            "[dim]      Negative correlation = higher value decreases z_mean (better)[/dim]"
        )

    def get_data(self) -> dict:
        importance = self._compute_importance()

        if importance.empty:
            return {"parameters": []}

        return {"parameters": importance.to_dict("records")}

    def get_csv_rows(self) -> list:
        data = self.get_data()
        params = data.get("parameters", [])
        if not params:
            return []

        rows = [["parameter", "importance", "std", "correlation"]]
        for p in params:
            rows.append(
                [
                    p.get("parameter", ""),
                    p.get("importance", ""),
                    p.get("std", ""),
                    p.get("correlation", ""),
                ]
            )
        return rows

    def to_plotly(self) -> Optional["plotly.graph_objects.Figure"]:  # noqa: F821
        """Return Plotly horizontal bar chart of parameter importance."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        importance = self._compute_importance()
        if importance.empty:
            return None

        top = importance.head(15).iloc[::-1]

        colors = []
        for corr in top["correlation"]:
            if corr > 0.2:
                colors.append("#ef4444")  # red - higher value = worse
            elif corr < -0.2:
                colors.append("#22c55e")  # green - higher value = better
            else:
                colors.append("#eab308")  # yellow - neutral

        params = [p.replace("config/", "") for p in top["parameter"]]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=params,
                x=top["importance"],
                error_x=dict(type="data", array=top["std"], visible=True),
                orientation="h",
                marker_color=colors,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Importance: %{x:.4f}<br>"
                    "Correlation: %{customdata:+.3f}<extra></extra>"
                ),
                customdata=top["correlation"],
            )
        )

        fig.update_layout(
            title="Parameter Importance (Permutation-based)",
            xaxis_title="Importance",
            yaxis_title="Parameter",
            height=max(400, len(params) * 30),
            width=800,
            showlegend=False,
        )

        # add annotation for color meaning
        fig.add_annotation(
            text="🟢 negative corr (better) | 🟡 neutral | 🔴 positive corr (worse)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.12,
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

        return fig
