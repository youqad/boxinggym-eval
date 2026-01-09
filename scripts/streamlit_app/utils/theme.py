"""Dark theme CSS injection for BoxingGym dashboard."""

import streamlit as st

# Color constants (matching Observatory theme)
COLORS = {
    "bg_base": "#080c14",
    "bg_elevated": "#0a1628",
    "bg_surface": "rgba(14, 26, 47, 0.85)",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_tertiary": "#64748b",
    "accent_primary": "#f59e0b",
    "accent_secondary": "#14b8a6",
    "positive": "#34d399",
    "negative": "#fb7185",
    "neutral": "#fbbf24",
    "border_subtle": "rgba(148, 163, 184, 0.12)",
}


def inject_custom_css():
    """Inject custom CSS for dark theme styling."""
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Base typography */
    .stApp {
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Crimson Pro', Georgia, serif !important;
        font-weight: 600 !important;
    }

    /* Metric cards styling */
    [data-testid="stMetric"] {
        background: rgba(14, 26, 47, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 8px;
        padding: 1rem;
    }

    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #f59e0b !important;
    }

    /* DataFrame styling */
    .stDataFrame {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #0a1628;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f59e0b !important;
    }

    /* Positive/negative value colors */
    .val-positive {
        color: #34d399 !important;
        font-weight: 600;
    }

    .val-negative {
        color: #fb7185 !important;
        font-weight: 600;
    }

    .val-neutral {
        color: #fbbf24 !important;
    }

    /* Chart container */
    .chart-container {
        background: rgba(14, 26, 47, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #080c14;
    }

    ::-webkit-scrollbar-thumb {
        background: #64748b;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def style_metric_delta(delta: float) -> str:
    """Return CSS class for delta value."""
    if delta <= -0.1:
        return "val-positive"
    elif delta >= 0.1:
        return "val-negative"
    return "val-neutral"


def format_delta_html(value: float) -> str:
    """Format delta value with color styling."""
    css_class = style_metric_delta(value)
    sign = "+" if value > 0 else ""
    return f'<span class="{css_class}">{sign}{value:.3f}</span>'
