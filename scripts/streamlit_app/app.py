"""BoxingGym Dashboard - Main Entry Point.

Run with:
    streamlit run scripts/streamlit_app/app.py

Or via CLI:
    uv run python scripts/analyze_sweep_results.py --sweep-id <ID> --web
"""

import os

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="BoxingGym Observatory",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.theme import inject_custom_css

# Inject custom CSS
inject_custom_css()

# Main page content
st.title("🔬 BoxingGym Observatory")

st.markdown("""
Benchmark for evaluating LLM agents on experimental design and model discovery.

Agents interact with simulated environments (predator-prey dynamics, human decision-making, etc.) to infer underlying models through iterative experiments.
""")

with st.expander("Metrics"):
    st.markdown("""
**z_mean:** Prediction error in standard deviations (lower is better). z=0 is perfect, z=-1 beats baseline, z=+2 is way off.

**budget:** Number of experiments allowed (0 = zero-shot guess, 10 = run experiments first)

**environment:** Simulated domain (`hyperbolic_temporal_discount` = human patience, `lotka_volterra` = predator-prey)
    """)

st.markdown("---")

st.markdown("""
**Sidebar pages:**
- **Leaderboard**: Model rankings with bootstrap CIs and significance testing
- **Benchmark Dashboard**: Compare results against paper baselines
- **Sweep Analysis**: Interactive W&B sweep charts
- **Paper Comparison**: Replicate paper figures with your data
- **PPL Examples**: View generated probabilistic programs
- **LLM Call Logs**: Cost and latency tracking
""")

with st.expander("Usage"):
    st.code("""
# web dashboard
uv run python scripts/analyze_sweep_results.py --sweep-id <SWEEP_ID> --web

# CLI analysis (no web)
uv run python scripts/analyze_sweep_results.py --sweep-id <SWEEP_ID>

# or run streamlit directly
streamlit run scripts/streamlit_app/app.py
    """, language="bash")

st.markdown("---")

st.markdown("""
### Credits

Dashboard for [**BoxingGym**](https://github.com/kanishkg/boxing-gym) (Gandhi et al., Stanford).

**Paper:** [arXiv:2501.01540](https://arxiv.org/abs/2501.01540)
**Original:** [kanishkg/boxing-gym](https://github.com/kanishkg/boxing-gym)
**Fork:** [youqad/boxing-gym](https://github.com/youqad/boxing-gym)
""")

# Show environment info if sweep IDs are set
sweep_ids = os.environ.get("SWEEP_IDS", "")
if sweep_ids:
    st.info(f"🎯 Sweep IDs from CLI: `{sweep_ids}`")
    st.markdown("Navigate to **Sweep Analysis** to view results.")
