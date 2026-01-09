"""BoxingGym Dashboard - Main Entry Point.

Run with:
    streamlit run scripts/streamlit_app/app.py

Or via CLI:
    uv run python scripts/analyze_sweep_results.py --sweep-id <ID> --web
"""

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
st.markdown(
    """
Sidebar pages:

- **Benchmark Dashboard**: Compare results against paper baselines
- **Sweep Analysis**: Interactive W&B sweep charts

---

### Quick Start

```bash
# web dashboard
uv run python scripts/analyze_sweep_results.py --sweep-id <SWEEP_ID> --web

# CLI analysis (no web)
uv run python scripts/analyze_sweep_results.py --sweep-id <SWEEP_ID>
```

```bash
streamlit run scripts/streamlit_app/app.py
```
"""
)

# Show environment info if sweep IDs are set
import os

sweep_ids = os.environ.get("SWEEP_IDS", "")
if sweep_ids:
    st.info(f"🎯 Sweep IDs from CLI: `{sweep_ids}`")
    st.markdown("Navigate to **Sweep Analysis** to view results.")
