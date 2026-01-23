"""Hugging Face Space Entry Point for BoxingGym Dashboard.

This file serves as the entry point for the HF Space.
It launches the Streamlit dashboard.
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Set default environment variables for HF Space
os.environ.setdefault("WANDB_ENTITY", os.getenv("WANDB_ENTITY", ""))
os.environ.setdefault("WANDB_PROJECT", os.getenv("WANDB_PROJECT", "boxing-gym"))
os.environ.setdefault("BOXING_GYM_DEMO_MODE", "1")

# Run the Streamlit app
if __name__ == "__main__":
    import subprocess
    subprocess.run([
        "streamlit", "run",
        "scripts/streamlit_app/app.py",
        "--server.headless", "true",
        "--server.port", "7860",  # HF Spaces default port
        "--server.address", "0.0.0.0",
    ])
