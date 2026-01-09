import os
import sys
import tempfile
from pathlib import Path


# Ensure the package is importable when running pytest without installation.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# Point PyTensor compiledir to a writable location to avoid permission errors.
compiledir = Path(tempfile.gettempdir()) / "pytensor_compiledir"
compiledir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={compiledir}")

# Force fake LLM responses during tests to avoid real network calls / API keys.
os.environ.setdefault("BOXINGGYM_FAKE_LLM", "1")
