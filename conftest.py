"""
Pytest configuration — CardioVision AI.

Adds both backend and ML source directories to sys.path so all test
modules can import from backend and ML source code.

IMPORTANT: Run tests with the system Python (Python 3.14) which has all
ML packages installed:
    python -m pytest tests/ -v
or via pytest.ini testpaths.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Backend imports (FastAPI app, model_service, etc.)
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# ML source imports (feature_config, preprocessing, etc.)
ML_SRC_DIR = ROOT / "ml" / "src"
sys.path.insert(0, str(ML_SRC_DIR))
