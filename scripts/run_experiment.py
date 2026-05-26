"""Thin shim so `python scripts/run_experiment.py --experiment baseline` works
from a checkout without `pip install -e .`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from price_model.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
