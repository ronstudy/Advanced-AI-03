"""Pytest bootstrap ensuring `src/` is importable regardless of working directory."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
