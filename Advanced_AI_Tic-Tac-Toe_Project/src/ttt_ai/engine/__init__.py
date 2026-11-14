"""Game engine package for Extended Tic-Tac-Toe (`M x N, K`).

This package must stay pure Python + NumPy with zero dependencies on `torch`,
`rich`, `matplotlib`, or higher-level agent/training modules.
"""

from __future__ import annotations

from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError, TicTacToeError

__all__ = [
    "TicTacToeError",
    "InvalidConfigError",
    "IllegalMoveError",
]
