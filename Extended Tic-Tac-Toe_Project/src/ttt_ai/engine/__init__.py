"""Game engine package for Extended Tic-Tac-Toe (`M x N, K`).

This package must stay pure Python + NumPy with zero dependencies on `torch`,
`rich`, `matplotlib`, or higher-level agent/training modules.
"""

from __future__ import annotations

from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError, TicTacToeError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.engine.rules import (
    DIRECTIONS,
    EMPTY,
    PLAYER_O,
    PLAYER_X,
    Player,
    action_to_coords,
    check_win_around,
    coords_to_action,
    encode_canonical_board,
    positional_weights,
    precompute_win_windows,
    win_windows,
)

__all__ = [
    "DIRECTIONS",
    "EMPTY",
    "PLAYER_O",
    "PLAYER_X",
    "Player",
    "ExtendedTicTacToe",
    "TicTacToeError",
    "InvalidConfigError",
    "IllegalMoveError",
    "action_to_coords",
    "coords_to_action",
    "check_win_around",
    "precompute_win_windows",
    "win_windows",
    "positional_weights",
    "encode_canonical_board",
]
