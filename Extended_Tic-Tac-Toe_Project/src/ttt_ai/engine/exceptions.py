"""Custom exception hierarchy for the Extended Tic-Tac-Toe (`ttt_ai`) package."""

from __future__ import annotations


class TicTacToeError(Exception):
    """Base exception for all domain and configuration errors in `ttt_ai`."""


class InvalidConfigError(TicTacToeError):
    """Raised when board parameters, seeds, or YAML configurations are invalid."""


class IllegalMoveError(TicTacToeError):
    """Raised when an action violates game rules or state transition invariants."""
