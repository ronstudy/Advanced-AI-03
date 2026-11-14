"""Shared game rules, win detection, coordinate helpers, and board encoding for `M x N, K`.

This module stays strictly pure Python + NumPy so the game engine has zero
dependencies on neural-network or UI libraries.
"""

from __future__ import annotations

from enum import IntEnum
from functools import lru_cache
from typing import Any

import numpy as np

from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError

EMPTY: int = 0
PLAYER_X: int = 1
PLAYER_O: int = -1

# Four undirected scan axes: horizontal, vertical, main diagonal, anti-diagonal.
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)


class Player(IntEnum):
    """Canonical integer identifiers for the two alternating players."""

    X = PLAYER_X
    O = PLAYER_O  # noqa: E741

    @property
    def opponent(self) -> Player:
        """Return the opposing player (`Player.O` for `X`, `Player.X` for `O`)."""
        return Player(-self.value)

    @property
    def symbol(self) -> str:
        """Return the single-character ASCII marker (`'X'` or `'O'`)."""
        return "X" if self is Player.X else "O"


def _is_strict_int(value: Any) -> bool:
    """Return True if `value` is an integer and not a boolean."""
    # Reject booleans explicitly because `bool` subclasses `int` in Python.
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_board_dims(rows: int, cols: int, win_k: int) -> None:
    """Validate `(rows, cols, win_k)` dimensions for an `M x N, K` board."""
    if not (_is_strict_int(rows) and _is_strict_int(cols) and _is_strict_int(win_k)):
        raise InvalidConfigError(
            "Board dimensions and win_k must be integers, "
            f"got rows={rows!r}, cols={cols!r}, win_k={win_k!r}."
        )
    if rows < 3 or cols < 3:
        raise InvalidConfigError(
            f"Board dimensions must be at least 3x3, got {rows}x{cols}."
        )
    max_valid_k = min(rows, cols)
    if win_k < 3 or win_k > max_valid_k:
        raise InvalidConfigError(
            f"win_k must satisfy 3 <= win_k <= min(rows, cols) ({max_valid_k}), "
            f"got {win_k}."
        )


def action_to_coords(action: int, cols: int) -> tuple[int, int]:
    """Convert a flat cell index into `(row, col)` coordinates.

    Args:
        action: Non-negative flat board index in row-major order.
        cols: Number of columns on the board (`cols >= 3`).

    Returns:
        Tuple `(row, col)` corresponding to `divmod(action, cols)`.

    Raises:
        InvalidConfigError: If `cols` is not an integer `>= 3`.
        IllegalMoveError: If `action` is not a non-negative integer.
    """
    if not _is_strict_int(cols) or cols < 3:
        raise InvalidConfigError(f"cols must be an integer >= 3, got {cols!r}.")
    if not _is_strict_int(action) or action < 0:
        raise IllegalMoveError(f"action must be a non-negative integer, got {action!r}.")
    return divmod(action, cols)


def coords_to_action(row: int, col: int, cols: int) -> int:
    """Convert `(row, col)` board coordinates into a flat row-major cell index.

    Args:
        row: Non-negative row index.
        col: Column index in `[0, cols)`.
        cols: Number of columns on the board (`cols >= 3`).

    Returns:
        Flat integer cell index `row * cols + col`.

    Raises:
        InvalidConfigError: If `cols` is not an integer `>= 3`.
        IllegalMoveError: If `row` or `col` is invalid or out of column bounds.
    """
    if not _is_strict_int(cols) or cols < 3:
        raise InvalidConfigError(f"cols must be an integer >= 3, got {cols!r}.")
    if not (_is_strict_int(row) and _is_strict_int(col)) or row < 0 or not (0 <= col < cols):
        raise IllegalMoveError(
            f"Invalid coordinates ({row!r}, {col!r}) for board with {cols} columns."
        )
    return row * cols + col


def check_win_around(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    win_k: int,
) -> list[tuple[int, int]] | None:
    """Detect a `win_k`-in-a-row line passing through `(row, col)` in `O(K)` time.

    Scanning only outward along the four axes centered on the last played cell
    avoids full-board `O(M * N)` scans during high-frequency Alpha-Beta / MCTS
    simulations.

    Args:
        board: 2D NumPy array with values in `{-1, 0, +1}`.
        row: Row index of the last placed mark.
        col: Column index of the last placed mark.
        player: Player mark (`+1` for `X`, `-1` for `O`) to check.
        win_k: Consecutive marks required to win (`3 <= win_k <= min(rows, cols)`).

    Returns:
        Ordered list of `win_k` winning `(row, col)` coordinates containing
        `(row, col)` if `player` achieved a line of length `>= win_k`, else `None`.

    Raises:
        InvalidConfigError: If `board` is not 2D, dimensions/win_k are invalid,
            or `player` is not `+1` or `-1`.
        IllegalMoveError: If `(row, col)` lies outside the board boundaries.
    """
    if not isinstance(board, np.ndarray) or board.ndim != 2:
        raise InvalidConfigError("board must be a 2D numpy.ndarray.")

    rows, cols = board.shape
    _validate_board_dims(rows, cols, win_k)

    if not _is_strict_int(player) or player not in (PLAYER_X, PLAYER_O):
        raise InvalidConfigError(
            f"player must be {PLAYER_X} (X) or {PLAYER_O} (O), got {player!r}."
        )
    if (
        not (_is_strict_int(row) and _is_strict_int(col))
        or not (0 <= row < rows and 0 <= col < cols)
    ):
        raise IllegalMoveError(
            f"Coordinates ({row!r}, {col!r}) out of bounds for board shape {board.shape}."
        )

    target = int(player)
    if int(board[row, col]) != target:
        return None

    for dr, dc in DIRECTIONS:
        backward: list[tuple[int, int]] = []
        for step in range(1, win_k):
            nr = row - dr * step
            nc = col - dc * step
            if 0 <= nr < rows and 0 <= nc < cols and int(board[nr, nc]) == target:
                backward.append((nr, nc))
            else:
                break

        forward: list[tuple[int, int]] = []
        for step in range(1, win_k):
            nr = row + dr * step
            nc = col + dc * step
            if 0 <= nr < rows and 0 <= nc < cols and int(board[nr, nc]) == target:
                forward.append((nr, nc))
            else:
                break

        line = list(reversed(backward)) + [(row, col)] + forward
        if len(line) >= win_k:
            # Because `len(backward) <= win_k - 1`, `(row, col)` sits at index
            # `<= win_k - 1` and is guaranteed to be included in `line[:win_k]`.
            return line[:win_k]

    return None


@lru_cache(maxsize=64)
def precompute_win_windows(
    rows: int,
    cols: int,
    win_k: int,
) -> tuple[tuple[int, ...], ...]:
    """Precompute every contiguous `win_k`-cell winning line as flat action indices.

    Cached per `(rows, cols, win_k)` so heuristic evaluators and rule-based
    agents reuse the exact same static window geometry without recomputation.

    Args:
        rows: Number of board rows (`>= 3`).
        cols: Number of board columns (`>= 3`).
        win_k: Consecutive marks required to win (`3 <= win_k <= min(rows, cols)`).

    Returns:
        Immutable tuple of windows, where each window is a tuple of `win_k`
        row-major flat cell indices.

    Raises:
        InvalidConfigError: If `(rows, cols, win_k)` violates board constraints.
    """
    _validate_board_dims(rows, cols, win_k)
    windows: list[tuple[int, ...]] = []

    # Horizontal windows (dr=0, dc=1).
    for r in range(rows):
        for c in range(cols - win_k + 1):
            windows.append(tuple(r * cols + (c + offset) for offset in range(win_k)))

    # Vertical windows (dr=1, dc=0).
    for r in range(rows - win_k + 1):
        for c in range(cols):
            windows.append(tuple((r + offset) * cols + c for offset in range(win_k)))

    # Main diagonal windows (dr=1, dc=1).
    for r in range(rows - win_k + 1):
        for c in range(cols - win_k + 1):
            windows.append(
                tuple((r + offset) * cols + (c + offset) for offset in range(win_k))
            )

    # Anti-diagonal windows (dr=1, dc=-1).
    for r in range(rows - win_k + 1):
        for c in range(win_k - 1, cols):
            windows.append(
                tuple((r + offset) * cols + (c - offset) for offset in range(win_k))
            )

    return tuple(windows)


# Alias matching ARCHITECTURE.md directory listing (`win_windows()`).
win_windows = precompute_win_windows


def positional_weights(
    rows: int,
    cols: int,
    win_k: int,
    normalize: bool = False,
) -> np.ndarray:
    """Compute strategic cell weights by counting how many win windows cross each cell.

    Cells traversed by more `win_k` lines offer higher tactical leverage (e.g.,
    center > corners > edges on `3x3, K=3`).

    Args:
        rows: Number of board rows (`>= 3`).
        cols: Number of board columns (`>= 3`).
        win_k: Consecutive marks required to win (`3 <= win_k <= min(rows, cols)`).
        normalize: If True, scales weights to `[0.0, 1.0]` as `float32` divided by
            the maximum cell weight; otherwise returns raw `int32` window counts.

    Returns:
        2D NumPy array of shape `(rows, cols)` with strategic cell weights.

    Raises:
        InvalidConfigError: If `(rows, cols, win_k)` violates board constraints.
    """
    windows = precompute_win_windows(rows, cols, win_k)
    counts = np.zeros(rows * cols, dtype=np.int32)
    for window in windows:
        for cell_idx in window:
            counts[cell_idx] += 1

    grid = counts.reshape((rows, cols))
    if not normalize:
        return grid

    max_weight = int(grid.max())
    return (grid.astype(np.float32) / np.float32(max_weight)).astype(np.float32)


def encode_canonical_board(board: np.ndarray, current_player: int) -> np.ndarray:
    """Transform a board state into the current player's canonical perspective.

    Multiplying by `current_player` maps friendly pieces to `+1`, opponent pieces
    to `-1`, and empty squares to `0`, allowing neural networks and Q-tables to
    learn a single player-invariant representation (reference: course slide 38).

    Args:
        board: 2D NumPy array representing the board with values in `{-1, 0, +1}`.
        current_player: Active player to move (`+1` for `X`, `-1` for `O`).

    Returns:
        Fresh 2D `np.int8` array of shape `board.shape` from `current_player`'s
        perspective without mutating the input array.

    Raises:
        InvalidConfigError: If `board` is not 2D or `current_player` is invalid.
    """
    if not isinstance(board, np.ndarray) or board.ndim != 2:
        raise InvalidConfigError("board must be a 2D numpy.ndarray.")
    if not _is_strict_int(current_player) or current_player not in (PLAYER_X, PLAYER_O):
        raise InvalidConfigError(
            f"current_player must be {PLAYER_X} (X) or {PLAYER_O} (O), got {current_player!r}."
        )
    return (np.array(board, dtype=np.int8, copy=True) * np.int8(current_player)).astype(
        np.int8,
        copy=False,
    )
