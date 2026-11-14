"""Core game state machine and unified API for Extended Tic-Tac-Toe (`M x N, K`).

This module implements `ExtendedTicTacToe` using pure Python and NumPy only,
keeping the engine isolated from neural-network, search, and UI layers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError
from ttt_ai.engine.rules import (
    EMPTY,
    PLAYER_O,
    PLAYER_X,
    Player,
    action_to_coords,
    check_win_around,
    encode_canonical_board,
    precompute_win_windows,
)


class ExtendedTicTacToe:
    """Deterministic two-player zero-sum `M x N, K` Tic-Tac-Toe environment.

    Follows the unified 7+1 Game API (`reset`, `legal_actions`, `step`,
    `is_terminal`, `result`, `render`, `clone`, `undo`) and stores move history
    on an undo stack so tree-search agents can explore via `step()`/`undo()` in
    `O(1)` time per transition without repeated allocations.
    """

    def __init__(self, rows: int = 4, cols: int = 4, win_k: int = 4) -> None:
        """Initialize a new game board with validated `(rows, cols, win_k)` rules.

        Args:
            rows: Number of rows on the board (`rows >= 3`).
            cols: Number of columns on the board (`cols >= 3`).
            win_k: Consecutive marks needed to win (`3 <= win_k <= min(rows, cols)`).

        Raises:
            InvalidConfigError: If any dimension or `win_k` violates board rules.
        """
        # Delegate validation to `precompute_win_windows` so all board rules stay
        # in one place (DRY) while warming the cached window geometry early.
        precompute_win_windows(rows, cols, win_k)

        self._rows: int = int(rows)
        self._cols: int = int(cols)
        self._win_k: int = int(win_k)
        self._board: np.ndarray = np.zeros((self._rows, self._cols), dtype=np.int8)
        self._current_player: Player = Player.X
        self._history: list[tuple[int, Player]] = []
        self._winner: Player | None = None
        self._winning_line: tuple[tuple[int, int], ...] | None = None
        self._terminal: bool = False

    @property
    def rows(self) -> int:
        """Number of rows (`M`) on the board."""
        return self._rows

    @property
    def cols(self) -> int:
        """Number of columns (`N`) on the board."""
        return self._cols

    @property
    def win_k(self) -> int:
        """Number of consecutive marks (`K`) required to win."""
        return self._win_k

    @property
    def num_cells(self) -> int:
        """Total number of cells (`M * N`) on the board."""
        return self._rows * self._cols

    @property
    def shape(self) -> tuple[int, int]:
        """Board dimensions as `(rows, cols)`."""
        return (self._rows, self._cols)

    @property
    def current_player(self) -> Player:
        """Player whose turn it is to move (`Player.X` or `Player.O`)."""
        return self._current_player

    @property
    def board(self) -> np.ndarray:
        """Defensive copy of the 2D board array (`dtype=np.int8`)."""
        # Returning a copy prevents external callers from mutating engine state.
        return self._board.copy()

    @property
    def winner(self) -> Player | None:
        """Winning player if the game ended in a win, else `None`."""
        return self._winner

    @property
    def winning_line(self) -> tuple[tuple[int, int], ...] | None:
        """Tuple of `win_k` winning `(row, col)` coordinates, or `None`."""
        return self._winning_line

    @property
    def move_history(self) -> tuple[int, ...]:
        """Chronological tuple of flat action indices played so far."""
        return tuple(action for action, _ in self._history)

    @property
    def last_move(self) -> int | None:
        """Most recent flat action played, or `None` if no moves have occurred."""
        return self._history[-1][0] if self._history else None

    @property
    def move_count(self) -> int:
        """Total number of moves currently on the board."""
        return len(self._history)

    def canonical_board(self) -> np.ndarray:
        """Return the board encoded from `current_player`'s perspective.

        Returns:
            Fresh 2D `np.int8` array where `+1` is the active player's mark,
            `-1` is the opponent's mark, and `0` is empty.
        """
        return encode_canonical_board(self._board, int(self._current_player))

    def reset(self) -> np.ndarray:
        """Clear the board, reset turn to `Player.X`, and return initial state.

        Returns:
            Fresh 2D `np.int8` copy of the cleared `(rows, cols)` board.
        """
        self._board.fill(EMPTY)
        self._current_player = Player.X
        self._history.clear()
        self._winner = None
        self._winning_line = None
        self._terminal = False
        return self._board.copy()

    def legal_actions(self) -> list[int]:
        """Return sorted flat indices of all valid moves in the current state.

        Returns:
            Ascending list of empty cell indices `a in [0, M*N - 1]`, or `[]`
            if the game has already reached a terminal state.
        """
        if self._terminal:
            return []
        return [int(idx) for idx in np.flatnonzero(self._board == EMPTY)]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply a move for `current_player` and advance the game state.

        Args:
            action: Flat row-major cell index in `[0, rows * cols - 1]`.

        Returns:
            Tuple `(board_copy, reward, done, info)` where `reward` is `1.0` if
            the acting player won on this move and `0.0` otherwise, `done` equals
            `is_terminal()`, and `info` summarizes move and outcome metadata.

        Raises:
            IllegalMoveError: If the game is already terminal, `action` is not a
                valid integer index within bounds, or the target cell is occupied.
        """
        if self._terminal:
            raise IllegalMoveError("Cannot call step() after the game has ended.")
        if not isinstance(action, (int, np.integer)) or isinstance(action, bool):
            raise IllegalMoveError(f"action must be an integer, got {action!r}.")

        action_idx = int(action)
        if not (0 <= action_idx < self.num_cells):
            raise IllegalMoveError(
                f"action {action_idx} out of bounds for [0, {self.num_cells - 1}]."
            )

        row, col = action_to_coords(action_idx, self._cols)
        if int(self._board[row, col]) != EMPTY:
            raise IllegalMoveError(
                f"Cell ({row}, {col}) [action={action_idx}] is already occupied."
            )

        acting_player = self._current_player
        self._board[row, col] = int(acting_player)
        self._history.append((action_idx, acting_player))

        win_coords = check_win_around(
            self._board,
            row,
            col,
            int(acting_player),
            self._win_k,
        )
        if win_coords is not None:
            self._winner = acting_player
            self._winning_line = tuple(win_coords)
            self._terminal = True
            reward = 1.0
        elif len(self._history) == self.num_cells:
            self._winner = None
            self._winning_line = None
            self._terminal = True
            reward = 0.0
        else:
            self._current_player = acting_player.opponent
            reward = 0.0

        info: dict[str, Any] = {
            "player": int(acting_player),
            "current_player": int(self._current_player),
            "winner": int(self._winner) if self._winner is not None else None,
            "winning_line": list(self._winning_line) if self._winning_line is not None else None,
            "move_count": len(self._history),
        }
        return self._board.copy(), reward, self._terminal, info

    def is_terminal(self) -> bool:
        """Return True if the game has ended in a win or a full-board draw."""
        return self._terminal

    def result(self, player: int | None = None) -> float | None:
        """Return the zero-sum terminal payoff from `player`'s perspective.

        Args:
            player: Perspective player (`+1` for `X`, `-1` for `O`). Defaults to
                `Player.X (+1)` when `None`.

        Returns:
            `None` if non-terminal, `0.0` on draw, `+1.0` if `player` won, or
            `-1.0` if the opponent won.

        Raises:
            InvalidConfigError: If `player` is provided and not `+1` or `-1`.
        """
        if player is None:
            target_player = PLAYER_X
        else:
            if (
                not isinstance(player, (int, np.integer))
                or isinstance(player, bool)
                or int(player) not in (PLAYER_X, PLAYER_O)
            ):
                raise InvalidConfigError(
                    f"player must be {PLAYER_X} (X), {PLAYER_O} (O), or None, got {player!r}."
                )
            target_player = int(player)

        if not self._terminal:
            return None
        if self._winner is None:
            return 0.0
        return 1.0 if int(self._winner) == target_player else -1.0

    def render(self) -> str:
        """Render the board as a readable ASCII grid with coordinates and win markers.

        Returns:
            Multi-line ASCII string showing board rules, turn/result status,
            row/column indices, and `[X]`/`[O]` highlights on winning cells.
        """
        win_cells = set(self._winning_line) if self._winning_line is not None else set()

        if self._terminal:
            if self._winner is not None:
                status = f"Result: Player {self._winner.symbol} wins!"
            else:
                status = "Result: Draw"
        else:
            status = f"Turn: Player {self._current_player.symbol} (move #{self.move_count})"

        lines = [
            f"Board: {self._rows}x{self._cols} (K={self._win_k}) | {status}",
            "    " + " ".join(f" {col} " for col in range(self._cols)),
            "   +" + "---+" * self._cols,
        ]
        for row in range(self._rows):
            row_cells: list[str] = []
            for col in range(self._cols):
                val = int(self._board[row, col])
                if val == PLAYER_X:
                    token = "[X]" if (row, col) in win_cells else " X "
                elif val == PLAYER_O:
                    token = "[O]" if (row, col) in win_cells else " O "
                else:
                    token = " . "
                row_cells.append(token)
            lines.append(f"{row:>2} |" + "|".join(row_cells) + "|")
            lines.append("   +" + "---+" * self._cols)

        if self._winning_line is not None:
            lines.append(f"Winning line: {list(self._winning_line)}")

        return "\n".join(lines)

    def clone(self) -> ExtendedTicTacToe:
        """Create an independent deep copy of the game state for search trees.

        Returns:
            New `ExtendedTicTacToe` instance sharing no mutable buffers with `self`.
        """
        cloned = ExtendedTicTacToe.__new__(ExtendedTicTacToe)
        cloned._rows = self._rows
        cloned._cols = self._cols
        cloned._win_k = self._win_k
        cloned._board = self._board.copy()
        cloned._current_player = self._current_player
        cloned._history = list(self._history)
        cloned._winner = self._winner
        cloned._winning_line = self._winning_line
        cloned._terminal = self._terminal
        return cloned

    def undo(self) -> None:
        """Revert the most recent move in `O(1)` time without re-scanning the board.

        Raises:
            IllegalMoveError: If called when no moves have been played.
        """
        if not self._history:
            raise IllegalMoveError("Cannot call undo() when move history is empty.")

        last_action, prev_player = self._history.pop()
        row, col = divmod(last_action, self._cols)
        self._board[row, col] = EMPTY

        # Because a game terminates immediately upon a win or full board, the state
        # directly preceding the last move is guaranteed to be non-terminal with no winner.
        self._current_player = prev_player
        self._winner = None
        self._winning_line = None
        self._terminal = False
