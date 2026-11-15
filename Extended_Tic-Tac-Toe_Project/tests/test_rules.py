"""Unit tests for Task T1.2: exceptions, coordinate helpers, and shared game rules."""

from __future__ import annotations

import numpy as np
import pytest

from ttt_ai.engine import (
    EMPTY,
    PLAYER_O,
    PLAYER_X,
    IllegalMoveError,
    InvalidConfigError,
    Player,
    TicTacToeError,
    action_to_coords,
    check_win_around,
    coords_to_action,
    encode_canonical_board,
    positional_weights,
    precompute_win_windows,
    win_windows,
)


def test_exception_hierarchy_and_player_enum() -> None:
    """Verify domain exception inheritance and Player enum properties."""
    assert issubclass(InvalidConfigError, TicTacToeError)
    assert issubclass(IllegalMoveError, TicTacToeError)
    assert int(Player.X) == PLAYER_X == 1
    assert int(Player.O) == PLAYER_O == -1
    assert Player.X.opponent is Player.O
    assert Player.O.opponent is Player.X
    assert Player.X.symbol == "X"
    assert Player.O.symbol == "O"
    assert EMPTY == 0


def test_coordinate_conversion_round_trip_and_errors() -> None:
    """Verify flat action <-> (row, col) conversions and boundary errors."""
    rows, cols = 4, 5
    for r in range(rows):
        for c in range(cols):
            action = coords_to_action(r, c, cols)
            assert action == r * cols + c
            assert action_to_coords(action, cols) == (r, c)

    with pytest.raises(InvalidConfigError):
        action_to_coords(0, cols=2)
    with pytest.raises(InvalidConfigError):
        coords_to_action(0, 0, cols=2)
    with pytest.raises(IllegalMoveError):
        action_to_coords(-1, cols=4)
    with pytest.raises(IllegalMoveError):
        action_to_coords(True, cols=4)  # type: ignore[arg-type]
    with pytest.raises(IllegalMoveError):
        coords_to_action(0, 4, cols=4)
    with pytest.raises(IllegalMoveError):
        coords_to_action(-1, 0, cols=4)


@pytest.mark.parametrize(
    ("shape", "win_k", "coords", "last_move"),
    [
        # 3x3 horizontal & vertical & diagonals
        ((3, 3), 3, [(0, 0), (0, 1), (0, 2)], (0, 2)),
        ((3, 3), 3, [(0, 2), (1, 2), (2, 2)], (1, 2)),
        ((3, 3), 3, [(0, 0), (1, 1), (2, 2)], (1, 1)),
        ((3, 3), 3, [(0, 2), (1, 1), (2, 0)], (2, 0)),
        # 4x4 K=4 boundary and diagonal wins
        ((4, 4), 4, [(3, 0), (3, 1), (3, 2), (3, 3)], (3, 1)),
        ((4, 4), 4, [(0, 0), (1, 0), (2, 0), (3, 0)], (0, 0)),
        ((4, 4), 4, [(0, 0), (1, 1), (2, 2), (3, 3)], (3, 3)),
        ((4, 4), 4, [(0, 3), (1, 2), (2, 1), (3, 0)], (1, 2)),
        # 4x4 K=3 offset sub-lines
        ((4, 4), 3, [(1, 1), (2, 2), (3, 3)], (2, 2)),
        ((4, 4), 3, [(1, 3), (2, 2), (3, 1)], (1, 3)),
        # 5x5 K=4 and rectangular 4x5 K=4
        ((5, 5), 4, [(1, 0), (2, 1), (3, 2), (4, 3)], (2, 1)),
        ((4, 5), 4, [(0, 4), (1, 3), (2, 2), (3, 1)], (3, 1)),
    ],
)
def test_check_win_around_detects_all_axes(
    shape: tuple[int, int],
    win_k: int,
    coords: list[tuple[int, int]],
    last_move: tuple[int, int],
) -> None:
    """Verify O(K) win detection across 4 axes, corners, edges, and board sizes."""
    for player in (Player.X, Player.O):
        board = np.zeros(shape, dtype=np.int8)
        for r, c in coords:
            board[r, c] = int(player)

        win_line = check_win_around(board, last_move[0], last_move[1], int(player), win_k)
        assert win_line is not None
        assert len(win_line) == win_k
        assert set(win_line) == set(coords)
        assert last_move in win_line


def test_check_win_around_overline_includes_last_move() -> None:
    """When a move bridges a line longer than win_k, return K cells containing last_move."""
    board = np.zeros((5, 5), dtype=np.int8)
    for c in range(5):
        board[2, c] = PLAYER_X

    # Even when playing at the right end (2, 4), returned K=4 slice must contain (2, 4).
    win_line = check_win_around(board, 2, 4, PLAYER_X, win_k=4)
    assert win_line == [(2, 1), (2, 2), (2, 3), (2, 4)]

    # Playing in the middle (2, 2) on a 5-cell row also includes (2, 2).
    mid_line = check_win_around(board, 2, 2, PLAYER_X, win_k=4)
    assert mid_line is not None
    assert len(mid_line) == 4
    assert (2, 2) in mid_line


def test_check_win_around_non_winning_cases() -> None:
    """Return None for <K streaks, blocked lines, empty cells, or opponent cells."""
    board = np.zeros((4, 4), dtype=np.int8)
    # 3 in a row (K=4 not satisfied)
    board[0, 0] = PLAYER_X
    board[0, 1] = PLAYER_X
    board[0, 2] = PLAYER_X
    assert check_win_around(board, 0, 2, PLAYER_X, win_k=4) is None

    # Blocked by opponent in the middle
    board[1, 0] = PLAYER_X
    board[1, 1] = PLAYER_O
    board[1, 2] = PLAYER_X
    board[1, 3] = PLAYER_X
    assert check_win_around(board, 1, 3, PLAYER_X, win_k=3) is None

    # Empty cell or mismatched player at (row, col)
    assert check_win_around(board, 3, 3, PLAYER_X, win_k=4) is None
    assert check_win_around(board, 0, 2, PLAYER_O, win_k=3) is None


def test_check_win_around_invalid_inputs() -> None:
    """Validate defensive exceptions for invalid boards, players, win_k, or coordinates."""
    board = np.zeros((4, 4), dtype=np.int8)

    with pytest.raises(InvalidConfigError):
        check_win_around(np.zeros(16, dtype=np.int8), 0, 0, PLAYER_X, win_k=4)
    with pytest.raises(InvalidConfigError):
        check_win_around(board, 0, 0, EMPTY, win_k=4)
    with pytest.raises(InvalidConfigError):
        check_win_around(board, 0, 0, True, win_k=4)
    with pytest.raises(InvalidConfigError):
        check_win_around(board, 0, 0, PLAYER_X, win_k=5)
    with pytest.raises(IllegalMoveError):
        check_win_around(board, -1, 0, PLAYER_X, win_k=4)
    with pytest.raises(IllegalMoveError):
        check_win_around(board, 0, 4, PLAYER_X, win_k=4)


@pytest.mark.parametrize(
    ("rows", "cols", "win_k", "expected_count"),
    [
        (3, 3, 3, 8),
        (4, 4, 4, 10),
        (4, 4, 3, 24),
        (5, 5, 4, 28),
    ],
)
def test_precompute_win_windows_counts_and_geometry(
    rows: int,
    cols: int,
    win_k: int,
    expected_count: int,
) -> None:
    """Verify exact window counts and collinear constant-step geometry."""
    assert win_windows is precompute_win_windows
    windows = precompute_win_windows(rows, cols, win_k)
    assert len(windows) == expected_count
    assert len(set(windows)) == expected_count

    for window in windows:
        assert len(window) == win_k
        coords = [action_to_coords(idx, cols) for idx in window]
        deltas = {
            (coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
            for i in range(win_k - 1)
        }
        assert len(deltas) == 1
        assert next(iter(deltas)) in {(0, 1), (1, 0), (1, 1), (1, -1)}


@pytest.mark.parametrize(
    ("rows", "cols", "win_k"),
    [(2, 3, 3), (3, 3, 2), (3, 4, 4), (True, 4, 3)],
)
def test_precompute_win_windows_rejects_invalid_dims(rows: int, cols: int, win_k: int) -> None:
    """Reject out-of-range or boolean board dimensions in precompute_win_windows."""
    with pytest.raises(InvalidConfigError):
        precompute_win_windows(rows, cols, win_k)


def test_positional_weights_values_normalization_and_sum_invariant() -> None:
    """Verify exact strategic weights for 3x3/4x4 and mathematical sum invariant."""
    w3 = positional_weights(3, 3, 3)
    expected_w3 = np.array(
        [
            [3, 2, 3],
            [2, 4, 2],
            [3, 2, 3],
        ],
        dtype=np.int32,
    )
    assert w3.dtype == np.int32
    np.testing.assert_array_equal(w3, expected_w3)

    w4 = positional_weights(4, 4, 4)
    expected_w4 = np.array(
        [
            [3, 2, 2, 3],
            [2, 3, 3, 2],
            [2, 3, 3, 2],
            [3, 2, 2, 3],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(w4, expected_w4)

    for rows, cols, win_k in [(3, 3, 3), (4, 4, 3), (4, 4, 4), (5, 5, 4), (4, 5, 4)]:
        weights = positional_weights(rows, cols, win_k)
        assert int(weights.sum()) == len(precompute_win_windows(rows, cols, win_k)) * win_k

    norm_w3 = positional_weights(3, 3, 3, normalize=True)
    assert norm_w3.dtype == np.float32
    assert norm_w3[1, 1] == pytest.approx(1.0)
    assert norm_w3[0, 0] == pytest.approx(0.75)
    assert norm_w3[0, 1] == pytest.approx(0.5)


def test_encode_canonical_board_perspectives_and_immutability() -> None:
    """Verify perspective flipping, non-mutation of input, and involution property."""
    board = np.array(
        [
            [PLAYER_X, EMPTY, PLAYER_O],
            [EMPTY, PLAYER_O, PLAYER_X],
            [PLAYER_X, EMPTY, EMPTY],
        ],
        dtype=np.int8,
    )
    original_snapshot = board.copy()

    canon_x = encode_canonical_board(board, PLAYER_X)
    assert canon_x.dtype == np.int8
    assert canon_x is not board
    np.testing.assert_array_equal(canon_x, original_snapshot)

    canon_o = encode_canonical_board(board, PLAYER_O)
    expected_o = np.array(
        [
            [-1, 0, 1],
            [0, 1, -1],
            [-1, 0, 0],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(canon_o, expected_o)

    # Double flip restores original board, and source board stayed untouched.
    np.testing.assert_array_equal(encode_canonical_board(canon_o, PLAYER_O), original_snapshot)
    np.testing.assert_array_equal(board, original_snapshot)

    with pytest.raises(InvalidConfigError):
        encode_canonical_board(np.zeros(9, dtype=np.int8), PLAYER_X)
    with pytest.raises(InvalidConfigError):
        encode_canonical_board(board, EMPTY)
    with pytest.raises(InvalidConfigError):
        encode_canonical_board(board, True)
