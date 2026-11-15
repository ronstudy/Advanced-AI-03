"""Unit and integration tests for Task T1.3 / T1.4: ExtendedTicTacToe engine API."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ttt_ai.config import load_game_config
from ttt_ai.engine import (
    EMPTY,
    PLAYER_O,
    PLAYER_X,
    ExtendedTicTacToe,
    IllegalMoveError,
    InvalidConfigError,
    Player,
    action_to_coords,
    precompute_win_windows,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
ENGINE_DIR = PROJECT_ROOT / "src" / "ttt_ai" / "engine"


def _play_moves(game: ExtendedTicTacToe, moves: list[int]) -> tuple[np.ndarray, float, bool, dict]:
    """Replay a deterministic move sequence on `game` and return the final step tuple."""
    assert moves, "moves sequence must be non-empty"
    last_step = None
    for action in moves:
        last_step = game.step(action)
    assert last_step is not None
    return last_step


def test_initialization_defaults_custom_shapes_and_invalid_configs() -> None:
    """Verify default 4x4 K=4 state, rectangular/custom sizes, and invalid rejections."""
    default_game = ExtendedTicTacToe()
    assert default_game.rows == 4
    assert default_game.cols == 4
    assert default_game.win_k == 4
    assert default_game.num_cells == 16
    assert default_game.shape == (4, 4)
    assert default_game.current_player is Player.X
    assert default_game.winner is None
    assert default_game.winning_line is None
    assert default_game.last_move is None
    assert default_game.move_count == 0
    assert default_game.move_history == ()
    assert not default_game.is_terminal()
    assert default_game.result() is None
    assert default_game.legal_actions() == list(range(16))

    rect_game = ExtendedTicTacToe(rows=4, cols=5, win_k=4)
    assert rect_game.shape == (4, 5)
    assert rect_game.num_cells == 20
    assert rect_game.legal_actions() == list(range(20))

    for bad_args in [(2, 3, 3), (3, 2, 3), (4, 4, 2), (4, 4, 5), (3, 5, 4), (True, 4, 3)]:
        with pytest.raises(InvalidConfigError):
            ExtendedTicTacToe(*bad_args)


def test_board_copy_immutability_canonical_board_and_reset() -> None:
    """Verify `board` returns a defensive copy, canonical flips perspective, and reset clears."""
    game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    board_view = game.board
    board_view[0, 0] = PLAYER_O
    assert int(game.board[0, 0]) == EMPTY

    obs_after_x, reward, done, info = game.step(4)  # X at center (1, 1)
    assert reward == 0.0
    assert not done
    assert info["player"] == PLAYER_X
    assert info["current_player"] == PLAYER_O
    assert info["winner"] is None
    assert info["move_count"] == 1
    assert game.current_player is Player.O
    assert game.last_move == 4
    assert game.move_history == (4,)

    # Returned observation from step() must also be decoupled from internal board.
    obs_after_x[0, 0] = PLAYER_X
    assert int(game.board[0, 0]) == EMPTY

    # From O's perspective, X's mark at (1, 1) must appear as -1.
    canon = game.canonical_board()
    assert canon.dtype == np.int8
    assert int(canon[1, 1]) == -1

    reset_board = game.reset()
    assert reset_board.shape == (3, 3)
    assert np.all(reset_board == EMPTY)
    assert game.current_player is Player.X
    assert game.move_count == 0
    assert game.last_move is None
    assert not game.is_terminal()
    assert game.legal_actions() == list(range(9))


@pytest.mark.parametrize(
    ("rows", "cols", "win_k", "moves", "expected_winner"),
    [
        # 3x3 K=3: X wins horizontally along top edge
        (3, 3, 3, [0, 3, 1, 4, 2], Player.X),
        # 3x3 K=3: O wins vertically along right edge
        (3, 3, 3, [0, 2, 1, 5, 6, 8], Player.O),
        # 3x3 K=3: X wins on main diagonal
        (3, 3, 3, [0, 1, 4, 2, 8], Player.X),
        # 4x4 K=3: O wins on anti-diagonal offset
        (4, 4, 3, [0, 7, 1, 10, 5, 13], Player.O),
        # 4x4 K=4: X wins on bottom border row
        (4, 4, 4, [12, 0, 13, 1, 14, 2, 15], Player.X),
        # 4x4 K=4: O wins on main diagonal (0, 5, 10, 15)
        (4, 4, 4, [1, 0, 2, 5, 3, 10, 4, 15], Player.O),
        # 5x5 K=4: X wins on anti-diagonal
        (5, 5, 4, [3, 0, 7, 1, 11, 2, 15], Player.X),
    ],
)
def test_win_outcomes_across_boards_axes_and_players(
    rows: int,
    cols: int,
    win_k: int,
    moves: list[int],
    expected_winner: Player,
) -> None:
    """Verify win detection, terminal state, rewards, and result perspectives."""
    game = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)
    _, reward, done, info = _play_moves(game, moves)

    assert done is True
    assert reward == 1.0
    assert game.is_terminal() is True
    assert game.winner is expected_winner
    assert game.winning_line is not None
    assert len(game.winning_line) == win_k
    assert game.legal_actions() == []
    assert info["winner"] == int(expected_winner)
    assert info["winning_line"] == list(game.winning_line)

    assert game.result(expected_winner) == 1.0
    assert game.result(expected_winner.opponent) == -1.0
    # Default result() uses Player.X perspective.
    expected_x_score = 1.0 if expected_winner is Player.X else -1.0
    assert game.result() == expected_x_score


def test_full_board_draws_3x3_and_4x4() -> None:
    """Verify full-board draws report terminal=True, winner=None, and 0.0 payoffs."""
    # Classic 3x3 draw sequence:
    # X O X
    # X O O
    # O X X
    game_3x3 = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _, reward, done, info = _play_moves(game_3x3, [0, 1, 2, 4, 3, 5, 7, 6, 8])
    assert done is True
    assert reward == 0.0
    assert game_3x3.is_terminal() is True
    assert game_3x3.winner is None
    assert game_3x3.winning_line is None
    assert game_3x3.legal_actions() == []
    assert game_3x3.result(Player.X) == 0.0
    assert game_3x3.result(Player.O) == 0.0
    assert info["winner"] is None

    # 4x4 K=4 draw (every row/col/diag blocked by both X and O):
    # X O X O
    # O X O X
    # O X O X
    # X O X O
    draw_4x4_moves = [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14, 15]
    game_4x4 = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    _, reward_4, done_4, _ = _play_moves(game_4x4, draw_4x4_moves)
    assert done_4 is True
    assert reward_4 == 0.0
    assert game_4x4.winner is None
    assert game_4x4.result() == 0.0
    assert game_4x4.move_count == 16


def test_illegal_moves_and_invalid_result_player_raise() -> None:
    """Ensure every illegal step, post-terminal move, empty undo, and bad player raises."""
    game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)

    with pytest.raises(IllegalMoveError):
        game.undo()

    for bad_action in [-1, 9, True, 1.5, "4", None]:
        with pytest.raises(IllegalMoveError):
            game.step(bad_action)  # type: ignore[arg-type]

    game.step(0)
    with pytest.raises(IllegalMoveError):
        game.step(0)

    # Bad player passed to result()
    with pytest.raises(InvalidConfigError):
        game.result(0)
    with pytest.raises(InvalidConfigError):
        game.result(True)  # type: ignore[arg-type]

    # Finish game by X win: [0, 3, 1, 4, 2]
    _play_moves(game, [3, 1, 4, 2])
    assert game.is_terminal()
    with pytest.raises(IllegalMoveError):
        game.step(5)


def test_clone_memory_independence() -> None:
    """Verify cloned games share no mutable board or history state with the original."""
    original = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    _play_moves(original, [0, 1, 5])

    cloned = original.clone()
    assert cloned is not original
    np.testing.assert_array_equal(cloned.board, original.board)
    assert cloned.current_player is original.current_player
    assert cloned.move_history == original.move_history

    # Mutate clone through play & undo; original must remain unaffected.
    # O plays 2, X plays 10, O plays 3, X plays 15 -> X completes main diagonal (0, 5, 10, 15).
    cloned.step(2)
    cloned.step(10)
    cloned.step(3)
    cloned.step(15)
    assert cloned.is_terminal()
    assert cloned.winner is Player.X
    assert not original.is_terminal()
    assert original.move_history == (0, 1, 5)
    assert original.current_player is Player.O

    # Mutate original; clone must remain unaffected.
    original.undo()
    assert original.move_history == (0, 1)
    assert cloned.move_history == (0, 1, 5, 2, 10, 3, 15)


def test_undo_restores_non_terminal_win_draw_and_search_tree_states() -> None:
    """Verify O(1) undo() perfectly reverts normal, winning, drawing, and tree-search paths."""
    game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)

    # Play to a win by X: [0, 3, 1, 4, 2]
    _play_moves(game, [0, 3, 1, 4, 2])
    assert game.is_terminal()
    assert game.winner is Player.X

    # Undo the winning move -> terminal flags and winner must clear completely.
    game.undo()
    assert not game.is_terminal()
    assert game.winner is None
    assert game.winning_line is None
    assert game.result() is None
    assert game.current_player is Player.X
    assert game.move_history == (0, 3, 1, 4)
    assert 2 in game.legal_actions()

    # Now replay to a full-board draw and undo back to start.
    game.reset()
    draw_moves = [0, 1, 2, 4, 3, 5, 7, 6, 8]
    _play_moves(game, draw_moves)
    assert game.is_terminal() and game.winner is None

    for expected_remaining in range(len(draw_moves) - 1, -1, -1):
        game.undo()
        assert not game.is_terminal()
        assert game.winner is None
        assert game.move_count == expected_remaining

    assert game.current_player is Player.X
    assert np.all(game.board == EMPTY)
    with pytest.raises(IllegalMoveError):
        game.undo()

    # Simulate Alpha-Beta backtracking: snapshot state, explore depth-3, undo back.
    _play_moves(game, [4, 0])
    snapshot_board = game.board
    snapshot_player = game.current_player
    snapshot_legal = game.legal_actions()

    for a1 in snapshot_legal[:3]:
        game.step(a1)
        for a2 in game.legal_actions()[:2]:
            game.step(a2)
            game.undo()
        game.undo()

    np.testing.assert_array_equal(game.board, snapshot_board)
    assert game.current_player is snapshot_player
    assert game.legal_actions() == snapshot_legal


def test_render_output_states() -> None:
    """Verify render() shows board dimensions, turn/result status, and winning highlights."""
    game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    initial_text = game.render()
    assert "Board: 3x3 (K=3)" in initial_text
    assert "Turn: Player X" in initial_text

    _play_moves(game, [0, 3, 1, 4, 2])
    win_text = game.render()
    assert "Result: Player X wins!" in win_text
    assert "[X]" in win_text
    assert "Winning line: [(0, 0), (0, 1), (0, 2)]" in win_text

    game.reset()
    _play_moves(game, [0, 1, 2, 4, 3, 5, 7, 6, 8])
    draw_text = game.render()
    assert "Result: Draw" in draw_text


def _pick_safe_filler_moves(
    rows: int,
    cols: int,
    forbidden_cells: set[int],
    count: int,
    windows: tuple[tuple[int, ...], ...],
) -> list[int]:
    """Select `count` cells outside `forbidden_cells` that never form a full win window."""
    chosen: list[int] = []
    for cell in range(rows * cols):
        if cell in forbidden_cells:
            continue
        candidate_set = set(chosen) | {cell}
        if any(set(window).issubset(candidate_set) for window in windows):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    raise AssertionError(f"Unable to select {count} non-winning filler cells.")


@pytest.mark.parametrize(
    ("rows", "cols", "win_k"),
    [(3, 3, 3), (4, 4, 3), (4, 4, 4), (5, 5, 4)],
)
def test_all_precomputed_win_windows_for_both_players(rows: int, cols: int, win_k: int) -> None:
    """Exhaustively test every win window across 4 board configs for both X and O."""
    windows = precompute_win_windows(rows, cols, win_k)

    for window in windows:
        window_set = set(window)
        expected_coords = tuple(action_to_coords(action, cols) for action in window)

        for target_player in (Player.X, Player.O):
            opponent_moves_needed = win_k - 1 if target_player is Player.X else win_k
            filler = _pick_safe_filler_moves(
                rows=rows,
                cols=cols,
                forbidden_cells=window_set,
                count=opponent_moves_needed,
                windows=windows,
            )

            game = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)
            if target_player is Player.X:
                for step_idx, x_move in enumerate(window):
                    _, reward, done, _ = game.step(x_move)
                    if step_idx < win_k - 1:
                        assert not done
                        assert reward == 0.0
                        game.step(filler[step_idx])
                    else:
                        assert done
                        assert reward == 1.0
            else:
                for step_idx, o_move in enumerate(window):
                    _, reward_x, done_x, _ = game.step(filler[step_idx])
                    assert not done_x
                    assert reward_x == 0.0
                    _, reward_o, done_o, _ = game.step(o_move)
                    if step_idx < win_k - 1:
                        assert not done_o
                        assert reward_o == 0.0
                    else:
                        assert done_o
                        assert reward_o == 1.0

            assert game.is_terminal() is True
            assert game.winner is target_player
            assert game.winning_line == expected_coords
            assert game.result(target_player) == 1.0
            assert game.result(target_player.opponent) == -1.0
            assert game.legal_actions() == []


def test_last_cell_win_takes_precedence_over_draw() -> None:
    """Verify winning on the very last empty cell reports a win rather than a draw."""
    # 3x3 K=3: X wins on move #9 via main diagonal (0, 4, 8).
    game_3x3 = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _, reward_3, done_3, info_3 = _play_moves(game_3x3, [0, 1, 2, 3, 4, 5, 7, 6, 8])
    assert done_3 is True
    assert reward_3 == 1.0
    assert game_3x3.move_count == 9
    assert game_3x3.winner is Player.X
    assert game_3x3.winning_line == ((0, 0), (1, 1), (2, 2))
    assert game_3x3.result(Player.X) == 1.0
    assert game_3x3.result(Player.O) == -1.0
    assert info_3["winner"] == PLAYER_X

    # 4x4 K=4: O wins on move #16 via right column (3, 7, 11, 15).
    game_4x4 = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    moves_4x4 = [0, 1, 2, 3, 5, 4, 6, 7, 8, 10, 9, 11, 12, 13, 14, 15]
    _, reward_4, done_4, info_4 = _play_moves(game_4x4, moves_4x4)
    assert done_4 is True
    assert reward_4 == 1.0
    assert game_4x4.move_count == 16
    assert game_4x4.winner is Player.O
    assert game_4x4.winning_line == ((0, 3), (1, 3), (2, 3), (3, 3))
    assert game_4x4.result(Player.O) == 1.0
    assert game_4x4.result(Player.X) == -1.0
    assert info_4["winner"] == PLAYER_O


def test_full_board_draw_on_5x5_k4() -> None:
    """Verify a complete 25-move 5x5 K=4 board with no 4-in-a-row ends in a draw."""
    # Pattern:
    # X O X O X
    # O X O X O
    # O X O X O
    # X O X O X
    # X O X O X
    x_moves = [0, 2, 4, 6, 8, 11, 13, 15, 17, 19, 20, 22, 24]
    o_moves = [1, 3, 5, 7, 9, 10, 12, 14, 16, 18, 21, 23]
    interleaved: list[int] = []
    for idx, x_cell in enumerate(x_moves):
        interleaved.append(x_cell)
        if idx < len(o_moves):
            interleaved.append(o_moves[idx])

    game = ExtendedTicTacToe(rows=5, cols=5, win_k=4)
    _, reward, done, info = _play_moves(game, interleaved)

    assert done is True
    assert reward == 0.0
    assert game.move_count == 25
    assert game.is_terminal() is True
    assert game.winner is None
    assert game.winning_line is None
    assert game.legal_actions() == []
    assert game.result(Player.X) == 0.0
    assert game.result(Player.O) == 0.0
    assert info["winner"] is None


@pytest.mark.parametrize(
    "config_name",
    [
        "game_3x3_k3.yaml",
        "game_4x4_k3.yaml",
        "game_4x4_k4.yaml",
        "game_5x5_k4.yaml",
    ],
)
def test_engine_from_yaml_configs(config_name: str) -> None:
    """Ensure every YAML benchmark preset initializes an identically sized engine."""
    cfg = load_game_config(CONFIGS_DIR / config_name)
    game = ExtendedTicTacToe(rows=cfg.board.rows, cols=cfg.board.cols, win_k=cfg.board.win_k)

    assert game.shape == cfg.board.shape
    assert game.win_k == cfg.board.win_k
    assert game.num_cells == cfg.board.num_cells


@pytest.mark.parametrize(
    ("rows", "cols", "win_k"),
    [(3, 3, 3), (4, 4, 3), (4, 4, 4), (5, 5, 4)],
)
def test_random_rollouts_state_snapshots_and_full_undo_invariants(
    rows: int,
    cols: int,
    win_k: int,
) -> None:
    """Fuzz random games to terminal and assert undo() exactly restores every step."""
    rng = np.random.default_rng(2026 + rows * 10 + win_k)
    game = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)

    for _ in range(10):
        game.reset()
        snapshots: list[tuple[np.ndarray, Player, list[int], bool, Player | None, int | None]] = [
            (
                game.board,
                game.current_player,
                game.legal_actions(),
                game.is_terminal(),
                game.winner,
                game.last_move,
            )
        ]

        while not game.is_terminal():
            legal = game.legal_actions()
            assert legal
            action = int(legal[int(rng.integers(0, len(legal)))])
            game.step(action)
            snapshots.append(
                (
                    game.board,
                    game.current_player,
                    game.legal_actions(),
                    game.is_terminal(),
                    game.winner,
                    game.last_move,
                )
            )

        # Unwind step-by-step from terminal state back to the empty board.
        for step_idx in range(len(snapshots) - 2, -1, -1):
            game.undo()
            exp_board, exp_player, exp_legal, exp_term, exp_winner, exp_last = snapshots[step_idx]
            np.testing.assert_array_equal(game.board, exp_board)
            assert game.current_player is exp_player
            assert game.legal_actions() == exp_legal
            assert game.is_terminal() is exp_term
            assert game.winner is exp_winner
            assert game.last_move == exp_last
            assert game.winning_line is None
            assert game.result() is None


def test_engine_module_isolation_and_no_print_statements() -> None:
    """Enforce ARCHITECTURE.md boundary: pure Python + NumPy, no UI/DL libs or print()."""
    forbidden_modules = {"torch", "rich", "matplotlib", "ttt_ai.agents", "ttt_ai.ui"}

    for filename in ("__init__.py", "exceptions.py", "rules.py", "game.py"):
        source = (ENGINE_DIR / filename).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filename)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_modules
                    assert not alias.name.startswith(tuple(forbidden_modules))
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                root_mod = node.module.split(".")[0]
                assert root_mod not in forbidden_modules
                assert not node.module.startswith(tuple(forbidden_modules))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", f"Forbidden print() found in engine/{filename}"

