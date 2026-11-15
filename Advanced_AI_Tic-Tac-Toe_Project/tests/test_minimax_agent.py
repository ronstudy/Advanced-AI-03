"""Unit tests and tactical/pruning checks for Task T2.3: Minimax & AlphaBeta agents."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ttt_ai import AlphaBetaAgent, ExtendedTicTacToe, MinimaxAgent, RandomAgent, evaluate_board
from ttt_ai.engine import IllegalMoveError, InvalidConfigError, Player

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = PROJECT_ROOT / "src" / "ttt_ai" / "agents"


def _play_sequence(game: ExtendedTicTacToe, moves: list[int]) -> None:
    """Apply a sequence of flat action indices to `game`."""
    for action in moves:
        game.step(action)


def test_evaluate_board_symmetry_bounds_monotonicity_and_validation() -> None:
    """Verify evaluate_board properties: zero-sum symmetry, (-1, 1) range, and monotonicity."""
    with pytest.raises(TypeError):
        evaluate_board("not_a_game")  # type: ignore[arg-type]

    game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    with pytest.raises(InvalidConfigError):
        evaluate_board(game, perspective_player=0)
    with pytest.raises(InvalidConfigError):
        evaluate_board(game, perspective_player=True)  # type: ignore[arg-type]

    # Empty board is completely neutral.
    assert evaluate_board(game, Player.X) == 0.0
    assert evaluate_board(game, Player.O) == 0.0

    # Monotonic progression along row 0 for X: 1 -> 2 -> 3 marks (with harmless O moves).
    game.step(0)  # X at (0, 0)
    s1 = evaluate_board(game, Player.X)
    assert 0.0 < s1 < 1.0
    assert evaluate_board(game, Player.O) == pytest.approx(-s1)

    game.step(12)  # O at (3, 0)
    game.step(1)  # X at (0, 1)
    s2 = evaluate_board(game, Player.X)
    assert s1 < s2 < 1.0
    assert evaluate_board(game, Player.O) == pytest.approx(-s2)

    game.step(13)  # O at (3, 1)
    game.step(2)  # X at (0, 2)
    s3 = evaluate_board(game, Player.X)
    assert s2 < s3 < 1.0
    assert evaluate_board(game, Player.O) == pytest.approx(-s3)

    # Terminal win returns exact +/-10.0.
    game.step(14)  # O at (3, 2)
    game.step(3)  # X wins along row 0
    assert evaluate_board(game, Player.X) == pytest.approx(10.0)
    assert evaluate_board(game, Player.O) == pytest.approx(-10.0)

    # Terminal draw returns 0.0.
    draw_game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _play_sequence(draw_game, [0, 1, 2, 4, 3, 5, 7, 6, 8])
    assert evaluate_board(draw_game, Player.X) == 0.0
    assert evaluate_board(draw_game, Player.O) == 0.0


def test_minimax_and_alphabeta_config_validation() -> None:
    """Verify Level 3 configuration validation and metadata serialization."""
    ab = AlphaBetaAgent(max_depth=5, time_budget_ms=25.0)
    assert ab.level == 3
    assert ab.name == "AlphaBetaAgent"
    assert ab.max_depth == 5
    assert ab.use_alpha_beta is True
    assert ab.use_move_ordering is True
    assert ab.iterative_deepening is True
    assert ab.get_config()["use_alpha_beta"] is True
    assert ab.get_config()["max_depth"] == 5

    mm = MinimaxAgent(max_depth=3, use_alpha_beta=False, use_move_ordering=False)
    assert mm.use_alpha_beta is False
    assert mm.use_move_ordering is False
    assert mm.iterative_deepening is False

    # Updating time_budget_ms dynamically turns on automatic Iterative Deepening.
    mm.set_time_budget_ms(20.0)
    assert mm.iterative_deepening is True

    for bad_depth in (0, -2, True, 2.5):
        with pytest.raises(InvalidConfigError):
            MinimaxAgent(max_depth=bad_depth)  # type: ignore[arg-type]
    with pytest.raises(InvalidConfigError):
        MinimaxAgent(use_alpha_beta=1)  # type: ignore[arg-type]
    with pytest.raises(InvalidConfigError):
        MinimaxAgent(use_move_ordering="yes")  # type: ignore[arg-type]
    with pytest.raises(InvalidConfigError):
        MinimaxAgent(iterative_deepening="auto")  # type: ignore[arg-type]


def test_tactical_immediate_win_faster_ply_preference_block_and_fork() -> None:
    """Verify AlphaBetaAgent finds immediate wins, prefers shorter plies, blocks, and forks."""
    agent = AlphaBetaAgent(max_depth=4)

    # 1. Immediate win (ply 1) on 4x4 K=4 -> value_estimate = 10.0 - 0.01 = 9.99.
    win_game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    _play_sequence(win_game, [0, 4, 1, 5, 2, 6])
    d_win = agent.select_action(win_game)
    assert d_win.action == 3
    assert d_win.value_estimate == pytest.approx(9.99)

    # 2. Immediate win (ply 1 = 9.99) strictly beats slower forced win (ply 3 = 9.97).
    # On 3x3 K=3: X has (0, 2) -> wins at 1 in 1 ply; O at (3, 7).
    ply_game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _play_sequence(ply_game, [0, 3, 2, 7])
    d_ply = agent.select_action(ply_game)
    assert d_ply.action == 1
    assert d_ply.value_estimate == pytest.approx(9.99)

    # 3. Immediate block on 4x4 K=4 when O has no immediate win.
    block_game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    _play_sequence(block_game, [3, 0, 6, 1, 9, 4, 2])
    d_block = agent.select_action(block_game)
    assert d_block.action == 12

    # 4. 3-ply forced fork win on 3x3 K=3: X at (1, 3), O at (5, 7) -> X plays 0 to force win.
    fork_game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _play_sequence(fork_game, [1, 5, 3, 7])
    d_fork = agent.select_action(fork_game)
    assert d_fork.action == 0
    assert d_fork.value_estimate == pytest.approx(9.97)


@pytest.mark.parametrize(
    ("rows", "cols", "win_k", "moves", "depth"),
    [
        (3, 3, 3, [4, 0], 4),
        (3, 3, 3, [0, 4, 8], 5),
        (3, 3, 3, [1, 5, 3, 7], 4),
        (4, 4, 4, [5, 0, 6, 3], 3),
        (4, 4, 4, [0, 5, 1, 10, 4, 15], 3),
        (4, 4, 3, [5, 0, 6, 12], 3),
    ],
)
def test_minimax_equals_alphabeta_and_pruning_reduces_nodes(
    rows: int,
    cols: int,
    win_k: int,
    moves: list[int],
    depth: int,
) -> None:
    """Verify Minimax == AlphaBeta root value and optimal action, with huge node savings."""
    game = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)
    _play_sequence(game, moves)
    board_before = game.board
    history_before = game.move_history

    mm_plain = MinimaxAgent(
        name="MinimaxPlain",
        max_depth=depth,
        use_alpha_beta=False,
        use_move_ordering=False,
    )
    ab_no_order = MinimaxAgent(
        name="ABNoOrder",
        max_depth=depth,
        use_alpha_beta=True,
        use_move_ordering=False,
    )
    ab_ordered = AlphaBetaAgent(
        name="ABOrdered",
        max_depth=depth,
        use_move_ordering=True,
    )

    dec_mm = mm_plain.select_action(game)
    dec_ab_no_order = ab_no_order.select_action(game)
    dec_ab_ordered = ab_ordered.select_action(game)

    np.testing.assert_array_equal(game.board, board_before)
    assert game.move_history == history_before

    assert dec_ab_no_order.action == dec_mm.action
    assert dec_ab_ordered.action == dec_mm.action
    assert dec_ab_no_order.value_estimate == pytest.approx(dec_mm.value_estimate, abs=1e-12)
    assert dec_ab_ordered.value_estimate == pytest.approx(dec_mm.value_estimate, abs=1e-12)
    assert dec_mm.depth_reached == min(depth, len(game.legal_actions()))
    assert dec_ab_ordered.depth_reached == dec_mm.depth_reached

    # Alpha-Beta pruning must visit strictly fewer nodes than unpruned Minimax.
    assert dec_ab_no_order.nodes_visited < dec_mm.nodes_visited
    assert dec_ab_ordered.nodes_visited < dec_mm.nodes_visited


def test_iterative_deepening_timeout_handling_and_source_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Iterative Deepening stops cleanly mid-recursion on SearchTimeoutError."""
    game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    board_before = game.board
    history_before = game.move_history

    # Deterministic timeout simulation: allow depth 1 and depth 2 to finish,
    # then trigger timeout mid-recursion at node 256 inside depth 3.
    call_count = 0

    def fake_perf_counter() -> float:
        nonlocal call_count
        call_count += 1
        return 100.0 if call_count <= 3 else 100.5

    monkeypatch.setattr("ttt_ai.agents.base.time.perf_counter", fake_perf_counter)

    timed_agent = AlphaBetaAgent(max_depth=6, time_budget_ms=10.0, iterative_deepening=True)
    decision = timed_agent.select_action(game)

    monkeypatch.undo()

    # Must unwind cleanly and return the last completed depth (< 6) without mutating game.
    assert decision.action in game.legal_actions()
    assert 1 <= decision.depth_reached < 6
    assert decision.nodes_visited >= 256
    np.testing.assert_array_equal(game.board, board_before)
    assert game.move_history == history_before

    # Real wall-clock tight budget on full empty 4x4 with deep max_depth=16.
    wall_timed = AlphaBetaAgent(max_depth=16, time_budget_ms=15.0)
    empty_4x4 = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    wall_decision = wall_timed.select_action(empty_4x4)
    assert wall_decision.action in empty_4x4.legal_actions()
    assert wall_decision.latency_ms < 100.0
    assert wall_decision.depth_reached >= 1


def test_immediate_timeout_before_depth1_uses_move_ordering_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fallback returns best move-ordered action when even depth=1 times out."""
    game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    agent = AlphaBetaAgent(max_depth=4, time_budget_ms=5.0, use_move_ordering=True)

    # Raise SearchTimeoutError inside _negamax on depth=1 to force depth-0 fallback.
    def timeout_negamax(*args: object, **kwargs: object) -> float:
        from ttt_ai.agents.base import SearchTimeoutError

        raise SearchTimeoutError("Simulated immediate depth-1 timeout")

    monkeypatch.setattr(AlphaBetaAgent, "_negamax", timeout_negamax)
    decision = agent.select_action(game)
    monkeypatch.undo()

    # Move ordering ranks center cell 4 highest on empty 3x3 board.
    assert decision.action == 4
    assert decision.depth_reached == 0
    assert decision.value_estimate == 0.0
    assert not game.is_terminal()
    assert game.move_count == 0


def test_edge_cases_empty_last_cell_and_terminal_boards() -> None:
    """Verify empty board, single-remaining-cell win/draw, and terminal rejection."""
    agent = AlphaBetaAgent(max_depth=4)

    # 1. Empty 3x3 board.
    empty_3x3 = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    d_empty = agent.select_action(empty_3x3)
    assert d_empty.action in range(9)
    assert d_empty.nodes_visited > 1

    # 2. Near-full 3x3 draw cell (only cell 8 left).
    near_draw = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _play_sequence(near_draw, [0, 1, 2, 4, 3, 5, 7, 6])
    d_last_draw = agent.select_action(near_draw)
    assert d_last_draw.action == 8
    assert d_last_draw.value_estimate == 0.0
    assert d_last_draw.depth_reached == 1
    near_draw.step(d_last_draw.action)
    assert near_draw.is_terminal()

    with pytest.raises(IllegalMoveError):
        agent.select_action(near_draw)

    # 3. Near-full 3x3 win on very last cell (X wins via main diagonal 0, 4, 8).
    near_win = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _play_sequence(near_win, [0, 1, 2, 3, 4, 5, 7, 6])
    d_last_win = agent.select_action(near_win)
    assert d_last_win.action == 8
    assert d_last_win.value_estimate == pytest.approx(9.99)


def test_alphabeta_perfect_draw_on_3x3_and_dominates_random_on_4x4() -> None:
    """Verify AlphaBeta draws self on 3x3 (full depth) and dominates Random on 4x4 K=4."""
    ab_full_x = AlphaBetaAgent(name="AB_X", max_depth=9)
    ab_full_o = AlphaBetaAgent(name="AB_O", max_depth=9)
    game_3x3 = ExtendedTicTacToe(rows=3, cols=3, win_k=3)

    while not game_3x3.is_terminal():
        active = ab_full_x if game_3x3.current_player is Player.X else ab_full_o
        game_3x3.step(active.select_action(game_3x3).action)

    assert game_3x3.result(Player.X) == 0.0

    # Paired 10 matches (20 games) AlphaBeta(d=3) vs Random on 4x4 K=4.
    ab = AlphaBetaAgent(max_depth=3, time_budget_ms=50.0)
    rnd = RandomAgent(seed=77)
    wins = 0
    draws = 0
    losses = 0
    for idx in range(10):
        for ab_is_x in (True, False):
            game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
            ab_player = Player.X if ab_is_x else Player.O
            rnd.reset(seed=77 + idx)
            while not game.is_terminal():
                if game.current_player is ab_player:
                    action = ab.select_action(game).action
                else:
                    action = rnd.select_action(game).action
                game.step(action)
            res = game.result(ab_player)
            if res == 1.0:
                wins += 1
            elif res == 0.0:
                draws += 1
            else:
                losses += 1

    assert losses == 0
    assert (wins + 0.5 * draws) / 20.0 >= 0.90


def test_minimax_module_has_no_print_statements() -> None:
    """Enforce RULES.md Section 4.4: minimax_agent.py must never call print()."""
    source = (AGENTS_DIR / "minimax_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="minimax_agent.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "print", "Forbidden print() found in agents/minimax_agent.py"
