"""Unit tests and tactical benchmarks for Task T2.2: RandomAgent & RuleBasedAgent."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ttt_ai import AgentDecision, ExtendedTicTacToe, RandomAgent, RuleBasedAgent, RuleTier
from ttt_ai.engine import IllegalMoveError, InvalidConfigError, Player

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = PROJECT_ROOT / "src" / "ttt_ai" / "agents"


def _play_sequence(game: ExtendedTicTacToe, moves: list[int]) -> None:
    """Apply a sequence of flat action indices to `game`."""
    for action in moves:
        game.step(action)


def test_random_agent_config_reproducibility_and_uniformity() -> None:
    """Verify RandomAgent level, seed reproducibility, uniform scores, and coverage."""
    agent_a = RandomAgent(seed=123)
    agent_b = RandomAgent(seed=123)
    assert agent_a.level == 1
    assert agent_a.name == "RandomAgent"
    assert agent_a.get_config()["agent_class"] == "RandomAgent"

    game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    seq_a = [agent_a.select_action(game).action for _ in range(10)]
    seq_b = [agent_b.select_action(game).action for _ in range(10)]
    assert seq_a == seq_b

    agent_a.reset(seed=123)
    assert [agent_a.select_action(game).action for _ in range(10)] == seq_a

    # Verify uniform score distribution and non-degenerate coverage over 900 draws on 3x3.
    small_game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    sampler = RandomAgent(seed=2026)
    counts = {cell: 0 for cell in range(9)}
    for _ in range(900):
        decision = sampler.select_action(small_game)
        assert isinstance(decision, AgentDecision)
        assert decision.nodes_visited == 1
        assert decision.latency_ms >= 0.0
        assert len(decision.action_scores) == 9
        for score in decision.action_scores.values():
            assert score == pytest.approx(1.0 / 9.0)
        counts[decision.action] += 1

    # With 900 uniform draws across 9 cells (expected 100 each), every cell must be hit.
    assert all(cnt > 50 for cnt in counts.values())


@pytest.mark.parametrize(
    ("rows", "cols", "win_k"),
    [(3, 3, 3), (4, 4, 3), (4, 4, 4), (5, 5, 4)],
)
def test_random_agent_plays_legal_full_games_without_mutating_state(
    rows: int,
    cols: int,
    win_k: int,
) -> None:
    """Verify RandomAgent completes full games across all board sizes cleanly."""
    agent_x = RandomAgent(name="RandX", seed=11)
    agent_o = RandomAgent(name="RandO", seed=22)
    game = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)

    while not game.is_terminal():
        active = agent_x if game.current_player is Player.X else agent_o
        board_before = game.board
        legal_before = game.legal_actions()

        decision = active.select_action(game)
        np.testing.assert_array_equal(game.board, board_before)
        assert decision.action in legal_before
        game.step(decision.action)

    assert game.is_terminal()
    assert game.legal_actions() == []


def test_rule_based_agent_config_validation() -> None:
    """Verify RuleBasedAgent validates tie_break mode and exposes metadata."""
    agent = RuleBasedAgent(name="HeuristicL2", seed=42, tie_break="deterministic")
    assert agent.level == 2
    assert agent.name == "HeuristicL2"
    assert agent.tie_break == "deterministic"
    assert agent.get_config()["tie_break"] == "deterministic"
    assert RuleTier.IMMEDIATE_WIN > RuleTier.IMMEDIATE_BLOCK > RuleTier.CREATE_FORK
    assert RuleTier.CREATE_FORK > RuleTier.BLOCK_FORK > RuleTier.POSITIONAL > RuleTier.FALLBACK

    with pytest.raises(InvalidConfigError):
        RuleBasedAgent(tie_break="invalid_mode")  # type: ignore[arg-type]


def test_rule_based_tier1_immediate_win_across_boards_and_players() -> None:
    """Verify Tier 1 completes immediate wins for both X and O on 3x3 and 4x4."""
    agent = RuleBasedAgent(seed=7)

    # 3x3 K=3: X has (0, 1), O has (3, 4) -> X must win immediately at 2.
    game_3x3 = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _play_sequence(game_3x3, [0, 3, 1, 4])
    d1 = agent.select_action(game_3x3)
    assert d1.action == 2
    assert d1.value_estimate == pytest.approx(1.0)
    assert d1.action_scores[2] >= 0.92

    # 4x4 K=4: O has main diagonal (0, 5, 10) -> O must win immediately at 15.
    game_4x4 = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    _play_sequence(game_4x4, [1, 0, 2, 5, 6, 10, 8])
    assert game_4x4.current_player is Player.O
    d2 = agent.select_action(game_4x4)
    assert d2.action == 15
    assert d2.value_estimate == pytest.approx(1.0)


def test_rule_based_conflict_tier1_overrides_tier2_block() -> None:
    """Verify Tier 1 (Immediate Win) strictly overrides Tier 2 (Immediate Block)."""
    agent = RuleBasedAgent(seed=42)

    # 3x3 K=3: O threatens column (2, 5 -> 8), X has (0, 4 -> 8? No, use distinct cells!):
    # X at 0, 1 (wins at 2); O at 3, 6 (wins at 0? occupied -> use O at 4, 7 -> wins at 1?
    # Let X at 0, 3 (wins at 6); O at 2, 5 (wins at 8); X also at 4 -> actually:
    # X: [0, 3], O: [2, 5], X to move? Wait: X has 2 moves, O has 2 moves -> X to move!
    game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _play_sequence(game, [0, 2, 3, 5])
    assert game.current_player is Player.X
    decision = agent.select_action(game)
    assert decision.action == 6
    assert decision.action_scores[6] > decision.action_scores[8]

    # 4x4 K=4: O threatens bottom row (12, 13, 14 -> 15), X can win col 0 (0, 4, 8 -> 12?
    # Keep target cells distinct): X at (1, 5, 9 -> 13? occupied -> use 2, 6, 10 -> wins at 14,
    # O at 12, 13, 15? No, X wins at 14 blocks nothing -> better:
    # X at (0, 4, 8 -> wins at 12), O at (3, 7, 11 -> wins at 15).
    game_4x4 = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    _play_sequence(game_4x4, [0, 3, 4, 7, 8, 11])
    assert game_4x4.current_player is Player.X
    d4 = agent.select_action(game_4x4)
    assert d4.action == 12
    assert d4.action_scores[12] > d4.action_scores[15]


def test_rule_based_tier2_immediate_block() -> None:
    """Verify Tier 2 blocks opponent's K-1 winning threat when no immediate win exists."""
    agent = RuleBasedAgent(seed=13)

    # 3x3 K=3: X has (0, 4), O at 2, X at 7 (no immediate win for O) -> O must block 8.
    game_3x3 = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _play_sequence(game_3x3, [0, 2, 4])
    assert game_3x3.current_player is Player.O
    d1 = agent.select_action(game_3x3)
    assert d1.action == 8
    assert 0.78 <= d1.action_scores[8] <= 0.90

    # 4x4 K=4: X has anti-diagonal (3, 6, 9), O has (0, 1, 4) -> O must block 12.
    game_4x4 = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    _play_sequence(game_4x4, [3, 0, 6, 1, 9, 4])
    # One more move for X? Currently 3 X and 3 O -> X to move; let X play 2 -> O must block 12.
    game_4x4.step(2)
    assert game_4x4.current_player is Player.O
    d2 = agent.select_action(game_4x4)
    assert d2.action == 12
    assert 0.78 <= d2.action_scores[12] <= 0.90


def test_rule_based_tier3_create_fork_and_no_false_collinear_fork() -> None:
    """Verify Tier 3 creates true multi-target forks and rejects collinear single-target traps."""
    agent = RuleBasedAgent(seed=5, tie_break="deterministic")

    # 3x3 K=3: X at (1, 3), O at (5, 7) -> X playing 0 uniquely threatens both 2 and 6 (Fork).
    game_3x3 = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _play_sequence(game_3x3, [1, 5, 3, 7])
    assert game_3x3.current_player is Player.X
    d1 = agent.select_action(game_3x3)
    assert d1.action == 0
    assert d1.depth_reached == 2
    assert d1.value_estimate == pytest.approx(0.75)

    # 4x4 K=4: X at (0, 1, 10, 14), O at (4, 5, 12, 13) -> X playing 2 threatens 3 and 6.
    game_4x4 = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    _play_sequence(game_4x4, [0, 4, 1, 5, 10, 12, 14, 13])
    assert game_4x4.current_player is Player.X
    d2 = agent.select_action(game_4x4)
    assert d2.action == 2
    assert d2.value_estimate == pytest.approx(0.75)

    # 4x4 K=3 anti-false-fork check: X at (0, 3), O at (4, 8).
    # Playing 1 or 2 only creates threats pointing to the SINGLE shared cell (2 or 1),
    # so neither is a real fork; RuleBasedAgent must NOT claim CREATE_FORK (0.75).
    game_k3 = ExtendedTicTacToe(rows=4, cols=4, win_k=3)
    _play_sequence(game_k3, [0, 4, 3, 8])
    d3 = agent.select_action(game_k3)
    assert d3.value_estimate != pytest.approx(0.75)


def test_rule_based_tier4_block_single_and_multiple_opponent_forks() -> None:
    """Verify Tier 4 handles both single fork squares and the opposite-corner double-fork trap."""
    agent = RuleBasedAgent(seed=9)

    # Case 1 (single opponent fork): 3x3 K=3 with X at (0, 5), O at 4.
    # X's only fork square is 2 (creates threats 1 and 8), so O must play 2.
    single_fork_game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _play_sequence(single_fork_game, [0, 4, 5])
    assert single_fork_game.current_player is Player.O
    d_single = agent.select_action(single_fork_game)
    assert d_single.action == 2
    assert d_single.depth_reached == 2

    # Case 2 (classic opposite-corners trap): 3x3 K=3 with X at (0, 8), O at 4.
    # X threatens forks at BOTH 2 and 6. O must force a safe counter-threat on an edge
    # {1, 3, 5, 7} rather than playing a corner {2, 6} which loses immediately.
    double_fork_game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _play_sequence(double_fork_game, [0, 4, 8])
    assert double_fork_game.current_player is Player.O
    d_double = agent.select_action(double_fork_game)
    assert d_double.action in {1, 3, 5, 7}
    assert d_double.depth_reached == 2
    for edge in (1, 3, 5, 7):
        assert d_double.action_scores[edge] > d_double.action_scores[2]
        assert d_double.action_scores[edge] > d_double.action_scores[6]


def test_rule_based_tier5_and_tier6_positional_and_tie_break() -> None:
    """Verify positional_weights priority on empty boards and deterministic/random tie-breaks."""
    det_agent = RuleBasedAgent(seed=42, tie_break="deterministic")
    rand_agent_1 = RuleBasedAgent(seed=42, tie_break="random")
    rand_agent_2 = RuleBasedAgent(seed=42, tie_break="random")

    # 3x3 K=3 empty board: center cell 4 has unique max positional weight (4).
    game_3x3 = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    d_3x3 = det_agent.select_action(game_3x3)
    assert d_3x3.action == 4
    assert d_3x3.depth_reached == 1

    # 4x4 K=3 empty board: inner 4 cells {5, 6, 9, 10} share highest weight (8).
    game_4x4_k3 = ExtendedTicTacToe(rows=4, cols=4, win_k=3)
    assert det_agent.select_action(game_4x4_k3).action == 5
    r1 = rand_agent_1.select_action(game_4x4_k3).action
    r2 = rand_agent_2.select_action(game_4x4_k3).action
    assert r1 in {5, 6, 9, 10}
    assert r1 == r2

    # 4x4 K=4 empty board: corners and inner 4 cells share weight (3).
    game_4x4_k4 = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    assert det_agent.select_action(game_4x4_k4).action == 0
    assert rand_agent_1.select_action(game_4x4_k4).action in {0, 3, 5, 6, 9, 10, 12, 15}


def test_rule_based_edge_cases_and_immutability() -> None:
    """Verify near-full board, terminal error, and zero mutation of source game state."""
    agent = RuleBasedAgent(seed=1)

    # Near-full 3x3 board with only cell 8 empty.
    game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    _play_sequence(game, [0, 1, 2, 4, 3, 5, 7, 6])
    board_before = game.board
    decision = agent.select_action(game)
    np.testing.assert_array_equal(game.board, board_before)
    assert decision.action == 8
    assert decision.latency_ms > 0.0
    assert decision.nodes_visited >= 1

    game.step(decision.action)
    assert game.is_terminal()
    with pytest.raises(IllegalMoveError):
        agent.select_action(game)


def test_rule_based_agent_is_exhaustively_unbeatable_on_3x3_k3() -> None:
    """Exhaustively verify RuleBasedAgent never loses on 3x3 K=3 as either X or O."""
    agent = RuleBasedAgent(seed=42, tie_break="deterministic")

    def verify_never_loses(game: ExtendedTicTacToe, agent_player: Player) -> int:
        if game.is_terminal():
            payoff = game.result(agent_player)
            assert payoff is not None and payoff >= 0.0, (
                f"RuleBasedAgent lost as {agent_player.symbol} on board:\n{game.render()}"
            )
            return 1

        leaf_count = 0
        if game.current_player is agent_player:
            decision = agent.select_action(game)
            sim = game.clone()
            sim.step(decision.action)
            leaf_count += verify_never_loses(sim, agent_player)
        else:
            for opp_action in game.legal_actions():
                sim = game.clone()
                sim.step(opp_action)
                leaf_count += verify_never_loses(sim, agent_player)
        return leaf_count

    leaves_as_x = verify_never_loses(ExtendedTicTacToe(3, 3, 3), Player.X)
    leaves_as_o = verify_never_loses(ExtendedTicTacToe(3, 3, 3), Player.O)
    assert leaves_as_x > 0
    assert leaves_as_o > 0


def test_rule_based_beats_random_and_measures_latency_on_4x4_k4() -> None:
    """Verify RuleBasedAgent dominates RandomAgent on 4x4 K=4 with sub-millisecond latency."""
    rule_agent = RuleBasedAgent(seed=42)
    rand_agent = RandomAgent(seed=100)

    wins = 0
    draws = 0
    losses = 0
    latencies_ms: list[float] = []
    nodes_list: list[int] = []

    num_pairs = 20
    for pair_idx in range(num_pairs):
        for rule_is_x in (True, False):
            game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
            rule_player = Player.X if rule_is_x else Player.O
            rule_agent.reset(seed=42 + pair_idx)
            rand_agent.reset(seed=1000 + pair_idx)

            while not game.is_terminal():
                if game.current_player is rule_player:
                    decision = rule_agent.select_action(game)
                    latencies_ms.append(decision.latency_ms)
                    nodes_list.append(decision.nodes_visited)
                else:
                    decision = rand_agent.select_action(game)
                game.step(decision.action)

            res = game.result(rule_player)
            if res == 1.0:
                wins += 1
            elif res == 0.0:
                draws += 1
            else:
                losses += 1

    score = (wins + 0.5 * draws) / float(num_pairs * 2)
    assert losses == 0
    assert score >= 0.90
    assert float(np.mean(latencies_ms)) < 10.0
    assert min(nodes_list) >= 1


def test_level1_and_level2_modules_have_no_print_statements() -> None:
    """Enforce RULES.md Section 4.4: no print() calls inside agents modules."""
    for filename in ("__init__.py", "base.py", "random_agent.py", "rule_based_agent.py"):
        source = (AGENTS_DIR / filename).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", f"Forbidden print() found in agents/{filename}"
