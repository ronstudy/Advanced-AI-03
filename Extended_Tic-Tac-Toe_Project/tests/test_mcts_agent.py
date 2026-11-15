"""Unit and integration tests for Level 4 MCTSAgent (Task T2.4)."""

from __future__ import annotations

import ast
import math
import weakref
from pathlib import Path

import numpy as np
import pytest

import ttt_ai.agents.base as base_mod
from ttt_ai import (
    AgentDecision,
    AlphaBetaAgent,
    ExtendedTicTacToe,
    MCTSAgent,
    RandomAgent,
    RuleBasedAgent,
)
from ttt_ai.agents.mcts_agent import _MCTSNode, _release_tree
from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError
from ttt_ai.engine.rules import Player


def _make_board(rows: int, cols: int, win_k: int, moves: list[int]) -> ExtendedTicTacToe:
    """Create a board populated deterministically by alternating moves."""
    game = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)
    for move in moves:
        game.step(move)
    return game


def _play_match(
    game_factory,
    agent_x,
    agent_o,
) -> float | None:
    """Play one full game using public APIs and return X's outcome."""
    game = game_factory()
    agent_x.reset()
    agent_o.reset()
    while not game.is_terminal():
        actor = agent_x if game.current_player == Player.X else agent_o
        decision = actor.select_action(game)
        assert isinstance(decision, AgentDecision)
        assert decision.action in game.legal_actions()
        assert decision.latency_ms >= 0.0
        game.step(decision.action)
    return game.result(int(Player.X))


class TestMCTSConfigAndValidation:
    """Tests for MCTSAgent initialization, setters, exports, and validation."""

    def test_defaults_and_exports(self) -> None:
        agent = MCTSAgent()
        assert agent.level == 4
        assert agent.name == "MCTSAgent"
        assert agent.n_simulations == 200
        assert math.isclose(agent.c_uct, math.sqrt(2.0), rel_tol=1e-12)
        assert agent.seed is None
        assert agent.time_budget_ms is None
        assert agent.last_tree_nodes_released == 0

        cfg = agent.get_config()
        assert cfg["agent_class"] == "MCTSAgent"
        assert cfg["level"] == 4
        assert cfg["n_simulations"] == 200
        assert math.isclose(cfg["c_uct"], math.sqrt(2.0), rel_tol=1e-12)

    @pytest.mark.parametrize("bad_sims", [0, -5, True, False, 10.5, "200", None])
    def test_invalid_n_simulations_rejected(self, bad_sims) -> None:
        with pytest.raises(InvalidConfigError):
            MCTSAgent(n_simulations=bad_sims)
        agent = MCTSAgent()
        with pytest.raises(InvalidConfigError):
            agent.set_n_simulations(bad_sims)

    @pytest.mark.parametrize(
        "bad_c",
        [-0.01, -1, math.nan, math.inf, -math.inf, True, False, "1.4", None],
    )
    def test_invalid_c_uct_rejected(self, bad_c) -> None:
        with pytest.raises(InvalidConfigError):
            MCTSAgent(c_uct=bad_c)
        agent = MCTSAgent()
        with pytest.raises(InvalidConfigError):
            agent.set_c_uct(bad_c)

    def test_valid_setters_including_zero_exploration(self) -> None:
        agent = MCTSAgent(n_simulations=50, c_uct=1.0, name="UCT-Custom")
        assert agent.name == "UCT-Custom"
        agent.set_n_simulations(120)
        agent.set_c_uct(0.0)
        assert agent.n_simulations == 120
        assert agent.c_uct == 0.0


class TestMCTSTelemetryAndTreeLifecycle:
    """Tests for UCT distributions, backpropagation signs, determinism, and cleanup."""

    def test_visit_distribution_and_partial_root_coverage(self) -> None:
        game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
        agent = MCTSAgent(n_simulations=5, seed=11)
        decision = agent.select_action(game)

        legal = game.legal_actions()
        assert set(decision.action_scores.keys()) == set(legal)
        assert decision.simulations == 5
        assert decision.nodes_visited == 6  # root + 5 expanded root children
        assert decision.depth_reached == 1
        assert math.isclose(sum(decision.action_scores.values()), 1.0, abs_tol=1e-9)

        nonzero = [score for score in decision.action_scores.values() if score > 0.0]
        zeros = [score for score in decision.action_scores.values() if score == 0.0]
        assert len(nonzero) == 5
        assert len(zeros) == 11
        for score in nonzero:
            assert math.isclose(score, 1.0 / 5.0, abs_tol=1e-9)
        assert decision.action_scores[decision.action] == max(
            decision.action_scores.values()
        )

    def test_robust_child_matches_highest_visit_share(self) -> None:
        game = _make_board(3, 3, 3, [0, 4, 1, 6])
        agent = MCTSAgent(n_simulations=150, seed=7)
        decision = agent.select_action(game)

        max_score = max(decision.action_scores.values())
        assert math.isclose(
            decision.action_scores[decision.action], max_score, abs_tol=1e-12
        )
        assert math.isclose(sum(decision.action_scores.values()), 1.0, abs_tol=1e-9)
        assert -1.0 <= float(decision.value_estimate) <= 1.0
        assert decision.simulations == 150
        assert decision.depth_reached >= 1

    def test_backpropagate_flips_zero_sum_sign_per_ply(self) -> None:
        root = _MCTSNode(
            parent=None,
            action_from_parent=None,
            just_moved_player=Player.O,
            player_to_move=Player.X,
            depth=0,
            untried_actions=[],
        )
        ply1 = _MCTSNode(
            parent=root,
            action_from_parent=4,
            just_moved_player=Player.X,
            player_to_move=Player.O,
            depth=1,
            untried_actions=[],
        )
        ply2 = _MCTSNode(
            parent=ply1,
            action_from_parent=0,
            just_moved_player=Player.O,
            player_to_move=Player.X,
            depth=2,
            untried_actions=[],
        )
        root.children[4] = ply1
        ply1.children[0] = ply2

        # Suppose X wins after O's ply2 move => O's outcome at ply2 is -1.0.
        MCTSAgent._backpropagate(ply2, -1.0)
        assert ply2.visits == 1 and math.isclose(ply2.q_value, -1.0)
        assert ply1.visits == 1 and math.isclose(ply1.q_value, +1.0)
        assert root.visits == 1 and math.isclose(root.q_value, -1.0)

    def test_seed_reproducibility_and_reset(self) -> None:
        game = _make_board(4, 4, 4, [5, 0, 10, 3])
        agent_a = MCTSAgent(n_simulations=120, seed=1234)
        agent_b = MCTSAgent(n_simulations=120, seed=1234)

        dec_a = agent_a.select_action(game)
        dec_b = agent_b.select_action(game)

        assert dec_a.action == dec_b.action
        assert dec_a.simulations == dec_b.simulations
        assert dec_a.nodes_visited == dec_b.nodes_visited
        assert dec_a.depth_reached == dec_b.depth_reached
        assert dec_a.value_estimate == dec_b.value_estimate
        assert dec_a.action_scores == dec_b.action_scores

        # Running another move advances RNG, then reset(seed=1234) reproduces.
        agent_a.select_action(game)
        agent_a.reset(seed=1234)
        dec_a_reset = agent_a.select_action(game)
        assert dec_a_reset.action == dec_b.action
        assert dec_a_reset.action_scores == dec_b.action_scores

    def test_tree_release_breaks_cycles_without_gc(self) -> None:
        root = _MCTSNode(
            parent=None,
            action_from_parent=None,
            just_moved_player=Player.O,
            player_to_move=Player.X,
            depth=0,
            untried_actions=[1, 2],
        )
        child = _MCTSNode(
            parent=root,
            action_from_parent=0,
            just_moved_player=Player.X,
            player_to_move=Player.O,
            depth=1,
            untried_actions=[],
        )
        grandchild = _MCTSNode(
            parent=child,
            action_from_parent=3,
            just_moved_player=Player.O,
            player_to_move=Player.X,
            depth=2,
            untried_actions=[],
        )
        root.children[0] = child
        child.children[3] = grandchild

        refs = [weakref.ref(root), weakref.ref(child), weakref.ref(grandchild)]
        released = _release_tree(root)
        assert released == 3
        del root, child, grandchild
        # Without cyclic GC, reference counting frees every detached node at once.
        assert all(ref() is None for ref in refs)
        assert _release_tree(None) == 0

    def test_agent_releases_every_created_node_across_moves(self) -> None:
        game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
        agent = MCTSAgent(n_simulations=100, seed=99)

        for _ in range(10):
            if game.is_terminal():
                break
            decision = agent.select_action(game)
            assert agent.last_tree_nodes_released == decision.nodes_visited
            assert not hasattr(agent, "_root")
            game.step(decision.action)


class TestMCTSTacticsAndEdgeCases:
    """Tests for tactical win/block accuracy, time budgets, and boundary states."""

    def test_finds_immediate_win_3x3_and_4x4(self) -> None:
        # 3x3 K=3: X has (0,0)=0 and (0,1)=1; cell 2 wins immediately for X.
        game_3x3 = _make_board(3, 3, 3, [0, 3, 1, 4])
        dec_3x3 = MCTSAgent(n_simulations=200, seed=5).select_action(game_3x3)
        assert dec_3x3.action == 2
        assert float(dec_3x3.value_estimate) > 0.8

        # 4x4 K=3: O has (1,0)=4 and (1,1)=5; cell 6 wins immediately for O.
        game_4x3 = _make_board(4, 4, 3, [0, 4, 15, 5, 12])
        dec_4x3 = MCTSAgent(n_simulations=200, seed=5).select_action(game_4x3)
        assert dec_4x3.action == 6
        assert float(dec_4x3.value_estimate) > 0.8

        # 4x4 K=4: X has main diagonal 0, 5, 10; cell 15 wins immediately.
        game_4x4 = _make_board(4, 4, 4, [0, 1, 5, 2, 10, 12])
        dec_4x4 = MCTSAgent(n_simulations=250, seed=5).select_action(game_4x4)
        assert dec_4x4.action == 15
        assert float(dec_4x4.value_estimate) > 0.8

    def test_blocks_urgent_threat_as_x_and_o(self) -> None:
        # 3x3 K=3: X threatens row 0 via cells 0, 1; O must block cell 2.
        game_o_blocks = _make_board(3, 3, 3, [0, 4, 1])
        dec_o = MCTSAgent(n_simulations=200, seed=19).select_action(game_o_blocks)
        assert dec_o.action == 2

        # 3x3 K=3: X={0,1,3}, O={2,5,6}; O threatens col 2 (2,5,8), X must block 8.
        game_x_blocks = _make_board(3, 3, 3, [0, 2, 3, 5, 1, 6])
        dec_x = MCTSAgent(n_simulations=200, seed=19).select_action(game_x_blocks)
        assert dec_x.action == 8

        # 4x4 K=4: X={0,1,2}, O={4,8}; O to move must block cell 3.
        game_4x4_block = _make_board(4, 4, 4, [0, 4, 1, 8, 2])
        dec_4x4 = MCTSAgent(n_simulations=300, seed=19).select_action(game_4x4_block)
        assert dec_4x4.action == 3

    def test_prefers_winning_fork_on_3x3(self) -> None:
        # X={1, 3}, O={2, 6} -> X to move; only cell 4 creates a double threat (5 & 7).
        unique_fork = _make_board(3, 3, 3, [1, 2, 3, 6])
        dec_unique = MCTSAgent(n_simulations=300, seed=42).select_action(unique_fork)
        assert dec_unique.action == 4
        assert float(dec_unique.value_estimate) > 0.3

        # X={0, 6}, O={1, 3} -> both 4 (0-4-8 / 2-4-6) and 8 (0-4-8 / 6-7-8) fork.
        dual_fork = _make_board(3, 3, 3, [0, 1, 6, 3])
        dec_dual = MCTSAgent(n_simulations=300, seed=42).select_action(dual_fork)
        assert dec_dual.action in {4, 8}
        assert dec_dual.action_scores[4] + dec_dual.action_scores[8] > 0.70
        assert float(dec_dual.value_estimate) > 0.3

    def test_time_budget_stops_cleanly_after_at_least_one_sim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ticks = iter([10.0, 10.005, 10.010, 10.015, 10.025, 10.050, 10.100])
        last_val = 10.100

        def fake_perf_counter() -> float:
            nonlocal last_val
            try:
                last_val = next(ticks)
            except StopIteration:
                pass
            return last_val

        monkeypatch.setattr(base_mod.time, "perf_counter", fake_perf_counter)

        # Budget = 20 ms -> deadline = 10.020; ticks cross deadline after 3 sims:
        # start=10.0, sim1=10.005, sim2=10.010, sim3=10.015, sim4=10.025 (>= 10.020).
        game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
        agent = MCTSAgent(n_simulations=500, time_budget_ms=20.0, seed=3)
        decision = agent.select_action(game)

        assert 1 <= decision.simulations < 500
        assert decision.simulations == 4
        assert math.isclose(sum(decision.action_scores.values()), 1.0, abs_tol=1e-9)
        assert decision.action in game.legal_actions()

    def test_tiny_time_budget_still_completes_minimum_one_simulation(self) -> None:
        game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
        agent = MCTSAgent(n_simulations=10_000, time_budget_ms=0.001, seed=2)
        decision = agent.select_action(game)
        assert decision.simulations >= 1
        assert decision.simulations < 10_000
        assert decision.action in game.legal_actions()
        assert math.isclose(sum(decision.action_scores.values()), 1.0, abs_tol=1e-9)

    def test_wall_clock_time_budget_bounds_large_simulation_target(self) -> None:
        game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
        agent = MCTSAgent(n_simulations=50_000, time_budget_ms=25.0, seed=1)
        decision = agent.select_action(game)
        assert 1 <= decision.simulations < 50_000
        assert decision.latency_ms < 250.0

    def test_edge_cases_single_empty_terminal_and_immutability(self) -> None:
        # 3x3 board with only cell 8 empty (draw position).
        draw_prefix = _make_board(3, 3, 3, [0, 1, 2, 4, 3, 6, 7, 5])
        assert draw_prefix.legal_actions() == [8]
        before_board = draw_prefix.board.copy()
        before_history = draw_prefix.move_history

        agent = MCTSAgent(n_simulations=30, seed=0)
        decision = agent.select_action(draw_prefix)
        assert decision.action == 8
        assert math.isclose(decision.action_scores[8], 1.0, abs_tol=1e-12)
        assert math.isclose(float(decision.value_estimate), 0.0, abs_tol=1e-12)
        assert decision.simulations == 30
        assert np.array_equal(draw_prefix.board, before_board)
        assert draw_prefix.move_history == before_history

        # Terminal board raises IllegalMoveError.
        terminal_game = _make_board(3, 3, 3, [0, 3, 1, 4, 2])
        assert terminal_game.is_terminal()
        with pytest.raises(IllegalMoveError):
            agent.select_action(terminal_game)

    def test_mcts_module_has_no_print_or_private_engine_access(self) -> None:
        source_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "ttt_ai"
            / "agents"
            / "mcts_agent.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        forbidden_attrs = {"_board", "_history", "_current_player", "_terminal", "_winner"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print"
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attrs


class TestMCTSIntegrationMatches:
    """Multi-level game-loop integration tests for Level 4 vs Levels 1, 2, 3."""

    def test_mcts_dominates_random_on_4x4(self) -> None:
        mcts = MCTSAgent(n_simulations=120, seed=101)
        rand = RandomAgent(seed=202)
        points = 0.0
        n_pairs = 5

        for pair_idx in range(n_pairs):
            mcts.set_seed(1000 + pair_idx)
            rand.set_seed(2000 + pair_idx)
            res_x = _play_match(lambda: ExtendedTicTacToe(4, 4, 4), mcts, rand)
            assert res_x in (-1.0, 0.0, 1.0)
            points += 1.0 if res_x > 0 else (0.5 if res_x == 0.0 else 0.0)

            mcts.set_seed(3000 + pair_idx)
            rand.set_seed(4000 + pair_idx)
            res_o = _play_match(lambda: ExtendedTicTacToe(4, 4, 4), rand, mcts)
            assert res_o in (-1.0, 0.0, 1.0)
            points += 1.0 if res_o < 0 else (0.5 if res_o == 0.0 else 0.0)

        assert points / (2 * n_pairs) >= 0.75

    def test_mcts_completes_valid_games_against_rule_based_and_alphabeta(self) -> None:
        mcts = MCTSAgent(n_simulations=100, time_budget_ms=80.0, seed=17)
        rule_agent = RuleBasedAgent(seed=18)
        ab_agent = AlphaBetaAgent(max_depth=3, time_budget_ms=80.0, seed=19)

        for opponent in (rule_agent, ab_agent):
            res_x = _play_match(lambda: ExtendedTicTacToe(4, 4, 4), mcts, opponent)
            res_o = _play_match(lambda: ExtendedTicTacToe(4, 4, 4), opponent, mcts)
            assert res_x in (-1.0, 0.0, 1.0)
            assert res_o in (-1.0, 0.0, 1.0)
