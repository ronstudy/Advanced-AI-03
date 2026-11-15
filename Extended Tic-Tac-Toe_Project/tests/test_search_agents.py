"""Search equivalence, pruning efficiency, tactical, and tournament tests (Task T2.5)."""

from __future__ import annotations

import math

import pytest

from ttt_ai import (
    AlphaBetaAgent,
    ExtendedTicTacToe,
    MCTSAgent,
    MinimaxAgent,
    RandomAgent,
    RuleBasedAgent,
)
from ttt_ai.engine.rules import Player


def _make_board(
    rows: int, cols: int, win_k: int, moves: list[int]
) -> ExtendedTicTacToe:
    """Create a deterministic game state by applying sequential actions."""
    game = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)
    for move in moves:
        game.step(move)
    return game


def _play_game(
    rows: int,
    cols: int,
    win_k: int,
    agent_x,
    agent_o,
) -> float:
    """Run one complete game via public APIs and return Player.X's outcome."""
    game = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)
    agent_x.reset()
    agent_o.reset()
    while not game.is_terminal():
        actor = agent_x if game.current_player == Player.X else agent_o
        decision = actor.select_action(game)
        assert decision.action in game.legal_actions()
        game.step(decision.action)
    outcome = game.result(int(Player.X))
    assert outcome is not None
    return outcome


class TestSearchEquivalenceAndPruning:
    """Verify exact Minimax vs Alpha-Beta equivalence and pruning efficiency."""

    @pytest.mark.parametrize(
        ("rows", "cols", "win_k", "moves", "depth"),
        [
            (3, 3, 3, [], 4),
            (3, 3, 3, [4], 5),
            (3, 3, 3, [4, 0, 8], 5),
            (3, 3, 3, [0, 4, 1, 3], 5),
            (3, 3, 3, [1, 2, 3, 6], 5),
            (4, 4, 3, [5, 0, 10, 3], 3),
            (4, 4, 3, [0, 5, 1, 9, 6], 4),
            (4, 4, 4, [], 3),
            (4, 4, 4, [5, 0, 10, 3], 4),
            (4, 4, 4, [0, 5, 1, 6, 2, 9], 4),
        ],
    )
    def test_minimax_and_alphabeta_exact_equivalence_on_fixed_trees(
        self,
        rows: int,
        cols: int,
        win_k: int,
        moves: list[int],
        depth: int,
    ) -> None:
        game = _make_board(rows, cols, win_k, moves)

        pure_mm = MinimaxAgent(
            name="MM-Fixed",
            max_depth=depth,
            use_alpha_beta=False,
            use_move_ordering=False,
            iterative_deepening=False,
        )
        ab_plain = MinimaxAgent(
            name="AB-NoOrder",
            max_depth=depth,
            use_alpha_beta=True,
            use_move_ordering=False,
            iterative_deepening=False,
        )
        ab_ordered = AlphaBetaAgent(
            name="AB-Ordered",
            max_depth=depth,
            use_alpha_beta=True,
            use_move_ordering=True,
            iterative_deepening=False,
        )

        dec_mm = pure_mm.select_action(game)
        dec_ab_plain = ab_plain.select_action(game)
        dec_ab_ord = ab_ordered.select_action(game)

        assert dec_ab_plain.action == dec_mm.action
        assert dec_ab_ord.action == dec_mm.action
        assert math.isclose(
            float(dec_ab_plain.value_estimate),
            float(dec_mm.value_estimate),
            abs_tol=1e-9,
        )
        assert math.isclose(
            float(dec_ab_ord.value_estimate),
            float(dec_mm.value_estimate),
            abs_tol=1e-9,
        )
        assert dec_ab_plain.depth_reached == dec_mm.depth_reached
        assert dec_ab_ord.depth_reached == dec_mm.depth_reached

        # Both Alpha-Beta modes strictly prune nodes compared to pure Minimax.
        assert dec_ab_plain.nodes_visited < dec_mm.nodes_visited
        assert dec_ab_ord.nodes_visited < dec_mm.nodes_visited

    def test_alphabeta_and_move_ordering_node_reduction_gate(self) -> None:
        # 4x4 K=4 empty board at depth=4: Sprint 2 gate requires sharp node drop.
        game_4x4 = _make_board(4, 4, 4, [])
        pure_mm = MinimaxAgent(
            max_depth=4,
            use_alpha_beta=False,
            use_move_ordering=False,
            iterative_deepening=False,
        )
        ab_plain = MinimaxAgent(
            max_depth=4,
            use_alpha_beta=True,
            use_move_ordering=False,
            iterative_deepening=False,
        )
        ab_ordered = AlphaBetaAgent(
            max_depth=4,
            use_alpha_beta=True,
            use_move_ordering=True,
            iterative_deepening=False,
        )

        dec_mm = pure_mm.select_action(game_4x4)
        dec_ab_plain = ab_plain.select_action(game_4x4)
        dec_ab_ord = ab_ordered.select_action(game_4x4)

        assert dec_ab_ord.action == dec_mm.action
        assert dec_ab_ord.nodes_visited < dec_ab_plain.nodes_visited < dec_mm.nodes_visited
        # Alpha-Beta + Move Ordering should cut >85% of Minimax nodes on empty 4x4 d=4.
        assert dec_ab_ord.nodes_visited < 0.15 * dec_mm.nodes_visited

        # 3x3 K=3 partial tree at depth=5.
        game_3x3 = _make_board(3, 3, 3, [4, 0])
        dec_mm_3 = MinimaxAgent(
            max_depth=5,
            use_alpha_beta=False,
            use_move_ordering=False,
            iterative_deepening=False,
        ).select_action(game_3x3)
        dec_ab_3 = AlphaBetaAgent(
            max_depth=5,
            use_alpha_beta=True,
            use_move_ordering=True,
            iterative_deepening=False,
        ).select_action(game_3x3)

        assert dec_ab_3.action == dec_mm_3.action
        assert dec_ab_3.nodes_visited < 0.5 * dec_mm_3.nodes_visited


class TestTacticalConsensusAndBudgets:
    """Verify tactical agreement across Levels 2-4 and compute-budget scaling."""

    @pytest.mark.parametrize(
        ("rows", "cols", "win_k", "moves", "expected_action"),
        [
            # Immediate 1-ply wins
            (3, 3, 3, [0, 3, 1, 4], 2),
            (3, 3, 3, [0, 3, 2, 4, 6], 5),
            (4, 4, 3, [0, 4, 1, 8], 2),
            (4, 4, 4, [0, 1, 5, 2, 10, 12], 15),
            (4, 4, 4, [0, 3, 1, 6, 4, 9, 15], 12),
            # Urgent 1-ply blocks
            (3, 3, 3, [0, 4, 1], 2),
            (3, 3, 3, [0, 1, 8, 4], 7),
            (4, 4, 3, [0, 5, 1], 2),
            (4, 4, 4, [0, 4, 1, 8, 2], 3),
            (4, 4, 4, [4, 0, 8, 5, 9, 10], 15),
            # Unique winning fork (3x3)
            (3, 3, 3, [1, 2, 3, 6], 4),
        ],
    )
    def test_tactical_consensus_across_levels_2_3_4(
        self,
        rows: int,
        cols: int,
        win_k: int,
        moves: list[int],
        expected_action: int,
    ) -> None:
        game = _make_board(rows, cols, win_k, moves)
        agents = [
            RuleBasedAgent(seed=42),
            MinimaxAgent(max_depth=3, use_alpha_beta=False, iterative_deepening=False),
            AlphaBetaAgent(max_depth=4),
            MCTSAgent(n_simulations=300, seed=42),
        ]
        for agent in agents:
            decision = agent.select_action(game)
            assert (
                decision.action == expected_action
            ), f"{agent.name} chose {decision.action} instead of {expected_action}"

    def test_search_scaling_and_time_budget_enforcement(self) -> None:
        game = _make_board(4, 4, 4, [5, 0])

        # Alpha-Beta Iterative Deepening scales depth and nodes monotonically.
        ab_depths = [
            AlphaBetaAgent(max_depth=d, iterative_deepening=True).select_action(game)
            for d in (1, 2, 3)
        ]
        assert [dec.depth_reached for dec in ab_depths] == [1, 2, 3]
        assert (
            ab_depths[0].nodes_visited
            < ab_depths[1].nodes_visited
            < ab_depths[2].nodes_visited
        )

        # Alpha-Beta with tight time budget stays bounded.
        ab_timed = AlphaBetaAgent(
            max_depth=6,
            time_budget_ms=15.0,
        ).select_action(game)
        assert ab_timed.action in game.legal_actions()
        assert ab_timed.latency_ms < 200.0

        # MCTS scales simulations and nodes monotonically, freeing all tree nodes.
        mcts_runs = []
        for n_sims in (25, 75, 150):
            agent = MCTSAgent(n_simulations=n_sims, seed=11)
            dec = agent.select_action(game)
            assert agent.last_tree_nodes_released == dec.nodes_visited
            mcts_runs.append(dec)

        assert [dec.simulations for dec in mcts_runs] == [25, 75, 150]
        assert (
            mcts_runs[0].nodes_visited
            < mcts_runs[1].nodes_visited
            < mcts_runs[2].nodes_visited
        )

        mcts_timed_agent = MCTSAgent(
            n_simulations=50_000,
            time_budget_ms=20.0,
            seed=11,
        )
        mcts_timed = mcts_timed_agent.select_action(game)
        assert 1 <= mcts_timed.simulations < 50_000
        assert mcts_timed.latency_ms < 200.0
        assert mcts_timed_agent.last_tree_nodes_released == mcts_timed.nodes_visited


class TestMultiLevelHierarchyVerification:
    """Paired match sanity tests across Level 1-4 hierarchy on 3x3 and 4x4."""

    def test_unbeaten_3x3_alphabeta_oracle_against_levels_1_2_4(self) -> None:
        oracle = AlphaBetaAgent(max_depth=9, use_alpha_beta=True, use_move_ordering=True)
        opponents = [
            RandomAgent(seed=10),
            RuleBasedAgent(seed=20),
            MCTSAgent(n_simulations=150, seed=30),
        ]

        for opponent in opponents:
            for pair_idx in range(3):
                opponent.set_seed(100 + pair_idx)
                assert _play_game(3, 3, 3, oracle, opponent) >= 0.0
                opponent.set_seed(200 + pair_idx)
                assert _play_game(3, 3, 3, opponent, oracle) <= 0.0

    def test_hierarchy_dominates_random_on_4x4(self) -> None:
        contenders = [
            RuleBasedAgent(seed=1),
            AlphaBetaAgent(max_depth=3, seed=2),
            MCTSAgent(n_simulations=100, seed=3),
        ]
        random_opp = RandomAgent(seed=99)
        n_pairs = 5

        for contender in contenders:
            points = 0.0
            for pair_idx in range(n_pairs):
                contender.set_seed(500 + pair_idx)
                random_opp.set_seed(700 + pair_idx)
                res_x = _play_game(4, 4, 4, contender, random_opp)
                points += 1.0 if res_x > 0 else (0.5 if res_x == 0.0 else 0.0)

                contender.set_seed(600 + pair_idx)
                random_opp.set_seed(800 + pair_idx)
                res_o = _play_game(4, 4, 4, random_opp, contender)
                points += 1.0 if res_o < 0 else (0.5 if res_o == 0.0 else 0.0)

            assert (
                points / (2 * n_pairs) >= 0.80
            ), f"{contender.name} scored {points / (2 * n_pairs):.3f} < 0.80 vs Random"
