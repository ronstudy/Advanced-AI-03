"""Unified API and behavioral contract tests for Level 1-7 AI agents (T2.5 & T3.4)."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ttt_ai import (
    AgentDecision,
    AlphaBetaAgent,
    BaseAgent,
    DQNAgent,
    ExtendedTicTacToe,
    HybridMCTSAgent,
    MCTSAgent,
    MinimaxAgent,
    PolicyNetworkAgent,
    RandomAgent,
    RuleBasedAgent,
    TabularQAgent,
    ValueGreedyAgent,
)
from ttt_ai.engine.exceptions import IllegalMoveError
from ttt_ai.engine.rules import Player

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def _make_board(
    rows: int, cols: int, win_k: int, moves: list[int]
) -> ExtendedTicTacToe:
    """Create a board instance by applying deterministic sequential moves."""
    game = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)
    for move in moves:
        game.step(move)
    return game


def _make_level1_to_4_agents(
    seed: int = 42,
    search_depth: int = 3,
    n_simulations: int = 60,
    time_budget_ms: float | None = None,
) -> list[BaseAgent]:
    """Instantiate one representative agent per variant across Levels 1 through 4."""
    return [
        RandomAgent(seed=seed, time_budget_ms=time_budget_ms),
        RuleBasedAgent(seed=seed, time_budget_ms=time_budget_ms),
        MinimaxAgent(
            name="MinimaxPure",
            max_depth=search_depth,
            use_alpha_beta=False,
            use_move_ordering=False,
            seed=seed,
            time_budget_ms=time_budget_ms,
        ),
        AlphaBetaAgent(
            max_depth=search_depth,
            seed=seed,
            time_budget_ms=time_budget_ms,
        ),
        MCTSAgent(
            n_simulations=n_simulations,
            seed=seed,
            time_budget_ms=time_budget_ms,
        ),
    ]


class TestAgentContractsLevel1To4:
    """Contract tests verifying shared BaseAgent guarantees for Levels 1-4."""

    def test_agent_hierarchy_levels_and_serializable_configs(self) -> None:
        agents = _make_level1_to_4_agents(seed=7)
        expected_levels = [1, 2, 3, 3, 4]
        required_keys = {"agent_class", "name", "level", "seed", "time_budget_ms"}

        for agent, expected_level in zip(agents, expected_levels, strict=True):
            assert isinstance(agent, BaseAgent)
            assert agent.level == expected_level
            cfg = agent.get_config()
            assert required_keys.issubset(cfg.keys())
            assert cfg["level"] == expected_level
            assert cfg["seed"] == 7
            # Config metadata must serialize cleanly for JSONL tournament headers.
            serialized = json.dumps(cfg)
            assert isinstance(serialized, str)

    @pytest.mark.parametrize(
        ("rows", "cols", "win_k", "moves", "depth"),
        [
            (3, 3, 3, [], 3),
            (3, 3, 3, [4, 0, 8, 2], 3),
            (3, 3, 3, [0, 1, 2, 4, 3, 6, 7, 5], 3),
            (4, 4, 3, [], 3),
            (4, 4, 3, [5, 0, 10, 3], 3),
            (4, 4, 4, [], 3),
            (4, 4, 4, [5, 0, 10, 3], 3),
            (4, 4, 4, [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14], 3),
            (5, 5, 4, [], 2),
            (5, 5, 4, [12, 0, 6, 24, 18, 4], 2),
        ],
    )
    def test_source_game_immutability_across_boards_and_stages(
        self,
        rows: int,
        cols: int,
        win_k: int,
        moves: list[int],
        depth: int,
    ) -> None:
        game = _make_board(rows, cols, win_k, moves)
        before_board = game.board.copy()
        before_history = game.move_history
        before_player = game.current_player
        before_terminal = game.is_terminal()
        before_winner = game.winner

        for agent in _make_level1_to_4_agents(seed=13, search_depth=depth, n_simulations=40):
            decision = agent.select_action(game)
            assert decision.action in game.legal_actions()
            assert np.array_equal(game.board, before_board)
            assert game.move_history == before_history
            assert game.current_player == before_player
            assert game.is_terminal() == before_terminal
            assert game.winner == before_winner

    @pytest.mark.parametrize(
        ("rows", "cols", "win_k", "moves"),
        [
            (3, 3, 3, []),
            (3, 3, 3, [4, 0, 8]),
            (4, 4, 4, []),
            (4, 4, 4, [5, 0, 10]),
        ],
    )
    def test_legal_action_and_decision_telemetry_contract(
        self,
        rows: int,
        cols: int,
        win_k: int,
        moves: list[int],
    ) -> None:
        game = _make_board(rows, cols, win_k, moves)
        legal = game.legal_actions()
        legal_set = set(legal)

        for agent in _make_level1_to_4_agents(seed=21, search_depth=3, n_simulations=60):
            decision = agent.select_action(game)
            assert isinstance(decision, AgentDecision)
            assert decision.action in legal_set
            assert math.isfinite(decision.latency_ms) and decision.latency_ms >= 0.0
            assert decision.value_estimate is not None
            assert math.isfinite(decision.value_estimate)
            assert set(decision.action_scores.keys()) == legal_set

            top3 = decision.top_actions(k=3)
            assert len(top3) == min(3, len(legal))
            for idx in range(len(top3) - 1):
                a_curr, s_curr = top3[idx]
                a_next, s_next = top3[idx + 1]
                assert (s_curr > s_next) or (s_curr == s_next and a_curr < a_next)

            # Decision dictionary must be strictly JSON-serializable.
            assert isinstance(json.dumps(decision.to_dict()), str)

            if isinstance(agent, RandomAgent):
                assert decision.nodes_visited == 1
                assert decision.simulations == 0
                assert math.isclose(
                    sum(decision.action_scores.values()), 1.0, abs_tol=1e-9
                )
            elif isinstance(agent, RuleBasedAgent):
                assert decision.nodes_visited >= 3 * len(legal)
                assert decision.simulations == 0
                assert decision.depth_reached in (1, 2)
                assert math.isclose(
                    decision.action_scores[decision.action],
                    max(decision.action_scores.values()),
                    abs_tol=1e-12,
                )
            elif isinstance(agent, MinimaxAgent):
                assert decision.nodes_visited >= len(legal)
                assert decision.simulations == 0
                assert decision.depth_reached >= 1
                assert math.isclose(
                    decision.action_scores[decision.action],
                    max(decision.action_scores.values()),
                    abs_tol=1e-12,
                )
            elif isinstance(agent, MCTSAgent):
                assert decision.simulations == 60
                assert decision.nodes_visited >= 2
                assert decision.depth_reached >= 1
                assert math.isclose(
                    sum(decision.action_scores.values()), 1.0, abs_tol=1e-9
                )
                assert math.isclose(
                    decision.action_scores[decision.action],
                    max(decision.action_scores.values()),
                    abs_tol=1e-12,
                )
                assert agent.last_tree_nodes_released == decision.nodes_visited

    def test_single_remaining_move_and_terminal_or_invalid_game_rejection(self) -> None:
        near_full_3x3 = _make_board(3, 3, 3, [0, 1, 2, 4, 3, 6, 7, 5])
        near_full_4x4 = _make_board(
            4, 4, 4, [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14]
        )
        terminal_win = _make_board(3, 3, 3, [0, 3, 1, 4, 2])
        terminal_draw = _make_board(3, 3, 3, [0, 1, 2, 4, 3, 6, 7, 5, 8])

        assert terminal_win.is_terminal() and terminal_draw.is_terminal()

        for agent in _make_level1_to_4_agents(seed=5):
            assert agent.select_action(near_full_3x3).action == 8
            assert agent.select_action(near_full_4x4).action == 15

            with pytest.raises(IllegalMoveError):
                agent.select_action(terminal_win)
            with pytest.raises(IllegalMoveError):
                agent.select_action(terminal_draw)
            with pytest.raises(TypeError):
                agent.select_action("not_a_game")  # type: ignore[arg-type]

    def test_seed_reproducibility_and_reset_contract(self) -> None:
        game = _make_board(4, 4, 4, [5, 0, 10, 3])
        agents_a = _make_level1_to_4_agents(seed=99, search_depth=3, n_simulations=80)
        agents_b = _make_level1_to_4_agents(seed=99, search_depth=3, n_simulations=80)

        for agent_a, agent_b in zip(agents_a, agents_b, strict=True):
            dec_a = agent_a.select_action(game)
            dec_b = agent_b.select_action(game)
            assert dec_a.action == dec_b.action
            assert dec_a.action_scores == dec_b.action_scores
            assert dec_a.value_estimate == dec_b.value_estimate

            # Advance RNG with one extra turn, then reset(seed=99) to reproduce.
            agent_a.select_action(game)
            agent_a.reset(seed=99)
            dec_a_reset = agent_a.select_action(game)
            assert dec_a_reset.action == dec_b.action
            assert dec_a_reset.action_scores == dec_b.action_scores

    def test_all_agent_modules_ast_no_print_and_no_private_engine_access(self) -> None:
        agents_dir = (
            Path(__file__).resolve().parent.parent / "src" / "ttt_ai" / "agents"
        )
        forbidden_engine_attrs = {
            "_board",
            "_history",
            "_current_player",
            "_terminal",
            "_winner",
            "_winning_line",
        }
        module_paths = sorted(agents_dir.glob("*.py"))
        assert len(module_paths) >= 5

        for module_path in module_paths:
            tree = ast.parse(module_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "print", f"print() found in {module_path}"
                if isinstance(node, ast.Attribute):
                    assert (
                        node.attr not in forbidden_engine_attrs
                    ), f"Forbidden private attr {node.attr} accessed in {module_path}"

    @pytest.mark.parametrize("board_args", [(3, 3, 3), (4, 4, 4)])
    def test_cross_level_full_games_complete_cleanly(
        self,
        board_args: tuple[int, int, int],
    ) -> None:
        rows, cols, win_k = board_args
        roster_x = _make_level1_to_4_agents(seed=10, search_depth=2, n_simulations=30)
        roster_o = _make_level1_to_4_agents(seed=20, search_depth=2, n_simulations=30)

        for idx, agent_x in enumerate(roster_x):
            agent_o = roster_o[(idx + 1) % len(roster_o)]
            game = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)
            agent_x.reset()
            agent_o.reset()

            while not game.is_terminal():
                actor = agent_x if game.current_player == Player.X else agent_o
                decision = actor.select_action(game)
                assert decision.action in game.legal_actions()
                game.step(decision.action)

            assert game.result(int(Player.X)) in (-1.0, 0.0, 1.0)


def _make_all_7_level_agents(
    board_tag: str,
    *,
    seed: int = 42,
    search_depth: int = 3,
    n_simulations: int = 40,
) -> list[BaseAgent]:
    """Instantiate verified agents covering every tier from Level 1 through Level 7."""
    rl_agent: BaseAgent
    if board_tag == "3x3_k3":
        rl_agent = TabularQAgent(
            CHECKPOINTS_DIR / "tabular_q_3x3_k3_seed42.pkl",
            seed=seed,
        )
    else:
        rl_agent = DQNAgent(
            CHECKPOINTS_DIR / "dqnnet_4x4_k4_seed42.pt",
            seed=seed,
        )

    policy_fast = PolicyNetworkAgent(
        CHECKPOINTS_DIR / f"fastpolicynet_{board_tag}.pt",
        seed=seed,
    )
    policy_strong = PolicyNetworkAgent(
        CHECKPOINTS_DIR / f"strongpolicynet_{board_tag}.pt",
        seed=seed,
    )
    value_greedy = ValueGreedyAgent(
        CHECKPOINTS_DIR / f"valuenet_{board_tag}.pt",
        seed=seed,
    )
    hybrid_rollout = HybridMCTSAgent(
        policy_checkpoint_path=CHECKPOINTS_DIR / f"strongpolicynet_{board_tag}.pt",
        n_simulations=n_simulations,
        leaf_eval_mode="rollout",
        seed=seed,
        name="HybridMCTS_Rollout",
    )
    hybrid_mixed = HybridMCTSAgent(
        policy_checkpoint_path=CHECKPOINTS_DIR / f"strongpolicynet_{board_tag}.pt",
        value_checkpoint_path=CHECKPOINTS_DIR / f"valuenet_{board_tag}.pt",
        n_simulations=n_simulations,
        leaf_eval_mode="mixed",
        value_weight=0.5,
        seed=seed,
        name="HybridMCTS_Mixed",
    )
    return [
        RandomAgent(seed=seed),
        RuleBasedAgent(seed=seed),
        AlphaBetaAgent(max_depth=search_depth, seed=seed),
        MCTSAgent(n_simulations=n_simulations, seed=seed),
        policy_fast,
        policy_strong,
        value_greedy,
        rl_agent,
        hybrid_rollout,
        hybrid_mixed,
    ]


class TestAgentContractsAll7Levels:
    """Sprint 3 gate tests verifying all 7 AI levels obey unified contracts."""

    @pytest.mark.parametrize(
        ("board_tag", "rows", "cols", "win_k", "moves"),
        [
            ("3x3_k3", 3, 3, 3, []),
            ("3x3_k3", 3, 3, 3, [4, 0, 8]),
            ("4x4_k4", 4, 4, 4, []),
            ("4x4_k4", 4, 4, 4, [5, 0, 10]),
            ("4x4_k4", 4, 4, 4, [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14]),
        ],
    )
    def test_all_7_levels_preserve_state_and_emit_valid_telemetry(
        self,
        board_tag: str,
        rows: int,
        cols: int,
        win_k: int,
        moves: list[int],
    ) -> None:
        game = _make_board(rows, cols, win_k, moves)
        legal = game.legal_actions()
        legal_set = set(legal)
        before_board = game.board.copy()
        before_history = game.move_history
        before_player = game.current_player

        agents = _make_all_7_level_agents(board_tag, seed=42, search_depth=2, n_simulations=30)
        levels_present = {agent.level for agent in agents}
        assert levels_present == {1, 2, 3, 4, 5, 6, 7}

        for agent in agents:
            cfg = agent.get_config()
            assert isinstance(json.dumps(cfg), str)
            decision = agent.select_action(game)
            assert decision.action in legal_set
            assert np.array_equal(game.board, before_board)
            assert game.move_history == before_history
            assert game.current_player == before_player
            assert math.isfinite(decision.latency_ms) and decision.latency_ms >= 0.0
            assert decision.value_estimate is not None
            assert math.isfinite(decision.value_estimate)
            assert set(decision.action_scores.keys()) == legal_set
            assert isinstance(json.dumps(decision.to_dict()), str)

            if isinstance(agent, HybridMCTSAgent):
                assert decision.simulations == 30
                assert set(decision.prior_probs.keys()) == legal_set
                assert math.isclose(
                    sum(decision.prior_probs.values()), 1.0, abs_tol=1e-6
                )
                assert math.isclose(
                    sum(decision.action_scores.values()), 1.0, abs_tol=1e-6
                )
                assert agent.last_tree_nodes_released == decision.nodes_visited

    @pytest.mark.parametrize(
        ("board_tag", "rows", "cols", "win_k"),
        [("3x3_k3", 3, 3, 3), ("4x4_k4", 4, 4, 4)],
    )
    def test_all_7_levels_seed_reproducibility_and_full_games(
        self,
        board_tag: str,
        rows: int,
        cols: int,
        win_k: int,
    ) -> None:
        game = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)
        roster_a = _make_all_7_level_agents(board_tag, seed=77, search_depth=2, n_simulations=25)
        roster_b = _make_all_7_level_agents(board_tag, seed=77, search_depth=2, n_simulations=25)

        for agent_a, agent_b in zip(roster_a, roster_b, strict=True):
            dec_a = agent_a.select_action(game)
            dec_b = agent_b.select_action(game)
            assert dec_a.action == dec_b.action
            assert dec_a.action_scores == dec_b.action_scores

            agent_a.select_action(game)
            agent_a.reset(seed=77)
            dec_reset = agent_a.select_action(game)
            assert dec_reset.action == dec_b.action
            assert dec_reset.action_scores == dec_b.action_scores

        # Play full games pairing Level 7 against every other tier cleanly.
        hybrid_x = roster_a[-1]
        for opponent in roster_b[:-1]:
            match = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)
            hybrid_x.reset(seed=77)
            opponent.reset(seed=88)
            while not match.is_terminal():
                actor = hybrid_x if match.current_player == Player.X else opponent
                match.step(actor.select_action(match).action)
            assert match.result(int(Player.X)) in (-1.0, 0.0, 1.0)
