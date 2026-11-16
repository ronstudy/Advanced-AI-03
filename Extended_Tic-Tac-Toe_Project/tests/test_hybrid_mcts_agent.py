"""Unit and behavioral tests for Task T3.4: Level 7 HybridMCTSAgent (PUCT + Cache)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ttt_ai import (
    BoardConfig,
    CheckpointVerificationError,
    ExtendedTicTacToe,
    FastPolicyNet,
    HybridMCTSAgent,
    RandomAgent,
    StrongPolicyNet,
    ValueNet,
)
from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError
from ttt_ai.engine.rules import Player

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"


class TestHybridMCTSValidationAndCheckpoints:
    """Verify parameter validation, checkpoint security, and config serialization."""

    def test_rejects_untrained_missing_value_in_mixed_and_bad_hyperparams(self) -> None:
        with pytest.raises(CheckpointVerificationError):
            HybridMCTSAgent()

        strong_4x4 = StrongPolicyNet(rows=4, cols=4, win_k=4)
        with pytest.raises(CheckpointVerificationError):
            HybridMCTSAgent(policy_model=strong_4x4)

        with pytest.raises(CheckpointVerificationError):
            HybridMCTSAgent(
                policy_model=strong_4x4,
                leaf_eval_mode="mixed",
                allow_unverified_model=True,
            )

        with pytest.raises(InvalidConfigError):
            HybridMCTSAgent(
                policy_model=strong_4x4,
                leaf_eval_mode="invalid",  # type: ignore[arg-type]
                allow_unverified_model=True,
            )
        with pytest.raises(InvalidConfigError):
            HybridMCTSAgent(
                policy_model=strong_4x4,
                n_simulations=0,
                allow_unverified_model=True,
            )
        with pytest.raises(InvalidConfigError):
            HybridMCTSAgent(
                policy_model=strong_4x4,
                c_puct=-0.5,
                allow_unverified_model=True,
            )
        with pytest.raises(InvalidConfigError):
            HybridMCTSAgent(
                policy_model=strong_4x4,
                value_weight=1.5,
                allow_unverified_model=True,
            )
        with pytest.raises(InvalidConfigError):
            HybridMCTSAgent(
                policy_model=strong_4x4,
                max_cache_size=-1,
                allow_unverified_model=True,
            )

    def test_loads_real_checkpoints_and_rejects_mismatched_boards(self) -> None:
        policy_ckpt = CKPT_DIR / "strongpolicynet_4x4_k4.pt"
        value_ckpt = CKPT_DIR / "valuenet_4x4_k4.pt"
        assert policy_ckpt.is_file() and value_ckpt.is_file()

        agent_mixed = HybridMCTSAgent(
            policy_checkpoint_path=policy_ckpt,
            value_checkpoint_path=value_ckpt,
            n_simulations=40,
            leaf_eval_mode="mixed",
            value_weight=0.5,
            seed=42,
        )
        assert agent_mixed.level == 7
        cfg = agent_mixed.get_config()
        assert cfg["level"] == 7
        assert cfg["leaf_eval_mode"] == "mixed"
        assert cfg["policy_model_class"] == "StrongPolicyNet"
        assert cfg["value_model_class"] == "ValueNet"
        assert isinstance(json.dumps(cfg), str)

        with pytest.raises(CheckpointVerificationError):
            HybridMCTSAgent(
                policy_checkpoint_path=policy_ckpt,
                board_config=BoardConfig(3, 3, 3),
            )
        with pytest.raises(CheckpointVerificationError):
            HybridMCTSAgent(
                policy_checkpoint_path=policy_ckpt,
                value_checkpoint_path=CKPT_DIR / "valuenet_3x3_k3.pt",
                leaf_eval_mode="mixed",
            )
        with pytest.raises(InvalidConfigError):
            agent_mixed.select_action(ExtendedTicTacToe(3, 3, 3))


class TestPUCTSearchLRUCacheAndTacticalCorrection:
    """Verify PUCT tree search corrects misleading priors, caches boards, and honors budgets."""

    def test_puct_search_overrides_wrong_policy_prior_on_immediate_win(self) -> None:
        cfg = BoardConfig(3, 3, 3)
        policy = FastPolicyNet(board_config=cfg, conv_channels=8, hidden_dim=8)
        # Deliberately bias policy prior AGAINST winning cell 2 and IN FAVOR of cell 8.
        with torch.no_grad():
            policy.policy_head.weight.zero_()
            policy.policy_head.bias.fill_(-1.0)
            policy.policy_head.bias[8] = 3.0
            policy.policy_head.bias[2] = 0.0

        game = ExtendedTicTacToe(3, 3, 3)
        for mv in (0, 3, 1, 4):
            game.step(mv)  # X needs cell 2 to win immediately; legal = [2, 5, 6, 7, 8]
        before = game.board.copy()

        agent = HybridMCTSAgent(
            policy_model=policy,
            n_simulations=60,
            c_puct=1.5,
            leaf_eval_mode="rollout",
            allow_unverified_model=True,
            seed=42,
        )
        decision = agent.select_action(game)

        # PUCT search must discover cell 2 (+1.0 terminal) and visit it most, overriding prior[8].
        assert np.array_equal(game.board, before)
        assert decision.action == 2
        assert decision.prior_probs[8] > decision.prior_probs[2]
        assert decision.action_scores[2] > decision.action_scores[8]
        assert set(decision.prior_probs.keys()) == set(game.legal_actions())
        assert set(decision.action_scores.keys()) == set(game.legal_actions())
        assert math.isclose(sum(decision.prior_probs.values()), 1.0, abs_tol=1e-6)
        assert math.isclose(sum(decision.action_scores.values()), 1.0, abs_tol=1e-6)
        assert agent.last_tree_nodes_released == decision.nodes_visited

    def test_lru_cache_hits_and_bounded_eviction(self) -> None:
        cfg = BoardConfig(4, 4, 4)
        policy = StrongPolicyNet(board_config=cfg, conv_channels=8, hidden_dim=8)
        value = ValueNet(board_config=cfg, conv_channels=8, hidden_dim=8)

        # 1. Verify bounded eviction when tree nodes exceed max_cache_size.
        bounded_agent = HybridMCTSAgent(
            policy_model=policy,
            value_model=value,
            n_simulations=40,
            leaf_eval_mode="mixed",
            value_weight=0.5,
            max_cache_size=16,
            allow_unverified_model=True,
            seed=7,
        )
        game = ExtendedTicTacToe(4, 4, 4)
        game.step(5)
        game.step(0)
        bounded_agent.select_action(game)
        assert bounded_agent.cache_size == 16

        # 2. Verify high cache hit rate on transposition / repeated states with sufficient capacity.
        warm_agent = HybridMCTSAgent(
            policy_model=policy,
            value_model=value,
            n_simulations=40,
            leaf_eval_mode="mixed",
            value_weight=0.5,
            max_cache_size=256,
            allow_unverified_model=True,
            seed=7,
        )
        dec1 = warm_agent.select_action(game)
        misses_after_first = warm_agent.cache_misses
        assert misses_after_first >= 1

        warm_agent.reset(seed=7)
        dec2 = warm_agent.select_action(game)
        assert warm_agent.cache_hits > 0
        assert warm_agent.cache_hit_rate > 0.0
        assert dec2.prior_probs == dec1.prior_probs
        assert dec2.action_scores == dec1.action_scores

        warm_agent.clear_cache()
        assert warm_agent.cache_size == 0
        assert warm_agent.cache_hits == 0
        assert warm_agent.cache_misses == 0

    @pytest.mark.parametrize("leaf_mode", ["rollout", "mixed"])
    def test_rollout_and_mixed_modes_time_budget_and_full_game(
        self, leaf_mode: str
    ) -> None:
        policy_ckpt = CKPT_DIR / "strongpolicynet_4x4_k4.pt"
        value_ckpt = CKPT_DIR / "valuenet_4x4_k4.pt"

        agent = HybridMCTSAgent(
            policy_checkpoint_path=policy_ckpt,
            value_checkpoint_path=value_ckpt if leaf_mode == "mixed" else None,
            n_simulations=500,
            leaf_eval_mode=leaf_mode,  # type: ignore[arg-type]
            value_weight=0.5,
            time_budget_ms=25.0,
            seed=42,
        )
        game = ExtendedTicTacToe(4, 4, 4)
        dec = agent.select_action(game)
        assert 1 <= dec.simulations <= 500
        assert dec.action in game.legal_actions()
        assert dec.depth_reached >= 1
        assert agent.last_tree_nodes_released == dec.nodes_visited

        # Near-full (1 cell left) and terminal rejection.
        near_full = ExtendedTicTacToe(4, 4, 4)
        for mv in (0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14):
            near_full.step(mv)
        assert agent.select_action(near_full).action == 15

        terminal = ExtendedTicTacToe(4, 4, 4)
        for mv in (0, 4, 1, 5, 2, 6, 3):
            terminal.step(mv)
        with pytest.raises(IllegalMoveError):
            agent.select_action(terminal)

        # Complete match against RandomAgent cleanly.
        agent.set_time_budget_ms(None)
        agent.set_n_simulations(30)
        opp = RandomAgent(seed=101)
        match = ExtendedTicTacToe(4, 4, 4)
        while not match.is_terminal():
            actor = agent if match.current_player == Player.X else opp
            match.step(actor.select_action(match).action)
        assert match.result(int(Player.X)) in (-1.0, 0.0, 1.0)
