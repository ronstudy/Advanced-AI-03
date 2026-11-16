"""Unit & integration tests for Task T3.3: TabularQAgent, DQNAgent, and RL training."""

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
    DQNAgent,
    DQNNet,
    ExtendedTicTacToe,
    RandomAgent,
    TabularQAgent,
    TabularQLearningAgent,
)
from ttt_ai.config import load_yaml_dict
from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError
from ttt_ai.engine.rules import Player
from ttt_ai.training import (
    DQNTrainConfig,
    ReplayBuffer,
    TabularQTrainConfig,
    compute_zero_sum_bellman_targets,
    train_dqn_multiseed,
    train_tabular_q_multiseed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestReplayBufferAndZeroSumBellman:
    """Verify ring buffer invariants and masked zero-sum Bellman targets."""

    def test_train_rl_yaml_loads_with_three_seeds(self) -> None:
        cfg = load_yaml_dict(PROJECT_ROOT / "configs" / "train_rl.yaml")
        assert len(cfg["seeds"]) == 3
        assert "3x3_k3" in cfg["tabular_q"]["boards"]
        assert "4x4_k4" in cfg["dqn"]["boards"]
        assert "4x4_k3" in cfg["dqn"]["boards"]

    def test_replay_buffer_ring_wraparound_and_sampling(self) -> None:
        cfg = BoardConfig(4, 4, 4)
        buf = ReplayBuffer(capacity=4, board_config=cfg)
        rng = np.random.default_rng(42)

        with pytest.raises(InvalidConfigError):
            buf.sample(2, rng)

        for idx in range(6):
            state = np.full((4, 4), idx % 2, dtype=np.int8)
            next_state = np.full((4, 4), -1, dtype=np.int8)
            mask = np.ones(16, dtype=np.bool_)
            buf.push(state, idx % 16, float(idx == 5), next_state, idx == 5, mask)

        assert len(buf) == 4
        s_b, a_b, r_b, sn_b, d_b, nm_b = buf.sample(3, rng)
        assert s_b.shape == (3, 1, 4, 4) and s_b.dtype == torch.float32
        assert a_b.shape == (3,) and a_b.dtype == torch.long
        assert r_b.shape == (3,) and r_b.dtype == torch.float32
        assert sn_b.shape == (3, 1, 4, 4) and sn_b.dtype == torch.float32
        assert d_b.shape == (3,) and d_b.dtype == torch.bool
        assert nm_b.shape == (3, 16) and nm_b.dtype == torch.bool

    def test_zero_sum_bellman_targets_handles_terminal_and_masked_nonterminal(self) -> None:
        cfg = BoardConfig(4, 4, 4)
        target_net = DQNNet(board_config=cfg, conv_channels=8, hidden_dim=8)
        with torch.no_grad():
            target_net.q_head.weight.zero_()
            target_net.q_head.bias.fill_(-0.2)
            target_net.q_head.bias[0] = 0.9  # illegal in row 1
            target_net.q_head.bias[5] = 0.5  # highest legal in row 1

        rewards = torch.tensor([1.0, 0.0], dtype=torch.float32)
        next_states = torch.zeros(2, 1, 4, 4, dtype=torch.float32)
        dones = torch.tensor([True, False], dtype=torch.bool)
        next_masks = torch.tensor(
            [
                [False] * 16,  # terminal row has zero legal actions
                [False, True, False, False, False, True] + [False] * 10,
            ],
            dtype=torch.bool,
        )

        targets = compute_zero_sum_bellman_targets(
            rewards,
            next_states,
            dones,
            next_masks,
            target_net=target_net,
            gamma=0.95,
        )
        assert float(targets[0]) == pytest.approx(1.0)
        # Non-terminal target negates max legal Q (0.5 at cell 5, ignoring 0.9 at cell 0).
        assert float(targets[1]) == pytest.approx(-0.95 * 0.5, abs=1e-5)


class TestRLTrainingAndLevel6AgentContracts:
    """Verify multi-seed TabularQ & DQN training, checkpoints, and BaseAgent contracts."""

    def test_rejects_untrained_instantiation_and_invalid_configs(self) -> None:
        assert TabularQLearningAgent is TabularQAgent
        with pytest.raises(CheckpointVerificationError):
            TabularQAgent()
        with pytest.raises(CheckpointVerificationError):
            DQNAgent()
        with pytest.raises(CheckpointVerificationError):
            DQNAgent(model=DQNNet(rows=4, cols=4, win_k=4))
        with pytest.raises(CheckpointVerificationError):
            TabularQAgent(q_table={}, board_config=BoardConfig(3, 3, 3))
        with pytest.raises(InvalidConfigError):
            TabularQTrainConfig(episodes=0)
        with pytest.raises(InvalidConfigError):
            DQNTrainConfig(replay_capacity=16, batch_size=32)

    def test_multiseed_tabular_q_training_and_agent_contract(self, tmp_path: Path) -> None:
        cfg_3x3 = BoardConfig(3, 3, 3)
        base_cfg = TabularQTrainConfig(
            board=cfg_3x3,
            episodes=600,
            epsilon_decay_episodes=450,
            eval_every_episodes=300,
            eval_games=10,
        )
        results = train_tabular_q_multiseed(
            base_cfg,
            seeds=(42, 123),
            checkpoint_dir=tmp_path / "ckpts",
            log_dir=tmp_path / "logs",
        )
        assert set(results.keys()) == {42, 123}

        for seed, res in results.items():
            assert res.checkpoint_path is not None and res.checkpoint_path.is_file()
            assert res.meta_path is not None and res.meta_path.is_file()
            assert res.log_path is not None and res.log_path.is_file()
            assert res.num_states > 0

            agent = TabularQAgent(res.checkpoint_path, seed=seed)
            assert agent.level == 6
            assert agent.board_config == cfg_3x3

            # Reject incompatible 4x4 board.
            with pytest.raises(InvalidConfigError):
                agent.select_action(ExtendedTicTacToe(4, 4, 4))

            # Verify source immutability and action masking on occupied 3x3 board.
            game = ExtendedTicTacToe(3, 3, 3)
            game.step(4)
            game.step(0)
            before = game.board.copy()
            dec = agent.select_action(game)
            assert dec.action in game.legal_actions()
            assert np.array_equal(game.board, before)
            assert set(dec.action_scores.keys()) == set(game.legal_actions())
            assert dec.value_estimate is not None and math.isfinite(dec.value_estimate)

            lines = res.log_path.read_text(encoding="utf-8").strip().splitlines()
            assert json.loads(lines[0])["type"] == "meta"
            assert json.loads(lines[-1])["type"] == "summary"

    def test_multiseed_dqn_selfplay_training_masking_and_edge_cases(
        self, tmp_path: Path
    ) -> None:
        cfg_4x4 = BoardConfig(4, 4, 4)
        base_cfg = DQNTrainConfig(
            board=cfg_4x4,
            episodes=40,
            replay_capacity=1000,
            batch_size=16,
            min_replay_size=32,
            train_every_steps=2,
            target_update_steps=16,
            epsilon_decay_episodes=30,
            snapshot_every_episodes=20,
            eval_every_episodes=20,
            eval_games=6,
            conv_channels=16,
            hidden_dim=16,
        )
        results = train_dqn_multiseed(
            base_cfg,
            seeds=(42, 123),
            checkpoint_dir=tmp_path / "ckpts",
            log_dir=tmp_path / "logs",
        )
        assert set(results.keys()) == {42, 123}

        for seed, res in results.items():
            assert res.checkpoint_path is not None and res.checkpoint_path.is_file()
            assert res.meta_path is not None and res.meta_path.is_file()
            assert res.log_path is not None and res.log_path.is_file()
            assert res.optimizer_steps > 0

            agent = DQNAgent(res.checkpoint_path, seed=seed)
            assert agent.level == 6
            assert agent.board_config == cfg_4x4

            # Reject wrong board config at load and at inference.
            with pytest.raises(CheckpointVerificationError):
                DQNAgent(res.checkpoint_path, board_config=BoardConfig(3, 3, 3))
            with pytest.raises(InvalidConfigError):
                agent.select_action(ExtendedTicTacToe(3, 3, 3))

        # Explicit masking test: illegal cell has highest raw Q (+999.0), must be masked out.
        dqn_net = DQNNet(board_config=cfg_4x4, conv_channels=8, hidden_dim=8)
        with torch.no_grad():
            dqn_net.q_head.weight.zero_()
            dqn_net.q_head.bias.fill_(-1.0)
            dqn_net.q_head.bias[0] = 999.0
            dqn_net.q_head.bias[5] = 998.0
            dqn_net.q_head.bias[12] = 5.0

        masked_agent = DQNAgent(model=dqn_net, allow_unverified_model=True, seed=7)
        game = ExtendedTicTacToe(4, 4, 4)
        game.step(0)
        game.step(5)
        dec = masked_agent.select_action(game)
        assert dec.action == 12
        assert set(dec.action_scores.keys()) == set(game.legal_actions())

        # Empty, near-full (1 cell), terminal rejection, and full match against RandomAgent.
        empty_game = ExtendedTicTacToe(4, 4, 4)
        assert masked_agent.select_action(empty_game).action in empty_game.legal_actions()

        near_full = ExtendedTicTacToe(4, 4, 4)
        for mv in (0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14):
            near_full.step(mv)
        assert masked_agent.select_action(near_full).action == 15

        terminal_game = ExtendedTicTacToe(4, 4, 4)
        for mv in (0, 4, 1, 5, 2, 6, 3):
            terminal_game.step(mv)
        with pytest.raises(IllegalMoveError):
            masked_agent.select_action(terminal_game)

        opp = RandomAgent(seed=55)
        match = ExtendedTicTacToe(4, 4, 4)
        while not match.is_terminal():
            actor = masked_agent if match.current_player == Player.X else opp
            match.step(actor.select_action(match).action)
        assert match.result(int(Player.X)) in (-1.0, 0.0, 1.0)
