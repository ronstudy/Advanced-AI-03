"""Unit & integration tests for Task T3.2: dataset_gen, train_supervised, and Level 5 agents."""

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
    PolicyNetworkAgent,
    RandomAgent,
    StrongPolicyNet,
    ValueGreedyAgent,
    ValueNet,
    load_game_config,
)
from ttt_ai.config import load_yaml_dict
from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError
from ttt_ai.engine.rules import Player
from ttt_ai.training import (
    DatasetConfig,
    SupervisedSplit,
    SupervisedTrainConfig,
    generate_expert_dataset,
    load_supervised_dataset,
    save_supervised_dataset,
    split_by_game_id,
    train_all_supervised_models,
    train_policy_network,
    train_supervised_model,
    verify_no_game_leakage,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDatasetGenerationAndLeakFreeSplit:
    """Verify expert dataset generation, game-level splitting, and .npz persistence."""

    def test_train_supervised_yaml_loads_with_valid_schema(self) -> None:
        cfg = load_yaml_dict(PROJECT_ROOT / "configs" / "train_supervised.yaml")
        assert cfg["dataset"]["split_policy"] == "game_id_70_15_15"
        assert "4x4_k4" in cfg["dataset"]["boards"]
        assert "3x3_k3" in cfg["dataset"]["boards"]
        game_4x4 = load_game_config(PROJECT_ROOT / "configs" / "game_4x4_k4.yaml")
        assert game_4x4.board.rows == cfg["dataset"]["boards"]["4x4_k4"]["rows"]

    def test_dataset_config_validation_rejects_invalid_inputs(self) -> None:
        cfg = BoardConfig(3, 3, 3)
        with pytest.raises(InvalidConfigError):
            DatasetConfig(board=cfg, num_games=2)
        with pytest.raises(InvalidConfigError):
            DatasetConfig(board=cfg, opening_random_moves=-1)
        with pytest.raises(InvalidConfigError):
            DatasetConfig(board=cfg, explore_prob=1.0)
        with pytest.raises(InvalidConfigError):
            DatasetConfig(board=cfg, teacher_max_depth=0)
        with pytest.raises(InvalidConfigError):
            DatasetConfig(board=cfg, teacher_mode="mcts")  # type: ignore[arg-type]
        with pytest.raises(InvalidConfigError):
            DatasetConfig(board=cfg, split_ratios=(0.6, 0.2, 0.1))

    @pytest.mark.parametrize(
        ("rows", "cols", "win_k", "depth"),
        [(3, 3, 3, 5), (4, 4, 4, 3)],
    )
    def test_generate_expert_dataset_is_legal_leak_free_and_round_trips(
        self,
        tmp_path: Path,
        rows: int,
        cols: int,
        win_k: int,
        depth: int,
    ) -> None:
        board_cfg = BoardConfig(rows=rows, cols=cols, win_k=win_k)
        ds_cfg = DatasetConfig(
            board=board_cfg,
            num_games=24,
            opening_random_moves=2,
            explore_prob=0.10,
            teacher_max_depth=depth,
            seed=42,
        )
        dataset = generate_expert_dataset(ds_cfg)
        summary = dataset.summary()
        assert summary["train_games"] >= 1
        assert summary["val_games"] >= 1
        assert summary["test_games"] >= 1
        assert dataset.dataset_hash.startswith("sha256:")
        assert 0.0 <= dataset.duplicate_ratio < 1.0

        # Game IDs across all three splits must be strictly pairwise disjoint.
        t_ids = dataset.train.unique_game_ids
        v_ids = dataset.val.unique_game_ids
        te_ids = dataset.test.unique_game_ids
        assert not (t_ids & v_ids)
        assert not (t_ids & te_ids)
        assert not (v_ids & te_ids)

        # Every recorded expert action must lie inside the state's legal mask.
        for split in (dataset.train, dataset.val, dataset.test):
            assert len(split) > 0
            for idx in range(len(split)):
                act = int(split.actions[idx])
                assert bool(split.legal_masks[idx, act])
                assert float(split.values[idx]) in (-1.0, 0.0, 1.0)

        out_npz = tmp_path / f"dataset_{board_cfg.name}.npz"
        saved_path, file_sha = save_supervised_dataset(dataset, out_npz)
        assert saved_path.is_file()
        assert file_sha.startswith("sha256:")

        reloaded = load_supervised_dataset(saved_path)
        assert reloaded.dataset_hash == dataset.dataset_hash
        assert np.array_equal(reloaded.train.boards, dataset.train.boards)
        assert np.array_equal(reloaded.train.actions, dataset.train.actions)
        assert np.array_equal(reloaded.test.game_ids, dataset.test.game_ids)

    def test_verify_no_game_leakage_catches_overlap(self) -> None:
        boards = np.zeros((2, 3, 3), dtype=np.int8)
        actions = np.array([0, 1], dtype=np.int64)
        values = np.array([0.0, 0.0], dtype=np.float32)
        masks = np.ones((2, 9), dtype=np.bool_)
        shared_ids = np.array([7, 8], dtype=np.int64)

        s1 = SupervisedSplit(boards, actions, values, masks, shared_ids)
        s2 = SupervisedSplit(boards, actions, values, masks, np.array([8, 9], dtype=np.int64))
        s3 = SupervisedSplit(boards, actions, values, masks, np.array([10, 11], dtype=np.int64))
        with pytest.raises(InvalidConfigError):
            verify_no_game_leakage(s1, s2, s3)

        with pytest.raises(InvalidConfigError):
            split_by_game_id(
                boards,
                actions,
                values,
                masks,
                np.array([1, 1], dtype=np.int64),
                BoardConfig(3, 3, 3),
            )


class TestSupervisedTrainingPipeline:
    """Verify training loop convergence, val-best checkpointing, and JSONL logs."""

    def test_train_all_supervised_models_creates_verified_checkpoints_and_logs(
        self, tmp_path: Path
    ) -> None:
        board_cfg = BoardConfig(3, 3, 3)
        dataset = generate_expert_dataset(
            DatasetConfig(
                board=board_cfg,
                num_games=40,
                opening_random_moves=2,
                explore_prob=0.15,
                teacher_max_depth=5,
                seed=7,
            )
        )
        base_cfg = SupervisedTrainConfig(
            epochs=6,
            batch_size=16,
            learning_rate=2e-3,
            conv_channels=32,
            hidden_dim=32,
            seed=7,
        )
        ckpt_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"
        results = train_all_supervised_models(
            dataset,
            base_config=base_cfg,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
        )
        assert set(results.keys()) == {"FastPolicyNet", "StrongPolicyNet", "ValueNet"}

        for model_name, res in results.items():
            assert res.checkpoint_path is not None and res.checkpoint_path.is_file()
            assert res.meta_path is not None and res.meta_path.is_file()
            assert res.log_path is not None and res.log_path.is_file()
            assert len(res.history) == 6
            # Best epoch must match the minimum val_loss seen across epochs.
            expected_best_epoch = min(res.history, key=lambda rec: rec.val_loss).epoch
            assert res.best_epoch == expected_best_epoch
            assert math.isfinite(res.test_loss)
            assert res.metadata is not None
            assert res.metadata.dataset_hash == dataset.dataset_hash
            assert res.metadata.split_policy == dataset.split_policy

            if model_name in ("FastPolicyNet", "StrongPolicyNet"):
                assert res.history[-1].train_loss < res.history[0].train_loss
                assert res.test_accuracy is not None and 0.0 <= res.test_accuracy <= 1.0
            else:
                assert res.test_accuracy is None

            log_lines = res.log_path.read_text(encoding="utf-8").strip().splitlines()
            assert len(log_lines) == 8  # 1 meta + 6 epochs + 1 summary
            records = [json.loads(line) for line in log_lines]
            assert records[0]["type"] == "meta"
            assert records[-1]["type"] == "summary"

    def test_train_supervised_rejects_dqn_and_bad_configs(self) -> None:
        dataset = generate_expert_dataset(
            DatasetConfig(board=BoardConfig(3, 3, 3), num_games=12, teacher_max_depth=3, seed=3)
        )
        with pytest.raises(InvalidConfigError):
            SupervisedTrainConfig(epochs=0)
        with pytest.raises(InvalidConfigError):
            SupervisedTrainConfig(learning_rate=-0.1)
        with pytest.raises(InvalidConfigError):
            train_supervised_model(dataset, SupervisedTrainConfig(model_name="DQNNet"))


class TestLevel5PolicyAndValueAgents:
    """Verify Level 5 PolicyNetworkAgent & ValueGreedyAgent contracts and edge cases."""

    def test_rejects_untrained_direct_instantiation_and_mismatched_board(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(CheckpointVerificationError):
            PolicyNetworkAgent()
        with pytest.raises(CheckpointVerificationError):
            ValueGreedyAgent()

        untrained_fast = FastPolicyNet(rows=3, cols=3, win_k=3)
        with pytest.raises(CheckpointVerificationError):
            PolicyNetworkAgent(model=untrained_fast)

        # Train briefly to obtain real verified checkpoints for 3x3.
        dataset = generate_expert_dataset(
            DatasetConfig(board=BoardConfig(3, 3, 3), num_games=20, teacher_max_depth=4, seed=11)
        )
        ckpt_dir = tmp_path / "ckpts"
        results = train_all_supervised_models(
            dataset,
            base_config=SupervisedTrainConfig(epochs=3, conv_channels=16, hidden_dim=16, seed=11),
            checkpoint_dir=ckpt_dir,
            model_names=("FastPolicyNet", "ValueNet"),
        )

        policy_3x3 = PolicyNetworkAgent(results["FastPolicyNet"].checkpoint_path)
        value_3x3 = ValueGreedyAgent(results["ValueNet"].checkpoint_path)
        assert policy_3x3.level == 5
        assert value_3x3.level == 5

        with pytest.raises(CheckpointVerificationError):
            PolicyNetworkAgent(
                results["FastPolicyNet"].checkpoint_path,
                board_config=BoardConfig(4, 4, 4),
            )

        game_4x4 = ExtendedTicTacToe(4, 4, 4)
        with pytest.raises(InvalidConfigError):
            policy_3x3.select_action(game_4x4)
        with pytest.raises(InvalidConfigError):
            value_3x3.select_action(game_4x4)

    def test_action_masking_blocks_occupied_cells_even_with_extreme_logits(self) -> None:
        cfg = BoardConfig(4, 4, 4)
        model = StrongPolicyNet(board_config=cfg, conv_channels=8, hidden_dim=8)
        # Force policy head bias so occupied cells (0, 5, 10) have overwhelming raw logits.
        with torch.no_grad():
            model.policy_head.weight.zero_()
            model.policy_head.bias.fill_(-5.0)
            model.policy_head.bias[0] = 1000.0
            model.policy_head.bias[5] = 900.0
            model.policy_head.bias[10] = 800.0
            model.policy_head.bias[15] = 10.0  # highest legal cell

        agent = PolicyNetworkAgent(model=model, allow_unverified_model=True)
        game = ExtendedTicTacToe(4, 4, 4)
        game.step(0)
        game.step(5)
        game.step(10)
        snapshot = game.board.copy()

        decision = agent.select_action(game)
        assert decision.action == 15
        assert np.array_equal(game.board, snapshot)
        assert set(decision.action_scores.keys()) == set(game.legal_actions())
        assert math.isclose(sum(decision.action_scores.values()), 1.0, abs_tol=1e-6)
        for occupied in (0, 5, 10):
            assert occupied not in decision.action_scores

    def test_value_greedy_agent_takes_immediate_win_and_respects_zero_sum_negation(self) -> None:
        cfg = BoardConfig(3, 3, 3)
        value_net = ValueNet(board_config=cfg, conv_channels=8, hidden_dim=8)
        agent = ValueGreedyAgent(model=value_net, allow_unverified_model=True)

        # X has 0 and 1 -> cell 2 is immediate win (+1.0), regardless of ValueNet weights.
        game = ExtendedTicTacToe(3, 3, 3)
        for move in (0, 3, 1, 4):
            game.step(move)
        before_board = game.board.copy()

        decision = agent.select_action(game)
        assert decision.action == 2
        assert decision.value_estimate == pytest.approx(1.0)
        assert np.array_equal(game.board, before_board)
        assert set(decision.action_scores.keys()) == set(game.legal_actions())

    def test_level5_contract_empty_near_full_terminal_sampling_and_matches(
        self, tmp_path: Path
    ) -> None:
        cfg = BoardConfig(4, 4, 4)
        dataset = generate_expert_dataset(
            DatasetConfig(board=cfg, num_games=18, teacher_max_depth=2, seed=42)
        )
        res = train_policy_network(
            dataset,
            model_name="StrongPolicyNet",
            config=SupervisedTrainConfig(epochs=3, conv_channels=16, hidden_dim=16, seed=42),
            checkpoint_path=tmp_path / "strong_4x4.pt",
        )
        val_res = train_all_supervised_models(
            dataset,
            base_config=SupervisedTrainConfig(epochs=3, conv_channels=16, hidden_dim=16, seed=42),
            checkpoint_dir=tmp_path,
            model_names=("ValueNet",),
        )["ValueNet"]

        policy_det = PolicyNetworkAgent(res.checkpoint_path, sample=False, seed=42)
        policy_samp_a = PolicyNetworkAgent(
            res.checkpoint_path, sample=True, temperature=0.8, seed=99
        )
        policy_samp_b = PolicyNetworkAgent(
            res.checkpoint_path, sample=True, temperature=0.8, seed=99
        )
        value_agent = ValueGreedyAgent(val_res.checkpoint_path, seed=42)

        # Empty board, near-full board (1 cell left), and terminal rejection.
        empty_game = ExtendedTicTacToe(4, 4, 4)
        near_full = ExtendedTicTacToe(4, 4, 4)
        for mv in (0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14):
            near_full.step(mv)
        terminal_win = ExtendedTicTacToe(4, 4, 4)
        for mv in (0, 4, 1, 5, 2, 6, 3):
            terminal_win.step(mv)

        for agent in (policy_det, policy_samp_a, value_agent):
            dec_empty = agent.select_action(empty_game)
            assert dec_empty.action in empty_game.legal_actions()
            assert math.isfinite(dec_empty.latency_ms) and dec_empty.latency_ms >= 0.0
            assert dec_empty.value_estimate is not None
            assert agent.select_action(near_full).action == 15
            with pytest.raises(IllegalMoveError):
                agent.select_action(terminal_win)

        # Seed reproducibility for stochastic policy sampling.
        policy_samp_a.reset(seed=99)
        policy_samp_b.reset(seed=99)
        assert policy_samp_a.select_action(empty_game).action == policy_samp_b.select_action(
            empty_game
        ).action
        policy_samp_a.reset(seed=99)
        policy_samp_b.reset(seed=99)
        assert policy_samp_a.select_action(empty_game).action == policy_samp_b.select_action(
            empty_game
        ).action

        # Full match completion against RandomAgent without illegal moves.
        random_opp = RandomAgent(seed=123)
        for agent_x in (policy_det, value_agent):
            match_game = ExtendedTicTacToe(4, 4, 4)
            agent_x.reset()
            random_opp.reset(seed=123)
            while not match_game.is_terminal():
                actor = agent_x if match_game.current_player == Player.X else random_opp
                dec = actor.select_action(match_game)
                match_game.step(dec.action)
            assert match_game.result(int(Player.X)) in (-1.0, 0.0, 1.0)
