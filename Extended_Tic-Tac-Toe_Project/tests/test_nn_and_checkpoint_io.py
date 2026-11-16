"""Unit tests for Task T3.1: neural-network architectures, action masking, and checkpoint I/O."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from ttt_ai import (
    BoardConfig,
    CheckpointMetadata,
    CheckpointVerificationError,
    DQNNet,
    ExtendedTicTacToe,
    FastPolicyNet,
    StrongPolicyNet,
    ValueNet,
    load_pickle_checkpoint,
    load_torch_checkpoint,
    save_pickle_checkpoint,
    save_torch_checkpoint,
    seed_everything,
    verify_checkpoint,
)
from ttt_ai.agents.nn_models import (
    DEFAULT_CONV_CHANNELS,
    DEFAULT_HIDDEN_DIM,
    MASK_FILL_VALUE,
    BaseBoardNet,
    action_masking,
    apply_action_mask,
    board_to_tensor,
    build_legal_action_mask,
    create_model,
    masked_argmax,
    masked_softmax,
)
from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError
from ttt_ai.engine.rules import Player
from ttt_ai.training.checkpoint_io import (
    collect_hardware_info,
    compute_file_sha256,
    resolve_meta_path,
)


def _make_trained_metadata(
    agent_class: str,
    board_config: BoardConfig,
    *,
    training_type: str = "supervised",
    epochs_trained: int = 5,
    episodes_trained: int = 0,
    seed: int = 42,
) -> CheckpointMetadata:
    """Create valid trained metadata fixture for checkpoint I/O unit tests."""
    return CheckpointMetadata(
        agent_class=agent_class,
        board_config=board_config,
        seed=seed,
        training_type=training_type,
        epochs_trained=epochs_trained,
        episodes_trained=episodes_trained,
        dataset_hash="sha256:test_fixture_dataset",
        split_policy="game_id_70_15_15" if training_type == "supervised" else None,
        best_val_loss=0.45 if training_type == "supervised" else None,
        best_val_accuracy=0.82 if training_type == "supervised" else None,
        final_eval_score=0.90 if training_type != "supervised" else None,
        training_time_sec=1.25,
    )


class TestNeuralNetworkArchitectures:
    """Verify forward shapes, bounds, parameter hierarchy, and gradients."""

    @pytest.mark.parametrize(
        ("rows", "cols", "win_k"),
        [(3, 3, 3), (4, 4, 3), (4, 4, 4), (5, 5, 4), (4, 5, 4)],
    )
    def test_all_networks_support_general_mxn_boards_and_batches(
        self,
        rows: int,
        cols: int,
        win_k: int,
    ) -> None:
        seed_everything(42)
        cfg = BoardConfig(rows=rows, cols=cols, win_k=win_k)
        batch = torch.randn(4, 1, rows, cols, dtype=torch.float32)

        fast_net = FastPolicyNet(board_config=cfg)
        strong_net = StrongPolicyNet(board_config=cfg)
        value_net = ValueNet(board_config=cfg)
        dqn_net = DQNNet(board_config=cfg)

        for net in (fast_net, strong_net, value_net, dqn_net):
            assert isinstance(net, BaseBoardNet)
            assert isinstance(net, nn.Module)
            assert net.board_config == cfg
            assert net.rows == rows
            assert net.cols == cols
            assert net.win_k == win_k
            assert net.num_actions == rows * cols
            assert net.conv_channels == DEFAULT_CONV_CHANNELS
            assert net.hidden_dim == DEFAULT_HIDDEN_DIM
            arch_cfg = net.get_architecture_config()
            assert arch_cfg["board_config"] == cfg.to_dict()
            assert arch_cfg["num_parameters"] == net.count_parameters()

        fast_logits = fast_net(batch)
        strong_logits = strong_net(batch)
        values = value_net(batch)
        q_values = dqn_net(batch)

        assert fast_logits.shape == (4, rows * cols)
        assert strong_logits.shape == (4, rows * cols)
        assert q_values.shape == (4, rows * cols)
        assert values.shape == (4, 1)
        assert torch.all(values >= -1.0) and torch.all(values <= 1.0)
        assert torch.isfinite(fast_logits).all()
        assert torch.isfinite(strong_logits).all()
        assert torch.isfinite(values).all()
        assert torch.isfinite(q_values).all()

        # StrongPolicyNet adds an extra hidden Linear(64, 64) layer over FastPolicyNet.
        expected_extra_params = DEFAULT_HIDDEN_DIM * DEFAULT_HIDDEN_DIM + DEFAULT_HIDDEN_DIM
        assert strong_net.count_parameters() - fast_net.count_parameters() == expected_extra_params

    def test_create_model_registry_and_validation(self) -> None:
        cfg = BoardConfig(rows=4, cols=4, win_k=4)
        assert isinstance(create_model("FastPolicyNet", cfg), FastPolicyNet)
        assert isinstance(create_model("fast_policy", cfg), FastPolicyNet)
        assert isinstance(create_model("StrongPolicyNet", cfg), StrongPolicyNet)
        assert isinstance(create_model("strong-policy-net", cfg), StrongPolicyNet)
        assert isinstance(create_model("ValueNet", cfg), ValueNet)
        assert isinstance(create_model("value", cfg), ValueNet)
        assert isinstance(create_model("DQNNet", cfg), DQNNet)
        assert isinstance(create_model("dqn", cfg), DQNNet)

        with pytest.raises(InvalidConfigError):
            create_model("UnknownNet", cfg)
        with pytest.raises(InvalidConfigError):
            create_model("", cfg)
        with pytest.raises(InvalidConfigError):
            FastPolicyNet(rows=4, cols=4, win_k=4, conv_channels=0)
        with pytest.raises(InvalidConfigError):
            StrongPolicyNet(rows=4, cols=4, win_k=4, hidden_dim=-8)
        with pytest.raises(InvalidConfigError):
            ValueNet(board_config="4x4")  # type: ignore[arg-type]

    def test_forward_rejects_mismatched_shapes_and_dtypes(self) -> None:
        net = FastPolicyNet(rows=4, cols=4, win_k=4)
        with pytest.raises(InvalidConfigError):
            net(torch.zeros(1, 1, 3, 3))
        with pytest.raises(InvalidConfigError):
            net(torch.zeros(1, 2, 4, 4))
        with pytest.raises(InvalidConfigError):
            net(torch.zeros(4, 4))
        with pytest.raises(InvalidConfigError):
            net(torch.zeros(1, 1, 4, 4, dtype=torch.bool))
        with pytest.raises(InvalidConfigError):
            net(np.zeros((1, 1, 4, 4), dtype=np.float32))  # type: ignore[arg-type]

    def test_gradients_flow_cleanly_in_all_architectures(self) -> None:
        seed_everything(7)
        cfg = BoardConfig(rows=4, cols=4, win_k=4)
        x = torch.randn(8, 1, 4, 4)

        for policy_cls in (FastPolicyNet, StrongPolicyNet):
            model = policy_cls(board_config=cfg)
            loss = nn.CrossEntropyLoss()(model(x), torch.arange(8) % 16)
            loss.backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )

        value_model = ValueNet(board_config=cfg)
        v_loss = nn.MSELoss()(value_model(x), torch.ones(8, 1) * 0.5)
        v_loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in value_model.parameters()
        )

        dqn_model = DQNNet(board_config=cfg)
        q_loss = nn.MSELoss()(dqn_model(x)[:, 0], torch.ones(8))
        q_loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in dqn_model.parameters()
        )


class TestBoardEncodingAndActionMasking:
    """Verify canonical tensor conversion and shared legal-action masking."""

    def test_board_to_tensor_canonicalization_and_immutability(self) -> None:
        game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
        game.step(5)  # X at (1, 1)
        game.step(0)  # O at (0, 0)
        game.step(6)  # X at (1, 2) -> now O (-1) to move
        raw_board = game.board
        snapshot = raw_board.copy()

        canonical_tensor = board_to_tensor(raw_board, current_player=int(game.current_player))
        assert np.array_equal(raw_board, snapshot)
        assert canonical_tensor.shape == (1, 1, 4, 4)
        assert canonical_tensor.dtype == torch.float32
        # From O's perspective (-1), O's mark at (0,0) becomes +1 and X's at (1,1) becomes -1.
        assert float(canonical_tensor[0, 0, 0, 0]) == 1.0
        assert float(canonical_tensor[0, 0, 1, 1]) == -1.0
        assert float(canonical_tensor[0, 0, 3, 3]) == 0.0

        batch_arr = np.stack([raw_board, raw_board], axis=0)
        batch_tensor = board_to_tensor(batch_arr, current_player=Player.X)
        assert batch_tensor.shape == (2, 1, 4, 4)
        assert float(batch_tensor[1, 0, 0, 0]) == -1.0

        tensor_4d = board_to_tensor(canonical_tensor)
        assert torch.equal(tensor_4d, canonical_tensor)

    def test_board_to_tensor_rejects_invalid_inputs(self) -> None:
        with pytest.raises(InvalidConfigError):
            board_to_tensor(np.zeros((2, 2), dtype=np.int8))
        with pytest.raises(InvalidConfigError):
            board_to_tensor(np.zeros((4, 4), dtype=bool))
        with pytest.raises(InvalidConfigError):
            board_to_tensor(np.zeros((2, 3, 4, 4), dtype=np.float32))
        with pytest.raises(InvalidConfigError):
            board_to_tensor(np.zeros((4, 4), dtype=np.int8), current_player=0)
        with pytest.raises(InvalidConfigError):
            board_to_tensor(np.zeros((4, 4), dtype=np.int8), current_player=True)

    def test_action_mask_softmax_and_argmax_contracts(self) -> None:
        logits = torch.tensor([2.0, 5.0, 5.0, 10.0, -3.0], dtype=torch.float32)
        legal = [0, 1, 2, 4]  # cell 3 has largest raw logit (10.0) but is illegal

        masked = apply_action_mask(logits, legal)
        assert action_masking is apply_action_mask
        assert float(masked[3]) == pytest.approx(MASK_FILL_VALUE)
        assert float(masked[0]) == pytest.approx(2.0)

        probs = masked_softmax(logits, legal)
        assert float(probs[3]) == 0.0
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
        # Actions 1 and 2 tied on max legal logit -> deterministic smallest index 1 wins.
        best_action, best_score, legal_map = masked_argmax(probs, legal)
        assert best_action == 1
        assert best_score == pytest.approx(float(probs[1]))
        assert set(legal_map.keys()) == set(legal)

        # Boolean NumPy/Torch masks & 2D batch masking.
        batch_logits = torch.tensor(
            [[1.0, 100.0, 2.0, -1.0], [-5.0, -2.0, 50.0, -3.0]],
            dtype=torch.float32,
        )
        batch_probs = masked_softmax(batch_logits, [[0, 2], [1, 3]])
        assert float(batch_probs[0, 1]) == 0.0
        assert float(batch_probs[1, 2]) == 0.0
        assert float(batch_probs[0].sum()) == pytest.approx(1.0, abs=1e-6)
        assert float(batch_probs[1].sum()) == pytest.approx(1.0, abs=1e-6)
        assert int(batch_probs[0].argmax()) == 2
        assert int(batch_probs[1].argmax()) == 1

        # Batched terminal DQN mask support when allow_empty=True.
        bool_mask = torch.tensor(
            [[True, False, True, False], [False, False, False, False]],
            dtype=torch.bool,
        )
        dqn_masked = apply_action_mask(batch_logits, bool_mask, allow_empty=True)
        assert torch.all(dqn_masked[1] == MASK_FILL_VALUE)

        with pytest.raises(IllegalMoveError):
            apply_action_mask(batch_logits, bool_mask, allow_empty=False)

    def test_masking_edge_cases_raise_explicit_errors(self) -> None:
        scores = torch.zeros(9, dtype=torch.float32)
        with pytest.raises(IllegalMoveError):
            build_legal_action_mask(9, [0, 9])
        with pytest.raises(IllegalMoveError):
            build_legal_action_mask(9, [True])  # type: ignore[list-item]
        with pytest.raises(IllegalMoveError):
            masked_softmax(scores, [])
        with pytest.raises(IllegalMoveError):
            masked_argmax(scores, [])
        with pytest.raises(InvalidConfigError):
            masked_softmax(scores, [0, 1], temperature=0.0)
        with pytest.raises(InvalidConfigError):
            apply_action_mask(scores, [0, 1], mask_value=1.0)
        with pytest.raises(InvalidConfigError):
            masked_argmax(np.array([1.0, float("nan"), 0.0]), [0, 1, 2])

    def test_policy_and_dqn_helpers_integrate_with_game_state(self) -> None:
        seed_everything(123)
        game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
        for move in (0, 5, 1, 10, 2):
            game.step(move)

        cfg = BoardConfig(rows=4, cols=4, win_k=4)
        policy = StrongPolicyNet(board_config=cfg)
        dqn = DQNNet(board_config=cfg)

        x = board_to_tensor(game.canonical_board())
        legal = game.legal_actions()
        with torch.no_grad():
            probs = policy.predict_probs(x, legal)
            assert probs.shape == (1, 16)
            for occupied in (0, 5, 1, 10, 2):
                assert float(probs[0, occupied]) == 0.0

            chosen, score, score_map = masked_argmax(probs, legal)
            assert chosen in legal
            assert set(score_map.keys()) == set(legal)
            assert score == pytest.approx(score_map[chosen])

            masked_q = dqn.predict_masked_q(x, legal)
            q_action, _, q_map = masked_argmax(masked_q, legal)
            assert q_action in legal
            assert set(q_map.keys()) == set(legal)


class TestCheckpointIOAndVerification:
    """Verify .pt/.pkl checkpoint saving, loading, and tamper/mismatch rejection."""

    def test_torch_checkpoint_round_trip_all_models(self, tmp_path: Path) -> None:
        seed_everything(42)
        cfg = BoardConfig(rows=4, cols=4, win_k=4)
        sample_input = torch.randn(2, 1, 4, 4)

        for model_cls in (FastPolicyNet, StrongPolicyNet, ValueNet, DQNNet):
            model = model_cls(board_config=cfg)
            ckpt_path = tmp_path / f"{model_cls.__name__.lower()}.pt"
            meta = _make_trained_metadata(model_cls.__name__, cfg)

            saved_ckpt, saved_meta_path, final_meta = save_torch_checkpoint(
                model, ckpt_path, meta
            )
            assert saved_ckpt == ckpt_path
            assert saved_meta_path == resolve_meta_path(ckpt_path)
            assert saved_ckpt.is_file() and saved_meta_path.is_file()
            assert final_meta.weights_sha256 == compute_file_sha256(ckpt_path)
            assert "python" in final_meta.hardware

            verified_meta = verify_checkpoint(
                ckpt_path,
                board_config=cfg,
                expected_agent_class=model_cls.__name__,
                expected_training_type="supervised",
            )
            assert verified_meta == final_meta

            # Load automatically via metadata reconstruction.
            loaded_auto, loaded_meta = load_torch_checkpoint(ckpt_path, board_config=cfg)
            assert isinstance(loaded_auto, model_cls)
            assert not loaded_auto.training
            assert loaded_meta == final_meta

            # Load into pre-instantiated module instance.
            fresh_instance = model_cls(board_config=cfg)
            loaded_manual, _ = load_torch_checkpoint(ckpt_path, model=fresh_instance)
            assert loaded_manual is fresh_instance

            model.eval()
            with torch.no_grad():
                expected_out = model(sample_input)
                assert torch.allclose(loaded_auto(sample_input), expected_out)
                assert torch.allclose(loaded_manual(sample_input), expected_out)

    def test_pickle_checkpoint_round_trip(self, tmp_path: Path) -> None:
        cfg = BoardConfig(rows=3, cols=3, win_k=3)
        q_table = {b"0" * 9: np.zeros(9, dtype=np.float32)}
        ckpt_path = tmp_path / "tabular_q_3x3.pkl"
        meta = _make_trained_metadata(
            "TabularQAgent",
            cfg,
            training_type="rl_tabular",
            epochs_trained=0,
            episodes_trained=50000,
        )

        _, meta_path, final_meta = save_pickle_checkpoint(q_table, ckpt_path, meta)
        assert meta_path.is_file()
        assert final_meta.episodes_trained == 50000

        loaded_table, loaded_meta = load_pickle_checkpoint(
            ckpt_path,
            board_config=cfg,
            expected_agent_class="TabularQAgent",
        )
        assert loaded_meta == final_meta
        assert np.array_equal(loaded_table[b"0" * 9], q_table[b"0" * 9])

    def test_rejects_untrained_missing_meta_mismatched_board_and_tampered_files(
        self, tmp_path: Path
    ) -> None:
        cfg_4x4_k4 = BoardConfig(rows=4, cols=4, win_k=4)
        cfg_4x4_k3 = BoardConfig(rows=4, cols=4, win_k=3)
        cfg_3x3_k3 = BoardConfig(rows=3, cols=3, win_k=3)

        # 1. Untrained metadata must be rejected at construction & from_dict.
        with pytest.raises(CheckpointVerificationError):
            CheckpointMetadata(
                agent_class="StrongPolicyNet",
                board_config=cfg_4x4_k4,
                seed=42,
                training_type="supervised",
                epochs_trained=0,
                episodes_trained=0,
            )

        # Save a valid checkpoint first, then probe failure modes.
        model = StrongPolicyNet(board_config=cfg_4x4_k4)
        ckpt_path = tmp_path / "strong_4x4_k4.pt"
        meta = _make_trained_metadata("StrongPolicyNet", cfg_4x4_k4)
        save_torch_checkpoint(model, ckpt_path, meta)
        meta_path = resolve_meta_path(ckpt_path)

        # 2. overwrite=False rejects existing checkpoint.
        with pytest.raises(FileExistsError):
            save_torch_checkpoint(model, ckpt_path, meta, overwrite=False)

        # 3. Wrong board dimensions (same 4x4 grid, different win_k, and 3x3 grid).
        with pytest.raises(CheckpointVerificationError):
            load_torch_checkpoint(ckpt_path, board_config=cfg_4x4_k3)
        with pytest.raises(CheckpointVerificationError):
            load_torch_checkpoint(ckpt_path, board_config=cfg_3x3_k3)

        # 4. Wrong model class when loading into mismatched pre-instantiated module.
        wrong_cls_model = FastPolicyNet(board_config=cfg_4x4_k4)
        with pytest.raises(CheckpointVerificationError):
            load_torch_checkpoint(ckpt_path, model=wrong_cls_model)
        with pytest.raises(CheckpointVerificationError):
            verify_checkpoint(ckpt_path, expected_agent_class="ValueNet")

        # 5. Tampered weights file fails SHA-256 verification.
        tampered_ckpt = tmp_path / "tampered.pt"
        tampered_meta_path = resolve_meta_path(tampered_ckpt)
        save_torch_checkpoint(model, tampered_ckpt, meta)
        tampered_ckpt.write_bytes(tampered_ckpt.read_bytes() + b"\x00")
        with pytest.raises(CheckpointVerificationError):
            load_torch_checkpoint(tampered_ckpt, board_config=cfg_4x4_k4)
        assert tampered_meta_path.is_file()

        # 6. Metadata claiming 0 epochs/0 episodes on disk is rejected.
        untrained_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        untrained_meta["epochs_trained"] = 0
        untrained_meta["episodes_trained"] = 0
        meta_path.write_text(json.dumps(untrained_meta), encoding="utf-8")
        with pytest.raises(CheckpointVerificationError):
            load_torch_checkpoint(ckpt_path, board_config=cfg_4x4_k4)

        # 7. Missing metadata file rejected.
        meta_path.unlink()
        with pytest.raises(CheckpointVerificationError):
            load_torch_checkpoint(ckpt_path, board_config=cfg_4x4_k4)

        # 8. Missing checkpoint file raises FileNotFoundError.
        with pytest.raises(FileNotFoundError):
            load_torch_checkpoint(tmp_path / "does_not_exist.pt")

    def test_hardware_info_snapshot_fields(self) -> None:
        hw = collect_hardware_info()
        assert hw["device"] in ("cpu", "cuda")
        assert isinstance(hw["python"], str) and hw["python"]
        assert isinstance(hw["torch"], str) and hw["torch"]
        assert isinstance(hw["cpu_count"], int) and hw["cpu_count"] >= 1
        assert math.isfinite(float(hw["cpu_count"]))
