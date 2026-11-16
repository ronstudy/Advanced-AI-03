"""Level 6 reinforcement-learning agents: TabularQAgent and DQNAgent."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ttt_ai.agents.base import AgentDecision, BaseAgent, _is_finite_real
from ttt_ai.agents.nn_models import DQNNet, board_to_tensor, masked_argmax
from ttt_ai.config import BoardConfig
from ttt_ai.engine.exceptions import CheckpointVerificationError, InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.training.checkpoint_io import (
    CheckpointMetadata,
    load_pickle_checkpoint,
    load_torch_checkpoint,
)


def _validate_epsilon(epsilon: float) -> float:
    """Validate exploration probability lies in ``[0.0, 1.0]``."""
    if not _is_finite_real(epsilon) or not (0.0 <= float(epsilon) <= 1.0):
        raise InvalidConfigError(
            f"epsilon must be a finite number in [0.0, 1.0], got {epsilon!r}."
        )
    return float(epsilon)


def _validate_board_match(
    game: ExtendedTicTacToe,
    expected_board: BoardConfig,
    agent_name: str,
) -> None:
    """Verify the active game dimensions and win_k match the agent's trained config."""
    if (
        game.rows != expected_board.rows
        or game.cols != expected_board.cols
        or game.win_k != expected_board.win_k
    ):
        raise InvalidConfigError(
            f"Agent '{agent_name}' was trained for board {expected_board.name}, "
            f"cannot run on {game.rows}x{game.cols}_k{game.win_k}."
        )


class TabularQAgent(BaseAgent):
    """Level 6 Tabular Q-learning agent keyed by canonical board byte strings.

    Primarily designed for ``3x3, K=3`` (course slide 56 reference) where the
    reachable state space is small enough to learn an explicit state-action
    value table. Enforces verified ``.pkl`` + ``*.meta.json`` loading and
    strict legal-action masking.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        q_table: Mapping[bytes, np.ndarray] | None = None,
        metadata: CheckpointMetadata | None = None,
        board_config: BoardConfig | None = None,
        epsilon: float = 0.0,
        allow_unverified_table: bool = False,
        name: str | None = "TabularQAgent",
        seed: int | None = None,
        time_budget_ms: float | None = None,
    ) -> None:
        """Initialize a Level 6 Tabular Q-learning agent from a verified checkpoint."""
        resolved_eps = _validate_epsilon(epsilon)
        if checkpoint_path is not None and q_table is not None:
            raise InvalidConfigError(
                "Provide either checkpoint_path or q_table, not both simultaneously."
            )

        resolved_ckpt: str | None = None
        if checkpoint_path is not None:
            payload, loaded_meta = load_pickle_checkpoint(
                checkpoint_path,
                board_config=board_config,
                expected_agent_class=("TabularQAgent", "TabularQLearningAgent"),
            )
            if not isinstance(payload, Mapping):
                raise CheckpointVerificationError(
                    f"TabularQ checkpoint must contain a mapping, got {type(payload).__name__}."
                )
            resolved_table = {
                bytes(k): np.asarray(v, dtype=np.float32) for k, v in payload.items()
            }
            resolved_meta: CheckpointMetadata | None = loaded_meta
            resolved_board = loaded_meta.board_config
            resolved_ckpt = str(Path(checkpoint_path))
        else:
            if q_table is None:
                raise CheckpointVerificationError(
                    "Untrained TabularQ agents are forbidden: provide a verified "
                    "checkpoint_path or a trained q_table with CheckpointMetadata."
                )
            if not isinstance(q_table, Mapping):
                raise InvalidConfigError(
                    "q_table must be a mapping of {bytes: ndarray}, "
                    f"got {type(q_table).__name__}."
                )
            if metadata is None and not allow_unverified_table:
                raise CheckpointVerificationError(
                    "Direct q_table injection without CheckpointMetadata is rejected by default "
                    "(RULES.md Section 1.3)."
                )
            if metadata is not None:
                if not isinstance(metadata, CheckpointMetadata):
                    raise InvalidConfigError(
                        "metadata must be a CheckpointMetadata instance, "
                        f"got {type(metadata).__name__}."
                    )
                if board_config is not None and metadata.board_config != board_config:
                    raise CheckpointVerificationError(
                        f"Provided board_config {board_config.to_dict()} mismatches "
                        f"metadata {metadata.board_config.to_dict()}."
                    )
                resolved_board = metadata.board_config
            elif board_config is not None:
                resolved_board = board_config
            else:
                raise InvalidConfigError(
                    "board_config is required when providing an unverified in-memory q_table."
                )
            resolved_table = {
                bytes(k): np.asarray(v, dtype=np.float32) for k, v in q_table.items()
            }
            resolved_meta = metadata

        expected_cells = resolved_board.num_cells
        for state_key, values in resolved_table.items():
            if values.shape != (expected_cells,):
                raise CheckpointVerificationError(
                    f"Q-table vector for state {state_key!r} has shape {values.shape}, "
                    f"expected ({expected_cells},)."
                )

        super().__init__(
            name=name,
            level=6,
            seed=seed,
            time_budget_ms=time_budget_ms,
        )
        self._q_table: dict[bytes, np.ndarray] = resolved_table
        self._metadata: CheckpointMetadata | None = resolved_meta
        self._board_config: BoardConfig = resolved_board
        self._checkpoint_path: str | None = resolved_ckpt
        self._epsilon: float = resolved_eps

    @property
    def board_config(self) -> BoardConfig:
        """Validated :class:`BoardConfig` supported by this Q-table."""
        return self._board_config

    @property
    def metadata(self) -> CheckpointMetadata | None:
        """Verified checkpoint metadata if loaded/provided."""
        return self._metadata

    @property
    def epsilon(self) -> float:
        """Exploration probability used during action selection."""
        return self._epsilon

    @property
    def num_states(self) -> int:
        """Number of distinct canonical states stored in the Q-table."""
        return len(self._q_table)

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration metadata for tournament logs."""
        cfg = super().get_config()
        cfg.update(
            {
                "board_config": self._board_config.to_dict(),
                "checkpoint_path": self._checkpoint_path,
                "epsilon": self._epsilon,
                "num_states": len(self._q_table),
            }
        )
        return cfg

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        """Look up Q-values for ``game.canonical_board()`` and pick masked move."""
        _validate_board_match(game, self._board_config, self._name)
        legal_actions = game.legal_actions()
        key = game.canonical_board().tobytes()
        q_vec = self._q_table.get(key)
        if q_vec is None:
            q_vec = np.zeros(self._board_config.num_cells, dtype=np.float32)

        best_action, _, action_scores = masked_argmax(q_vec, legal_actions)
        if self._epsilon > 0.0 and float(self.rng.random()) < self._epsilon:
            chosen_action = int(legal_actions[int(self.rng.integers(0, len(legal_actions)))])
        else:
            chosen_action = best_action

        return AgentDecision(
            action=chosen_action,
            nodes_visited=1,
            simulations=0,
            depth_reached=0,
            value_estimate=float(action_scores[chosen_action]),
            action_scores=action_scores,
        )


# Alias matching PRD.md Section 3 naming.
TabularQLearningAgent = TabularQAgent


class DQNAgent(BaseAgent):
    """Level 6 Deep Q-Network agent for ``4x4`` and general ``M x N, K`` boards.

    Loads a trained :class:`DQNNet` verified against ``*.meta.json``, encodes
    ``game.canonical_board()`` into ``(1, 1, M, N)``, masks illegal cells with
    ``-1e9`` via :meth:`DQNNet.predict_masked_q`, and selects the legal action
    maximizing estimated Q-value.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        model: DQNNet | None = None,
        metadata: CheckpointMetadata | None = None,
        board_config: BoardConfig | None = None,
        epsilon: float = 0.0,
        device: str | torch.device = "cpu",
        allow_unverified_model: bool = False,
        name: str | None = "DQNAgent",
        seed: int | None = None,
        time_budget_ms: float | None = None,
    ) -> None:
        """Initialize a Level 6 DQN agent from a verified ``DQNNet`` checkpoint."""
        resolved_eps = _validate_epsilon(epsilon)
        if checkpoint_path is not None and model is not None:
            raise InvalidConfigError(
                "Provide either checkpoint_path or model, not both simultaneously."
            )

        resolved_ckpt: str | None = None
        if checkpoint_path is not None:
            loaded_model, loaded_meta = load_torch_checkpoint(
                checkpoint_path,
                board_config=board_config,
                expected_agent_class="DQNNet",
                device=device,
                eval_mode=True,
            )
            if not isinstance(loaded_model, DQNNet):
                raise CheckpointVerificationError(
                    f"Expected DQNNet checkpoint, got {loaded_model.__class__.__name__}."
                )
            resolved_model = loaded_model
            resolved_meta: CheckpointMetadata | None = loaded_meta
            resolved_ckpt = str(Path(checkpoint_path))
        else:
            if model is None:
                raise CheckpointVerificationError(
                    "Untrained DQN agents are forbidden: provide a verified "
                    "checkpoint_path or a trained DQNNet with CheckpointMetadata."
                )
            if not isinstance(model, DQNNet):
                raise InvalidConfigError(
                    f"Expected DQNNet instance, got {model.__class__.__name__}."
                )
            if metadata is None and not allow_unverified_model:
                raise CheckpointVerificationError(
                    "Direct DQNNet injection without CheckpointMetadata is rejected by default "
                    "(RULES.md Section 1.3)."
                )
            if metadata is not None:
                if not isinstance(metadata, CheckpointMetadata):
                    raise InvalidConfigError(
                        "metadata must be a CheckpointMetadata instance, "
                        f"got {type(metadata).__name__}."
                    )
                if metadata.board_config != model.board_config:
                    raise CheckpointVerificationError(
                        f"Model board_config {model.board_config.to_dict()} mismatches "
                        f"metadata {metadata.board_config.to_dict()}."
                    )
                if metadata.agent_class != "DQNNet":
                    raise CheckpointVerificationError(
                        f"Metadata agent_class '{metadata.agent_class}' does not match DQNNet."
                    )
            if board_config is not None and model.board_config != board_config:
                raise CheckpointVerificationError(
                    f"Model board_config {model.board_config.to_dict()} mismatches "
                    f"expected {board_config.to_dict()}."
                )
            resolved_model = model.to(device=device)
            resolved_model.eval()
            resolved_meta = metadata

        super().__init__(
            name=name,
            level=6,
            seed=seed,
            time_budget_ms=time_budget_ms,
        )
        self._model: DQNNet = resolved_model
        self._metadata: CheckpointMetadata | None = resolved_meta
        self._board_config: BoardConfig = resolved_model.board_config
        self._checkpoint_path: str | None = resolved_ckpt
        self._device: torch.device = torch.device(device)
        self._epsilon: float = resolved_eps

    @property
    def board_config(self) -> BoardConfig:
        """Validated :class:`BoardConfig` supported by this DQN model."""
        return self._board_config

    @property
    def model(self) -> DQNNet:
        """Underlying PyTorch :class:`DQNNet`."""
        return self._model

    @property
    def metadata(self) -> CheckpointMetadata | None:
        """Verified checkpoint metadata if loaded/provided."""
        return self._metadata

    @property
    def epsilon(self) -> float:
        """Exploration probability used during action selection."""
        return self._epsilon

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration metadata for tournament logs."""
        cfg = super().get_config()
        cfg.update(
            {
                "model_class": self._model.__class__.__name__,
                "board_config": self._board_config.to_dict(),
                "checkpoint_path": self._checkpoint_path,
                "epsilon": self._epsilon,
                "device": str(self._device),
            }
        )
        return cfg

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        """Evaluate masked Q-values on ``game.canonical_board()`` and pick move."""
        _validate_board_match(game, self._board_config, self._name)
        legal_actions = game.legal_actions()
        x_tensor = board_to_tensor(game.canonical_board(), device=self._device)

        with torch.no_grad():
            masked_q = self._model.predict_masked_q(x_tensor, legal_actions)

        q_row = masked_q[0].detach().cpu().numpy()
        best_action, _, action_scores = masked_argmax(q_row, legal_actions)
        if self._epsilon > 0.0 and float(self.rng.random()) < self._epsilon:
            chosen_action = int(legal_actions[int(self.rng.integers(0, len(legal_actions)))])
        else:
            chosen_action = best_action

        return AgentDecision(
            action=chosen_action,
            nodes_visited=1,
            simulations=0,
            depth_reached=1,
            value_estimate=float(action_scores[chosen_action]),
            action_scores=action_scores,
        )
