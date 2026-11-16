"""Level 5 supervised deep-learning agents: PolicyNetworkAgent and ValueGreedyAgent.

Both agents inherit from :class:`BaseAgent`, interact with
:class:`ExtendedTicTacToe` strictly via public Game APIs, enforce verified
checkpoint loading (refusing untrained random weights by default), and apply
mandatory legal-action masking prior to move selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ttt_ai.agents.base import AgentDecision, BaseAgent, _is_finite_real
from ttt_ai.agents.nn_models import (
    FastPolicyNet,
    StrongPolicyNet,
    ValueNet,
    board_to_tensor,
    masked_argmax,
)
from ttt_ai.config import BoardConfig
from ttt_ai.engine.exceptions import CheckpointVerificationError, InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.training.checkpoint_io import CheckpointMetadata, load_torch_checkpoint


def _validate_game_matches_board(
    game: ExtendedTicTacToe,
    expected_board: BoardConfig,
    agent_name: str,
) -> None:
    """Verify live game dimensions and win_k match the loaded model's BoardConfig."""
    if (
        game.rows != expected_board.rows
        or game.cols != expected_board.cols
        or game.win_k != expected_board.win_k
    ):
        raise InvalidConfigError(
            f"Agent '{agent_name}' was trained for board {expected_board.name}, "
            f"cannot run on {game.rows}x{game.cols}_k{game.win_k}."
        )


def _resolve_verified_torch_agent_model(
    *,
    checkpoint_path: str | Path | None,
    model: FastPolicyNet | StrongPolicyNet | ValueNet | None,
    metadata: CheckpointMetadata | None,
    board_config: BoardConfig | None,
    expected_classes: tuple[str, ...],
    expected_types: tuple[type[Any], ...],
    device: str | torch.device,
    allow_unverified_model: bool,
) -> tuple[Any, CheckpointMetadata | None, str | None]:
    """Load and verify a trained PyTorch network or validate an in-memory instance."""
    if checkpoint_path is not None and model is not None:
        raise InvalidConfigError(
            "Provide either checkpoint_path or model, not both simultaneously."
        )

    if checkpoint_path is not None:
        loaded_model, loaded_meta = load_torch_checkpoint(
            checkpoint_path,
            board_config=board_config,
            expected_agent_class=expected_classes,
            device=device,
            eval_mode=True,
        )
        if not isinstance(loaded_model, expected_types):
            raise CheckpointVerificationError(
                f"Expected model of type {expected_classes}, "
                f"got {loaded_model.__class__.__name__}."
            )
        return loaded_model, loaded_meta, str(Path(checkpoint_path))

    if model is None:
        raise CheckpointVerificationError(
            "Untrained neural agents are forbidden: provide a verified checkpoint_path "
            "or a trained model with CheckpointMetadata."
        )
    if not isinstance(model, expected_types):
        raise InvalidConfigError(
            f"Expected model instance of {expected_classes}, got {model.__class__.__name__}."
        )
    if metadata is None and not allow_unverified_model:
        raise CheckpointVerificationError(
            "Direct model injection without CheckpointMetadata is rejected by default "
            "(RULES.md Section 1.3). Pass trained CheckpointMetadata or set "
            "allow_unverified_model=True explicitly for unit testing."
        )
    if metadata is not None:
        if not isinstance(metadata, CheckpointMetadata):
            raise InvalidConfigError(
                f"metadata must be a CheckpointMetadata instance, got {type(metadata).__name__}."
            )
        if metadata.board_config != model.board_config:
            raise CheckpointVerificationError(
                f"Model board_config {model.board_config.to_dict()} mismatches "
                f"metadata {metadata.board_config.to_dict()}."
            )
        if metadata.agent_class != model.__class__.__name__:
            raise CheckpointVerificationError(
                f"Model class '{model.__class__.__name__}' mismatches "
                f"metadata '{metadata.agent_class}'."
            )
    if board_config is not None and model.board_config != board_config:
        raise CheckpointVerificationError(
            f"Model board_config {model.board_config.to_dict()} mismatches "
            f"expected {board_config.to_dict()}."
        )

    target_model = model.to(device=device)
    target_model.eval()
    return target_model, metadata, None


class PolicyNetworkAgent(BaseAgent):
    """Level 5 supervised policy-network agent (``FastPolicyNet`` / ``StrongPolicyNet``).

    Encodes ``game.canonical_board()`` into ``(1, 1, M, N)``, evaluates policy
    logits, applies mandatory legal-action masking (setting illegal logits to
    ``-1e9`` before Softmax and ``0.0`` after Softmax), and selects either the
    deterministic argmax move or a temperature-scaled sample via ``self.rng``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        model: FastPolicyNet | StrongPolicyNet | None = None,
        metadata: CheckpointMetadata | None = None,
        board_config: BoardConfig | None = None,
        temperature: float = 1.0,
        sample: bool = False,
        device: str | torch.device = "cpu",
        allow_unverified_model: bool = False,
        name: str | None = None,
        seed: int | None = None,
        time_budget_ms: float | None = None,
    ) -> None:
        """Initialize a Level 5 policy-network agent from a verified checkpoint."""
        if not _is_finite_real(temperature) or float(temperature) <= 0.0:
            raise InvalidConfigError(
                f"temperature must be finite > 0.0, got {temperature!r}."
            )
        if not isinstance(sample, bool):
            raise InvalidConfigError(f"sample must be a bool, got {type(sample).__name__}.")

        resolved_model, resolved_meta, resolved_ckpt = _resolve_verified_torch_agent_model(
            checkpoint_path=checkpoint_path,
            model=model,
            metadata=metadata,
            board_config=board_config,
            expected_classes=("FastPolicyNet", "StrongPolicyNet"),
            expected_types=(FastPolicyNet, StrongPolicyNet),
            device=device,
            allow_unverified_model=allow_unverified_model,
        )

        resolved_name = (
            name
            if name is not None
            else f"PolicyNetworkAgent({resolved_model.__class__.__name__})"
        )
        super().__init__(
            name=resolved_name,
            level=5,
            seed=seed,
            time_budget_ms=time_budget_ms,
        )
        self._model: FastPolicyNet | StrongPolicyNet = resolved_model
        self._metadata: CheckpointMetadata | None = resolved_meta
        self._checkpoint_path: str | None = resolved_ckpt
        self._board_config: BoardConfig = resolved_model.board_config
        self._device: torch.device = torch.device(device)
        self._temperature: float = float(temperature)
        self._sample: bool = sample

    @property
    def board_config(self) -> BoardConfig:
        """Validated :class:`BoardConfig` supported by this agent's model."""
        return self._board_config

    @property
    def model(self) -> FastPolicyNet | StrongPolicyNet:
        """Underlying PyTorch policy network."""
        return self._model

    @property
    def metadata(self) -> CheckpointMetadata | None:
        """Verified checkpoint metadata if loaded/provided."""
        return self._metadata

    @property
    def temperature(self) -> float:
        """Softmax temperature divisor used during inference."""
        return self._temperature

    @property
    def sample(self) -> bool:
        """Whether moves are sampled stochastically from masked probabilities."""
        return self._sample

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration metadata for tournament logs."""
        cfg = super().get_config()
        cfg.update(
            {
                "model_class": self._model.__class__.__name__,
                "board_config": self._board_config.to_dict(),
                "checkpoint_path": self._checkpoint_path,
                "temperature": self._temperature,
                "sample": self._sample,
                "device": str(self._device),
            }
        )
        return cfg

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        """Predict masked legal move probabilities and pick the next move."""
        _validate_game_matches_board(game, self._board_config, self._name)
        legal_actions = game.legal_actions()
        x_tensor = board_to_tensor(game.canonical_board(), device=self._device)

        with torch.no_grad():
            probs = self._model.predict_probs(
                x_tensor,
                legal_actions,
                temperature=self._temperature,
            )

        probs_row = probs[0].detach().cpu().numpy()
        action_scores = {action: float(probs_row[action]) for action in legal_actions}

        if self._sample:
            legal_weights = np.asarray(
                [action_scores[action] for action in legal_actions],
                dtype=np.float64,
            )
            legal_weights = legal_weights / legal_weights.sum()
            chosen_action = int(self.rng.choice(legal_actions, p=legal_weights))
        else:
            chosen_action, _, _ = masked_argmax(probs_row, legal_actions)

        return AgentDecision(
            action=chosen_action,
            nodes_visited=1,
            simulations=0,
            depth_reached=1,
            value_estimate=float(action_scores[chosen_action]),
            action_scores=action_scores,
        )


class ValueGreedyAgent(BaseAgent):
    """Level 5 1-ply lookahead agent driven by a trained :class:`ValueNet`.

    Implements the slide 61-62 state-value baseline: for every legal action ``a``,
    simulates one step on a clone (via ``step()``/``undo()``) and evaluates
    ``+1.0`` on immediate win, ``0.0`` on immediate draw, or ``-V(s')`` from the
    opponent's canonical viewpoint, selecting the highest-scoring legal move.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        model: ValueNet | None = None,
        metadata: CheckpointMetadata | None = None,
        board_config: BoardConfig | None = None,
        device: str | torch.device = "cpu",
        allow_unverified_model: bool = False,
        name: str | None = "ValueGreedyAgent",
        seed: int | None = None,
        time_budget_ms: float | None = None,
    ) -> None:
        """Initialize a Level 5 ValueNet 1-ply lookahead agent."""
        resolved_model, resolved_meta, resolved_ckpt = _resolve_verified_torch_agent_model(
            checkpoint_path=checkpoint_path,
            model=model,
            metadata=metadata,
            board_config=board_config,
            expected_classes=("ValueNet",),
            expected_types=(ValueNet,),
            device=device,
            allow_unverified_model=allow_unverified_model,
        )

        super().__init__(
            name=name,
            level=5,
            seed=seed,
            time_budget_ms=time_budget_ms,
        )
        self._model: ValueNet = resolved_model
        self._metadata: CheckpointMetadata | None = resolved_meta
        self._checkpoint_path: str | None = resolved_ckpt
        self._board_config: BoardConfig = resolved_model.board_config
        self._device: torch.device = torch.device(device)

    @property
    def board_config(self) -> BoardConfig:
        """Validated :class:`BoardConfig` supported by this agent's ValueNet."""
        return self._board_config

    @property
    def model(self) -> ValueNet:
        """Underlying PyTorch :class:`ValueNet`."""
        return self._model

    @property
    def metadata(self) -> CheckpointMetadata | None:
        """Verified checkpoint metadata if loaded/provided."""
        return self._metadata

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration metadata for tournament logs."""
        cfg = super().get_config()
        cfg.update(
            {
                "model_class": self._model.__class__.__name__,
                "board_config": self._board_config.to_dict(),
                "checkpoint_path": self._checkpoint_path,
                "device": str(self._device),
            }
        )
        return cfg

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        """Evaluate every legal 1-ply successor state using exact terminal checks or -V(s')."""
        _validate_game_matches_board(game, self._board_config, self._name)
        legal_actions = game.legal_actions()
        acting_player = int(game.current_player)
        sim = game.clone()

        action_scores: dict[int, float] = {}
        pending_actions: list[int] = []
        pending_boards: list[np.ndarray] = []

        for action in legal_actions:
            sim.step(action)
            if sim.is_terminal():
                payoff = sim.result(acting_player)
                action_scores[action] = float(payoff) if payoff is not None else 0.0
            else:
                pending_actions.append(action)
                pending_boards.append(sim.canonical_board())
            sim.undo()

        if pending_boards:
            batch_array = np.stack(pending_boards, axis=0)
            batch_tensor = board_to_tensor(batch_array, device=self._device)
            with torch.no_grad():
                opp_values = self._model(batch_tensor).squeeze(-1).detach().cpu().numpy()
            for action, opp_val in zip(pending_actions, opp_values, strict=True):
                # Negate opponent-perspective V(s') because Extended Tic-Tac-Toe is zero-sum.
                action_scores[action] = -float(opp_val)

        # Deterministic tie-breaking by ascending flat action index.
        best_action = legal_actions[0]
        best_score = action_scores[best_action]
        for action in legal_actions[1:]:
            score = action_scores[action]
            if score > best_score:
                best_score = score
                best_action = action

        return AgentDecision(
            action=best_action,
            nodes_visited=len(legal_actions),
            simulations=0,
            depth_reached=1,
            value_estimate=float(best_score),
            action_scores=action_scores,
        )
