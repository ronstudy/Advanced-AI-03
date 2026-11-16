"""PyTorch neural-network architectures and shared Action Masking utilities.

Implements the general ``(B, 1, M, N)`` CNN+MLP architectures specified in
``Planning/ARCHITECTURE.md`` Section 4.2 and 4.3:
- :class:`FastPolicyNet`: 2 hidden MLP layers outputting ``M * N`` action logits.
- :class:`StrongPolicyNet`: 3 hidden MLP layers outputting ``M * N`` action logits.
- :class:`ValueNet`: 2 hidden MLP layers + ``Tanh`` outputting scalar value in ``[-1, 1]``.
- :class:`DQNNet`: 2 hidden MLP layers outputting ``M * N`` Q-values.

Architectural design generalizes the 3x3 CNN-MLP layout from the course slides
(``Tic tac toe Game.pdf``, slides 44-48 and 61-62) and Mark Liu (2024,
*AlphaGo Simplified*) to arbitrary ``M x N, K`` boards with shared tensor
canonicalization and mandatory legal-action masking.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ttt_ai.agents.base import _is_finite_real, _is_strict_int
from ttt_ai.config import BoardConfig
from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError
from ttt_ai.engine.rules import PLAYER_O, PLAYER_X, encode_canonical_board

DEFAULT_CONV_CHANNELS: int = 128
DEFAULT_HIDDEN_DIM: int = 64
MASK_FILL_VALUE: float = -1e9


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that `value` is a strictly positive integer (excluding booleans)."""
    if not _is_strict_int(value) or int(value) <= 0:
        raise InvalidConfigError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _validate_player_mark(player: Any) -> int:
    """Validate that `player` is `PLAYER_X (+1)` or `PLAYER_O (-1)`."""
    if not _is_strict_int(player) or int(player) not in (PLAYER_X, PLAYER_O):
        raise InvalidConfigError(
            f"current_player must be {PLAYER_X} (X) or {PLAYER_O} (O), got {player!r}."
        )
    return int(player)


def board_to_tensor(
    board: np.ndarray | torch.Tensor,
    current_player: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a 2D/3D/4D board array into a float32 ``(B, 1, M, N)`` tensor.

    Args:
        board: Board state with shape ``(M, N)``, ``(B, M, N)``, or ``(B, 1, M, N)``.
        current_player: Optional active player (``+1`` or ``-1``). When provided,
            the board is canonicalized so friendly pieces become ``+1`` and
            opponent pieces become ``-1`` (via :func:`encode_canonical_board` for
            2D boards).
        device: Optional target PyTorch device for the output tensor.

    Returns:
        Contiguous ``torch.float32`` tensor of shape ``(B, 1, M, N)``.

    Raises:
        InvalidConfigError: If `board` has invalid type, boolean/non-numeric dtype,
            unsupported rank/shape, spatial size smaller than ``3x3``, or invalid
            `current_player`.
    """
    if isinstance(board, np.ndarray):
        if board.dtype == np.bool_ or not np.issubdtype(board.dtype, np.number):
            raise InvalidConfigError(
                f"board NumPy array must have a numeric non-boolean dtype, got {board.dtype}."
            )
        if board.ndim == 2:
            if board.shape[0] < 3 or board.shape[1] < 3:
                raise InvalidConfigError(
                    f"Board spatial shape must be at least 3x3, got {board.shape}."
                )
            arr = (
                encode_canonical_board(board, _validate_player_mark(current_player))
                if current_player is not None
                else board
            )
            tensor = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif board.ndim == 3:
            if board.shape[1] < 3 or board.shape[2] < 3:
                raise InvalidConfigError(
                    f"Board spatial shape must be at least 3x3, got {board.shape[1:]}."
                )
            tensor = torch.as_tensor(board, dtype=torch.float32).unsqueeze(1)
            if current_player is not None:
                tensor = tensor * float(_validate_player_mark(current_player))
        elif board.ndim == 4:
            if board.shape[1] != 1:
                raise InvalidConfigError(
                    f"4D board input must have channel dim 1, got shape {board.shape}."
                )
            if board.shape[2] < 3 or board.shape[3] < 3:
                raise InvalidConfigError(
                    f"Board spatial shape must be at least 3x3, got {board.shape[2:]}."
                )
            tensor = torch.as_tensor(board, dtype=torch.float32)
            if current_player is not None:
                tensor = tensor * float(_validate_player_mark(current_player))
        else:
            raise InvalidConfigError(
                f"board must have 2, 3, or 4 dimensions, got shape {board.shape}."
            )
    elif isinstance(board, torch.Tensor):
        if board.dtype == torch.bool or board.is_complex():
            raise InvalidConfigError(
                f"board tensor must be numeric and non-boolean/non-complex, got {board.dtype}."
            )
        if board.ndim == 2:
            if board.shape[0] < 3 or board.shape[1] < 3:
                raise InvalidConfigError(
                    f"Board spatial shape must be at least 3x3, got {tuple(board.shape)}."
                )
            tensor = board.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif board.ndim == 3:
            if board.shape[1] < 3 or board.shape[2] < 3:
                raise InvalidConfigError(
                    f"Board spatial shape must be at least 3x3, got {tuple(board.shape[1:])}."
                )
            tensor = board.to(dtype=torch.float32).unsqueeze(1)
        elif board.ndim == 4:
            if board.shape[1] != 1:
                raise InvalidConfigError(
                    f"4D board tensor must have channel dim 1, got shape {tuple(board.shape)}."
                )
            if board.shape[2] < 3 or board.shape[3] < 3:
                raise InvalidConfigError(
                    f"Board spatial shape must be at least 3x3, got {tuple(board.shape[2:])}."
                )
            tensor = board.to(dtype=torch.float32)
        else:
            raise InvalidConfigError(
                f"board tensor must have 2, 3, or 4 dimensions, got {tuple(board.shape)}."
            )
        if current_player is not None:
            tensor = tensor * float(_validate_player_mark(current_player))
    else:
        raise InvalidConfigError(
            f"board must be a numpy.ndarray or torch.Tensor, got {type(board).__name__}."
        )

    if device is not None:
        tensor = tensor.to(device=device)
    return tensor.contiguous()


def build_legal_action_mask(
    num_actions: int,
    legal_actions: Sequence[int] | np.ndarray,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a 1D boolean tensor mask marking valid cell indices as ``True``.

    Args:
        num_actions: Total action space size (``M * N >= 1``).
        legal_actions: Sequence of flat action indices in ``[0, num_actions - 1]``.
        device: Optional PyTorch device for the output boolean tensor.

    Returns:
        Boolean ``torch.Tensor`` of shape ``(num_actions,)``.

    Raises:
        InvalidConfigError: If `num_actions` is invalid or `legal_actions` is
            not a valid index sequence.
        IllegalMoveError: If any action index is non-integer or out of bounds.
    """
    total_actions = _validate_positive_int(num_actions, "num_actions")
    if isinstance(legal_actions, (str, bytes, dict)) or not isinstance(
        legal_actions, (Sequence, np.ndarray)
    ):
        raise InvalidConfigError(
            "legal_actions must be a sequence or 1D array of integer indices, "
            f"got {type(legal_actions).__name__}."
        )
    if isinstance(legal_actions, np.ndarray):
        if legal_actions.ndim != 1:
            raise InvalidConfigError(
                f"1D legal_actions array expected, got shape {legal_actions.shape}."
            )
        items = legal_actions.tolist()
    else:
        items = list(legal_actions)

    mask = torch.zeros(total_actions, dtype=torch.bool, device=device)
    for action in items:
        if not _is_strict_int(action):
            raise IllegalMoveError(
                f"Legal actions must be integer indices, got {action!r}."
            )
        action_idx = int(action)
        if not (0 <= action_idx < total_actions):
            raise IllegalMoveError(
                f"Action index {action_idx} out of bounds for [0, {total_actions - 1}]."
            )
        mask[action_idx] = True
    return mask


def _resolve_action_mask(
    scores: torch.Tensor,
    legal_actions_or_mask: Sequence[int] | Sequence[Sequence[int]] | np.ndarray | torch.Tensor,
    *,
    allow_empty: bool = False,
) -> torch.Tensor:
    """Resolve action indices or boolean masks into a boolean tensor matching `scores`."""
    if not isinstance(scores, torch.Tensor):
        raise InvalidConfigError(
            f"scores must be a torch.Tensor, got {type(scores).__name__}."
        )
    if scores.ndim not in (1, 2) or scores.shape[-1] < 1:
        raise InvalidConfigError(
            f"scores must have shape (A,) or (B, A) with A >= 1, got {tuple(scores.shape)}."
        )

    num_actions = scores.shape[-1]
    if isinstance(legal_actions_or_mask, torch.Tensor):
        if legal_actions_or_mask.dtype != torch.bool:
            raise InvalidConfigError(
                "Boolean torch.Tensor expected for mask inputs; got "
                f"{legal_actions_or_mask.dtype}. Pass a Sequence[int] for index lists."
            )
        mask = legal_actions_or_mask.to(device=scores.device)
    elif isinstance(legal_actions_or_mask, np.ndarray) and legal_actions_or_mask.dtype == np.bool_:
        mask = torch.as_tensor(legal_actions_or_mask, dtype=torch.bool, device=scores.device)
    elif isinstance(legal_actions_or_mask, np.ndarray):
        if scores.ndim == 2 and scores.shape[0] != 1:
            raise InvalidConfigError(
                "Flat NumPy index arrays cannot mask batches with B > 1; "
                "pass a 2D boolean mask (B, A) or per-row index sequences."
            )
        flat_mask = build_legal_action_mask(
            num_actions, legal_actions_or_mask, device=scores.device
        )
        mask = flat_mask if scores.ndim == 1 else flat_mask.unsqueeze(0)
    elif isinstance(legal_actions_or_mask, Sequence) and not isinstance(
        legal_actions_or_mask, (str, bytes)
    ):
        if scores.ndim == 1:
            mask = build_legal_action_mask(
                num_actions, legal_actions_or_mask, device=scores.device
            )
        else:
            batch_size = scores.shape[0]
            if len(legal_actions_or_mask) == 0:
                if batch_size == 0:
                    mask = torch.zeros((0, num_actions), dtype=torch.bool, device=scores.device)
                elif batch_size == 1:
                    mask = torch.zeros((1, num_actions), dtype=torch.bool, device=scores.device)
                else:
                    raise InvalidConfigError(
                        "Empty flat legal_actions sequence cannot mask a batch with B > 1."
                    )
            else:
                first_item = legal_actions_or_mask[0]
                is_nested = isinstance(first_item, (Sequence, np.ndarray)) and not isinstance(
                    first_item, (str, bytes)
                )
                if is_nested:
                    if len(legal_actions_or_mask) != batch_size:
                        raise InvalidConfigError(
                            f"Batch mask length ({len(legal_actions_or_mask)}) must match "
                            f"scores batch size ({batch_size})."
                        )
                    rows_masks = [
                        build_legal_action_mask(num_actions, row_legal, device=scores.device)
                        for row_legal in legal_actions_or_mask  # type: ignore[arg-type]
                    ]
                    mask = torch.stack(rows_masks, dim=0)
                else:
                    if batch_size != 1:
                        raise InvalidConfigError(
                            "A flat sequence of legal action indices can only mask a single "
                            f"state (batch size 1), got batch size {batch_size}."
                        )
                    mask = build_legal_action_mask(
                        num_actions,
                        legal_actions_or_mask,  # type: ignore[arg-type]
                        device=scores.device,
                    ).unsqueeze(0)
    else:
        raise InvalidConfigError(
            "legal_actions_or_mask must be a sequence of indices or boolean mask, "
            f"got {type(legal_actions_or_mask).__name__}."
        )

    if scores.ndim == 1:
        if mask.shape != scores.shape:
            raise InvalidConfigError(
                f"Mask shape {tuple(mask.shape)} does not match 1D scores {tuple(scores.shape)}."
            )
        if not allow_empty and not bool(mask.any()):
            raise IllegalMoveError("Cannot apply action mask when no legal actions remain.")
    else:
        if mask.ndim == 1 and mask.shape[0] == num_actions:
            if scores.shape[0] != 1:
                raise InvalidConfigError(
                    "1D boolean mask cannot broadcast across multi-item batch "
                    f"of size {scores.shape[0]}."
                )
            mask = mask.unsqueeze(0)
        elif mask.shape != scores.shape:
            raise InvalidConfigError(
                f"Mask shape {tuple(mask.shape)} does not match 2D scores {tuple(scores.shape)}."
            )
        if not allow_empty and scores.shape[0] > 0 and not bool(mask.any(dim=-1).all()):
            raise IllegalMoveError(
                "Every state in the batch must have at least one legal action."
            )
    return mask


def apply_action_mask(
    scores: torch.Tensor,
    legal_actions_or_mask: Sequence[int] | Sequence[Sequence[int]] | np.ndarray | torch.Tensor,
    mask_value: float = MASK_FILL_VALUE,
    *,
    allow_empty: bool = False,
) -> torch.Tensor:
    """Mask illegal action entries in logits or Q-values with a large negative value.

    Using ``-1e9`` by default (instead of ``-inf``) avoids ``NaN`` gradients or
    unstable softmax reductions while driving ``exp(mask_value)`` to zero in
    IEEE-754 float32 arithmetic.

    Args:
        scores: Floating-point tensor of shape ``(A,)`` or ``(B, A)``.
        legal_actions_or_mask: Sequence of legal action indices or boolean mask.
        mask_value: Negative fill value assigned to illegal action positions.
        allow_empty: When ``True``, allows rows with zero legal actions (useful
            for terminal rows in batched DQN target computation).

    Returns:
        New tensor of the same shape as `scores` with illegal positions filled
        by `mask_value`.

    Raises:
        InvalidConfigError: If `scores` is not a floating-point 1D/2D tensor or
            `mask_value` is invalid.
        IllegalMoveError: If `allow_empty` is ``False`` and any item has zero
            legal actions, or if an action index is out of bounds.
    """
    if not isinstance(scores, torch.Tensor) or not scores.is_floating_point():
        raise InvalidConfigError("scores must be a floating-point torch.Tensor.")
    if (
        not isinstance(mask_value, (int, float))
        or isinstance(mask_value, bool)
        or math.isnan(float(mask_value))
        or float(mask_value) >= 0.0
    ):
        raise InvalidConfigError(
            f"mask_value must be a negative number (finite or -inf), got {mask_value!r}."
        )

    mask = _resolve_action_mask(scores, legal_actions_or_mask, allow_empty=allow_empty)
    return scores.masked_fill(~mask, float(mask_value))


# Public alias matching RULES.md terminology.
action_masking = apply_action_mask


def masked_softmax(
    logits: torch.Tensor,
    legal_actions_or_mask: Sequence[int] | Sequence[Sequence[int]] | np.ndarray | torch.Tensor,
    *,
    temperature: float = 1.0,
    mask_value: float = MASK_FILL_VALUE,
) -> torch.Tensor:
    """Compute masked action probabilities with strictly ``0.0`` on illegal cells.

    Args:
        logits: Unnormalized floating-point policy logits of shape ``(A,)`` or ``(B, A)``.
        legal_actions_or_mask: Legal action indices or boolean mask.
        temperature: Positive temperature divisor applied to logits before softmax.
        mask_value: Negative mask value passed to :func:`apply_action_mask`.

    Returns:
        Probability tensor matching `logits.shape` where illegal cells equal ``0.0``
        and legal cells sum to ``1.0`` along the last dimension.

    Raises:
        InvalidConfigError: If `logits` or `temperature` is invalid.
        IllegalMoveError: If any state has zero legal actions.
    """
    if not _is_finite_real(temperature) or float(temperature) <= 0.0:
        raise InvalidConfigError(
            f"temperature must be a finite number > 0.0, got {temperature!r}."
        )
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise InvalidConfigError("logits must be a floating-point torch.Tensor.")

    mask = _resolve_action_mask(logits, legal_actions_or_mask, allow_empty=False)
    temp = float(temperature)
    scaled_logits = logits / temp if temp != 1.0 else logits
    masked_logits = apply_action_mask(
        scaled_logits,
        mask,
        mask_value=mask_value,
        allow_empty=False,
    )
    probs = torch.softmax(masked_logits, dim=-1)
    # Explicitly zero illegal slots and renormalize so floating-point underflow
    # or custom mask values can never leave non-zero probability on occupied cells.
    probs = probs.masked_fill(~mask, 0.0)
    if probs.shape[0] == 0 and probs.ndim == 2:
        return probs
    denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return probs / denom


def masked_argmax(
    scores: torch.Tensor | np.ndarray,
    legal_actions: Sequence[int],
) -> tuple[int, float, dict[int, float]]:
    """Select the highest-scoring legal action with deterministic tie-breaking.

    Ties in score are broken by choosing the smallest flat cell index so
    inference remains deterministic across runs and matches
    :meth:`ttt_ai.agents.base.AgentDecision.top_actions`.

    Args:
        scores: 1D ``(A,)`` or single-row ``(1, A)`` logits, probabilities, or Q-values.
        legal_actions: Non-empty sequence of legal cell indices in ``[0, A - 1]``.

    Returns:
        Tuple ``(best_action, best_score, legal_scores_dict)``.

    Raises:
        InvalidConfigError: If `scores` has invalid type/shape or non-finite values
            on legal indices.
        IllegalMoveError: If `legal_actions` is empty or contains invalid indices.
    """
    if isinstance(scores, torch.Tensor):
        arr = scores.detach().cpu().numpy()
    elif isinstance(scores, np.ndarray):
        arr = scores
    else:
        raise InvalidConfigError(
            f"scores must be a torch.Tensor or numpy.ndarray, got {type(scores).__name__}."
        )

    if arr.ndim == 2 and arr.shape[0] == 1:
        flat_scores = arr[0]
    elif arr.ndim == 1:
        flat_scores = arr
    else:
        raise InvalidConfigError(
            f"scores must have shape (A,) or (1, A), got {arr.shape}."
        )

    num_actions = int(flat_scores.shape[0])
    if isinstance(legal_actions, (str, bytes, dict)) or not isinstance(
        legal_actions, Sequence
    ):
        raise InvalidConfigError(
            f"legal_actions must be a sequence of ints, got {type(legal_actions).__name__}."
        )
    if not legal_actions:
        raise IllegalMoveError("Cannot select an action when legal_actions is empty.")

    validated_actions: list[int] = []
    for action in legal_actions:
        if not _is_strict_int(action):
            raise IllegalMoveError(
                f"Legal actions must be integer indices, got {action!r}."
            )
        action_idx = int(action)
        if not (0 <= action_idx < num_actions):
            raise IllegalMoveError(
                f"Action index {action_idx} out of bounds for [0, {num_actions - 1}]."
            )
        validated_actions.append(action_idx)

    unique_legal = sorted(set(validated_actions))
    legal_scores: dict[int, float] = {}
    best_action = -1
    best_score = -math.inf
    for action_idx in unique_legal:
        val = float(flat_scores[action_idx])
        if not math.isfinite(val):
            raise InvalidConfigError(
                f"Score for legal action {action_idx} must be finite, got {val!r}."
            )
        legal_scores[action_idx] = val
        if best_action < 0 or val > best_score:
            best_action = action_idx
            best_score = val

    return best_action, best_score, legal_scores


class BaseBoardNet(nn.Module):
    """Base CNN+MLP module for ``(B, 1, M, N)`` Extended Tic-Tac-Toe networks."""

    def __init__(
        self,
        board_config: BoardConfig | None = None,
        *,
        rows: int = 4,
        cols: int = 4,
        win_k: int = 4,
        conv_channels: int = DEFAULT_CONV_CHANNELS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        """Initialize shared spatial dimensions and convolutional stem.

        Args:
            board_config: Optional validated :class:`BoardConfig`. When omitted,
                one is built from `(rows, cols, win_k)`.
            rows: Board rows (`M >= 3`) used when `board_config` is ``None``.
            cols: Board columns (`N >= 3`) used when `board_config` is ``None``.
            win_k: Consecutive marks needed to win (`3 <= win_k <= min(M, N)`).
            conv_channels: Output channels of the ``3x3`` convolutional layer.
            hidden_dim: Width of the fully connected hidden layers.

        Raises:
            InvalidConfigError: If board config or channel/hidden dimensions are invalid.
        """
        super().__init__()
        if board_config is not None:
            if not isinstance(board_config, BoardConfig):
                raise InvalidConfigError(
                    f"board_config must be a BoardConfig, got {type(board_config).__name__}."
                )
            resolved_config = board_config
        else:
            resolved_config = BoardConfig(rows=rows, cols=cols, win_k=win_k)

        self._board_config: BoardConfig = resolved_config
        self._conv_channels: int = _validate_positive_int(conv_channels, "conv_channels")
        self._hidden_dim: int = _validate_positive_int(hidden_dim, "hidden_dim")

        # Keep padding=1 with kernel_size=3 so spatial dimensions remain (M, N)
        # regardless of board size, preserving exact flattened size conv_channels * M * N.
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, self._conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    @property
    def board_config(self) -> BoardConfig:
        """Validated :class:`BoardConfig` this network was built for."""
        return self._board_config

    @property
    def rows(self) -> int:
        """Board row count (`M`)."""
        return self._board_config.rows

    @property
    def cols(self) -> int:
        """Board column count (`N`)."""
        return self._board_config.cols

    @property
    def win_k(self) -> int:
        """Win line length (`K`)."""
        return self._board_config.win_k

    @property
    def num_actions(self) -> int:
        """Total board cells (`M * N`)."""
        return self._board_config.num_cells

    @property
    def conv_channels(self) -> int:
        """Number of filters in the convolutional feature extractor."""
        return self._conv_channels

    @property
    def hidden_dim(self) -> int:
        """Number of units per hidden dense layer."""
        return self._hidden_dim

    def _build_mlp_backbone(self, num_hidden_layers: int) -> nn.Sequential:
        """Construct `Flatten -> (Linear + ReLU) * num_hidden_layers`."""
        layer_count = _validate_positive_int(num_hidden_layers, "num_hidden_layers")
        in_features = self._conv_channels * self._board_config.num_cells
        layers: list[nn.Module] = [nn.Flatten(start_dim=1)]
        for _ in range(layer_count):
            layers.append(nn.Linear(in_features, self._hidden_dim))
            layers.append(nn.ReLU())
            in_features = self._hidden_dim
        return nn.Sequential(*layers)

    def _validate_input_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Validate `(B, 1, M, N)` tensor rank, spatial shape, and numeric dtype."""
        if not isinstance(x, torch.Tensor):
            raise InvalidConfigError(
                f"Network input must be a torch.Tensor, got {type(x).__name__}."
            )
        if x.dtype == torch.bool or x.is_complex():
            raise InvalidConfigError(
                f"Network input tensor must be real numeric, got dtype {x.dtype}."
            )
        expected_tail = (1, self._board_config.rows, self._board_config.cols)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected_tail:
            raise InvalidConfigError(
                f"Expected input shape (B, 1, {expected_tail[1]}, {expected_tail[2]}), "
                f"got {tuple(x.shape)}."
            )
        return x if x.dtype == torch.float32 else x.to(dtype=torch.float32)

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Return the total number of scalar parameters in this module."""
        return sum(
            param.numel()
            for param in self.parameters()
            if not trainable_only or param.requires_grad
        )

    def get_architecture_config(self) -> dict[str, Any]:
        """Return serializable architecture hyperparameters for checkpoint metadata."""
        return {
            "agent_class": self.__class__.__name__,
            "board_config": self._board_config.to_dict(),
            "conv_channels": self._conv_channels,
            "hidden_dim": self._hidden_dim,
            "num_parameters": self.count_parameters(),
        }


class _BasePolicyNet(BaseBoardNet):
    """Shared implementation for CNN+MLP policy networks outputting action logits."""

    _NUM_HIDDEN_LAYERS: int = 2

    def __init__(
        self,
        board_config: BoardConfig | None = None,
        *,
        rows: int = 4,
        cols: int = 4,
        win_k: int = 4,
        conv_channels: int = DEFAULT_CONV_CHANNELS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        """Initialize convolutional stem, MLP hidden layers, and logits head."""
        super().__init__(
            board_config=board_config,
            rows=rows,
            cols=cols,
            win_k=win_k,
            conv_channels=conv_channels,
            hidden_dim=hidden_dim,
        )
        self.mlp = self._build_mlp_backbone(self._NUM_HIDDEN_LAYERS)
        self.policy_head = nn.Linear(self._hidden_dim, self.num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized action logits of shape ``(B, M * N)``.

        Args:
            x: Canonical board batch tensor of shape ``(B, 1, M, N)``.

        Returns:
            Raw action logits of shape ``(B, M * N)`` suitable for
            ``CrossEntropyLoss`` during training or :func:`masked_softmax` at inference.
        """
        validated = self._validate_input_tensor(x)
        features = self.conv_block(validated)
        hidden = self.mlp(features)
        return self.policy_head(hidden)

    def predict_probs(
        self,
        x: torch.Tensor,
        legal_actions_or_mask: Sequence[int] | Sequence[Sequence[int]] | np.ndarray | torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Run forward pass and apply mandatory legal-action masking + softmax."""
        logits = self.forward(x)
        return masked_softmax(logits, legal_actions_or_mask, temperature=temperature)


class FastPolicyNet(_BasePolicyNet):
    """Lightweight supervised Policy Network (2 hidden MLP layers) for fast inference."""

    _NUM_HIDDEN_LAYERS = 2


class StrongPolicyNet(_BasePolicyNet):
    """Deeper supervised Policy Network (3 hidden MLP layers) & HybridMCTS prior."""

    _NUM_HIDDEN_LAYERS = 3


class ValueNet(BaseBoardNet):
    """Supervised State-Value Network outputting scalar values in ``[-1.0, +1.0]``."""

    def __init__(
        self,
        board_config: BoardConfig | None = None,
        *,
        rows: int = 4,
        cols: int = 4,
        win_k: int = 4,
        conv_channels: int = DEFAULT_CONV_CHANNELS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        """Initialize convolutional stem, 2-layer MLP backbone, and Tanh value head."""
        super().__init__(
            board_config=board_config,
            rows=rows,
            cols=cols,
            win_k=win_k,
            conv_channels=conv_channels,
            hidden_dim=hidden_dim,
        )
        self.mlp = self._build_mlp_backbone(2)
        self.value_head = nn.Sequential(
            nn.Linear(self._hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate canonical state value in ``[-1.0, +1.0]`` with shape ``(B, 1)``."""
        validated = self._validate_input_tensor(x)
        features = self.conv_block(validated)
        hidden = self.mlp(features)
        return self.value_head(hidden)


class DQNNet(BaseBoardNet):
    """Deep Q-Network estimating per-action Q-values of shape ``(B, M * N)``."""

    def __init__(
        self,
        board_config: BoardConfig | None = None,
        *,
        rows: int = 4,
        cols: int = 4,
        win_k: int = 4,
        conv_channels: int = DEFAULT_CONV_CHANNELS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        """Initialize convolutional stem, 2-layer MLP backbone, and Q-value head."""
        super().__init__(
            board_config=board_config,
            rows=rows,
            cols=cols,
            win_k=win_k,
            conv_channels=conv_channels,
            hidden_dim=hidden_dim,
        )
        self.mlp = self._build_mlp_backbone(2)
        self.q_head = nn.Linear(self._hidden_dim, self.num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw per-cell Q-values of shape ``(B, M * N)``."""
        validated = self._validate_input_tensor(x)
        features = self.conv_block(validated)
        hidden = self.mlp(features)
        return self.q_head(hidden)

    def predict_masked_q(
        self,
        x: torch.Tensor,
        legal_actions_or_mask: Sequence[int] | Sequence[Sequence[int]] | np.ndarray | torch.Tensor,
        *,
        mask_value: float = MASK_FILL_VALUE,
        allow_empty: bool = False,
    ) -> torch.Tensor:
        """Compute Q-values with illegal action positions masked to `mask_value`."""
        q_values = self.forward(x)
        return apply_action_mask(
            q_values,
            legal_actions_or_mask,
            mask_value=mask_value,
            allow_empty=allow_empty,
        )


_MODEL_REGISTRY: dict[str, type[BaseBoardNet]] = {
    "fastpolicynet": FastPolicyNet,
    "fast_policy": FastPolicyNet,
    "fast_policy_net": FastPolicyNet,
    "strongpolicynet": StrongPolicyNet,
    "strong_policy": StrongPolicyNet,
    "strong_policy_net": StrongPolicyNet,
    "valuenet": ValueNet,
    "value": ValueNet,
    "value_net": ValueNet,
    "dqnnet": DQNNet,
    "dqn": DQNNet,
    "dqn_net": DQNNet,
}


def create_model(
    model_name: str,
    board_config: BoardConfig | None = None,
    *,
    rows: int = 4,
    cols: int = 4,
    win_k: int = 4,
    conv_channels: int = DEFAULT_CONV_CHANNELS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
) -> BaseBoardNet:
    """Instantiate a neural-network architecture by class name or short alias.

    Args:
        model_name: Model class name (e.g., ``"StrongPolicyNet"``) or alias
            (e.g., ``"strong_policy"``, ``"dqn"``).
        board_config: Optional validated :class:`BoardConfig`.
        rows: Board rows when `board_config` is omitted.
        cols: Board columns when `board_config` is omitted.
        win_k: Win length when `board_config` is omitted.
        conv_channels: Convolutional channel count.
        hidden_dim: Hidden MLP layer width.

    Returns:
        Instantiated :class:`BaseBoardNet` subclass.

    Raises:
        InvalidConfigError: If `model_name` is invalid or unrecognized.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise InvalidConfigError(
            f"model_name must be a non-empty string, got {model_name!r}."
        )
    key = model_name.strip().lower().replace("-", "_")
    model_cls = _MODEL_REGISTRY.get(key)
    if model_cls is None:
        supported = ("FastPolicyNet", "StrongPolicyNet", "ValueNet", "DQNNet")
        raise InvalidConfigError(
            f"Unknown neural network model '{model_name}'. Supported models: {supported}."
        )
    return model_cls(
        board_config=board_config,
        rows=rows,
        cols=cols,
        win_k=win_k,
        conv_channels=conv_channels,
        hidden_dim=hidden_dim,
    )
