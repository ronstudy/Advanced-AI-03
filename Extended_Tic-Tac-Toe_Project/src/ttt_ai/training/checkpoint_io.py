"""Checkpoint persistence and integrity verification with ``*.meta.json`` sidecars.

Enforces ``Planning/RULES.md`` Section 1.3 and ``Planning/ARCHITECTURE.md``
Section 5.2 by requiring every ``.pt`` / ``.pkl`` checkpoint to carry validated
metadata (model class, board dimensions, seed, training progress, and SHA-256
weight digest), rejecting untrained or board-mismatched checkpoints on load.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ttt_ai.agents.base import _is_finite_real, _is_strict_int
from ttt_ai.agents.nn_models import (
    DEFAULT_CONV_CHANNELS,
    DEFAULT_HIDDEN_DIM,
    BaseBoardNet,
    create_model,
)
from ttt_ai.config import BoardConfig
from ttt_ai.engine.exceptions import CheckpointVerificationError, InvalidConfigError

_HASH_CHUNK_BYTES: int = 65536
_SHA256_HEX_CHARS: frozenset[str] = frozenset("0123456789abcdef")


def _validate_non_empty_str(value: Any, field_name: str) -> str:
    """Validate that `value` is a non-empty string and strip outer whitespace."""
    if not isinstance(value, str) or not value.strip():
        raise CheckpointVerificationError(
            f"CheckpointMetadata.{field_name} must be a non-empty string, got {value!r}."
        )
    return value.strip()


def _validate_sha256_hex(digest: Any, *, allow_empty: bool = False) -> str:
    """Validate a lowercase 64-character SHA-256 hexadecimal digest string."""
    if not isinstance(digest, str):
        raise CheckpointVerificationError(
            f"weights_sha256 must be a string, got {type(digest).__name__}."
        )
    cleaned = digest.strip().lower()
    if cleaned.startswith("sha256:"):
        cleaned = cleaned[len("sha256:") :]
    if allow_empty and cleaned == "":
        return ""
    if len(cleaned) != 64 or any(ch not in _SHA256_HEX_CHARS for ch in cleaned):
        raise CheckpointVerificationError(
            f"weights_sha256 must be a 64-character hex digest, got {digest!r}."
        )
    return cleaned


def resolve_meta_path(checkpoint_path: str | Path) -> Path:
    """Derive the sidecar ``*.meta.json`` path corresponding to a checkpoint file.

    Args:
        checkpoint_path: Path to a ``.pt``, ``.pth``, or ``.pkl`` checkpoint.

    Returns:
        Path with suffix replaced by ``.meta.json``.

    Raises:
        InvalidConfigError: If `checkpoint_path` is empty or points to a directory.
    """
    if isinstance(checkpoint_path, str) and not checkpoint_path.strip():
        raise InvalidConfigError("checkpoint_path must be a non-empty path.")
    path = Path(checkpoint_path)
    if path.exists() and path.is_dir():
        raise InvalidConfigError(f"checkpoint_path must be a file path, got directory: {path}")
    return path.with_suffix(".meta.json")


def compute_file_sha256(path: str | Path) -> str:
    """Compute the lowercase SHA-256 hexadecimal digest of a file on disk.

    Args:
        path: File path to hash.

    Returns:
        64-character lowercase SHA-256 hexadecimal digest.

    Raises:
        FileNotFoundError: If `path` does not exist or is not a file.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {target}")

    hasher = hashlib.sha256()
    with target.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def collect_hardware_info() -> dict[str, Any]:
    """Capture current CPU/GPU and runtime library metadata for reproducibility."""
    info: dict[str, Any] = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 1,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Validated metadata stored alongside every trained model checkpoint.

    Attributes:
        agent_class: Model or agent class identifier (e.g., ``"StrongPolicyNet"``).
        board_config: Exact :class:`BoardConfig` the checkpoint was trained on.
        seed: Non-negative RNG seed used for training.
        training_type: Pipeline category (e.g., ``"supervised"``, ``"rl_dqn"``).
        architecture: Human-readable network/model architecture description.
        trained_at: ISO-8601 UTC timestamp recorded at training/save completion.
        epochs_trained: Completed training epochs (`>= 0`).
        episodes_trained: Completed RL episodes (`>= 0`).
        dataset_hash: Optional dataset content digest for supervised runs.
        split_policy: Optional split policy identifier (e.g., ``"game_id_70_15_15"``).
        best_val_loss: Optional best validation loss achieved.
        best_val_accuracy: Optional best validation accuracy in ``[0.0, 1.0]``.
        final_eval_score: Optional post-training evaluation win-score.
        reward_summary: Optional dictionary summarizing RL reward/win progression.
        training_time_sec: Total wall-clock training duration in seconds.
        hardware: Runtime hardware/software snapshot from :func:`collect_hardware_info`.
        model_config: Hyperparameters needed to reconstruct the PyTorch module.
        extra: Optional additional hyperparameters or training notes.
        weights_sha256: Lowercase 64-char SHA-256 hex digest of the checkpoint file.
    """

    agent_class: str
    board_config: BoardConfig
    seed: int
    training_type: str
    architecture: str = ""
    trained_at: str = ""
    epochs_trained: int = 0
    episodes_trained: int = 0
    dataset_hash: str | None = None
    split_policy: str | None = None
    best_val_loss: float | None = None
    best_val_accuracy: float | None = None
    final_eval_score: float | None = None
    reward_summary: dict[str, Any] = field(default_factory=dict)
    training_time_sec: float = 0.0
    hardware: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    weights_sha256: str = ""

    def __post_init__(self) -> None:
        """Enforce metadata completeness, numeric bounds, and trained-model status."""
        object.__setattr__(
            self,
            "agent_class",
            _validate_non_empty_str(self.agent_class, "agent_class"),
        )
        if not isinstance(self.board_config, BoardConfig):
            raise CheckpointVerificationError(
                "CheckpointMetadata.board_config must be a BoardConfig, "
                f"got {type(self.board_config).__name__}."
            )
        if not _is_strict_int(self.seed) or int(self.seed) < 0:
            raise CheckpointVerificationError(
                f"CheckpointMetadata.seed must be a non-negative integer, got {self.seed!r}."
            )
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(
            self,
            "training_type",
            _validate_non_empty_str(self.training_type, "training_type"),
        )

        if not isinstance(self.architecture, str):
            raise CheckpointVerificationError(
                "CheckpointMetadata.architecture must be a string, "
                f"got {type(self.architecture).__name__}."
            )
        object.__setattr__(self, "architecture", self.architecture.strip())

        if not isinstance(self.trained_at, str):
            raise CheckpointVerificationError(
                "CheckpointMetadata.trained_at must be an ISO timestamp string, "
                f"got {type(self.trained_at).__name__}."
            )
        resolved_ts = (
            self.trained_at.strip()
            if self.trained_at.strip()
            else datetime.now(timezone.utc).isoformat()
        )
        object.__setattr__(self, "trained_at", resolved_ts)

        if not _is_strict_int(self.epochs_trained) or int(self.epochs_trained) < 0:
            raise CheckpointVerificationError(
                "CheckpointMetadata.epochs_trained must be an integer >= 0, "
                f"got {self.epochs_trained!r}."
            )
        if not _is_strict_int(self.episodes_trained) or int(self.episodes_trained) < 0:
            raise CheckpointVerificationError(
                "CheckpointMetadata.episodes_trained must be an integer >= 0, "
                f"got {self.episodes_trained!r}."
            )
        epochs = int(self.epochs_trained)
        episodes = int(self.episodes_trained)
        object.__setattr__(self, "epochs_trained", epochs)
        object.__setattr__(self, "episodes_trained", episodes)

        # RULES.md Section 1.3 forbids using untrained random weights for Levels 5-7.
        if epochs == 0 and episodes == 0:
            raise CheckpointVerificationError(
                "Checkpoint metadata indicates an untrained model "
                "(both epochs_trained and episodes_trained are 0)."
            )

        for str_field in ("dataset_hash", "split_policy"):
            val = getattr(self, str_field)
            if val is not None:
                object.__setattr__(self, str_field, _validate_non_empty_str(val, str_field))

        if self.best_val_loss is not None:
            if not _is_finite_real(self.best_val_loss):
                raise CheckpointVerificationError(
                    "CheckpointMetadata.best_val_loss must be finite or None, "
                    f"got {self.best_val_loss!r}."
                )
            object.__setattr__(self, "best_val_loss", float(self.best_val_loss))

        if self.best_val_accuracy is not None:
            if (
                not _is_finite_real(self.best_val_accuracy)
                or not (0.0 <= float(self.best_val_accuracy) <= 1.0)
            ):
                raise CheckpointVerificationError(
                    "CheckpointMetadata.best_val_accuracy must be in [0.0, 1.0] or None, "
                    f"got {self.best_val_accuracy!r}."
                )
            object.__setattr__(self, "best_val_accuracy", float(self.best_val_accuracy))

        if self.final_eval_score is not None:
            if not _is_finite_real(self.final_eval_score):
                raise CheckpointVerificationError(
                    "CheckpointMetadata.final_eval_score must be finite or None, "
                    f"got {self.final_eval_score!r}."
                )
            object.__setattr__(self, "final_eval_score", float(self.final_eval_score))

        if (
            not _is_finite_real(self.training_time_sec)
            or float(self.training_time_sec) < 0.0
        ):
            raise CheckpointVerificationError(
                "CheckpointMetadata.training_time_sec must be finite >= 0.0, "
                f"got {self.training_time_sec!r}."
            )
        object.__setattr__(self, "training_time_sec", float(self.training_time_sec))

        for map_field in ("reward_summary", "hardware", "model_config", "extra"):
            raw_map = getattr(self, map_field)
            if not isinstance(raw_map, Mapping):
                raise CheckpointVerificationError(
                    f"CheckpointMetadata.{map_field} must be a mapping, "
                    f"got {type(raw_map).__name__}."
                )
            object.__setattr__(self, map_field, dict(raw_map))

        object.__setattr__(
            self,
            "weights_sha256",
            _validate_sha256_hex(self.weights_sha256, allow_empty=True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a JSON-compatible dictionary."""
        return {
            "agent_class": self.agent_class,
            "board_config": self.board_config.to_dict(),
            "seed": self.seed,
            "training_type": self.training_type,
            "architecture": self.architecture,
            "trained_at": self.trained_at,
            "epochs_trained": self.epochs_trained,
            "episodes_trained": self.episodes_trained,
            "dataset_hash": self.dataset_hash,
            "split_policy": self.split_policy,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "final_eval_score": self.final_eval_score,
            "reward_summary": dict(self.reward_summary),
            "training_time_sec": self.training_time_sec,
            "hardware": dict(self.hardware),
            "model_config": dict(self.model_config),
            "extra": dict(self.extra),
            "weights_sha256": self.weights_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckpointMetadata:
        """Deserialize and strictly verify persisted checkpoint metadata.

        Args:
            data: Parsed JSON mapping loaded from a ``*.meta.json`` sidecar.

        Returns:
            Validated :class:`CheckpointMetadata` instance.

        Raises:
            CheckpointVerificationError: If required keys are missing, SHA-256
                digest is absent/malformed, or metadata validation fails.
        """
        if not isinstance(data, Mapping):
            raise CheckpointVerificationError(
                f"Checkpoint metadata root must be a JSON object, got {type(data).__name__}."
            )

        required_keys = (
            "agent_class",
            "board_config",
            "seed",
            "training_type",
            "trained_at",
            "weights_sha256",
        )
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise CheckpointVerificationError(
                f"Checkpoint metadata is missing required keys: {missing}."
            )

        try:
            board_cfg = BoardConfig.from_dict(data["board_config"])
        except InvalidConfigError as exc:
            raise CheckpointVerificationError(
                f"Invalid board_config in checkpoint metadata: {exc}"
            ) from exc

        trained_at = _validate_non_empty_str(data["trained_at"], "trained_at")
        digest = _validate_sha256_hex(data["weights_sha256"], allow_empty=False)

        return cls(
            agent_class=data["agent_class"],
            board_config=board_cfg,
            seed=data["seed"],
            training_type=data["training_type"],
            architecture=data.get("architecture", ""),
            trained_at=trained_at,
            epochs_trained=data.get("epochs_trained", 0),
            episodes_trained=data.get("episodes_trained", 0),
            dataset_hash=data.get("dataset_hash"),
            split_policy=data.get("split_policy"),
            best_val_loss=data.get("best_val_loss"),
            best_val_accuracy=data.get("best_val_accuracy"),
            final_eval_score=data.get("final_eval_score"),
            reward_summary=data.get("reward_summary", {}),
            training_time_sec=data.get("training_time_sec", 0.0),
            hardware=data.get("hardware", {}),
            model_config=data.get("model_config", {}),
            extra=data.get("extra", {}),
            weights_sha256=digest,
        )


def _prepare_save_paths(
    checkpoint_path: str | Path,
    meta_path: str | Path | None,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Validate destination paths and ensure parent directories exist."""
    ckpt_path = Path(checkpoint_path)
    resolved_meta = Path(meta_path) if meta_path is not None else resolve_meta_path(ckpt_path)
    if ckpt_path.exists() and ckpt_path.is_dir():
        raise InvalidConfigError(f"checkpoint_path cannot be a directory: {ckpt_path}")
    if resolved_meta.exists() and resolved_meta.is_dir():
        raise InvalidConfigError(f"meta_path cannot be a directory: {resolved_meta}")
    if not overwrite and (ckpt_path.exists() or resolved_meta.exists()):
        raise FileExistsError(
            f"Checkpoint or metadata already exists: {ckpt_path} / {resolved_meta}"
        )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_meta.parent.mkdir(parents=True, exist_ok=True)
    return ckpt_path, resolved_meta


def _finalize_and_write_metadata(
    ckpt_path: Path,
    resolved_meta: Path,
    metadata: CheckpointMetadata,
    *,
    extra_model_config: Mapping[str, Any] | None = None,
    default_architecture: str = "",
) -> CheckpointMetadata:
    """Compute file digest, enrich hardware/model fields, and write JSON sidecar."""
    digest = compute_file_sha256(ckpt_path)
    hardware = dict(metadata.hardware) if metadata.hardware else collect_hardware_info()
    merged_model_cfg = dict(metadata.model_config)
    if extra_model_config:
        for key, value in extra_model_config.items():
            merged_model_cfg.setdefault(key, value)
    architecture = metadata.architecture or default_architecture

    final_meta = replace(
        metadata,
        architecture=architecture,
        hardware=hardware,
        model_config=merged_model_cfg,
        weights_sha256=digest,
    )
    resolved_meta.write_text(
        json.dumps(final_meta.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return final_meta


def save_torch_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    metadata: CheckpointMetadata,
    *,
    meta_path: str | Path | None = None,
    overwrite: bool = True,
) -> tuple[Path, Path, CheckpointMetadata]:
    """Persist a trained PyTorch model ``state_dict`` and verified ``*.meta.json``.

    Saving pure tensor ``state_dict()`` rather than pickled module objects ensures
    loading with ``torch.load(..., weights_only=True)`` succeeds cleanly across
    PyTorch versions without arbitrary pickle execution risk.

    Args:
        model: Trained PyTorch :class:`torch.nn.Module`.
        checkpoint_path: Destination ``.pt`` / ``.pth`` file path.
        metadata: Validated :class:`CheckpointMetadata` describing the training run.
        meta_path: Optional override path for the ``*.meta.json`` sidecar.
        overwrite: When ``False``, raises :class:`FileExistsError` if target exists.

    Returns:
        Tuple ``(checkpoint_path, meta_path, finalized_metadata)``.

    Raises:
        InvalidConfigError: If arguments have invalid types or destination is a dir.
        CheckpointVerificationError: If `model` mismatches `metadata` board config
            or class name, or metadata represents an untrained model.
        FileExistsError: If `overwrite` is ``False`` and target file exists.
    """
    if not isinstance(model, nn.Module):
        raise InvalidConfigError(
            f"model must be a torch.nn.Module, got {type(model).__name__}."
        )
    if not isinstance(metadata, CheckpointMetadata):
        raise InvalidConfigError(
            f"metadata must be a CheckpointMetadata instance, got {type(metadata).__name__}."
        )

    extra_model_cfg: dict[str, Any] = {}
    default_arch = model.__class__.__name__
    if isinstance(model, BaseBoardNet):
        if model.board_config != metadata.board_config:
            raise CheckpointVerificationError(
                f"Model board_config {model.board_config.to_dict()} does not match "
                f"metadata board_config {metadata.board_config.to_dict()}."
            )
        if model.__class__.__name__ != metadata.agent_class:
            raise CheckpointVerificationError(
                f"Model class '{model.__class__.__name__}' does not match "
                f"metadata agent_class '{metadata.agent_class}'."
            )
        extra_model_cfg = {
            "conv_channels": model.conv_channels,
            "hidden_dim": model.hidden_dim,
            "num_parameters": model.count_parameters(),
        }
        default_arch = (
            f"{model.__class__.__name__}(conv={model.conv_channels}, "
            f"hidden={model.hidden_dim}, board={model.board_config.name})"
        )

    ckpt_path, resolved_meta = _prepare_save_paths(
        checkpoint_path, meta_path, overwrite=overwrite
    )
    torch.save(model.state_dict(), ckpt_path)
    final_meta = _finalize_and_write_metadata(
        ckpt_path,
        resolved_meta,
        metadata,
        extra_model_config=extra_model_cfg,
        default_architecture=default_arch,
    )
    return ckpt_path, resolved_meta, final_meta


def save_pickle_checkpoint(
    payload: Any,
    checkpoint_path: str | Path,
    metadata: CheckpointMetadata,
    *,
    meta_path: str | Path | None = None,
    overwrite: bool = True,
) -> tuple[Path, Path, CheckpointMetadata]:
    """Persist a trained non-PyTorch artifact (e.g., Tabular Q-table) and metadata.

    Args:
        payload: Picklable object such as a Q-table dictionary.
        checkpoint_path: Destination ``.pkl`` file path.
        metadata: Validated :class:`CheckpointMetadata` describing the training run.
        meta_path: Optional override path for the ``*.meta.json`` sidecar.
        overwrite: When ``False``, raises :class:`FileExistsError` if target exists.

    Returns:
        Tuple ``(checkpoint_path, meta_path, finalized_metadata)``.

    Raises:
        InvalidConfigError: If `metadata` is invalid or path points to a directory.
        FileExistsError: If `overwrite` is ``False`` and target file exists.
    """
    if not isinstance(metadata, CheckpointMetadata):
        raise InvalidConfigError(
            f"metadata must be a CheckpointMetadata instance, got {type(metadata).__name__}."
        )
    ckpt_path, resolved_meta = _prepare_save_paths(
        checkpoint_path, meta_path, overwrite=overwrite
    )
    with ckpt_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    final_meta = _finalize_and_write_metadata(
        ckpt_path,
        resolved_meta,
        metadata,
        default_architecture=metadata.agent_class,
    )
    return ckpt_path, resolved_meta, final_meta


def verify_checkpoint(
    checkpoint_path: str | Path,
    *,
    board_config: BoardConfig | None = None,
    expected_agent_class: str | Sequence[str] | None = None,
    expected_training_type: str | None = None,
    meta_path: str | Path | None = None,
) -> CheckpointMetadata:
    """Verify checkpoint presence, sidecar metadata, SHA-256 digest, and board rules.

    Args:
        checkpoint_path: Path to the ``.pt`` / ``.pth`` / ``.pkl`` weights file.
        board_config: Optional expected :class:`BoardConfig` that must match.
        expected_agent_class: Optional expected class name or sequence of allowed names.
        expected_training_type: Optional expected ``training_type`` string.
        meta_path: Optional override path to the ``*.meta.json`` sidecar.

    Returns:
        Validated :class:`CheckpointMetadata`.

    Raises:
        FileNotFoundError: If `checkpoint_path` does not exist.
        InvalidConfigError: If `board_config` or path arguments are invalid.
        CheckpointVerificationError: If the ``*.meta.json`` sidecar is missing,
            malformed, untrained, fails SHA-256 verification, or mismatches board/class.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    resolved_meta = Path(meta_path) if meta_path is not None else resolve_meta_path(ckpt_path)
    if not resolved_meta.is_file():
        raise CheckpointVerificationError(
            f"Missing required checkpoint metadata file: {resolved_meta}"
        )

    try:
        raw_text = resolved_meta.read_text(encoding="utf-8")
        raw_meta = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointVerificationError(
            f"Failed to read or parse checkpoint metadata '{resolved_meta}': {exc}"
        ) from exc

    metadata = CheckpointMetadata.from_dict(raw_meta)

    actual_sha256 = compute_file_sha256(ckpt_path)
    if actual_sha256 != metadata.weights_sha256:
        raise CheckpointVerificationError(
            f"Checkpoint integrity check failed for '{ckpt_path}': "
            f"expected SHA-256 {metadata.weights_sha256}, got {actual_sha256}."
        )

    if board_config is not None:
        if not isinstance(board_config, BoardConfig):
            raise InvalidConfigError(
                f"board_config must be a BoardConfig, got {type(board_config).__name__}."
            )
        if metadata.board_config != board_config:
            raise CheckpointVerificationError(
                f"Checkpoint board_config mismatch: expected {board_config.name} "
                f"({board_config.to_dict()}), found {metadata.board_config.name} "
                f"({metadata.board_config.to_dict()})."
            )

    if expected_agent_class is not None:
        if isinstance(expected_agent_class, str):
            allowed = (expected_agent_class,)
        elif isinstance(expected_agent_class, Sequence):
            allowed = tuple(str(name) for name in expected_agent_class)
        else:
            raise InvalidConfigError(
                "expected_agent_class must be a string or sequence of strings, "
                f"got {type(expected_agent_class).__name__}."
            )
        if metadata.agent_class not in allowed:
            raise CheckpointVerificationError(
                f"Checkpoint agent_class mismatch: expected one of {allowed}, "
                f"found '{metadata.agent_class}'."
            )

    if expected_training_type is not None:
        if metadata.training_type != expected_training_type:
            raise CheckpointVerificationError(
                f"Checkpoint training_type mismatch: expected '{expected_training_type}', "
                f"found '{metadata.training_type}'."
            )

    return metadata


def load_torch_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module | None = None,
    board_config: BoardConfig | None = None,
    expected_agent_class: str | Sequence[str] | None = None,
    expected_training_type: str | None = None,
    meta_path: str | Path | None = None,
    device: torch.device | str = "cpu",
    eval_mode: bool = True,
) -> tuple[nn.Module, CheckpointMetadata]:
    """Verify and load a trained PyTorch checkpoint into a matching model.

    Args:
        checkpoint_path: Path to the ``.pt`` / ``.pth`` checkpoint file.
        model: Optional pre-instantiated :class:`torch.nn.Module`. When ``None``,
            the network is constructed automatically from ``metadata.agent_class``,
            ``metadata.board_config``, and ``metadata.model_config``.
        board_config: Optional expected :class:`BoardConfig` to verify against.
        expected_agent_class: Optional expected class name(s) to enforce.
        expected_training_type: Optional expected ``training_type`` string.
        meta_path: Optional override path for the ``*.meta.json`` sidecar.
        device: Target PyTorch device for loading tensors.
        eval_mode: When ``True`` (default), puts the loaded model into ``eval()`` mode.

    Returns:
        Tuple ``(loaded_model, verified_metadata)``.

    Raises:
        FileNotFoundError: If `checkpoint_path` does not exist.
        InvalidConfigError: If `model` or `board_config` arguments are invalid.
        CheckpointVerificationError: If metadata verification fails, SHA-256
            mismatches, board dimensions conflict, or ``state_dict`` fails to load.
    """
    resolved_board_cfg = board_config
    resolved_agent_cls = expected_agent_class

    if model is not None:
        if not isinstance(model, nn.Module):
            raise InvalidConfigError(
                f"model must be a torch.nn.Module, got {type(model).__name__}."
            )
        if isinstance(model, BaseBoardNet):
            if resolved_board_cfg is None:
                resolved_board_cfg = model.board_config
            elif resolved_board_cfg != model.board_config:
                raise CheckpointVerificationError(
                    f"Provided model board_config {model.board_config.to_dict()} conflicts "
                    f"with requested board_config {resolved_board_cfg.to_dict()}."
                )
            if resolved_agent_cls is None:
                resolved_agent_cls = model.__class__.__name__

    metadata = verify_checkpoint(
        checkpoint_path,
        board_config=resolved_board_cfg,
        expected_agent_class=resolved_agent_cls,
        expected_training_type=expected_training_type,
        meta_path=meta_path,
    )

    try:
        loaded_obj = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise CheckpointVerificationError(
            f"Failed to safely load PyTorch weights from '{checkpoint_path}': {exc}"
        ) from exc

    if not isinstance(loaded_obj, Mapping):
        raise CheckpointVerificationError(
            f"Expected state_dict mapping in '{checkpoint_path}', got {type(loaded_obj).__name__}."
        )

    if model is None:
        conv_channels = metadata.model_config.get("conv_channels", DEFAULT_CONV_CHANNELS)
        hidden_dim = metadata.model_config.get("hidden_dim", DEFAULT_HIDDEN_DIM)
        target_model: nn.Module = create_model(
            metadata.agent_class,
            board_config=metadata.board_config,
            conv_channels=conv_channels,
            hidden_dim=hidden_dim,
        )
    else:
        target_model = model

    try:
        target_model.load_state_dict(loaded_obj, strict=True)
    except RuntimeError as exc:
        raise CheckpointVerificationError(
            f"State dict incompatible with '{target_model.__class__.__name__}': {exc}"
        ) from exc

    target_model = target_model.to(device=device)
    if eval_mode:
        target_model.eval()
    return target_model, metadata


def load_pickle_checkpoint(
    checkpoint_path: str | Path,
    *,
    board_config: BoardConfig | None = None,
    expected_agent_class: str | Sequence[str] | None = None,
    expected_training_type: str | None = None,
    meta_path: str | Path | None = None,
) -> tuple[Any, CheckpointMetadata]:
    """Verify and load a trained pickle checkpoint (e.g., Tabular Q-table).

    Args:
        checkpoint_path: Path to the ``.pkl`` checkpoint file.
        board_config: Optional expected :class:`BoardConfig` to verify against.
        expected_agent_class: Optional expected class name(s) to enforce.
        expected_training_type: Optional expected ``training_type`` string.
        meta_path: Optional override path for the ``*.meta.json`` sidecar.

    Returns:
        Tuple ``(unpickled_payload, verified_metadata)``.

    Raises:
        FileNotFoundError: If `checkpoint_path` does not exist.
        CheckpointVerificationError: If verification fails or unpickling fails.
    """
    metadata = verify_checkpoint(
        checkpoint_path,
        board_config=board_config,
        expected_agent_class=expected_agent_class,
        expected_training_type=expected_training_type,
        meta_path=meta_path,
    )
    ckpt_path = Path(checkpoint_path)
    try:
        with ckpt_path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError, ValueError) as exc:
        raise CheckpointVerificationError(
            f"Failed to unpickle checkpoint '{ckpt_path}': {exc}"
        ) from exc
    return payload, metadata
