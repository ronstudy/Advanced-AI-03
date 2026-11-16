"""Supervised training loops for FastPolicyNet, StrongPolicyNet, and ValueNet.

Trains Level 5 neural networks on game-level split :class:`SupervisedDataset`
archives, selects the best epoch purely on validation loss (zero test leakage),
persists verified checkpoints with ``*.meta.json`` sidecars, and logs per-epoch
learning curves as JSONL.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ttt_ai.agents.base import _is_finite_real, _is_strict_int
from ttt_ai.agents.nn_models import (
    DEFAULT_CONV_CHANNELS,
    DEFAULT_HIDDEN_DIM,
    BaseBoardNet,
    FastPolicyNet,
    StrongPolicyNet,
    ValueNet,
    apply_action_mask,
    create_model,
)
from ttt_ai.config import BoardConfig, seed_everything
from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.training.checkpoint_io import CheckpointMetadata, save_torch_checkpoint
from ttt_ai.training.dataset_gen import SupervisedDataset, SupervisedSplit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SupervisedTrainConfig:
    """Validated hyperparameters for supervised policy/value network training."""

    model_name: str = "StrongPolicyNet"
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    conv_channels: int = DEFAULT_CONV_CHANNELS
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate training hyperparameters and normalize numeric types."""
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise InvalidConfigError(
                f"model_name must be a non-empty string, got {self.model_name!r}."
            )
        object.__setattr__(self, "model_name", self.model_name.strip())

        if not _is_strict_int(self.epochs) or int(self.epochs) < 1:
            raise InvalidConfigError(f"epochs must be >= 1, got {self.epochs!r}.")
        object.__setattr__(self, "epochs", int(self.epochs))

        if not _is_strict_int(self.batch_size) or int(self.batch_size) < 1:
            raise InvalidConfigError(f"batch_size must be >= 1, got {self.batch_size!r}.")
        object.__setattr__(self, "batch_size", int(self.batch_size))

        if not _is_finite_real(self.learning_rate) or float(self.learning_rate) <= 0.0:
            raise InvalidConfigError(
                f"learning_rate must be finite > 0.0, got {self.learning_rate!r}."
            )
        object.__setattr__(self, "learning_rate", float(self.learning_rate))

        if not _is_finite_real(self.weight_decay) or float(self.weight_decay) < 0.0:
            raise InvalidConfigError(
                f"weight_decay must be finite >= 0.0, got {self.weight_decay!r}."
            )
        object.__setattr__(self, "weight_decay", float(self.weight_decay))

        if not _is_strict_int(self.conv_channels) or int(self.conv_channels) < 1:
            raise InvalidConfigError(
                f"conv_channels must be >= 1, got {self.conv_channels!r}."
            )
        object.__setattr__(self, "conv_channels", int(self.conv_channels))

        if not _is_strict_int(self.hidden_dim) or int(self.hidden_dim) < 1:
            raise InvalidConfigError(
                f"hidden_dim must be >= 1, got {self.hidden_dim!r}."
            )
        object.__setattr__(self, "hidden_dim", int(self.hidden_dim))

        if not _is_strict_int(self.seed) or int(self.seed) < 0:
            raise InvalidConfigError(
                f"seed must be a non-negative integer, got {self.seed!r}."
            )
        object.__setattr__(self, "seed", int(self.seed))

        if not isinstance(self.device, str) or not self.device.strip():
            raise InvalidConfigError(f"device must be a non-empty string, got {self.device!r}.")
        object.__setattr__(self, "device", self.device.strip())


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Per-epoch training and validation metrics recorded during supervised fitting."""

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float | None
    val_accuracy: float | None
    learning_rate: float
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize epoch metrics into a JSON-safe dictionary."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
            "elapsed_sec": self.elapsed_sec,
        }


@dataclass(slots=True)
class SupervisedTrainingResult:
    """Summary returned after training and testing a supervised Level 5 model."""

    model_name: str
    board_config: BoardConfig
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float | None
    test_loss: float
    test_accuracy: float | None
    training_time_sec: float
    model: BaseBoardNet
    history: list[EpochMetrics] = field(default_factory=list)
    checkpoint_path: Path | None = None
    meta_path: Path | None = None
    log_path: Path | None = None
    metadata: CheckpointMetadata | None = None


def _build_split_loader(
    split: SupervisedSplit,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Convert a :class:`SupervisedSplit` into a PyTorch ``DataLoader``."""
    x_tensor = torch.from_numpy(split.boards).to(dtype=torch.float32).unsqueeze(1)
    y_action = torch.from_numpy(split.actions).to(dtype=torch.long)
    y_value = torch.from_numpy(split.values).to(dtype=torch.float32).unsqueeze(1)
    legal_masks = torch.from_numpy(split.legal_masks).to(dtype=torch.bool)
    dataset = TensorDataset(x_tensor, y_action, y_value, legal_masks)

    generator: torch.Generator | None = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(split))),
        shuffle=shuffle,
        generator=generator,
    )


def _evaluate_policy_split(
    model: BaseBoardNet,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Compute mean cross-entropy loss and masked top-1 accuracy on a split."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_action, _, legal_masks in loader:
            x_dev = x_batch.to(device)
            y_dev = y_action.to(device)
            mask_dev = legal_masks.to(device)

            logits = model(x_dev)
            loss = criterion(logits, y_dev)
            masked_logits = apply_action_mask(logits, mask_dev, allow_empty=False)
            preds = masked_logits.argmax(dim=-1)

            batch_n = int(x_dev.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_correct += int((preds == y_dev).sum().item())
            total_samples += batch_n

    denom = float(max(1, total_samples))
    return total_loss / denom, float(total_correct) / denom


def _evaluate_value_split(
    model: BaseBoardNet,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: nn.MSELoss,
    device: torch.device,
) -> float:
    """Compute mean squared error on a value-regression split."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_batch, _, y_value, _ in loader:
            x_dev = x_batch.to(device)
            z_dev = y_value.to(device)
            preds = model(x_dev)
            loss = criterion(preds, z_dev)
            batch_n = int(x_dev.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_samples += batch_n

    return total_loss / float(max(1, total_samples))


def _write_learning_curve_jsonl(
    log_path: Path,
    result: SupervisedTrainingResult,
    dataset: SupervisedDataset,
    config: SupervisedTrainConfig,
) -> None:
    """Persist metadata, per-epoch curves, and test summary to a JSONL file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        json.dumps(
            {
                "type": "meta",
                "model_name": result.model_name,
                "board_config": result.board_config.to_dict(),
                "seed": config.seed,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "dataset_summary": dataset.summary(),
            }
        )
    ]
    for epoch_rec in result.history:
        payload = {"type": "epoch"}
        payload.update(epoch_rec.to_dict())
        lines.append(json.dumps(payload))
    lines.append(
        json.dumps(
            {
                "type": "summary",
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "best_val_accuracy": result.best_val_accuracy,
                "test_loss": result.test_loss,
                "test_accuracy": result.test_accuracy,
                "training_time_sec": result.training_time_sec,
            }
        )
    )
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_supervised_model(
    dataset: SupervisedDataset,
    config: SupervisedTrainConfig,
    *,
    checkpoint_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> SupervisedTrainingResult:
    """Train a supervised Policy or Value network on a game-split dataset.

    Selects the best checkpoint weights purely on ``val_loss`` and evaluates the
    restored best model once on the held-out ``test`` split.

    Args:
        dataset: Disjoint :class:`SupervisedDataset` generated by ``dataset_gen``.
        config: Validated :class:`SupervisedTrainConfig`.
        checkpoint_path: Optional ``.pt`` destination to persist best weights +
            ``*.meta.json``.
        log_path: Optional ``.jsonl`` destination to save per-epoch learning curves.

    Returns:
        :class:`SupervisedTrainingResult` containing trained model, curves, and
        held-out test evaluation metrics.

    Raises:
        InvalidConfigError: If `dataset` or `config` is invalid or `model_name` is
            a reinforcement-learning model (e.g. ``DQNNet``).
    """
    if not isinstance(dataset, SupervisedDataset):
        raise InvalidConfigError(
            f"dataset must be a SupervisedDataset, got {type(dataset).__name__}."
        )
    if not isinstance(config, SupervisedTrainConfig):
        raise InvalidConfigError(
            f"config must be a SupervisedTrainConfig, got {type(config).__name__}."
        )

    seed_everything(config.seed)
    device = torch.device(config.device)
    model = create_model(
        config.model_name,
        board_config=dataset.board_config,
        conv_channels=config.conv_channels,
        hidden_dim=config.hidden_dim,
    ).to(device)

    if not isinstance(model, (FastPolicyNet, StrongPolicyNet, ValueNet)):
        raise InvalidConfigError(
            f"Model '{model.__class__.__name__}' is not a supervised model. "
            "Use FastPolicyNet, StrongPolicyNet, or ValueNet."
        )

    is_policy = isinstance(model, (FastPolicyNet, StrongPolicyNet))
    train_loader = _build_split_loader(
        dataset.train, batch_size=config.batch_size, shuffle=True, seed=config.seed
    )
    val_loader = _build_split_loader(
        dataset.val, batch_size=config.batch_size, shuffle=False, seed=config.seed
    )
    test_loader = _build_split_loader(
        dataset.test, batch_size=config.batch_size, shuffle=False, seed=config.seed
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    history: list[EpochMetrics] = []
    best_val_loss = math.inf
    best_val_acc: float | None = None
    best_epoch = 1
    best_state_dict = copy.deepcopy(model.state_dict())

    train_start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for x_batch, y_action, y_value, legal_masks in train_loader:
            x_dev = x_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            batch_n = int(x_dev.shape[0])

            if is_policy:
                y_dev = y_action.to(device)
                mask_dev = legal_masks.to(device)
                logits = model(x_dev)
                loss = ce_loss_fn(logits, y_dev)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    masked_logits = apply_action_mask(logits, mask_dev, allow_empty=False)
                    preds = masked_logits.argmax(dim=-1)
                    running_correct += int((preds == y_dev).sum().item())
            else:
                z_dev = y_value.to(device)
                preds = model(x_dev)
                loss = mse_loss_fn(preds, z_dev)
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item()) * batch_n
            running_samples += batch_n

        train_loss = running_loss / float(max(1, running_samples))
        train_acc: float | None = (
            float(running_correct) / float(max(1, running_samples)) if is_policy else None
        )

        if is_policy:
            val_loss, val_acc_val = _evaluate_policy_split(
                model, val_loader, ce_loss_fn, device
            )
            val_acc: float | None = val_acc_val
        else:
            val_loss = _evaluate_value_split(model, val_loader, mse_loss_fn, device)
            val_acc = None

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                learning_rate=config.learning_rate,
                elapsed_sec=time.perf_counter() - epoch_start,
            )
        )

    training_time_sec = time.perf_counter() - train_start

    # Restore the best validation checkpoint before evaluating on test split.
    model.load_state_dict(best_state_dict)
    model.eval()
    if is_policy:
        test_loss, test_acc_val = _evaluate_policy_split(
            model, test_loader, ce_loss_fn, device
        )
        test_acc: float | None = test_acc_val
    else:
        test_loss = _evaluate_value_split(model, test_loader, mse_loss_fn, device)
        test_acc = None

    saved_ckpt: Path | None = None
    saved_meta_path: Path | None = None
    saved_meta: CheckpointMetadata | None = None
    if checkpoint_path is not None:
        ckpt_meta = CheckpointMetadata(
            agent_class=model.__class__.__name__,
            board_config=dataset.board_config,
            seed=config.seed,
            training_type="supervised",
            epochs_trained=config.epochs,
            dataset_hash=dataset.dataset_hash,
            split_policy=dataset.split_policy,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_acc,
            training_time_sec=training_time_sec,
            extra={
                "best_epoch": best_epoch,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "raw_samples": dataset.raw_samples,
                "deduplicated_samples": dataset.deduplicated_samples,
                "duplicate_ratio": dataset.duplicate_ratio,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
            },
        )
        saved_ckpt, saved_meta_path, saved_meta = save_torch_checkpoint(
            model, checkpoint_path, ckpt_meta
        )

    resolved_log: Path | None = Path(log_path) if log_path is not None else None
    result = SupervisedTrainingResult(
        model_name=model.__class__.__name__,
        board_config=dataset.board_config,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_acc,
        test_loss=test_loss,
        test_accuracy=test_acc,
        training_time_sec=training_time_sec,
        model=model,
        history=history,
        checkpoint_path=saved_ckpt,
        meta_path=saved_meta_path,
        log_path=resolved_log,
        metadata=saved_meta,
    )
    if resolved_log is not None:
        _write_learning_curve_jsonl(resolved_log, result, dataset, config)

    logger.info(
        "Trained %s (%s): best_epoch=%d val_loss=%.4f test_loss=%.4f test_acc=%s",
        result.model_name,
        result.board_config.name,
        result.best_epoch,
        result.best_val_loss,
        result.test_loss,
        f"{result.test_accuracy:.4f}" if result.test_accuracy is not None else "n/a",
    )
    return result


def train_policy_network(
    dataset: SupervisedDataset,
    model_name: str = "StrongPolicyNet",
    config: SupervisedTrainConfig | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> SupervisedTrainingResult:
    """Convenience wrapper to train ``FastPolicyNet`` or ``StrongPolicyNet``."""
    if config is None:
        resolved = SupervisedTrainConfig(model_name=model_name)
    else:
        resolved = SupervisedTrainConfig(
            model_name=model_name,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            conv_channels=config.conv_channels,
            hidden_dim=config.hidden_dim,
            seed=config.seed,
            device=config.device,
        )
    return train_supervised_model(
        dataset,
        resolved,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
    )


def train_value_network(
    dataset: SupervisedDataset,
    config: SupervisedTrainConfig | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> SupervisedTrainingResult:
    """Convenience wrapper to train :class:`ValueNet` on outcome targets."""
    if config is None:
        resolved = SupervisedTrainConfig(model_name="ValueNet")
    else:
        resolved = SupervisedTrainConfig(
            model_name="ValueNet",
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            conv_channels=config.conv_channels,
            hidden_dim=config.hidden_dim,
            seed=config.seed,
            device=config.device,
        )
    return train_supervised_model(
        dataset,
        resolved,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
    )


def train_all_supervised_models(
    dataset: SupervisedDataset,
    *,
    base_config: SupervisedTrainConfig | None = None,
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    model_names: Sequence[str] = ("FastPolicyNet", "StrongPolicyNet", "ValueNet"),
) -> dict[str, SupervisedTrainingResult]:
    """Train all configured supervised architectures for a board configuration.

    Args:
        dataset: Disjoint :class:`SupervisedDataset` for the target board.
        base_config: Optional base training hyperparameters.
        checkpoint_dir: Optional directory for ``<model>_<board>.pt`` artifacts.
        log_dir: Optional directory for ``train_<model>_<board>.jsonl`` curves.
        model_names: Sequence of model names to train.

    Returns:
        Dictionary mapping canonical model class names to training results.
    """
    cfg_template = base_config if base_config is not None else SupervisedTrainConfig()
    board_tag = dataset.board_config.name
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    log_root = Path(log_dir) if log_dir is not None else None

    results: dict[str, SupervisedTrainingResult] = {}
    for model_name in model_names:
        per_model_cfg = SupervisedTrainConfig(
            model_name=model_name,
            epochs=cfg_template.epochs,
            batch_size=cfg_template.batch_size,
            learning_rate=cfg_template.learning_rate,
            weight_decay=cfg_template.weight_decay,
            conv_channels=cfg_template.conv_channels,
            hidden_dim=cfg_template.hidden_dim,
            seed=cfg_template.seed,
            device=cfg_template.device,
        )
        slug = model_name.strip().lower()
        ckpt_file = (
            ckpt_root / f"{slug}_{board_tag}.pt" if ckpt_root is not None else None
        )
        log_file = (
            log_root / f"train_{slug}_{board_tag}.jsonl" if log_root is not None else None
        )
        res = train_supervised_model(
            dataset,
            per_model_cfg,
            checkpoint_path=ckpt_file,
            log_path=log_file,
        )
        results[res.model_name] = res
    return results
