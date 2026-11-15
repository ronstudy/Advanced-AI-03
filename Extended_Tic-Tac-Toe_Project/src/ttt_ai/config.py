"""Configuration dataclasses, YAML loaders, and global reproducibility seeding."""

from __future__ import annotations

import logging
import math
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ttt_ai.engine.exceptions import InvalidConfigError

logger = logging.getLogger(__name__)


def _is_strict_int(value: Any) -> bool:
    """Return True if value is an integer and not a boolean."""
    # In Python `bool` subclasses `int`, so `isinstance(True, int)` is True.
    # Rejecting booleans avoids silent config bugs like `rows: true` -> `1`.
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class BoardConfig:
    """Validated board dimensions and win-length configuration for `M x N, K`.

    Attributes:
        rows: Number of board rows (`M >= 3`).
        cols: Number of board columns (`N >= 3`).
        win_k: Consecutive marks required to win (`3 <= win_k <= min(rows, cols)`).
    """

    rows: int = 4
    cols: int = 4
    win_k: int = 4

    def __post_init__(self) -> None:
        """Validate board dimensions and win-length invariants.

        Raises:
            InvalidConfigError: If any parameter is not an integer or violates
                `rows >= 3`, `cols >= 3`, `win_k >= 3`, or `win_k <= min(rows, cols)`.
        """
        if not (_is_strict_int(self.rows) and _is_strict_int(self.cols)):
            raise InvalidConfigError(
                f"Board dimensions must be integers, got rows={self.rows!r}, cols={self.cols!r}."
            )
        if not _is_strict_int(self.win_k):
            raise InvalidConfigError(f"win_k must be an integer, got {self.win_k!r}.")
        if self.rows < 3 or self.cols < 3:
            raise InvalidConfigError(
                f"Board dimensions must be at least 3x3, got {self.rows}x{self.cols}."
            )
        # Require `win_k <= min(rows, cols)` so horizontal, vertical, and both
        # diagonal axes can each form at least one valid winning window of length K.
        max_valid_k = min(self.rows, self.cols)
        if self.win_k < 3 or self.win_k > max_valid_k:
            raise InvalidConfigError(
                f"win_k must satisfy 3 <= win_k <= min(rows, cols) ({max_valid_k}), "
                f"got {self.win_k}."
            )

    @property
    def num_cells(self) -> int:
        """Total number of cells (`rows * cols`) on the board."""
        return self.rows * self.cols

    @property
    def shape(self) -> tuple[int, int]:
        """Board matrix shape as `(rows, cols)`."""
        return (self.rows, self.cols)

    @property
    def name(self) -> str:
        """Canonical short identifier such as `'4x4_k4'`."""
        return f"{self.rows}x{self.cols}_k{self.win_k}"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoardConfig:
        """Construct a `BoardConfig` from a dictionary mapping.

        Args:
            data: Mapping containing `rows`, `cols`, and `win_k`.

        Returns:
            A validated `BoardConfig` instance.

        Raises:
            InvalidConfigError: If `data` is not a mapping, lacks required keys,
                or contains invalid values.
        """
        if not isinstance(data, Mapping):
            raise InvalidConfigError(f"board config must be a mapping, got {type(data).__name__}.")
        required = ("rows", "cols", "win_k")
        missing = [key for key in required if key not in data]
        if missing:
            raise InvalidConfigError(f"Missing required board config keys: {missing}.")
        return cls(rows=data["rows"], cols=data["cols"], win_k=data["win_k"])

    def to_dict(self) -> dict[str, int]:
        """Serialize board configuration to a plain JSON/YAML-safe dictionary."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "win_k": self.win_k,
        }


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Top-level configuration loaded from `configs/game_*.yaml`.

    Attributes:
        name: Short configuration identifier (e.g., `'game_4x4_k4'`).
        description: Human-readable purpose of this benchmark/variant.
        board: Validated `BoardConfig` specifying `rows`, `cols`, and `win_k`.
        seed: Default non-negative random seed for reproducibility.
        default_time_budget_ms: Default per-move compute budget in milliseconds.
        default_n_simulations: Default simulation budget for tree-search agents.
    """

    name: str
    description: str
    board: BoardConfig
    seed: int = 42
    default_time_budget_ms: float = 50.0
    default_n_simulations: int = 200

    def __post_init__(self) -> None:
        """Validate all configuration fields and normalize numeric types.

        Raises:
            InvalidConfigError: If any field fails type or range checks.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidConfigError(f"GameConfig.name must be a non-empty string: {self.name!r}.")
        if not isinstance(self.description, str):
            raise InvalidConfigError(
                f"GameConfig.description must be a string, got {type(self.description).__name__}."
            )
        if not isinstance(self.board, BoardConfig):
            raise InvalidConfigError(
                f"GameConfig.board must be a BoardConfig, got {type(self.board).__name__}."
            )
        if not _is_strict_int(self.seed) or self.seed < 0:
            raise InvalidConfigError(f"seed must be a non-negative integer, got {self.seed!r}.")
        if (
            not isinstance(self.default_time_budget_ms, (int, float))
            or isinstance(self.default_time_budget_ms, bool)
            or not math.isfinite(float(self.default_time_budget_ms))
            or self.default_time_budget_ms <= 0
        ):
            raise InvalidConfigError(
                "default_time_budget_ms must be a positive finite number, "
                f"got {self.default_time_budget_ms!r}."
            )
        if not _is_strict_int(self.default_n_simulations) or self.default_n_simulations <= 0:
            raise InvalidConfigError(
                "default_n_simulations must be a positive integer, "
                f"got {self.default_n_simulations!r}."
            )
        # Normalize integer YAML values (e.g., `50`) to float for consistent typing.
        object.__setattr__(self, "default_time_budget_ms", float(self.default_time_budget_ms))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GameConfig:
        """Construct a `GameConfig` from a parsed YAML dictionary.

        Args:
            data: Root mapping parsed from a game YAML file.

        Returns:
            A validated `GameConfig` instance.

        Raises:
            InvalidConfigError: If required fields are missing or invalid.
        """
        if not isinstance(data, Mapping):
            raise InvalidConfigError(f"Game config must be a mapping, got {type(data).__name__}.")
        if "name" not in data:
            raise InvalidConfigError("Missing required field 'name' in game configuration.")
        if "board" not in data:
            raise InvalidConfigError("Missing required field 'board' in game configuration.")

        board_cfg = BoardConfig.from_dict(data["board"])
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            board=board_cfg,
            seed=data.get("seed", 42),
            default_time_budget_ms=data.get("default_time_budget_ms", 50.0),
            default_n_simulations=data.get("default_n_simulations", 200),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full game configuration to a plain dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "board": self.board.to_dict(),
            "seed": self.seed,
            "default_time_budget_ms": self.default_time_budget_ms,
            "default_n_simulations": self.default_n_simulations,
        }


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """Load a YAML file safely and verify its root element is a dictionary.

    Args:
        path: Path to the `.yaml` / `.yml` file.

    Returns:
        Parsed dictionary representation of the YAML file.

    Raises:
        FileNotFoundError: If `path` does not exist or is not a file.
        InvalidConfigError: If YAML parsing fails or the root is not a mapping.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise InvalidConfigError(f"Failed to parse YAML file '{file_path}': {exc}") from exc

    if not isinstance(parsed, dict):
        raise InvalidConfigError(
            f"YAML root in '{file_path}' must be a mapping, got {type(parsed).__name__}."
        )
    return parsed


def load_game_config(path: str | Path) -> GameConfig:
    """Load and validate a `GameConfig` from a YAML configuration file.

    Args:
        path: Path to a `configs/game_*.yaml` file.

    Returns:
        Validated `GameConfig` object.

    Raises:
        FileNotFoundError: If `path` does not exist.
        InvalidConfigError: If the YAML structure or values are invalid.
    """
    raw = load_yaml_dict(path)
    return GameConfig.from_dict(raw)


def seed_everything(seed: int, deterministic_torch: bool = True) -> np.random.Generator:
    """Synchronize RNG seeds across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Non-negative integer seed applied to all PRNG backends.
        deterministic_torch: When True, enforces deterministic cuDNN execution
            and disables cuDNN autotuning to ensure repeatable GPU/CPU runs.

    Returns:
        An isolated `np.random.Generator` seeded with `seed` for local agent use.

    Raises:
        InvalidConfigError: If `seed` is not a non-negative integer.
    """
    if not _is_strict_int(seed) or seed < 0:
        raise InvalidConfigError(f"seed must be a non-negative integer, got {seed!r}.")

    # Lock Python's hash randomization alongside all PRNG backends so dictionary/set
    # iteration order and stochastic operations remain consistent across runs.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch and hasattr(torch.backends, "cudnn"):
        # Disable cuDNN benchmark autotuning because selecting fastest kernels
        # dynamically can introduce non-deterministic floating-point reductions.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.debug(
        "Synchronized random seeds to %d (deterministic_torch=%s).",
        seed,
        deterministic_torch,
    )
    return np.random.default_rng(seed)
