"""Unit tests for Task T1.1: configuration loading, validation, and seeding."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from ttt_ai.config import BoardConfig, GameConfig, load_game_config, seed_everything
from ttt_ai.engine.exceptions import InvalidConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


@pytest.mark.parametrize(
    ("filename", "expected_rows", "expected_cols", "expected_k", "expected_sims", "expected_ms"),
    [
        ("game_4x4_k4.yaml", 4, 4, 4, 200, 50.0),
        ("game_4x4_k3.yaml", 4, 4, 3, 200, 50.0),
        ("game_3x3_k3.yaml", 3, 3, 3, 150, 50.0),
        ("game_5x5_k4.yaml", 5, 5, 4, 400, 100.0),
    ],
)
def test_load_standard_game_configs(
    filename: str,
    expected_rows: int,
    expected_cols: int,
    expected_k: int,
    expected_sims: int,
    expected_ms: float,
) -> None:
    """Verify all four standard game YAML files load cleanly with expected metadata."""
    cfg = load_game_config(CONFIGS_DIR / filename)

    assert isinstance(cfg, GameConfig)
    assert cfg.board.rows == expected_rows
    assert cfg.board.cols == expected_cols
    assert cfg.board.win_k == expected_k
    assert cfg.board.num_cells == expected_rows * expected_cols
    assert cfg.board.shape == (expected_rows, expected_cols)
    assert cfg.board.name == f"{expected_rows}x{expected_cols}_k{expected_k}"
    assert cfg.seed == 42
    assert cfg.default_n_simulations == expected_sims
    assert cfg.default_time_budget_ms == pytest.approx(expected_ms)


def test_board_config_round_trip() -> None:
    """Ensure BoardConfig and GameConfig serialize and reconstruct losslessly."""
    board = BoardConfig(rows=4, cols=5, win_k=4)
    assert board.num_cells == 20
    assert board.shape == (4, 5)
    assert BoardConfig.from_dict(board.to_dict()) == board

    game_cfg = GameConfig(
        name="custom_4x5",
        description="Rectangular board test",
        board=board,
        seed=7,
        default_time_budget_ms=25,
        default_n_simulations=100,
    )
    assert isinstance(game_cfg.default_time_budget_ms, float)
    assert GameConfig.from_dict(game_cfg.to_dict()) == game_cfg


@pytest.mark.parametrize(
    ("rows", "cols", "win_k"),
    [
        (2, 3, 3),
        (3, 2, 3),
        (4, 4, 2),
        (4, 4, 5),
        (3, 5, 4),
        (True, 4, 3),
        (4, 4, 3.0),
        ("4", 4, 4),
    ],
)
def test_board_config_rejects_invalid_dimensions(rows: object, cols: object, win_k: object) -> None:
    """Reject out-of-range, boolean, or non-integer board dimensions."""
    with pytest.raises(InvalidConfigError):
        BoardConfig(rows=rows, cols=cols, win_k=win_k)  # type: ignore[arg-type]


def test_load_game_config_missing_file_raises(tmp_path: Path) -> None:
    """Raise FileNotFoundError when the YAML configuration file does not exist."""
    missing_path = tmp_path / "non_existent.yaml"
    with pytest.raises(FileNotFoundError):
        load_game_config(missing_path)


def test_load_game_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    """Raise InvalidConfigError when YAML root is a list or scalar instead of a mapping."""
    bad_yaml = tmp_path / "list_root.yaml"
    bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(InvalidConfigError):
        load_game_config(bad_yaml)


@pytest.mark.parametrize(
    "yaml_content",
    [
        "name: bad_seed\nboard:\n  rows: 4\n  cols: 4\n  win_k: 4\nseed: -1\n",
        (
            "name: bad_budget\nboard:\n  rows: 4\n  cols: 4\n  win_k: 4\n"
            "default_time_budget_ms: 0\n"
        ),
        (
            "name: bad_sims\nboard:\n  rows: 4\n  cols: 4\n  win_k: 4\n"
            "default_n_simulations: 0\n"
        ),
        "name: ''\nboard:\n  rows: 4\n  cols: 4\n  win_k: 4\n",
    ],
)
def test_load_game_config_rejects_invalid_values(tmp_path: Path, yaml_content: str) -> None:
    """Reject negative seeds, non-positive budgets, or blank config names."""
    cfg_path = tmp_path / "invalid.yaml"
    cfg_path.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(InvalidConfigError):
        load_game_config(cfg_path)


def test_seed_everything_reproducibility() -> None:
    """Verify identical sequences across random, numpy, torch, and returned Generator."""
    rng_a = seed_everything(1234)
    py_a = [random.random() for _ in range(5)]
    np_a = np.random.rand(5).tolist()
    gen_a = rng_a.random(5).tolist()
    torch_a = torch.rand(5).tolist()
    assert os.environ["PYTHONHASHSEED"] == "1234"

    rng_b = seed_everything(1234)
    py_b = [random.random() for _ in range(5)]
    np_b = np.random.rand(5).tolist()
    gen_b = rng_b.random(5).tolist()
    torch_b = torch.rand(5).tolist()

    assert py_a == py_b
    assert np_a == np_b
    assert gen_a == gen_b
    assert torch_a == torch_b


@pytest.mark.parametrize("bad_seed", [-1, True, 3.14, "42"])
def test_seed_everything_rejects_invalid_seed(bad_seed: object) -> None:
    """Reject negative, boolean, or non-integer seeds with InvalidConfigError."""
    with pytest.raises(InvalidConfigError):
        seed_everything(bad_seed)  # type: ignore[arg-type]
