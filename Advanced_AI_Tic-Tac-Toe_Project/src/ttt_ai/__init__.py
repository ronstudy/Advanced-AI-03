"""Extended Tic-Tac-Toe (`M x N, K`) multi-level AI package."""

from __future__ import annotations

from ttt_ai.config import BoardConfig, GameConfig, load_game_config, seed_everything

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BoardConfig",
    "GameConfig",
    "load_game_config",
    "seed_everything",
]
