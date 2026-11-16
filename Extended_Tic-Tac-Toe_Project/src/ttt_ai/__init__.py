"""Extended Tic-Tac-Toe (`M x N, K`) multi-level AI package."""

from __future__ import annotations

from ttt_ai.agents import (
    AgentDecision,
    AlphaBetaAgent,
    BaseAgent,
    BaseBoardNet,
    DQNNet,
    FastPolicyNet,
    MCTSAgent,
    MinimaxAgent,
    RandomAgent,
    RuleBasedAgent,
    RuleTier,
    SearchTimeoutError,
    StrongPolicyNet,
    ValueNet,
    evaluate_board,
)
from ttt_ai.config import BoardConfig, GameConfig, load_game_config, seed_everything
from ttt_ai.engine.exceptions import CheckpointVerificationError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.training import (
    CheckpointMetadata,
    load_pickle_checkpoint,
    load_torch_checkpoint,
    save_pickle_checkpoint,
    save_torch_checkpoint,
    verify_checkpoint,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AgentDecision",
    "AlphaBetaAgent",
    "BaseAgent",
    "BaseBoardNet",
    "BoardConfig",
    "CheckpointMetadata",
    "CheckpointVerificationError",
    "DQNNet",
    "ExtendedTicTacToe",
    "FastPolicyNet",
    "GameConfig",
    "MCTSAgent",
    "MinimaxAgent",
    "RandomAgent",
    "RuleBasedAgent",
    "RuleTier",
    "SearchTimeoutError",
    "StrongPolicyNet",
    "ValueNet",
    "evaluate_board",
    "load_game_config",
    "load_pickle_checkpoint",
    "load_torch_checkpoint",
    "save_pickle_checkpoint",
    "save_torch_checkpoint",
    "seed_everything",
    "verify_checkpoint",
]
