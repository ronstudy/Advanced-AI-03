"""AI agent hierarchy (Level 1 through Level 7) for Extended Tic-Tac-Toe."""

from __future__ import annotations

from ttt_ai.agents.base import (
    AgentDecision,
    BaseAgent,
    SearchTimeoutError,
    uniform_action_scores,
)
from ttt_ai.agents.hybrid_mcts_agent import HybridMCTSAgent
from ttt_ai.agents.mcts_agent import MCTSAgent
from ttt_ai.agents.minimax_agent import AlphaBetaAgent, MinimaxAgent, evaluate_board
from ttt_ai.agents.nn_models import (
    DEFAULT_CONV_CHANNELS,
    DEFAULT_HIDDEN_DIM,
    MASK_FILL_VALUE,
    BaseBoardNet,
    DQNNet,
    FastPolicyNet,
    StrongPolicyNet,
    ValueNet,
    action_masking,
    apply_action_mask,
    board_to_tensor,
    build_legal_action_mask,
    create_model,
    masked_argmax,
    masked_softmax,
)
from ttt_ai.agents.policy_agent import PolicyNetworkAgent, ValueGreedyAgent
from ttt_ai.agents.random_agent import RandomAgent
from ttt_ai.agents.rl_agent import DQNAgent, TabularQAgent, TabularQLearningAgent
from ttt_ai.agents.rule_based_agent import RuleBasedAgent, RuleTier

__all__: list[str] = [
    "DEFAULT_CONV_CHANNELS",
    "DEFAULT_HIDDEN_DIM",
    "MASK_FILL_VALUE",
    "AgentDecision",
    "AlphaBetaAgent",
    "BaseAgent",
    "BaseBoardNet",
    "DQNAgent",
    "DQNNet",
    "FastPolicyNet",
    "HybridMCTSAgent",
    "MCTSAgent",
    "MinimaxAgent",
    "PolicyNetworkAgent",
    "RandomAgent",
    "RuleBasedAgent",
    "RuleTier",
    "SearchTimeoutError",
    "StrongPolicyNet",
    "TabularQAgent",
    "TabularQLearningAgent",
    "ValueGreedyAgent",
    "ValueNet",
    "action_masking",
    "apply_action_mask",
    "board_to_tensor",
    "build_legal_action_mask",
    "create_model",
    "evaluate_board",
    "masked_argmax",
    "masked_softmax",
    "uniform_action_scores",
]
