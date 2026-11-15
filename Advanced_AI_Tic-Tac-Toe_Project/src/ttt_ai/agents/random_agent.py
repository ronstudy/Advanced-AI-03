"""Level 1 baseline agent: uniform random move selection with isolated RNG."""

from __future__ import annotations

from ttt_ai.agents.base import AgentDecision, BaseAgent, uniform_action_scores
from ttt_ai.engine.game import ExtendedTicTacToe


class RandomAgent(BaseAgent):
    """Level 1 AI agent selecting uniformly at random from `game.legal_actions()`.

    Uses an isolated :class:`numpy.random.Generator` owned by :class:`BaseAgent`
    so match sequences remain reproducible per seed without interfering with
    global NumPy/Python RNG state.
    """

    def __init__(
        self,
        name: str | None = "RandomAgent",
        seed: int | None = None,
        time_budget_ms: float | None = None,
    ) -> None:
        """Initialize a Level 1 random baseline agent.

        Args:
            name: Human-readable display name. Defaults to `"RandomAgent"`.
            seed: Optional non-negative seed for the agent's local RNG.
            time_budget_ms: Optional per-move compute budget in milliseconds.

        Raises:
            InvalidConfigError: If `name`, `seed`, or `time_budget_ms` is invalid.
        """
        super().__init__(name=name, level=1, seed=seed, time_budget_ms=time_budget_ms)

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        """Sample one legal action uniformly at random.

        Args:
            game: Active, non-terminal :class:`ExtendedTicTacToe` instance.

        Returns:
            :class:`AgentDecision` containing the sampled move and uniform scores.
        """
        legal_actions = game.legal_actions()
        scores = uniform_action_scores(legal_actions)
        chosen_idx = int(self.rng.integers(0, len(legal_actions)))
        chosen_action = legal_actions[chosen_idx]

        return AgentDecision(
            action=chosen_action,
            nodes_visited=1,
            simulations=0,
            depth_reached=0,
            value_estimate=0.0,
            action_scores=scores,
        )
