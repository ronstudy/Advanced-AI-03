"""Base agent contract, decision telemetry schema, and search timeout exception.

All Level 1 through Level 7 agents inherit from :class:`BaseAgent` and interact
with :class:`ttt_ai.engine.game.ExtendedTicTacToe` strictly through its public
Game API without mutating the caller's game state.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError, TicTacToeError
from ttt_ai.engine.game import ExtendedTicTacToe


class SearchTimeoutError(TicTacToeError):
    """Raised internally when a search agent exceeds its per-move deadline."""


def _is_strict_int(value: Any) -> bool:
    """Return True if `value` is a Python/NumPy integer and not a boolean."""
    # Reject booleans explicitly because `bool` subclasses `int` in Python.
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_finite_real(value: Any) -> bool:
    """Return True if `value` is a finite non-boolean numeric scalar."""
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def uniform_action_scores(legal_actions: Sequence[int]) -> dict[int, float]:
    """Build a uniform probability distribution over a sequence of legal actions.

    Args:
        legal_actions: Sequence of valid flat board indices.

    Returns:
        Dictionary mapping each action index to `1.0 / len(legal_actions)`, or an
        empty dictionary when `legal_actions` is empty.

    Raises:
        IllegalMoveError: If any entry in `legal_actions` is not a non-negative integer.
    """
    if not legal_actions:
        return {}

    validated: list[int] = []
    for action in legal_actions:
        if not _is_strict_int(action) or int(action) < 0:
            raise IllegalMoveError(
                f"Legal actions must be non-negative integers, got {action!r}."
            )
        validated.append(int(action))

    prob = 1.0 / float(len(validated))
    return {action: prob for action in validated}


def _normalize_score_map(
    raw_map: Mapping[int, float],
    field_name: str,
) -> dict[int, float]:
    """Validate and copy an `{action: score}` mapping to plain `(int, float)` pairs."""
    if not isinstance(raw_map, Mapping):
        raise InvalidConfigError(
            f"{field_name} must be a mapping of {{action: score}}, "
            f"got {type(raw_map).__name__}."
        )

    normalized: dict[int, float] = {}
    for action, score in raw_map.items():
        if not _is_strict_int(action) or int(action) < 0:
            raise IllegalMoveError(
                f"{field_name} action keys must be non-negative integers, got {action!r}."
            )
        if not _is_finite_real(score):
            raise InvalidConfigError(
                f"{field_name}[{action!r}] must be a finite numeric value, got {score!r}."
            )
        normalized[int(action)] = float(score)
    return normalized


@dataclass(slots=True)
class AgentDecision:
    """Structured output and explainability telemetry returned by an AI agent.

    Attributes:
        action: Chosen flat row-major cell index in `[0, M * N - 1]`.
        latency_ms: Total wall-clock time spent inside `select_action()` in ms.
        nodes_visited: Number of search tree nodes evaluated (Minimax/Alpha-Beta).
        simulations: Number of playout/PUCT simulations executed (MCTS/HybridMCTS).
        depth_reached: Maximum completed search depth (Iterative Deepening).
        value_estimate: Root state value estimate from search or Value/Q network.
        action_scores: Mapping `{action: score}` over candidate moves for Rich CLI
            heatmaps and Top-K diagnostics.
        prior_probs: Mapping `{action: P(s, a)}` from Policy Network prior (Level 7).
    """

    action: int
    latency_ms: float = 0.0
    nodes_visited: int = 0
    simulations: int = 0
    depth_reached: int = 0
    value_estimate: float | None = None
    action_scores: dict[int, float] = field(default_factory=dict)
    prior_probs: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate all telemetry fields and normalize NumPy scalars to Python types.

        Raises:
            IllegalMoveError: If `action` or any score-map key is not a non-negative int.
            InvalidConfigError: If any latency, counter, or score field is invalid.
        """
        if not _is_strict_int(self.action) or int(self.action) < 0:
            raise IllegalMoveError(
                f"AgentDecision.action must be a non-negative integer, got {self.action!r}."
            )
        self.action = int(self.action)

        if not _is_finite_real(self.latency_ms) or float(self.latency_ms) < 0.0:
            raise InvalidConfigError(
                "AgentDecision.latency_ms must be a finite number >= 0.0, "
                f"got {self.latency_ms!r}."
            )
        self.latency_ms = float(self.latency_ms)

        for counter_name in ("nodes_visited", "simulations", "depth_reached"):
            raw_val = getattr(self, counter_name)
            if not _is_strict_int(raw_val) or int(raw_val) < 0:
                raise InvalidConfigError(
                    f"AgentDecision.{counter_name} must be a non-negative integer, "
                    f"got {raw_val!r}."
                )
            setattr(self, counter_name, int(raw_val))

        if self.value_estimate is not None:
            if not _is_finite_real(self.value_estimate):
                raise InvalidConfigError(
                    "AgentDecision.value_estimate must be finite or None, "
                    f"got {self.value_estimate!r}."
                )
            self.value_estimate = float(self.value_estimate)

        self.action_scores = _normalize_score_map(self.action_scores, "action_scores")
        self.prior_probs = _normalize_score_map(self.prior_probs, "prior_probs")

    def top_actions(self, k: int = 5) -> list[tuple[int, float]]:
        """Return up to `k` highest-scoring `(action, score)` pairs.

        Ties in score are broken deterministically by ascending cell index so
        Rich CLI rankings and logs remain reproducible across runs.

        Args:
            k: Maximum number of candidate actions to return (`k >= 0`).

        Returns:
            List of `(action, score)` tuples sorted by descending score.

        Raises:
            InvalidConfigError: If `k` is not a non-negative integer.
        """
        if not _is_strict_int(k) or int(k) < 0:
            raise InvalidConfigError(f"k must be a non-negative integer, got {k!r}.")
        limit = int(k)
        if limit == 0 or not self.action_scores:
            return []

        ranked = sorted(
            self.action_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:limit]

    def to_dict(self) -> dict[str, Any]:
        """Serialize this decision into a plain JSON-safe dictionary."""
        return {
            "action": self.action,
            "latency_ms": self.latency_ms,
            "nodes_visited": self.nodes_visited,
            "simulations": self.simulations,
            "depth_reached": self.depth_reached,
            "value_estimate": self.value_estimate,
            "action_scores": dict(self.action_scores),
            "prior_probs": dict(self.prior_probs),
        }


class BaseAgent(ABC):
    """Abstract base contract for all Extended Tic-Tac-Toe AI agents (Levels 1-7).

    Subclasses implement :meth:`_select_action`, while :meth:`select_action`
    enforces the shared Game API guarantees:
    1. Rejects calls on terminal games or invalid game objects.
    2. Wraps the entire decision pipeline with `time.perf_counter()`.
    3. Ensures the source `game` state is left completely unaltered.
    4. Ensures the returned action belongs to `game.legal_actions()`.
    """

    def __init__(
        self,
        name: str | None = None,
        level: int = 0,
        seed: int | None = None,
        time_budget_ms: float | None = None,
    ) -> None:
        """Initialize shared agent identity, RNG state, and compute budget.

        Args:
            name: Human-readable agent name. Defaults to the subclass name.
            level: Hierarchy level (`0` for base/custom, `1..7` for AI levels).
            seed: Optional non-negative RNG seed for isolated `np.random.Generator`.
            time_budget_ms: Optional positive per-move compute budget in ms.

        Raises:
            InvalidConfigError: If any argument fails validation.
        """
        if name is None:
            resolved_name = self.__class__.__name__
        elif isinstance(name, str) and name.strip():
            resolved_name = name.strip()
        else:
            raise InvalidConfigError(f"Agent name must be a non-empty string, got {name!r}.")

        if not _is_strict_int(level) or int(level) < 0:
            raise InvalidConfigError(
                f"Agent level must be a non-negative integer, got {level!r}."
            )

        self._name: str = resolved_name
        self._level: int = int(level)
        self._seed: int | None = None
        self._time_budget_ms: float | None = None
        self._rng: np.random.Generator = np.random.default_rng()
        self._turn_start_time: float = 0.0

        self.set_seed(seed)
        self.set_time_budget_ms(time_budget_ms)

    @property
    def name(self) -> str:
        """Display name of the agent."""
        return self._name

    @property
    def level(self) -> int:
        """Agent tier in the Level 1-7 hierarchy (`0` if unspecified)."""
        return self._level

    @property
    def seed(self) -> int | None:
        """Configured RNG seed, or `None` if unseeded."""
        return self._seed

    @property
    def time_budget_ms(self) -> float | None:
        """Per-move compute budget in milliseconds, or `None` if unbounded."""
        return self._time_budget_ms

    @property
    def rng(self) -> np.random.Generator:
        """Isolated NumPy random generator owned by this agent instance."""
        return self._rng

    def set_seed(self, seed: int | None) -> None:
        """Validate and apply a new isolated RNG seed for this agent.

        Args:
            seed: Non-negative integer seed, or `None` for non-deterministic RNG.

        Raises:
            InvalidConfigError: If `seed` is not `None` or a non-negative integer.
        """
        if seed is not None:
            if not _is_strict_int(seed) or int(seed) < 0:
                raise InvalidConfigError(
                    f"seed must be a non-negative integer or None, got {seed!r}."
                )
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)
        else:
            self._seed = None
            self._rng = np.random.default_rng()

    def set_time_budget_ms(self, time_budget_ms: float | None) -> None:
        """Validate and update the per-move compute budget in milliseconds.

        Args:
            time_budget_ms: Positive finite float budget in ms, or `None`.

        Raises:
            InvalidConfigError: If `time_budget_ms` is not `None` and not `> 0`.
        """
        if time_budget_ms is not None:
            if not _is_finite_real(time_budget_ms) or float(time_budget_ms) <= 0.0:
                raise InvalidConfigError(
                    "time_budget_ms must be a positive finite number or None, "
                    f"got {time_budget_ms!r}."
                )
            self._time_budget_ms = float(time_budget_ms)
        else:
            self._time_budget_ms = None

    def reset(self, seed: int | None = None) -> None:
        """Reset per-game agent state prior to starting a new match.

        Args:
            seed: Optional new seed to re-initialize this agent's RNG.
        """
        if seed is not None:
            self.set_seed(seed)

    def _compute_deadline(self, start_time: float | None = None) -> float | None:
        """Compute the monotonic `perf_counter()` deadline for the current move.

        When `start_time` is omitted inside `select_action()`, the deadline is
        anchored at the start of `select_action()` so preprocessing, tensor
        encoding, and search all share the same budget (RULES.md Section 3.2).

        Args:
            start_time: Optional explicit `perf_counter()` timestamp origin.

        Returns:
            Deadline timestamp in seconds, or `None` when `time_budget_ms` is unset.
        """
        if self._time_budget_ms is None:
            return None

        if start_time is not None:
            origin = float(start_time)
        elif self._turn_start_time > 0.0:
            origin = self._turn_start_time
        else:
            origin = time.perf_counter()

        return origin + (self._time_budget_ms / 1000.0)

    def _check_timeout(self, deadline: float | None) -> None:
        """Raise :class:`SearchTimeoutError` if monotonic time reached `deadline`.

        Args:
            deadline: Monotonic timestamp returned by :meth:`_compute_deadline`.

        Raises:
            SearchTimeoutError: If `deadline` is set and `perf_counter() >= deadline`.
        """
        if deadline is not None and time.perf_counter() >= deadline:
            raise SearchTimeoutError(
                f"Agent '{self._name}' reached time budget ({self._time_budget_ms} ms)."
            )

    def select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        """Select a validated legal move and measure total inference latency.

        Args:
            game: Active, non-terminal :class:`ExtendedTicTacToe` instance.

        Returns:
            Validated :class:`AgentDecision` with `latency_ms` measured over the
            full `select_action()` execution.

        Raises:
            TypeError: If `game` is not an :class:`ExtendedTicTacToe` instance.
            IllegalMoveError: If `game` is terminal, has no legal actions, is
                mutated by `_select_action()`, or returns an illegal action.
        """
        # Start timing before any preprocessing or validation so `latency_ms`
        # reflects the true end-to-end move cost (RULES.md Section 3.2).
        start_time = time.perf_counter()
        self._turn_start_time = start_time
        try:
            if not isinstance(game, ExtendedTicTacToe):
                raise TypeError(
                    f"game must be an ExtendedTicTacToe instance, got {type(game).__name__}."
                )
            if game.is_terminal():
                raise IllegalMoveError(
                    f"Agent '{self._name}' cannot select an action on a terminal game."
                )

            legal_actions = game.legal_actions()
            if not legal_actions:
                raise IllegalMoveError(
                    f"Agent '{self._name}' found no legal actions on a non-terminal game."
                )

            # Snapshot public game state to verify search routines either cloned
            # the game or balanced every step() with an exact undo().
            before_history = game.move_history
            before_player = game.current_player
            before_terminal = game.is_terminal()
            before_board = game.board

            raw_decision = self._select_action(game)
            if isinstance(raw_decision, AgentDecision):
                decision = raw_decision
            elif _is_strict_int(raw_decision):
                decision = AgentDecision(action=int(raw_decision))
            else:
                raise TypeError(
                    f"Agent '{self._name}'._select_action() must return AgentDecision "
                    f"or int, got {type(raw_decision).__name__}."
                )

            if (
                game.is_terminal() != before_terminal
                or game.current_player != before_player
                or game.move_history != before_history
                or not np.array_equal(game.board, before_board)
            ):
                raise IllegalMoveError(
                    f"Agent '{self._name}' mutated the source game state during select_action()."
                )

            if decision.action not in set(legal_actions):
                raise IllegalMoveError(
                    f"Agent '{self._name}' selected illegal action {decision.action}; "
                    f"legal actions are {legal_actions}."
                )

            decision.latency_ms = (time.perf_counter() - start_time) * 1000.0
            return decision
        finally:
            self._turn_start_time = 0.0

    @abstractmethod
    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        """Compute the next move for `game` without leaving mutations behind.

        Args:
            game: Active, non-terminal :class:`ExtendedTicTacToe` instance.

        Returns:
            An :class:`AgentDecision` specifying a valid move in `game.legal_actions()`.
        """

    def get_config(self) -> dict[str, Any]:
        """Return serializable agent metadata and hyperparameters for logging."""
        return {
            "agent_class": self.__class__.__name__,
            "name": self._name,
            "level": self._level,
            "seed": self._seed,
            "time_budget_ms": self._time_budget_ms,
        }
