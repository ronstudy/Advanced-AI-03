"""Level 3 adversarial search agents: Minimax, Alpha-Beta pruning, and Iterative Deepening."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ttt_ai.agents.base import AgentDecision, BaseAgent, SearchTimeoutError
from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.engine.rules import (
    PLAYER_O,
    PLAYER_X,
    Player,
    positional_weights,
    precompute_win_windows,
)

_TERMINAL_SCORE: float = 10.0
_PLY_PENALTY: float = 0.01
_TIE_EPS: float = 1e-12
_TIE_WINDOW_EPS: float = 1e-9
_DEADLINE_CHECK_MASK: int = 255  # Check perf_counter() every 256 visited nodes.


def _is_strict_int(value: Any) -> bool:
    """Return True if `value` is a Python/NumPy integer and not a boolean."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


@lru_cache(maxsize=64)
def _flat_positional_weights(rows: int, cols: int, win_k: int) -> tuple[int, ...]:
    """Return cached flattened positional weights for move ordering."""
    weights = positional_weights(rows, cols, win_k, normalize=False).reshape(-1)
    return tuple(int(w) for w in weights)


def evaluate_board(
    game: ExtendedTicTacToe,
    perspective_player: int | Player | None = None,
) -> float:
    """Evaluate a game state heuristically from `perspective_player`'s viewpoint.

    For terminal states, returns `+10.0` (win), `-10.0` (loss), or `0.0` (draw).
    For non-terminal states, sums `10^(c - 1)` across every open `win_k` window
    containing `c >= 1` marks of one side and zero marks of the opponent, then
    compresses the raw difference via `math.tanh` strictly into `(-1.0, +1.0)`
    so heuristic leaf scores never overlap with terminal payoffs (`±10.0`).

    Args:
        game: :class:`ExtendedTicTacToe` state to evaluate via public Game APIs.
        perspective_player: Player viewpoint (`+1` for `X`, `-1` for `O`).
            Defaults to `game.current_player` when `None`.

    Returns:
        Zero-sum score where positive values favor `perspective_player`.

    Raises:
        TypeError: If `game` is not an :class:`ExtendedTicTacToe` instance.
        InvalidConfigError: If `perspective_player` is not `None`, `+1`, or `-1`.
    """
    if not isinstance(game, ExtendedTicTacToe):
        raise TypeError(
            f"game must be an ExtendedTicTacToe instance, got {type(game).__name__}."
        )

    if perspective_player is None:
        target_player = int(game.current_player)
    else:
        if not _is_strict_int(perspective_player) or int(perspective_player) not in (
            PLAYER_X,
            PLAYER_O,
        ):
            raise InvalidConfigError(
                f"perspective_player must be {PLAYER_X} (X), {PLAYER_O} (O), or None, "
                f"got {perspective_player!r}."
            )
        target_player = int(perspective_player)

    if game.is_terminal():
        payoff = game.result(target_player)
        if payoff is None or payoff == 0.0:
            return 0.0
        return _TERMINAL_SCORE if payoff > 0.0 else -_TERMINAL_SCORE

    flat_board = [int(v) for v in game.board.reshape(-1)]
    windows = precompute_win_windows(game.rows, game.cols, game.win_k)
    opp_player = -target_player

    raw_score = 0.0
    for window in windows:
        self_count = 0
        opp_count = 0
        for idx in window:
            cell = flat_board[idx]
            if cell == target_player:
                self_count += 1
            elif cell == opp_player:
                opp_count += 1

        if self_count > 0 and opp_count == 0:
            raw_score += float(10 ** (self_count - 1))
        elif opp_count > 0 and self_count == 0:
            raw_score -= float(10 ** (opp_count - 1))

    if raw_score == 0.0:
        return 0.0

    scale = float(10 ** max(1, game.win_k - 1))
    return math.tanh(raw_score / scale)


@dataclass(slots=True)
class _SearchContext:
    """Mutable per-move search telemetry shared across recursive Negamax calls."""

    deadline: float | None
    nodes_visited: int = 0


class MinimaxAgent(BaseAgent):
    """Level 3 adversarial search agent supporting Minimax and Alpha-Beta pruning.

    Explores the game tree on a single `game.clone()` via O(1) `step()`/`undo()`
    transitions, supports configurable depth limits, Alpha-Beta pruning,
    positional/PV move ordering, and Iterative Deepening under `time_budget_ms`.
    """

    def __init__(
        self,
        name: str | None = "MinimaxAgent",
        max_depth: int = 4,
        use_alpha_beta: bool = True,
        use_move_ordering: bool = True,
        iterative_deepening: bool | None = None,
        seed: int | None = None,
        time_budget_ms: float | None = None,
    ) -> None:
        """Initialize a Level 3 Minimax / Alpha-Beta search agent.

        Args:
            name: Human-readable agent name. Defaults to `"MinimaxAgent"`.
            max_depth: Maximum search ply depth (`max_depth >= 1`).
            use_alpha_beta: Enable fail-soft Alpha-Beta pruning (toggleable for Ablation 1).
            use_move_ordering: Order moves by previous PV and `positional_weights`
                to maximize Alpha-Beta cutoffs (toggleable for Ablation 1).
            iterative_deepening: Whether to search depths `1..max_depth` iteratively.
                When `None`, automatically enables whenever `time_budget_ms` is set.
            seed: Optional RNG seed (kept for unified agent configuration).
            time_budget_ms: Optional per-move compute budget in milliseconds.

        Raises:
            InvalidConfigError: If any parameter fails type or range validation.
        """
        if not _is_strict_int(max_depth) or int(max_depth) < 1:
            raise InvalidConfigError(
                f"max_depth must be an integer >= 1, got {max_depth!r}."
            )
        if not isinstance(use_alpha_beta, bool):
            raise InvalidConfigError(
                f"use_alpha_beta must be a bool, got {type(use_alpha_beta).__name__}."
            )
        if not isinstance(use_move_ordering, bool):
            raise InvalidConfigError(
                f"use_move_ordering must be a bool, got {type(use_move_ordering).__name__}."
            )
        if iterative_deepening is not None and not isinstance(iterative_deepening, bool):
            raise InvalidConfigError(
                "iterative_deepening must be a bool or None, "
                f"got {type(iterative_deepening).__name__}."
            )

        super().__init__(name=name, level=3, seed=seed, time_budget_ms=time_budget_ms)
        self._max_depth: int = int(max_depth)
        self._use_alpha_beta: bool = use_alpha_beta
        self._use_move_ordering: bool = use_move_ordering
        self._iterative_deepening_override: bool | None = iterative_deepening

    @property
    def max_depth(self) -> int:
        """Configured maximum search depth in plies."""
        return self._max_depth

    @property
    def use_alpha_beta(self) -> bool:
        """Whether Alpha-Beta pruning is enabled."""
        return self._use_alpha_beta

    @property
    def use_move_ordering(self) -> bool:
        """Whether PV + positional move ordering is enabled."""
        return self._use_move_ordering

    @property
    def iterative_deepening(self) -> bool:
        """Whether Iterative Deepening is currently active for move selection."""
        if self._iterative_deepening_override is not None:
            return self._iterative_deepening_override
        return self.time_budget_ms is not None

    def get_config(self) -> dict[str, Any]:
        """Return serializable agent metadata and search hyperparameters."""
        config = super().get_config()
        config.update(
            {
                "max_depth": self._max_depth,
                "use_alpha_beta": self._use_alpha_beta,
                "use_move_ordering": self._use_move_ordering,
                "iterative_deepening": self.iterative_deepening,
            }
        )
        return config

    def _order_moves(
        self,
        legal_actions: list[int],
        rows: int,
        cols: int,
        win_k: int,
        pv_action: int | None = None,
    ) -> list[int]:
        """Order candidate moves to promote early Alpha-Beta cutoffs without mutating ties."""
        if not self._use_move_ordering:
            return legal_actions

        pos_weights = _flat_positional_weights(rows, cols, win_k)
        return sorted(
            legal_actions,
            key=lambda action: (
                0 if action == pv_action else 1,
                -pos_weights[action],
                action,
            ),
        )

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        """Execute Minimax / Alpha-Beta search on an isolated game clone.

        Args:
            game: Active, non-terminal :class:`ExtendedTicTacToe` instance.

        Returns:
            :class:`AgentDecision` with optimal move, root `action_scores`,
            `value_estimate`, `depth_reached`, and total `nodes_visited`.
        """
        legal_actions = game.legal_actions()
        rows, cols, win_k = game.rows, game.cols, game.win_k
        effective_max_depth = min(self._max_depth, len(legal_actions))
        root_player = game.current_player

        deadline = self._compute_deadline()
        ctx = _SearchContext(deadline=deadline, nodes_visited=0)
        sim = game.clone()

        if self.iterative_deepening:
            depth_schedule = range(1, effective_max_depth + 1)
        else:
            depth_schedule = range(effective_max_depth, effective_max_depth + 1)

        best_action: int | None = None
        best_value: float = 0.0
        best_scores: dict[int, float] = {}
        completed_depth: int = 0

        for depth in depth_schedule:
            try:
                if depth > 1:
                    self._check_timeout(ctx.deadline)
                cand_action, cand_value, cand_scores = self._search_root(
                    sim=sim,
                    legal_actions=legal_actions,
                    depth=depth,
                    root_player=root_player,
                    pv_action=best_action,
                    ctx=ctx,
                )
            except SearchTimeoutError:
                break

            best_action = cand_action
            best_value = cand_value
            best_scores = cand_scores
            completed_depth = depth

            # Once Iterative Deepening reaches a proven win at the shallowest
            # possible ply (`depth`), no shorter winning path can exist deeper.
            if (
                self.iterative_deepening
                and best_value >= _TERMINAL_SCORE - (_PLY_PENALTY * float(depth)) - _TIE_EPS
            ):
                break

        if best_action is None:
            ordered_fallback = self._order_moves(legal_actions, rows, cols, win_k)
            best_action = ordered_fallback[0]
            best_value = 0.0
            best_scores = {action: 0.0 for action in legal_actions}
            completed_depth = 0

        return AgentDecision(
            action=best_action,
            nodes_visited=max(1, ctx.nodes_visited),
            simulations=0,
            depth_reached=completed_depth,
            value_estimate=best_value,
            action_scores=best_scores,
        )

    def _search_root(
        self,
        sim: ExtendedTicTacToe,
        legal_actions: list[int],
        depth: int,
        root_player: Player,
        pv_action: int | None,
        ctx: _SearchContext,
    ) -> tuple[int, float, dict[int, float]]:
        """Evaluate every root move at fixed `depth`, breaking ties by lowest cell index."""
        ctx.nodes_visited += 1
        ordered_moves = self._order_moves(
            legal_actions=legal_actions,
            rows=sim.rows,
            cols=sim.cols,
            win_k=sim.win_k,
            pv_action=pv_action,
        )

        best_action: int | None = None
        best_value: float = -math.inf
        root_scores: dict[int, float] = {}

        for action in ordered_moves:
            # Because ties are broken deterministically in favor of smaller cell indices,
            # a candidate with `action < best_action` needs an exact tie check (`alpha - eps`)
            # so fail-soft pruning never hides a genuine tie behind an upper bound <= alpha.
            if not self._use_alpha_beta or best_action is None:
                alpha_bound = -math.inf
            elif action < best_action:
                alpha_bound = best_value - _TIE_WINDOW_EPS
            else:
                alpha_bound = best_value

            sim.step(action)
            try:
                score = -self._negamax(
                    sim=sim,
                    depth=depth - 1,
                    ply=1,
                    player=root_player.opponent,
                    alpha=-math.inf,
                    beta=-alpha_bound,
                    ctx=ctx,
                )
            finally:
                # Always unwind sim even if SearchTimeoutError aborts this branch mid-search.
                sim.undo()

            root_scores[action] = score
            if best_action is None or score > best_value + _TIE_EPS:
                best_value = score
                best_action = action
            elif abs(score - best_value) <= _TIE_EPS and action < best_action:
                best_action = action

        assert best_action is not None
        return best_action, best_value, root_scores

    def _negamax(
        self,
        sim: ExtendedTicTacToe,
        depth: int,
        ply: int,
        player: Player,
        alpha: float,
        beta: float,
        ctx: _SearchContext,
    ) -> float:
        """Recursive Negamax with fail-soft Alpha-Beta pruning and periodic deadline checks."""
        ctx.nodes_visited += 1
        if ctx.deadline is not None and (ctx.nodes_visited & _DEADLINE_CHECK_MASK) == 0:
            self._check_timeout(ctx.deadline)

        # Evaluate terminal states from `player`'s perspective (explicit ply player)
        # because ExtendedTicTacToe preserves `current_player` as the winner on terminal moves.
        if sim.is_terminal():
            payoff = sim.result(int(player))
            if payoff is None or payoff == 0.0:
                return 0.0
            terminal_val = _TERMINAL_SCORE - (float(ply) * _PLY_PENALTY)
            return terminal_val if payoff > 0.0 else -terminal_val

        if depth <= 0:
            return evaluate_board(sim, int(player))

        legal = sim.legal_actions()
        ordered_moves = self._order_moves(
            legal_actions=legal,
            rows=sim.rows,
            cols=sim.cols,
            win_k=sim.win_k,
        )

        best_score = -math.inf
        local_alpha = alpha if self._use_alpha_beta else -math.inf
        local_beta = beta if self._use_alpha_beta else math.inf

        for action in ordered_moves:
            sim.step(action)
            try:
                score = -self._negamax(
                    sim=sim,
                    depth=depth - 1,
                    ply=ply + 1,
                    player=player.opponent,
                    alpha=-local_beta,
                    beta=-local_alpha,
                    ctx=ctx,
                )
            finally:
                sim.undo()

            if score > best_score:
                best_score = score
            if self._use_alpha_beta:
                if score > local_alpha:
                    local_alpha = score
                if local_alpha >= local_beta:
                    break

        return best_score


class AlphaBetaAgent(MinimaxAgent):
    """Level 3 adversarial search agent preconfigured with Alpha-Beta pruning."""

    def __init__(
        self,
        name: str | None = "AlphaBetaAgent",
        max_depth: int = 4,
        use_alpha_beta: bool = True,
        use_move_ordering: bool = True,
        iterative_deepening: bool | None = None,
        seed: int | None = None,
        time_budget_ms: float | None = None,
    ) -> None:
        """Initialize an Alpha-Beta search agent (Level 3).

        Args:
            name: Human-readable agent name. Defaults to `"AlphaBetaAgent"`.
            max_depth: Maximum search ply depth (`max_depth >= 1`).
            use_alpha_beta: Enable Alpha-Beta pruning (defaults to `True`).
            use_move_ordering: Order candidate moves by PV and positional weights.
            iterative_deepening: Optional explicit override for Iterative Deepening.
            seed: Optional RNG seed.
            time_budget_ms: Optional per-move compute budget in milliseconds.
        """
        super().__init__(
            name=name,
            max_depth=max_depth,
            use_alpha_beta=use_alpha_beta,
            use_move_ordering=use_move_ordering,
            iterative_deepening=iterative_deepening,
            seed=seed,
            time_budget_ms=time_budget_ms,
        )
