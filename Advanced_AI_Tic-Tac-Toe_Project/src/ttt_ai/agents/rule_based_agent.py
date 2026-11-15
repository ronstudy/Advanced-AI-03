"""Level 2 heuristic agent: generalized Newell & Simon 6-tier priority rules."""

from __future__ import annotations

from enum import IntEnum
from functools import lru_cache
from typing import Any, Literal

from ttt_ai.agents.base import AgentDecision, BaseAgent
from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.engine.rules import EMPTY, positional_weights, precompute_win_windows

_SELF_MARK: int = 1
_OPP_MARK: int = -1

_TIER_SCORE_BANDS: dict[int, tuple[float, float]] = {
    6: (0.92, 1.00),  # IMMEDIATE_WIN
    5: (0.78, 0.90),  # IMMEDIATE_BLOCK
    4: (0.64, 0.76),  # CREATE_FORK
    3: (0.48, 0.62),  # BLOCK_FORK
    2: (0.05, 0.45),  # POSITIONAL
    1: (0.01, 0.04),  # FALLBACK
}


class RuleTier(IntEnum):
    """Strict priority hierarchy for generalized Newell & Simon rules."""

    FALLBACK = 1
    POSITIONAL = 2
    BLOCK_FORK = 3
    CREATE_FORK = 4
    IMMEDIATE_BLOCK = 5
    IMMEDIATE_WIN = 6


@lru_cache(maxsize=64)
def _cell_to_window_indices(
    rows: int,
    cols: int,
    win_k: int,
) -> tuple[tuple[int, ...], ...]:
    """Map each flat cell index to all `win_k`-line window indices passing through it."""
    windows = precompute_win_windows(rows, cols, win_k)
    mapping: list[list[int]] = [[] for _ in range(rows * cols)]
    for win_idx, window in enumerate(windows):
        for cell in window:
            mapping[cell].append(win_idx)
    return tuple(tuple(indices) for indices in mapping)


@lru_cache(maxsize=64)
def _flat_positional_weights(rows: int, cols: int, win_k: int) -> tuple[int, ...]:
    """Return cached flattened positional weights (`win_k` window counts per cell)."""
    weights = positional_weights(rows, cols, win_k, normalize=False).reshape(-1)
    return tuple(int(weight) for weight in weights)


def _fork_threats_for_move(
    flat_board: list[int],
    cell: int,
    player_mark: int,
    windows: tuple[tuple[int, ...], ...],
    cell_windows: tuple[tuple[int, ...], ...],
) -> tuple[set[int], int]:
    """Return `(distinct_threat_cells, threat_window_count)` created by playing `cell`.

    Tracking distinct threat cells rather than raw window counts avoids false
    forks on boards with `win_k < min(rows, cols)` (such as `4x4, K=3`) where two
    overlapping collinear windows can share the exact same single finishing cell.
    """
    opp_mark = -player_mark
    threat_cells: set[int] = set()
    threat_windows = 0

    for win_idx in cell_windows[cell]:
        window = windows[win_idx]
        blocked = False
        other_empties: list[int] = []
        for idx in window:
            if idx == cell:
                continue
            val = flat_board[idx]
            if val == opp_mark:
                blocked = True
                break
            if val == EMPTY:
                other_empties.append(idx)

        # An open window with exactly one remaining empty cell after placing
        # `player_mark` at `cell` represents an immediate K-1 threat.
        if not blocked and len(other_empties) == 1:
            threat_cells.add(other_empties[0])
            threat_windows += 1

    return threat_cells, threat_windows


def _normalize_band_scores(
    tier_ranks: dict[int, tuple[RuleTier, tuple[int, ...]]],
) -> dict[int, float]:
    """Scale `(RuleTier, rank_key)` entries into non-overlapping `[0, 1]` bands."""
    grouped_keys: dict[RuleTier, list[tuple[int, ...]]] = {}
    for tier, key in tier_ranks.values():
        grouped_keys.setdefault(tier, []).append(key)

    unique_sorted_by_tier: dict[RuleTier, dict[tuple[int, ...], int]] = {}
    for tier, keys in grouped_keys.items():
        distinct = sorted(set(keys))
        unique_sorted_by_tier[tier] = {key: idx for idx, key in enumerate(distinct)}

    scores: dict[int, float] = {}
    for action, (tier, key) in tier_ranks.items():
        low, high = _TIER_SCORE_BANDS[int(tier)]
        key_index_map = unique_sorted_by_tier[tier]
        num_distinct = len(key_index_map)
        if num_distinct == 1:
            scores[action] = high
        else:
            ratio = key_index_map[key] / float(num_distinct - 1)
            scores[action] = low + (high - low) * ratio
    return scores


class RuleBasedAgent(BaseAgent):
    """Level 2 AI agent generalizing Newell & Simon's 6-tier rules to `M x N, K`.

    Priority order (strictly from highest to lowest):
        1. Immediate Win (`IMMEDIATE_WIN`): Complete `K` in a row immediately.
        2. Immediate Block (`IMMEDIATE_BLOCK`): Block opponent's `K - 1` threat.
        3. Create Fork (`CREATE_FORK`): Create >= 2 distinct `K - 1` winning cells.
        4. Block Opponent Fork (`BLOCK_FORK`): Prevent opponent fork directly or
           force a defense whose reply does not form an opponent fork.
        5. Positional Control (`POSITIONAL`): Maximize `positional_weights()` and
           open-line tactical synergy.
        6. Tie-Break (`FALLBACK`): Controlled random (via isolated RNG) or
           deterministic lowest-index selection among top-ranked ties.
    """

    def __init__(
        self,
        name: str | None = "RuleBasedAgent",
        seed: int | None = None,
        time_budget_ms: float | None = None,
        tie_break: Literal["random", "deterministic"] = "random",
    ) -> None:
        """Initialize a Level 2 rule-based agent.

        Args:
            name: Human-readable display name. Defaults to `"RuleBasedAgent"`.
            seed: Optional non-negative RNG seed used when `tie_break="random"`.
            time_budget_ms: Optional per-move compute budget in milliseconds.
            tie_break: Strategy to resolve exact ties within the winning tier
                (`"random"` uses `self.rng`; `"deterministic"` picks min index).

        Raises:
            InvalidConfigError: If `tie_break` or base agent arguments are invalid.
        """
        if tie_break not in ("random", "deterministic"):
            raise InvalidConfigError(
                f"tie_break must be 'random' or 'deterministic', got {tie_break!r}."
            )
        super().__init__(name=name, level=2, seed=seed, time_budget_ms=time_budget_ms)
        self._tie_break: Literal["random", "deterministic"] = tie_break

    @property
    def tie_break(self) -> Literal["random", "deterministic"]:
        """Configured tie-breaking mode (`'random'` or `'deterministic'`)."""
        return self._tie_break

    def get_config(self) -> dict[str, Any]:
        """Return serializable agent metadata including tie-break configuration."""
        config = super().get_config()
        config["tie_break"] = self._tie_break
        return config

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        """Apply the 6-tier Newell & Simon hierarchy using canonical board views.

        Args:
            game: Active, non-terminal :class:`ExtendedTicTacToe` instance.

        Returns:
            :class:`AgentDecision` with chosen move, tier-aligned `action_scores`,
            evaluated `nodes_visited`, and heuristic `value_estimate`.
        """
        rows, cols, win_k = game.rows, game.cols, game.win_k
        legal_actions = game.legal_actions()
        flat_board = [int(v) for v in game.canonical_board().reshape(-1)]

        windows = precompute_win_windows(rows, cols, win_k)
        cell_windows = _cell_to_window_indices(rows, cols, win_k)
        pos_weights = _flat_positional_weights(rows, cols, win_k)

        # Single pass over precomputed win windows to collect line occupancies.
        win_self_counts = [0] * len(windows)
        win_opp_counts = [0] * len(windows)
        for win_idx, window in enumerate(windows):
            s_cnt = 0
            o_cnt = 0
            for idx in window:
                val = flat_board[idx]
                if val == _SELF_MARK:
                    s_cnt += 1
                elif val == _OPP_MARK:
                    o_cnt += 1
            win_self_counts[win_idx] = s_cnt
            win_opp_counts[win_idx] = o_cnt

        nodes_visited = len(legal_actions)
        win_line_counts: dict[int, int] = {}
        block_line_counts: dict[int, int] = {}
        open_self_progress: dict[int, int] = {}
        open_opp_disrupt: dict[int, int] = {}
        self_threats: dict[int, tuple[set[int], int]] = {}
        opp_threats: dict[int, tuple[set[int], int]] = {}

        for cell in legal_actions:
            w_count = 0
            b_count = 0
            s_progress = 0
            o_disrupt = 0
            for win_idx in cell_windows[cell]:
                s_cnt = win_self_counts[win_idx]
                o_cnt = win_opp_counts[win_idx]
                if o_cnt == 0:
                    s_progress += s_cnt + 1
                    if s_cnt == win_k - 1:
                        w_count += 1
                if s_cnt == 0:
                    o_disrupt += o_cnt
                    if o_cnt == win_k - 1:
                        b_count += 1

            win_line_counts[cell] = w_count
            block_line_counts[cell] = b_count
            open_self_progress[cell] = s_progress
            open_opp_disrupt[cell] = o_disrupt

            self_threats[cell] = _fork_threats_for_move(
                flat_board, cell, _SELF_MARK, windows, cell_windows
            )
            opp_threats[cell] = _fork_threats_for_move(
                flat_board, cell, _OPP_MARK, windows, cell_windows
            )
            nodes_visited += 2

        opp_fork_cells = {
            cell for cell in legal_actions if len(opp_threats[cell][0]) >= 2
        }

        # When opponent has >= 2 fork squares, check if playing `cell` creates an
        # immediate K-1 threat whose forced blocking cell is NOT a post-move fork
        # for opponent (classic anti-opposite-corner defense from Newell & Simon).
        safe_counter_keys: dict[int, tuple[int, ...]] = {}
        if len(opp_fork_cells) >= 2:
            for cell in legal_actions:
                s_threat_cells, s_threat_wins = self_threats[cell]
                if len(s_threat_cells) != 1:
                    continue
                forced_block = next(iter(s_threat_cells))
                scratch_board = list(flat_board)
                scratch_board[cell] = _SELF_MARK
                post_opp_cells, _ = _fork_threats_for_move(
                    scratch_board,
                    forced_block,
                    _OPP_MARK,
                    windows,
                    cell_windows,
                )
                nodes_visited += 1
                if len(post_opp_cells) < 2:
                    safe_counter_keys[cell] = (
                        1,
                        1 if cell in opp_fork_cells else 0,
                        s_threat_wins,
                        pos_weights[cell],
                        open_self_progress[cell],
                    )

        tier_ranks: dict[int, tuple[RuleTier, tuple[int, ...]]] = {}
        for cell in legal_actions:
            s_threat_cells, s_threat_wins = self_threats[cell]
            if win_line_counts[cell] > 0:
                tier_ranks[cell] = (
                    RuleTier.IMMEDIATE_WIN,
                    (win_line_counts[cell], pos_weights[cell]),
                )
            elif block_line_counts[cell] > 0:
                tier_ranks[cell] = (
                    RuleTier.IMMEDIATE_BLOCK,
                    (
                        block_line_counts[cell],
                        len(s_threat_cells),
                        pos_weights[cell],
                    ),
                )
            elif len(s_threat_cells) >= 2:
                tier_ranks[cell] = (
                    RuleTier.CREATE_FORK,
                    (len(s_threat_cells), s_threat_wins, pos_weights[cell]),
                )
            elif len(opp_fork_cells) == 1 and cell in opp_fork_cells:
                o_cells, o_wins = opp_threats[cell]
                tier_ranks[cell] = (
                    RuleTier.BLOCK_FORK,
                    (1, len(o_cells), o_wins, pos_weights[cell]),
                )
            elif len(opp_fork_cells) >= 2 and safe_counter_keys:
                if cell in safe_counter_keys:
                    tier_ranks[cell] = (RuleTier.BLOCK_FORK, safe_counter_keys[cell])
                else:
                    tier_ranks[cell] = (
                        RuleTier.POSITIONAL,
                        (
                            pos_weights[cell],
                            open_self_progress[cell],
                            open_opp_disrupt[cell],
                        ),
                    )
            elif len(opp_fork_cells) >= 2 and cell in opp_fork_cells:
                o_cells, o_wins = opp_threats[cell]
                tier_ranks[cell] = (
                    RuleTier.BLOCK_FORK,
                    (
                        0,
                        len(o_cells),
                        o_wins,
                        pos_weights[cell],
                        open_self_progress[cell],
                    ),
                )
            else:
                tier_ranks[cell] = (
                    RuleTier.POSITIONAL,
                    (
                        pos_weights[cell],
                        open_self_progress[cell],
                        open_opp_disrupt[cell],
                    ),
                )

        best_priority = max(tier_ranks.values())
        best_tier = best_priority[0]
        best_candidates = [
            cell for cell in legal_actions if tier_ranks[cell] == best_priority
        ]

        if self._tie_break == "deterministic" or len(best_candidates) == 1:
            chosen_action = min(best_candidates)
        else:
            pick_idx = int(self.rng.integers(0, len(best_candidates)))
            chosen_action = best_candidates[pick_idx]

        value_estimate = self._estimate_tier_value(
            best_tier=best_tier,
            num_block_targets=sum(1 for cnt in block_line_counts.values() if cnt > 0),
        )
        depth_reached = (
            2 if best_tier in (RuleTier.CREATE_FORK, RuleTier.BLOCK_FORK) else 1
        )
        action_scores = _normalize_band_scores(tier_ranks)

        return AgentDecision(
            action=chosen_action,
            nodes_visited=nodes_visited,
            simulations=0,
            depth_reached=depth_reached,
            value_estimate=value_estimate,
            action_scores=action_scores,
        )

    @staticmethod
    def _estimate_tier_value(best_tier: RuleTier, num_block_targets: int) -> float:
        """Map the fired rule tier to a canonical heuristic value in `[-1, +1]`."""
        if best_tier is RuleTier.IMMEDIATE_WIN:
            return 1.0
        if best_tier is RuleTier.IMMEDIATE_BLOCK:
            return 0.2 if num_block_targets <= 1 else -0.3
        if best_tier is RuleTier.CREATE_FORK:
            return 0.75
        if best_tier is RuleTier.BLOCK_FORK:
            return -0.1
        return 0.1
