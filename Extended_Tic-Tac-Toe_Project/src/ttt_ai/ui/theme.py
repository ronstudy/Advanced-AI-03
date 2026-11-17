"""Rich CLI color palette, latency thresholds, heatmap tiers, and terminal fallbacks."""

from __future__ import annotations

import math
import os
import sys

from rich import box

from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.engine.rules import action_to_coords

STYLE_PLAYER_X: str = "bold bright_cyan"
STYLE_PLAYER_O: str = "bold bright_red"
STYLE_LAST_MOVE: str = "bold bright_white on grey23"
STYLE_WIN_CELL: str = "bold black on bright_green"
STYLE_EMPTY_CELL: str = "dim grey50"
STYLE_HEADER_BANNER: str = "bold bright_white on navy_blue"
STYLE_WARNING: str = "bold yellow"
STYLE_ERROR: str = "bold red"
STYLE_INFO: str = "cyan"

HEATMAP_TOP1_STYLE: str = "bold black on bright_yellow"
HEATMAP_HIGH_STYLE: str = "bold bright_green"
HEATMAP_MED_STYLE: str = "yellow"
HEATMAP_LOW_STYLE: str = "dim white"
HEATMAP_OCCUPIED_STYLE: str = "dim grey42"

_HEATMAP_TIER_STYLES: dict[int, str] = {
    0: HEATMAP_OCCUPIED_STYLE,
    1: HEATMAP_LOW_STYLE,
    2: HEATMAP_MED_STYLE,
    3: HEATMAP_HIGH_STYLE,
    4: HEATMAP_TOP1_STYLE,
}


def should_use_ascii(force_ascii: bool = False) -> bool:
    """Determine whether terminal rendering should use ASCII-only box characters.

    Checks explicit override, the ``TTT_ASCII_UI`` environment variable, and
    the active ``sys.stdout.encoding`` so Windows legacy terminals do not fail
    on Unicode box-drawing glyphs.

    Args:
        force_ascii: Explicit caller override to force ASCII boxes.

    Returns:
        ``True`` if ASCII box drawing should be used; ``False`` for Unicode.
    """
    if force_ascii:
        return True

    env_flag = os.environ.get("TTT_ASCII_UI", "").strip().lower()
    if env_flag in ("1", "true", "yes", "on"):
        return True

    encoding = getattr(sys.stdout, "encoding", None)
    if isinstance(encoding, str) and encoding.strip():
        return "utf" not in encoding.lower()
    return False


def get_box_style(ascii_only: bool = False) -> box.Box:
    """Return the Rich box-drawing style compatible with the terminal mode.

    Args:
        ascii_only: When ``True``, returns pure ASCII ``box.ASCII``; otherwise
            returns ``box.ROUNDED``.

    Returns:
        A :class:`rich.box.Box` instance.
    """
    return box.ASCII if ascii_only else box.ROUNDED


def format_latency_style(
    latency_ms: float,
    budget_ms: float | None = None,
) -> str:
    """Select a Rich color style reflecting per-move latency against compute budget.

    Args:
        latency_ms: Measured inference or move latency in milliseconds (`>= 0`).
        budget_ms: Optional compute budget in milliseconds (`> 0`). When ``None``,
            uses ``50.0 ms`` as the soft green baseline and ``200.0 ms`` as high.

    Returns:
        Rich style string (``"bold green"``, ``"bold yellow"``, or ``"bold red"``).

    Raises:
        InvalidConfigError: If ``latency_ms`` is negative/non-finite or ``budget_ms``
            is non-positive/non-finite.
    """
    if (
        isinstance(latency_ms, bool)
        or not isinstance(latency_ms, (int, float))
        or not math.isfinite(float(latency_ms))
        or float(latency_ms) < 0.0
    ):
        raise InvalidConfigError(
            f"latency_ms must be a finite number >= 0.0, got {latency_ms!r}."
        )

    lat = float(latency_ms)
    if budget_ms is None:
        if lat < 50.0:
            return "bold green"
        if lat <= 200.0:
            return "bold yellow"
        return "bold red"

    if (
        isinstance(budget_ms, bool)
        or not isinstance(budget_ms, (int, float))
        or not math.isfinite(float(budget_ms))
        or float(budget_ms) <= 0.0
    ):
        raise InvalidConfigError(
            f"budget_ms must be a positive finite number or None, got {budget_ms!r}."
        )

    budget = float(budget_ms)
    if lat <= 0.8 * budget:
        return "bold green"
    if lat <= budget:
        return "bold yellow"
    return "bold red"


def classify_heatmap_tier(
    score: float,
    min_score: float,
    max_score: float,
    is_top1: bool,
) -> int:
    """Classify a legal move's score into a 4-level active heatmap tier (1..4).

    Tier ``4`` designates the top-ranked move, tier ``3`` marks strong moves
    (`>= 60%` of the normalized `[min_score, max_score]` span), tier ``2``
    marks moderate moves (`25%..60%`), and tier ``1`` marks low-priority moves.

    Args:
        score: Candidate move score from ``AgentDecision.action_scores``.
        min_score: Minimum score among legal moves on the current board.
        max_score: Maximum score among legal moves on the current board.
        is_top1: Whether this cell is the highest-scoring / chosen move.

    Returns:
        Integer tier in ``{1, 2, 3, 4}``.
    """
    if is_top1:
        return 4

    span = float(max_score) - float(min_score)
    if span <= 1e-12:
        # Uniform distributions (e.g., RandomAgent) map non-selected ties to tier 2.
        return 2

    normalized = (float(score) - float(min_score)) / span
    if normalized >= 0.60:
        return 3
    if normalized >= 0.25:
        return 2
    return 1


def style_for_heatmap_tier(tier: int) -> str:
    """Return the Rich style string corresponding to heatmap tier ``0..4``.

    Args:
        tier: Tier index where ``0`` is occupied, ``1`` is low, ``2`` is medium,
            ``3`` is high, and ``4`` is top-1.

    Returns:
        Rich style string from the module palette.

    Raises:
        InvalidConfigError: If ``tier`` is not an integer in ``0..4``.
    """
    if isinstance(tier, bool) or not isinstance(tier, int) or tier not in _HEATMAP_TIER_STYLES:
        raise InvalidConfigError(f"Heatmap tier must be an integer in 0..4, got {tier!r}.")
    return _HEATMAP_TIER_STYLES[tier]


def format_move_coord(action: int, cols: int) -> str:
    """Format a flat cell index as ``(row, col) [#action]``.

    Args:
        action: Non-negative flat row-major cell index.
        cols: Number of board columns (`>= 3`).

    Returns:
        Formatted coordinate string such as ``"(1, 2) [#6]"``.
    """
    row, col = action_to_coords(action, cols)
    return f"({row}, {col}) [#{action}]"
