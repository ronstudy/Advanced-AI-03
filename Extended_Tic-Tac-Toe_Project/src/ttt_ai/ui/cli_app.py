"""Interactive game controller and Rich CLI dashboard for Extended Tic-Tac-Toe.

Provides setup wizard, live match loop supporting Human vs AI (Easy/Medium/Hard
and Levels 1-7), AI vs AI, and Human vs Human modes, real-time move latency and
search telemetry, 5-tier action heatmaps (including Level 7 prior vs. PUCT
visit share comparison), and resilient input validation.

Interacts with :class:`ttt_ai.engine.game.ExtendedTicTacToe` and
:class:`ttt_ai.agents.base.BaseAgent` strictly via their public interfaces.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ttt_ai.agents.base import AgentDecision, BaseAgent
from ttt_ai.agents.hybrid_mcts_agent import HybridMCTSAgent
from ttt_ai.agents.mcts_agent import MCTSAgent
from ttt_ai.agents.minimax_agent import AlphaBetaAgent
from ttt_ai.agents.policy_agent import PolicyNetworkAgent
from ttt_ai.agents.random_agent import RandomAgent
from ttt_ai.agents.rl_agent import DQNAgent, TabularQAgent
from ttt_ai.agents.rule_based_agent import RuleBasedAgent
from ttt_ai.config import BoardConfig
from ttt_ai.engine.exceptions import (
    CheckpointVerificationError,
    IllegalMoveError,
    InvalidConfigError,
)
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.engine.rules import (
    PLAYER_O,
    PLAYER_X,
    Player,
    action_to_coords,
    coords_to_action,
)
from ttt_ai.training.checkpoint_io import resolve_meta_path
from ttt_ai.ui.theme import (
    STYLE_EMPTY_CELL,
    STYLE_ERROR,
    STYLE_HEADER_BANNER,
    STYLE_INFO,
    STYLE_LAST_MOVE,
    STYLE_PLAYER_O,
    STYLE_PLAYER_X,
    STYLE_WARNING,
    STYLE_WIN_CELL,
    classify_heatmap_tier,
    format_latency_style,
    format_move_coord,
    get_box_style,
    should_use_ascii,
    style_for_heatmap_tier,
)

_BOARD_PRESETS: dict[str, tuple[str, BoardConfig]] = {
    "1": ("4x4, K=4 (Primary Benchmark)", BoardConfig(rows=4, cols=4, win_k=4)),
    "2": ("4x4, K=3 (First-Mover Sensitivity)", BoardConfig(rows=4, cols=4, win_k=3)),
    "3": ("3x3, K=3 (Classic Reference)", BoardConfig(rows=3, cols=3, win_k=3)),
    "4": ("5x5, K=4 (Generalization)", BoardConfig(rows=5, cols=5, win_k=4)),
}

_HELP_TEXT: str = (
    "Input 'r,c' or 'r c' (0-indexed row & col) or flat cell index [0..M*N-1]. "
    "Commands: 'u'/'undo' revert turn, 'h'/'help' show help, 'q'/'quit' exit match."
)


class GameMode(str, Enum):
    """Supported interactive match modes."""

    HUMAN_VS_AI = "human_vs_ai"
    AI_VS_AI = "ai_vs_ai"
    HUMAN_VS_HUMAN = "human_vs_human"


class DifficultyPreset(str, Enum):
    """Human-friendly difficulty aliases mapped onto the AI hierarchy."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True, slots=True)
class ParsedUserCommand:
    """Validated user input action or control command."""

    kind: Literal["move", "undo", "help", "quit"]
    action: int | None = None


@dataclass(slots=True)
class MatchTurnRecord:
    """Telemetry record for a single completed ply in an interactive match."""

    turn_index: int
    player: Player
    action: int
    row: int
    col: int
    latency_ms: float
    agent_name: str
    decision: AgentDecision | None = None


@dataclass(slots=True)
class MatchResultSummary:
    """Structured summary emitted when an interactive match finishes or exits."""

    winner: Player | None
    is_draw: bool
    quit_early: bool
    move_count: int
    move_history: tuple[int, ...]
    winning_line: tuple[tuple[int, int], ...] | None
    turn_records: list[MatchTurnRecord] = field(default_factory=list)
    final_board_text: str = ""
    invalid_input_count: int = 0


def _parse_strict_int_token(token: str) -> int:
    """Parse an integer token strictly without accepting floats or punctuation."""
    stripped = token.strip()
    if not stripped:
        raise IllegalMoveError("Empty numeric token in move input.")
    try:
        return int(stripped, 10)
    except ValueError as exc:
        raise IllegalMoveError(
            f"Invalid move input {token!r}. Use 'row,col' (e.g. '1,2'), cell index, "
            "or commands 'u', 'h', 'q'."
        ) from exc


def parse_user_input(raw_text: str, game: ExtendedTicTacToe) -> ParsedUserCommand:
    """Parse and validate raw user console input against the live game state.

    Supports commands (``u``/``undo``, ``h``/``help``, ``q``/``quit``) and both
    coordinate inputs (``r,c`` or ``r c``) and flat row-major cell indices.

    Args:
        raw_text: Raw line entered by the user.
        game: Active :class:`ExtendedTicTacToe` instance.

    Returns:
        Validated :class:`ParsedUserCommand`.

    Raises:
        IllegalMoveError: If the command is malformed, coordinates/index are
            out of bounds, the cell is already occupied, or the game ended.
    """
    if not isinstance(raw_text, str):
        raise IllegalMoveError(f"Expected string input, got {type(raw_text).__name__}.")

    cleaned = raw_text.strip()
    if not cleaned:
        raise IllegalMoveError(
            "Empty input. Enter 'row,col', cell index, or 'u'/'h'/'q'."
        )

    lowered = cleaned.lower()
    if lowered in ("u", "undo"):
        return ParsedUserCommand(kind="undo")
    if lowered in ("h", "help", "?"):
        return ParsedUserCommand(kind="help")
    if lowered in ("q", "quit", "exit"):
        return ParsedUserCommand(kind="quit")

    if game.is_terminal():
        raise IllegalMoveError("Game has already ended; no further moves allowed.")

    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1].strip()

    if "," in cleaned:
        parts = [part.strip() for part in cleaned.split(",")]
        if len(parts) != 2 or any(not part for part in parts):
            raise IllegalMoveError(
                f"Invalid coordinate syntax {raw_text!r}. Expected 'row,col' such as '1,2'."
            )
        row = _parse_strict_int_token(parts[0])
        col = _parse_strict_int_token(parts[1])
        action = _validate_row_col_bounds(row, col, game)
    else:
        tokens = cleaned.split()
        if len(tokens) == 2:
            row = _parse_strict_int_token(tokens[0])
            col = _parse_strict_int_token(tokens[1])
            action = _validate_row_col_bounds(row, col, game)
        elif len(tokens) == 1:
            idx = _parse_strict_int_token(tokens[0])
            if not (0 <= idx < game.num_cells):
                raise IllegalMoveError(
                    f"Cell index {idx} is out of bounds for [0, {game.num_cells - 1}]."
                )
            action = idx
        else:
            raise IllegalMoveError(
                f"Invalid input {raw_text!r}. Expected 'row,col' or a single cell index."
            )

    legal = set(game.legal_actions())
    if action not in legal:
        row, col = action_to_coords(action, game.cols)
        raise IllegalMoveError(
            f"Cell ({row}, {col}) [#{action}] is already occupied. Choose an empty cell."
        )

    return ParsedUserCommand(kind="move", action=action)


def _validate_row_col_bounds(row: int, col: int, game: ExtendedTicTacToe) -> int:
    """Validate `(row, col)` against board shape and convert to flat action index."""
    if not (0 <= row < game.rows and 0 <= col < game.cols):
        raise IllegalMoveError(
            f"Coordinates ({row}, {col}) out of bounds for {game.rows}x{game.cols} board "
            f"(valid rows 0..{game.rows - 1}, cols 0..{game.cols - 1})."
        )
    return coords_to_action(row, col, game.cols)


def resolve_default_checkpoints(
    board_config: BoardConfig,
    checkpoints_dir: str | Path = "checkpoints",
) -> dict[str, Path]:
    """Discover verified checkpoints matching `board_config` under `checkpoints_dir`.

    Only checkpoints whose weights file AND ``*.meta.json`` sidecar exist on
    disk are returned.

    Args:
        board_config: Target board dimensions and win-length configuration.
        checkpoints_dir: Directory containing trained `.pt`/`.pkl` checkpoints.

    Returns:
        Dictionary mapping logical roles (``"fast_policy"``, ``"strong_policy"``,
        ``"value"``, ``"dqn"``, ``"tabular_q"``) to verified checkpoint paths.
    """
    base_dir = Path(checkpoints_dir)
    if not base_dir.is_dir():
        return {}

    tag = board_config.name
    candidates: dict[str, tuple[str, ...]] = {
        "fast_policy": (f"fastpolicynet_{tag}.pt",),
        "strong_policy": (f"strongpolicynet_{tag}.pt",),
        "value": (f"valuenet_{tag}.pt",),
        "dqn": (
            f"dqnnet_{tag}_seed42.pt",
            f"dqnnet_{tag}_seed123.pt",
            f"dqnnet_{tag}_seed999.pt",
        ),
        "tabular_q": (
            f"tabular_q_{tag}_seed42.pkl",
            f"tabular_q_{tag}_seed123.pkl",
            f"tabular_q_{tag}_seed999.pkl",
        ),
    }

    found: dict[str, Path] = {}
    for role, filenames in candidates.items():
        for filename in filenames:
            ckpt_path = base_dir / filename
            meta_path = resolve_meta_path(ckpt_path)
            if ckpt_path.is_file() and meta_path.is_file():
                found[role] = ckpt_path
                break
    return found


def create_agent_for_cli(
    level_or_difficulty: int | str | DifficultyPreset,
    board_config: BoardConfig,
    *,
    seed: int = 42,
    time_budget_ms: float | None = 50.0,
    n_simulations: int = 200,
    checkpoints_dir: str | Path = "checkpoints",
) -> BaseAgent:
    """Instantiate an AI agent (Difficulty preset or Level 1-7) for interactive play.

    Args:
        level_or_difficulty: Difficulty preset (``"easy"``, ``"medium"``,
            ``"hard"``) or hierarchy level ``1..7``.
        board_config: Target board configuration (used to match checkpoints).
        seed: Reproducible seed for the instantiated agent.
        time_budget_ms: Optional per-move compute budget in milliseconds.
        n_simulations: Simulation count for MCTS / HybridMCTS agents.
        checkpoints_dir: Directory searched for trained Level 5-7 checkpoints.

    Returns:
        Configured :class:`BaseAgent` ready for :meth:`BaseAgent.select_action`.

    Raises:
        InvalidConfigError: If `level_or_difficulty` is unrecognized.
        CheckpointVerificationError: If a neural/RL level (5-7) is requested for a
            board lacking a verified checkpoint.
    """
    if isinstance(level_or_difficulty, bool):
        raise InvalidConfigError("level_or_difficulty cannot be a boolean.")

    normalized: int | str
    if isinstance(level_or_difficulty, DifficultyPreset):
        normalized = level_or_difficulty.value
    elif isinstance(level_or_difficulty, int):
        normalized = level_or_difficulty
    elif isinstance(level_or_difficulty, str):
        raw = level_or_difficulty.strip().lower()
        if raw.isdigit():
            normalized = int(raw, 10)
        else:
            normalized = raw
    else:
        raise InvalidConfigError(
            f"Unsupported level_or_difficulty type: {type(level_or_difficulty).__name__}."
        )

    if normalized == DifficultyPreset.EASY.value:
        return RandomAgent(
            name="EasyAI (Random)",
            seed=seed,
            time_budget_ms=time_budget_ms,
        )
    if normalized == DifficultyPreset.MEDIUM.value:
        return RuleBasedAgent(
            name="MediumAI (RuleBased)",
            seed=seed,
            time_budget_ms=time_budget_ms,
        )
    if normalized == DifficultyPreset.HARD.value:
        return AlphaBetaAgent(
            name="HardAI (AlphaBeta)",
            max_depth=4,
            seed=seed,
            time_budget_ms=time_budget_ms,
        )

    if not isinstance(normalized, int) or not (1 <= normalized <= 7):
        raise InvalidConfigError(
            f"Expected difficulty ('easy', 'medium', 'hard') or level 1..7, "
            f"got {level_or_difficulty!r}."
        )

    if normalized == 1:
        return RandomAgent(name="L1_Random", seed=seed, time_budget_ms=time_budget_ms)
    if normalized == 2:
        return RuleBasedAgent(name="L2_RuleBased", seed=seed, time_budget_ms=time_budget_ms)
    if normalized == 3:
        return AlphaBetaAgent(
            name="L3_AlphaBeta",
            max_depth=4,
            seed=seed,
            time_budget_ms=time_budget_ms,
        )
    if normalized == 4:
        return MCTSAgent(
            name="L4_MCTS",
            n_simulations=n_simulations,
            seed=seed,
            time_budget_ms=time_budget_ms,
        )

    checkpoints = resolve_default_checkpoints(board_config, checkpoints_dir=checkpoints_dir)
    if normalized == 5:
        policy_ckpt = checkpoints.get("strong_policy") or checkpoints.get("fast_policy")
        if policy_ckpt is None:
            raise CheckpointVerificationError(
                f"No verified Level 5 policy checkpoint found for board {board_config.name} "
                f"in '{checkpoints_dir}'."
            )
        return PolicyNetworkAgent(
            checkpoint_path=policy_ckpt,
            board_config=board_config,
            name="L5_PolicyNet",
            seed=seed,
            time_budget_ms=time_budget_ms,
        )

    if normalized == 6:
        if "dqn" in checkpoints:
            return DQNAgent(
                checkpoint_path=checkpoints["dqn"],
                board_config=board_config,
                name="L6_DQN",
                seed=seed,
                time_budget_ms=time_budget_ms,
            )
        if "tabular_q" in checkpoints:
            return TabularQAgent(
                checkpoint_path=checkpoints["tabular_q"],
                board_config=board_config,
                name="L6_TabularQ",
                seed=seed,
                time_budget_ms=time_budget_ms,
            )
        raise CheckpointVerificationError(
            f"No verified Level 6 RL checkpoint (DQN or TabularQ) found for board "
            f"{board_config.name} in '{checkpoints_dir}'."
        )

    policy_ckpt = checkpoints.get("strong_policy") or checkpoints.get("fast_policy")
    if policy_ckpt is None:
        raise CheckpointVerificationError(
            f"No verified policy checkpoint found for Level 7 HybridMCTS on board "
            f"{board_config.name} in '{checkpoints_dir}'."
        )
    value_ckpt = checkpoints.get("value")
    leaf_mode: Literal["rollout", "mixed"] = "mixed" if value_ckpt is not None else "rollout"
    return HybridMCTSAgent(
        policy_checkpoint_path=policy_ckpt,
        value_checkpoint_path=value_ckpt,
        board_config=board_config,
        n_simulations=n_simulations,
        leaf_eval_mode=leaf_mode,
        name="L7_HybridMCTS",
        seed=seed,
        time_budget_ms=time_budget_ms,
    )


def format_extended_board_ascii(
    game: ExtendedTicTacToe,
    *,
    last_move: int | None = None,
    last_latency_ms: float | None = None,
) -> str:
    """Render an `M x N` board in clean ASCII with coordinates and move highlights.

    Args:
        game: :class:`ExtendedTicTacToe` instance inspected via public properties.
        last_move: Optional flat cell index override to highlight; defaults to
            `game.last_move`.
        last_latency_ms: Optional last-turn thinking duration in milliseconds.

    Returns:
        Multi-line ASCII string suitable for plain logs or non-ANSI terminals.
    """
    board = game.board
    rows, cols, win_k = game.rows, game.cols, game.win_k
    resolved_last = game.last_move if last_move is None else last_move
    last_rc = action_to_coords(resolved_last, cols) if resolved_last is not None else None
    win_cells = set(game.winning_line) if game.winning_line is not None else set()

    if game.is_terminal():
        if game.winner is not None:
            status = f"Result: Player {game.winner.symbol} wins!"
        else:
            status = "Result: Draw"
    else:
        status = f"Turn #{game.move_count + 1} | Current: Player {game.current_player.symbol}"

    header_parts = [f"Board: {rows}x{cols} (K={win_k})", status]
    if last_latency_ms is not None:
        header_parts.append(f"Think Time: {last_latency_ms:.2f} ms")

    lines: list[str] = [
        " | ".join(header_parts),
        "    " + " ".join(f" {c} " for c in range(cols)),
        "   +" + "---+" * cols,
    ]

    for r in range(rows):
        cells: list[str] = []
        for c in range(cols):
            val = int(board[r, c])
            mark = "X" if val == PLAYER_X else ("O" if val == PLAYER_O else ".")
            if (r, c) in win_cells:
                token = f"[{mark}]"
            elif last_rc is not None and (r, c) == last_rc:
                token = f"<{mark}>"
            else:
                token = f" {mark} "
            cells.append(token)
        row_suffix = " <- Last" if (last_rc is not None and r == last_rc[0]) else ""
        lines.append(f"{r:>2} |" + "|".join(cells) + "|" + row_suffix)
        lines.append("   +" + "---+" * cols)

    if last_rc is not None and resolved_last is not None:
        lines.append(f"Last Move: ({last_rc[0]}, {last_rc[1]}) [#{resolved_last}] (< > = last)")
    if game.winning_line is not None:
        lines.append(f"Winning Line: {list(game.winning_line)} ([ ] = winning line)")
    return "\n".join(lines)


def build_board_panel(
    game: ExtendedTicTacToe,
    *,
    ascii_only: bool = False,
) -> Panel:
    """Build the Rich Panel rendering the game board with coordinates & highlights."""
    board = game.board
    rows, cols = game.rows, game.cols
    last_move = game.last_move
    last_rc = action_to_coords(last_move, cols) if last_move is not None else None
    win_cells = set(game.winning_line) if game.winning_line is not None else set()

    table = Table(
        box=get_box_style(ascii_only),
        show_header=True,
        show_lines=True,
        padding=(0, 1),
        header_style="bold dim",
    )
    table.add_column("r\\c", justify="right", style="bold dim", width=3)
    for col in range(cols):
        table.add_column(str(col), justify="center", width=5)

    for row in range(rows):
        row_cells: list[Text] = [Text(str(row), style="bold dim")]
        for col in range(cols):
            val = int(board[row, col])
            idx = row * cols + col
            is_win = (row, col) in win_cells
            is_last = last_rc is not None and (row, col) == last_rc

            if val == PLAYER_X:
                label = "[X]" if is_win else ("<X>" if is_last else " X ")
                style = (
                    STYLE_WIN_CELL
                    if is_win
                    else (STYLE_LAST_MOVE if is_last else STYLE_PLAYER_X)
                )
            elif val == PLAYER_O:
                label = "[O]" if is_win else ("<O>" if is_last else " O ")
                style = (
                    STYLE_WIN_CELL
                    if is_win
                    else (STYLE_LAST_MOVE if is_last else STYLE_PLAYER_O)
                )
            else:
                label = f"{idx:>2} "
                style = STYLE_EMPTY_CELL
            row_cells.append(Text(label, style=style))
        table.add_row(*row_cells)

    legend = Text("Legend: ", style="dim")
    legend.append("X", style=STYLE_PLAYER_X)
    legend.append(" | ")
    legend.append("O", style=STYLE_PLAYER_O)
    legend.append(" | < > Last Move | [ ] Winning Line", style="dim")

    return Panel(
        Group(table, legend),
        title="[bold]GAME BOARD[/bold]",
        box=get_box_style(ascii_only),
        border_style="bright_blue",
    )


def build_telemetry_panel(
    turn_record: MatchTurnRecord | None,
    *,
    budget_ms: float | None = None,
    ascii_only: bool = False,
) -> Panel:
    """Build the Rich Panel displaying last-move player/AI telemetry."""
    if turn_record is None:
        return Panel(
            Text("Waiting for first move...", style="dim"),
            title="[bold]TURN TELEMETRY[/bold]",
            box=get_box_style(ascii_only),
            border_style="cyan",
        )

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="bold dim", no_wrap=True)
    table.add_column("Value")

    player_style = STYLE_PLAYER_X if turn_record.player is Player.X else STYLE_PLAYER_O
    lat_style = format_latency_style(turn_record.latency_ms, budget_ms=budget_ms)
    move_str = f"({turn_record.row}, {turn_record.col}) [#{turn_record.action}]"

    actor_label = f"{turn_record.agent_name} ({turn_record.player.symbol})"
    table.add_row("Actor", Text(actor_label, style=player_style))
    table.add_row("Move", Text(move_str, style="bold"))
    table.add_row("Latency", Text(f"{turn_record.latency_ms:.3f} ms", style=lat_style))

    decision = turn_record.decision
    if decision is not None:
        search_str = (
            f"nodes={decision.nodes_visited}, sims={decision.simulations}, "
            f"depth={decision.depth_reached}"
        )
        table.add_row("Search", Text(search_str))
        val_str = (
            f"{decision.value_estimate:+.4f}"
            if decision.value_estimate is not None
            else "n/a"
        )
        table.add_row("Value Est.", Text(val_str))
    else:
        table.add_row("Search", Text("Human input", style="dim"))

    return Panel(
        table,
        title="[bold]TURN TELEMETRY (Last Move)[/bold]",
        box=get_box_style(ascii_only),
        border_style="cyan",
    )


def build_heatmap_panel(
    game: ExtendedTicTacToe,
    decision: AgentDecision | None,
    *,
    ascii_only: bool = False,
) -> Panel:
    """Build the Rich Panel rendering the 5-tier action heatmap and Top-5 moves."""
    if decision is None or not decision.action_scores:
        return Panel(
            Text("Heatmap available after an AI decision.", style="dim"),
            title="[bold]ACTION HEATMAP & TOP-5 MOVES[/bold]",
            box=get_box_style(ascii_only),
            border_style="magenta",
        )

    rows, cols = game.rows, game.cols
    scores = decision.action_scores
    min_score = min(scores.values())
    max_score = max(scores.values())

    grid = Table(
        box=get_box_style(ascii_only),
        show_header=True,
        padding=(0, 1),
        header_style="bold dim",
    )
    grid.add_column("r\\c", justify="right", style="bold dim", width=3)
    for col in range(cols):
        grid.add_column(str(col), justify="center", width=7)

    for row in range(rows):
        row_cells: list[Text] = [Text(str(row), style="bold dim")]
        for col in range(cols):
            idx = row * cols + col
            if idx in scores:
                val = scores[idx]
                is_top1 = idx == decision.action
                tier = classify_heatmap_tier(val, min_score, max_score, is_top1=is_top1)
                cell_style = style_for_heatmap_tier(tier)
                formatted = f"[{val:+.2f}]" if is_top1 else f"{val:+.2f}"
                row_cells.append(Text(formatted, style=cell_style))
            else:
                row_cells.append(Text("  .  ", style=style_for_heatmap_tier(0)))
        grid.add_row(*row_cells)

    top_moves = decision.top_actions(k=5)
    has_priors = bool(decision.prior_probs)
    top_lines: list[Text] = []
    header_lbl = (
        "Top-5 Moves (Visit/Score vs Policy Prior P(s,a)):"
        if has_priors
        else "Top-5 Moves:"
    )
    top_lines.append(Text(header_lbl, style="bold"))

    for rank, (action, score) in enumerate(top_moves, start=1):
        coord = format_move_coord(action, cols)
        if has_priors:
            prior_val = decision.prior_probs.get(action, 0.0)
            line_text = (
                f"  {rank}. {coord:<14} score={score:+.3f} | prior={prior_val * 100.0:5.1f}%"
            )
        else:
            line_text = f"  {rank}. {coord:<14} score={score:+.4f}"
        top_lines.append(Text(line_text, style="bold bright_yellow" if rank == 1 else "white"))

    return Panel(
        Group(grid, *top_lines),
        title="[bold]ACTION HEATMAP & TOP-5 MOVES[/bold]",
        box=get_box_style(ascii_only),
        border_style="magenta",
    )


def build_dashboard_panel(
    game: ExtendedTicTacToe,
    *,
    mode: GameMode,
    actor_x_name: str,
    actor_o_name: str,
    turn_records: list[MatchTurnRecord],
    status_message: str = "",
    status_kind: Literal["info", "warning", "error"] = "info",
    budget_ms: float | None = None,
    ascii_only: bool = False,
) -> Panel:
    """Assemble the complete live match dashboard Panel."""
    if game.is_terminal():
        if game.winner is not None:
            turn_status = f"FINISHED — Winner: Player {game.winner.symbol}"
        else:
            turn_status = "FINISHED — Draw"
    else:
        turn_status = (
            f"Turn #{game.move_count + 1} | "
            f"To Move: Player {game.current_player.symbol}"
        )

    header = Text(
        f"Extended Tic-Tac-Toe ({game.rows}x{game.cols}, K={game.win_k}) | "
        f"Mode: {mode.value} | X={actor_x_name} vs O={actor_o_name}\n{turn_status}",
        style=STYLE_HEADER_BANNER,
    )

    last_record = turn_records[-1] if turn_records else None
    last_decision = last_record.decision if last_record is not None else None

    board_panel = build_board_panel(game, ascii_only=ascii_only)
    telemetry_panel = build_telemetry_panel(
        last_record,
        budget_ms=budget_ms,
        ascii_only=ascii_only,
    )
    heatmap_panel = build_heatmap_panel(game, last_decision, ascii_only=ascii_only)

    layout_grid = Table.grid(expand=True)
    layout_grid.add_column(ratio=1)
    layout_grid.add_column(ratio=1)
    right_col = Group(telemetry_panel, heatmap_panel)
    layout_grid.add_row(board_panel, right_col)

    recent_history = turn_records[-6:]
    hist_str = " -> ".join(
        f"{rec.turn_index}.{rec.player.symbol}({rec.row},{rec.col})"
        for rec in recent_history
    )
    history_text = Text(f"History: {hist_str or '(none)'}", style="dim")

    parts: list[object] = [header, layout_grid, history_text]
    if status_message:
        msg_style = (
            STYLE_ERROR
            if status_kind == "error"
            else (STYLE_WARNING if status_kind == "warning" else STYLE_INFO)
        )
        parts.append(Text(status_message, style=msg_style))

    return Panel(Group(*parts), box=get_box_style(ascii_only), border_style="white")


class InteractiveMatchController:
    """Orchestrates Human vs AI, AI vs AI, and Human vs Human matches via Public APIs."""

    def __init__(
        self,
        game: ExtendedTicTacToe,
        mode: GameMode | str,
        *,
        agent_x: BaseAgent | None = None,
        agent_o: BaseAgent | None = None,
        human_x_name: str = "Human_X",
        human_o_name: str = "Human_O",
        input_fn: Callable[[str], str] = input,
        console: Console | None = None,
        step_delay_ms: float = 0.0,
        step_by_step_ai: bool = False,
        ascii_only: bool | None = None,
        budget_ms: float | None = None,
    ) -> None:
        """Initialize the match controller and validate mode-to-agent assignments.

        Args:
            game: Active :class:`ExtendedTicTacToe` instance.
            mode: One of ``GameMode`` values (``human_vs_ai``, ``ai_vs_ai``,
                ``human_vs_human``).
            agent_x: AI agent playing ``X`` (or ``None`` if ``X`` is human).
            agent_o: AI agent playing ``O`` (or ``None`` if ``O`` is human).
            human_x_name: Display label when ``X`` is human-controlled.
            human_o_name: Display label when ``O`` is human-controlled.
            input_fn: Callable used to prompt the user (injectable for tests).
            console: Optional Rich :class:`Console` instance.
            step_delay_ms: Delay between AI moves in `AI vs AI` auto-play mode.
            step_by_step_ai: If ``True``, waits for Enter before each AI move in
                `AI vs AI` mode.
            ascii_only: Optional override for ASCII box drawing; auto-detects if ``None``.
            budget_ms: Optional reference time budget shown in latency telemetry.

        Raises:
            InvalidConfigError: If `game`, `mode`, `agent_x`/`agent_o`, or
                `step_delay_ms` violates controller invariants.
        """
        if not isinstance(game, ExtendedTicTacToe):
            raise InvalidConfigError(
                f"game must be an ExtendedTicTacToe instance, got {type(game).__name__}."
            )
        resolved_mode = mode if isinstance(mode, GameMode) else GameMode(str(mode).strip().lower())

        for side, agent in (("agent_x", agent_x), ("agent_o", agent_o)):
            if agent is not None and not isinstance(agent, BaseAgent):
                raise InvalidConfigError(
                    f"{side} must be a BaseAgent instance or None, got {type(agent).__name__}."
                )

        if resolved_mode is GameMode.AI_VS_AI:
            if agent_x is None or agent_o is None:
                raise InvalidConfigError("AI vs AI mode requires both agent_x and agent_o.")
        elif resolved_mode is GameMode.HUMAN_VS_AI:
            if (agent_x is None) == (agent_o is None):
                raise InvalidConfigError(
                    "Human vs AI mode requires exactly one AI agent (agent_x or agent_o)."
                )
        elif resolved_mode is GameMode.HUMAN_VS_HUMAN:
            if agent_x is not None or agent_o is not None:
                raise InvalidConfigError(
                    "Human vs Human mode requires both agent_x and agent_o to be None."
                )

        if isinstance(step_delay_ms, bool) or step_delay_ms < 0.0:
            raise InvalidConfigError(f"step_delay_ms must be >= 0.0, got {step_delay_ms!r}.")

        self._game: ExtendedTicTacToe = game
        self._mode: GameMode = resolved_mode
        self._agent_x: BaseAgent | None = agent_x
        self._agent_o: BaseAgent | None = agent_o
        self._human_x_name: str = human_x_name
        self._human_o_name: str = human_o_name
        self._input_fn: Callable[[str], str] = input_fn
        self._ascii_only: bool = should_use_ascii() if ascii_only is None else bool(ascii_only)
        self._console: Console = (
            console
            if console is not None
            else Console(safe_box=self._ascii_only, highlight=False)
        )
        self._step_delay_ms: float = float(step_delay_ms)
        self._step_by_step_ai: bool = bool(step_by_step_ai)
        self._budget_ms: float | None = budget_ms
        self._turn_records: list[MatchTurnRecord] = []
        self._invalid_input_count: int = 0

    @property
    def invalid_input_count(self) -> int:
        """Total number of invalid user inputs safely caught during the match."""
        return self._invalid_input_count

    @property
    def mode(self) -> GameMode:
        """Active match mode."""
        return self._mode

    def _actor_name(self, player: Player) -> str:
        """Return display name for the actor controlling `player`."""
        if player is Player.X:
            return self._agent_x.name if self._agent_x is not None else self._human_x_name
        return self._agent_o.name if self._agent_o is not None else self._human_o_name

    def _active_agent(self) -> BaseAgent | None:
        """Return the AI agent assigned to `game.current_player`, or ``None`` if human."""
        return self._agent_x if self._game.current_player is Player.X else self._agent_o

    def _human_player(self) -> Player | None:
        """Return the human-controlled player in ``HUMAN_VS_AI`` mode."""
        if self._mode is not GameMode.HUMAN_VS_AI:
            return None
        return Player.X if self._agent_x is None else Player.O

    def _render_dashboard(
        self,
        *,
        status_message: str = "",
        status_kind: Literal["info", "warning", "error"] = "info",
    ) -> None:
        """Render the current match dashboard to the Rich Console."""
        panel = build_dashboard_panel(
            self._game,
            mode=self._mode,
            actor_x_name=self._actor_name(Player.X),
            actor_o_name=self._actor_name(Player.O),
            turn_records=self._turn_records,
            status_message=status_message,
            status_kind=status_kind,
            budget_ms=self._budget_ms,
            ascii_only=self._ascii_only,
        )
        self._console.print(panel)

    def _handle_undo(self) -> tuple[str, Literal["info", "warning"]]:
        """Revert previous turn(s) cleanly while keeping turn history synchronized."""
        if self._mode is GameMode.HUMAN_VS_AI:
            human_side = self._human_player()
            min_moves = 2 if human_side is Player.X else 3
            if self._game.move_count < min_moves:
                return ("No completed human move available to undo.", "warning")
            self._game.undo()
            self._game.undo()
            del self._turn_records[-2:]
            return ("Undid last round (human + AI moves reverted).", "info")

        if self._game.move_count < 1:
            return ("No moves available to undo.", "warning")
        self._game.undo()
        if self._turn_records:
            self._turn_records.pop()
        return ("Undid last move.", "info")

    def run_match(self) -> MatchResultSummary:
        """Execute the full interactive match loop until terminal state or user quit.

        Returns:
            :class:`MatchResultSummary` containing final outcome, move history,
            telemetry records, and plain ASCII board snapshot.
        """
        if self._agent_x is not None:
            self._agent_x.reset()
        if self._agent_o is not None:
            self._agent_o.reset()

        status_msg = "Match started. Enter a move when prompted."
        status_kind: Literal["info", "warning", "error"] = "info"
        quit_early = False

        while not self._game.is_terminal():
            self._render_dashboard(status_message=status_msg, status_kind=status_kind)
            active_agent = self._active_agent()

            if active_agent is not None:
                if self._mode is GameMode.AI_VS_AI and self._step_by_step_ai:
                    try:
                        step_cmd = self._input_fn(
                            f"Press Enter for {active_agent.name} move (or 'q' to quit): "
                        )
                    except (EOFError, KeyboardInterrupt):
                        quit_early = True
                        break
                    if step_cmd.strip().lower() in ("q", "quit", "exit"):
                        quit_early = True
                        break
                elif self._mode is GameMode.AI_VS_AI and self._step_delay_ms > 0.0:
                    time.sleep(self._step_delay_ms / 1000.0)

                acting_player = self._game.current_player
                decision = active_agent.select_action(self._game)
                self._game.step(decision.action)
                row, col = action_to_coords(decision.action, self._game.cols)
                self._turn_records.append(
                    MatchTurnRecord(
                        turn_index=self._game.move_count,
                        player=acting_player,
                        action=decision.action,
                        row=row,
                        col=col,
                        latency_ms=decision.latency_ms,
                        agent_name=active_agent.name,
                        decision=decision,
                    )
                )
                status_msg = (
                    f"{active_agent.name} ({acting_player.symbol}) played "
                    f"({row}, {col}) [#{decision.action}] in {decision.latency_ms:.2f} ms."
                )
                status_kind = "info"
                continue

            # Human-controlled turn: keep looping until valid move or exit command.
            current_sym = self._game.current_player.symbol
            actor_name = self._actor_name(self._game.current_player)
            prompt = (
                f"[{actor_name} / {current_sym}] Enter move 'r,c' or index "
                f"[0-{self._game.num_cells - 1}] ('u'=undo, 'h'=help, 'q'=quit): "
            )
            turn_start = time.perf_counter()
            while True:
                try:
                    raw = self._input_fn(prompt)
                except (EOFError, KeyboardInterrupt):
                    quit_early = True
                    break

                try:
                    cmd = parse_user_input(raw, self._game)
                except IllegalMoveError as exc:
                    self._invalid_input_count += 1
                    status_msg = f"Invalid input: {exc}"
                    status_kind = "warning"
                    self._console.print(f"[{STYLE_WARNING}]{status_msg}[/{STYLE_WARNING}]")
                    continue

                if cmd.kind == "quit":
                    quit_early = True
                    break
                if cmd.kind == "help":
                    status_msg = _HELP_TEXT
                    status_kind = "info"
                    self._console.print(f"[{STYLE_INFO}]{status_msg}[/{STYLE_INFO}]")
                    continue
                if cmd.kind == "undo":
                    status_msg, status_kind = self._handle_undo()
                    break

                assert cmd.action is not None
                acting_player = self._game.current_player
                human_latency_ms = (time.perf_counter() - turn_start) * 1000.0
                self._game.step(cmd.action)
                row, col = action_to_coords(cmd.action, self._game.cols)
                self._turn_records.append(
                    MatchTurnRecord(
                        turn_index=self._game.move_count,
                        player=acting_player,
                        action=cmd.action,
                        row=row,
                        col=col,
                        latency_ms=human_latency_ms,
                        agent_name=actor_name,
                        decision=None,
                    )
                )
                status_msg = (
                    f"{actor_name} ({acting_player.symbol}) played "
                    f"({row}, {col}) [#{cmd.action}]."
                )
                status_kind = "info"
                break

            if quit_early:
                break

        if quit_early:
            final_msg = "Match ended early by user."
        elif self._game.winner is not None:
            final_msg = (
                f"Match Complete: {self._actor_name(self._game.winner)} "
                f"({self._game.winner.symbol}) WINS!"
            )
        else:
            final_msg = "Match Complete: DRAW!"

        self._render_dashboard(status_message=final_msg, status_kind="info")
        last_latency = self._turn_records[-1].latency_ms if self._turn_records else None
        final_ascii = format_extended_board_ascii(
            self._game,
            last_latency_ms=last_latency,
        )

        return MatchResultSummary(
            winner=self._game.winner,
            is_draw=self._game.is_terminal() and self._game.winner is None,
            quit_early=quit_early,
            move_count=self._game.move_count,
            move_history=self._game.move_history,
            winning_line=self._game.winning_line,
            turn_records=list(self._turn_records),
            final_board_text=final_ascii,
            invalid_input_count=self._invalid_input_count,
        )


def _prompt_choice(
    prompt: str,
    valid_options: tuple[str, ...],
    default: str,
    input_fn: Callable[[str], str],
    console: Console,
) -> str:
    """Prompt repeatedly until the user selects a valid option or accepts `default`."""
    while True:
        raw = input_fn(prompt).strip().lower()
        if not raw:
            return default
        if raw in valid_options:
            return raw
        console.print(
            f"[{STYLE_WARNING}]Invalid choice {raw!r}. "
            f"Choose one of {list(valid_options)}.[/{STYLE_WARNING}]"
        )


def run_setup_wizard(
    *,
    input_fn: Callable[[str], str] = input,
    console: Console | None = None,
    checkpoints_dir: str | Path = "checkpoints",
    ascii_only: bool | None = None,
) -> InteractiveMatchController:
    """Run an interactive Setup Wizard to configure board, mode, AI level, and budget.

    Args:
        input_fn: Input reader function (injectable for automated tests).
        console: Optional Rich :class:`Console`.
        checkpoints_dir: Directory searched for trained Level 5-7 checkpoints.
        ascii_only: Optional override for ASCII box drawing.

    Returns:
        Configured :class:`InteractiveMatchController` ready for :meth:`run_match`.
    """
    use_ascii = should_use_ascii() if ascii_only is None else bool(ascii_only)
    out_console = (
        console if console is not None else Console(safe_box=use_ascii, highlight=False)
    )

    out_console.print("[bold cyan]=== Extended Tic-Tac-Toe Setup Wizard ===[/bold cyan]")
    for key, (desc, _) in _BOARD_PRESETS.items():
        out_console.print(f"  [{key}] {desc}")
    out_console.print("  [5] Custom M x N, K")

    board_choice = _prompt_choice(
        "Select board [1-5, default=1]: ",
        ("1", "2", "3", "4", "5"),
        "1",
        input_fn,
        out_console,
    )
    if board_choice in _BOARD_PRESETS:
        board_cfg = _BOARD_PRESETS[board_choice][1]
    else:
        while True:
            try:
                rows = int(input_fn("Rows M (>=3, default=4): ").strip() or "4", 10)
                cols = int(input_fn("Cols N (>=3, default=4): ").strip() or "4", 10)
                win_k = int(input_fn("Win K (>=3, default=4): ").strip() or "4", 10)
                board_cfg = BoardConfig(rows=rows, cols=cols, win_k=win_k)
                break
            except (ValueError, InvalidConfigError) as exc:
                out_console.print(f"[{STYLE_WARNING}]Invalid board: {exc}[/{STYLE_WARNING}]")

    out_console.print("\n[bold]Game Mode:[/bold]")
    out_console.print("  [1] Human vs AI")
    out_console.print("  [2] AI vs AI")
    out_console.print("  [3] Human vs Human")
    mode_choice = _prompt_choice(
        "Select mode [1-3, default=1]: ",
        ("1", "2", "3"),
        "1",
        input_fn,
        out_console,
    )

    available_ckpts = resolve_default_checkpoints(board_cfg, checkpoints_dir=checkpoints_dir)
    out_console.print(
        f"\nAvailable trained checkpoints for {board_cfg.name}: "
        f"{sorted(available_ckpts.keys()) or ['none (Levels 1-4 available)']}"
    )

    budget_raw = input_fn("Time budget ms (positive number or 'none', default=50): ").strip()
    if budget_raw.lower() == "none":
        budget_ms: float | None = None
    elif not budget_raw:
        budget_ms = 50.0
    else:
        try:
            parsed_budget = float(budget_raw)
            budget_ms = parsed_budget if parsed_budget > 0 else 50.0
        except ValueError:
            budget_ms = 50.0

    seed_raw = input_fn("RNG seed (default=42): ").strip()
    try:
        seed = int(seed_raw, 10) if seed_raw else 42
    except ValueError:
        seed = 42
    if seed < 0:
        seed = 42

    game = ExtendedTicTacToe(board_cfg.rows, board_cfg.cols, board_cfg.win_k)

    def _prompt_agent(label: str, agent_seed: int) -> BaseAgent:
        out_console.print(
            f"\nSelect {label}: [easy / medium / hard] or AI Level [1..7] (default=medium):"
        )
        while True:
            raw = input_fn(f"{label} choice: ").strip().lower() or "medium"
            try:
                return create_agent_for_cli(
                    raw,
                    board_cfg,
                    seed=agent_seed,
                    time_budget_ms=budget_ms,
                    checkpoints_dir=checkpoints_dir,
                )
            except (InvalidConfigError, CheckpointVerificationError) as exc:
                out_console.print(f"[{STYLE_WARNING}]{exc}[/{STYLE_WARNING}]")

    if mode_choice == "3":
        return InteractiveMatchController(
            game=game,
            mode=GameMode.HUMAN_VS_HUMAN,
            input_fn=input_fn,
            console=out_console,
            ascii_only=use_ascii,
            budget_ms=budget_ms,
        )

    if mode_choice == "2":
        agent_x = _prompt_agent("Agent X", seed)
        agent_o = _prompt_agent("Agent O", seed + 1)
        step_mode = _prompt_choice(
            "AI vs AI pacing ([a]uto / [s]tep-by-step, default=a): ",
            ("a", "auto", "s", "step"),
            "a",
            input_fn,
            out_console,
        )
        is_step = step_mode in ("s", "step")
        return InteractiveMatchController(
            game=game,
            mode=GameMode.AI_VS_AI,
            agent_x=agent_x,
            agent_o=agent_o,
            input_fn=input_fn,
            console=out_console,
            step_delay_ms=0.0 if is_step else 400.0,
            step_by_step_ai=is_step,
            ascii_only=use_ascii,
            budget_ms=budget_ms,
        )

    side_choice = _prompt_choice(
        "Play as [X] (first) or [O] (second) [default=X]: ",
        ("x", "o"),
        "x",
        input_fn,
        out_console,
    )
    ai_agent = _prompt_agent("AI Opponent", seed)
    if side_choice == "x":
        return InteractiveMatchController(
            game=game,
            mode=GameMode.HUMAN_VS_AI,
            agent_x=None,
            agent_o=ai_agent,
            input_fn=input_fn,
            console=out_console,
            ascii_only=use_ascii,
            budget_ms=budget_ms,
        )
    return InteractiveMatchController(
        game=game,
        mode=GameMode.HUMAN_VS_AI,
        agent_x=ai_agent,
        agent_o=None,
        input_fn=input_fn,
        console=out_console,
        ascii_only=use_ascii,
        budget_ms=budget_ms,
    )
