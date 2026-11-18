"""CLI entrypoint to play Extended Tic-Tac-Toe interactively via Rich CLI."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from ttt_ai.config import BoardConfig
from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.ui import (
    GameMode,
    InteractiveCLIApp,
    create_agent_for_cli,
    run_setup_wizard,
    should_use_ascii,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_board_tag(tag: str) -> BoardConfig:
    """Parse a board specification string such as `'4x4_k4'` or `'3x3_k3'`."""
    cleaned = tag.strip().lower()
    match = re.fullmatch(r"(\d+)x(\d+)_k(\d+)", cleaned)
    if match is None:
        raise InvalidConfigError(
            f"Invalid --board format {tag!r}. Expected '<rows>x<cols>_k<win_k>', e.g. '4x4_k4'."
        )
    return BoardConfig(
        rows=int(match.group(1), 10),
        cols=int(match.group(2), 10),
        win_k=int(match.group(3), 10),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for launching interactive or automated demo matches."""
    parser = argparse.ArgumentParser(
        description="Launch Extended Tic-Tac-Toe Rich CLI (Wizard or Direct Match)."
    )
    parser.add_argument(
        "--board",
        type=str,
        default=None,
        help="Board tag '<M>x<N>_k<K>' (e.g. '4x4_k4', '4x4_k3', '3x3_k3', '5x5_k4').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=("human_vs_ai", "ai_vs_ai", "human_vs_human"),
        help="Match mode. When omitted along with AI flags, runs the Setup Wizard.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="AI opponent difficulty ('easy', 'medium', 'hard') or level ('1'..'7').",
    )
    parser.add_argument(
        "--agent-x",
        "--level-x",
        dest="agent_x",
        type=str,
        default=None,
        help="AI difficulty ('easy'/'medium'/'hard') or level ('1'..'7') for Player X.",
    )
    parser.add_argument(
        "--agent-o",
        "--level-o",
        dest="agent_o",
        type=str,
        default=None,
        help="AI difficulty ('easy'/'medium'/'hard') or level ('1'..'7') for Player O.",
    )
    parser.add_argument(
        "--human-player",
        type=str,
        default=None,
        choices=("X", "O", "x", "o"),
        help="Side controlled by human in human_vs_ai mode (default: X, or O if --agent-x set).",
    )
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=50.0,
        help="Per-move time budget in milliseconds (<=0 disables budget; default: 50.0).",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=200,
        help="Simulation count for MCTS / HybridMCTS agents (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for AI agents (default: 42).",
    )
    parser.add_argument(
        "--step-delay",
        "--step-delay-ms",
        dest="step_delay_ms",
        type=float,
        default=600.0,
        help="Auto-play delay between moves in ai_vs_ai mode in ms (default: 600.0 ms).",
    )
    parser.add_argument(
        "--step-by-step",
        action="store_true",
        help="Wait for Enter before each AI move in ai_vs_ai mode.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
        help="Directory containing trained checkpoints for Levels 5-7.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Force ASCII-only box drawing for legacy Windows terminals.",
    )
    return parser.parse_args(argv)


def build_controller_from_args(
    args: argparse.Namespace,
    *,
    console: Console,
    ascii_only: bool,
) -> InteractiveCLIApp:
    """Instantiate an :class:`InteractiveCLIApp` from parsed direct-mode CLI flags."""
    board_cfg = (
        _parse_board_tag(args.board)
        if args.board is not None
        else BoardConfig(rows=4, cols=4, win_k=4)
    )
    budget_ms = float(args.budget_ms) if args.budget_ms > 0.0 else None

    if args.mode is not None:
        mode = GameMode(args.mode)
    elif args.agent_x is not None and args.agent_o is not None:
        mode = GameMode.AI_VS_AI
    else:
        mode = GameMode.HUMAN_VS_AI

    game = ExtendedTicTacToe(board_cfg.rows, board_cfg.cols, board_cfg.win_k)
    agent_x = None
    agent_o = None

    if mode is GameMode.AI_VS_AI:
        spec_x = args.agent_x or args.difficulty or "2"
        spec_o = args.agent_o or args.difficulty or "3"
        agent_x = create_agent_for_cli(
            spec_x,
            board_cfg,
            seed=args.seed,
            time_budget_ms=budget_ms,
            n_simulations=args.n_simulations,
            checkpoints_dir=args.checkpoints_dir,
        )
        agent_o = create_agent_for_cli(
            spec_o,
            board_cfg,
            seed=args.seed + 1,
            time_budget_ms=budget_ms,
            n_simulations=args.n_simulations,
            checkpoints_dir=args.checkpoints_dir,
        )
    elif mode is GameMode.HUMAN_VS_AI:
        if args.human_player is not None:
            human_side = args.human_player.upper()
        elif args.agent_x is not None and args.agent_o is None:
            human_side = "O"
        else:
            human_side = "X"

        ai_spec = (
            args.difficulty
            or (args.agent_o if human_side == "X" else args.agent_x)
            or "medium"
        )
        ai_agent = create_agent_for_cli(
            ai_spec,
            board_cfg,
            seed=args.seed,
            time_budget_ms=budget_ms,
            n_simulations=args.n_simulations,
            checkpoints_dir=args.checkpoints_dir,
        )
        if human_side == "X":
            agent_o = ai_agent
        else:
            agent_x = ai_agent

    return InteractiveCLIApp(
        game=game,
        mode=mode,
        agent_x=agent_x,
        agent_o=agent_o,
        console=console,
        step_delay_ms=0.0 if args.step_by_step else args.step_delay_ms,
        step_by_step_ai=args.step_by_step,
        ascii_only=ascii_only,
        budget_ms=budget_ms,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run Setup Wizard or direct match controller based on CLI arguments."""
    args = _parse_args(argv)
    ascii_only = should_use_ascii(force_ascii=args.ascii)
    console = Console(safe_box=ascii_only, highlight=False)

    direct_mode = any(
        val is not None
        for val in (args.mode, args.board, args.difficulty, args.agent_x, args.agent_o)
    )
    if not direct_mode:
        controller = run_setup_wizard(
            console=console,
            checkpoints_dir=args.checkpoints_dir,
            ascii_only=ascii_only,
        )
        controller.run_match()
        return

    controller = build_controller_from_args(args, console=console, ascii_only=ascii_only)
    controller.run_match()


if __name__ == "__main__":
    main()
