"""CLI entrypoint to run Paired Round-Robin tournaments (Task T4.1)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ttt_ai.benchmark import TournamentRunner, load_tournament_config
from ttt_ai.config import BoardConfig, seed_everything
from ttt_ai.engine.exceptions import InvalidConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_board_override(board_tag: str) -> BoardConfig:
    """Map short CLI board tags to validated :class:`BoardConfig` instances."""
    mapping: dict[str, BoardConfig] = {
        "4x4_k4": BoardConfig(4, 4, 4),
        "4x4_k3": BoardConfig(4, 4, 3),
        "3x3_k3": BoardConfig(3, 3, 3),
        "5x5_k4": BoardConfig(5, 5, 4),
    }
    key = board_tag.strip().lower()
    if key not in mapping:
        raise InvalidConfigError(
            f"Unknown board preset {board_tag!r}. Choose from {sorted(mapping.keys())}."
        )
    return mapping[key]


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for running Paired Round-Robin tournaments."""
    parser = argparse.ArgumentParser(
        description="Run Paired Round-Robin Extended Tic-Tac-Toe AI tournaments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tournament.yaml",
        help="Path to tournament YAML configuration.",
    )
    parser.add_argument(
        "--board",
        type=str,
        default=None,
        choices=("4x4_k4", "4x4_k3", "3x3_k3", "5x5_k4"),
        help="Optional board preset override.",
    )
    parser.add_argument(
        "--budget-mode",
        type=str,
        default="default",
        help="Budget mode key from tournament YAML (default, fixed_10ms, fixed_50ms).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=None,
        help="Override number of X/O paired rounds per matchup (2*n_pairs games/pair).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override base RNG seed.",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of agent_id names from config to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSONL path. Defaults to logs/tournament_<board>_<mode>.jsonl.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite any existing JSONL file instead of resuming completed matches.",
    )
    return parser.parse_args()


def main() -> None:
    """Configure and execute the Paired Round-Robin tournament."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    cfg = load_tournament_config(args.config)

    board_cfg = _parse_board_override(args.board) if args.board is not None else cfg.board
    if args.budget_mode not in cfg.budget_modes:
        raise InvalidConfigError(
            f"Budget mode {args.budget_mode!r} not defined in '{args.config}'. "
            f"Available: {sorted(cfg.budget_modes.keys())}."
        )
    mode_cfg = cfg.budget_modes[args.budget_mode]

    n_pairs = int(args.n_pairs if args.n_pairs is not None else cfg.n_pairs)
    base_seed = int(args.seed if args.seed is not None else cfg.seed)
    seed_everything(base_seed)

    roster = cfg.resolve_agents_for_board(board_cfg)
    if args.agents is not None:
        selected_set = set(args.agents)
        roster = tuple(spec for spec in roster if spec.agent_id in selected_set)
        if len(roster) < 2:
            raise InvalidConfigError(
                f"--agents filter {args.agents!r} matched fewer than 2 agents."
            )

    output_path = (
        args.output
        if args.output is not None
        else PROJECT_ROOT / "logs" / f"tournament_{board_cfg.name}_{mode_cfg.name}.jsonl"
    )

    runner = TournamentRunner(
        board_config=board_cfg,
        agent_specs=roster,
        n_pairs=n_pairs,
        base_seed=base_seed,
        budget_mode=mode_cfg.name,
        time_budget_override_ms=mode_cfg.time_budget_ms,
        n_simulations_override=mode_cfg.n_simulations,
        base_dir=PROJECT_ROOT,
        tournament_name=f"{cfg.name}_{board_cfg.name}_{mode_cfg.name}",
    )
    final_path = runner.run(output_path=output_path, resume=not args.no_resume)
    logging.info("Tournament JSONL log ready at: %s", final_path)


if __name__ == "__main__":
    main()
