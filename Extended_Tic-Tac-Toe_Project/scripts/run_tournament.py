"""CLI entrypoint to run, audit, and summarize Paired Round-Robin tournaments."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ttt_ai.benchmark import (
    DEFAULT_OFFICIAL_TOURNAMENT_TARGETS,
    TournamentIntegrityReport,
    TournamentRunner,
    export_summary_csv,
    format_summary_markdown,
    load_tournament_config,
    parse_board_preset,
    run_official_tournament_suite,
    save_summary_markdown,
    summarize_tournament_log,
    verify_tournament_log_integrity,
)
from ttt_ai.benchmark.tournament import (
    _export_suite_consolidated_csv,
    _export_suite_consolidated_markdown,
)
from ttt_ai.config import seed_everything
from ttt_ai.engine.exceptions import InvalidConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for running and auditing Paired Round-Robin tournaments.

    Args:
        argv: Optional sequence of CLI tokens (defaults to ``sys.argv[1:]``).

    Returns:
        Populated :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Run, audit, and summarize Paired Round-Robin Extended Tic-Tac-Toe tournaments."
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
        help="Optional board preset override for single-tournament mode.",
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
        help="Optional subset of agent_id names from config to include (single mode).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Destination JSONL path for single mode. "
            "Defaults to logs/tournament_<board>_<mode>.jsonl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "logs",
        help="Output directory for JSONL logs when using --suite.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Directory for generated CSV and Markdown summary reports.",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip exporting CSV and Markdown summary reports.",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Execute the full Task T4.3 official tournament suite (4 tournaments).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify integrity and regenerate reports for existing log(s) "
            "without executing matches."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite any existing JSONL file instead of resuming completed matches.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _verify_suite_logs(
    args: argparse.Namespace,
    cfg: Any,
    n_pairs: int,
    base_seed: int,
    reports_dir: Path | None,
) -> list[Path]:
    """Audit all existing T4.3 suite JSONL logs and optionally export consolidated reports."""
    output_dir = Path(args.output_dir)
    verified_paths: list[Path] = []
    ordered_runs: list[tuple[str, str, Any]] = []
    integrity_reports: dict[str, TournamentIntegrityReport] = {}

    for board_tag, mode_name in DEFAULT_OFFICIAL_TOURNAMENT_TARGETS:
        board_cfg = parse_board_preset(board_tag)
        if mode_name not in cfg.budget_modes:
            raise InvalidConfigError(
                f"Budget mode {mode_name!r} not defined in '{args.config}'."
            )
        roster = cfg.resolve_agents_for_board(board_cfg)
        key = f"{board_cfg.name}_{mode_name}"
        log_path = output_dir / f"tournament_{key}.jsonl"
        audit = verify_tournament_log_integrity(
            log_path,
            expected_agents=[spec.agent_id for spec in roster],
            expected_n_pairs=n_pairs,
            replay_moves=True,
        )
        if not audit.is_valid:
            raise InvalidConfigError(
                f"Integrity verification failed for '{log_path}': "
                f"total={audit.total_matches}/{audit.expected_matches}, "
                f"duplicates={audit.duplicate_match_ids}, "
                f"unbalanced={audit.unbalanced_matchups}, "
                f"illegal={audit.corrupt_or_illegal_matches[:5]}."
            )
        verified_paths.append(log_path)
        integrity_reports[key] = audit

        summary = summarize_tournament_log(log_path, seed=base_seed)
        ordered_runs.append((board_cfg.name, mode_name, summary))
        if reports_dir is not None:
            export_summary_csv(summary, reports_dir / f"tournament_{key}_metrics.csv")
            save_summary_markdown(summary, reports_dir / f"tournament_{key}_summary.md")

    if reports_dir is not None:
        _export_suite_consolidated_csv(
            ordered_runs,
            reports_dir / "tournament_suite_metrics.csv",
        )
        _export_suite_consolidated_markdown(
            ordered_runs,
            integrity_reports,
            format_summary_markdown,
            reports_dir / "tournament_suite_summary.md",
        )
    return verified_paths


def run_from_args(args: argparse.Namespace) -> list[Path]:
    """Execute or verify tournament(s) from parsed CLI arguments.

    Args:
        args: Parsed :class:`argparse.Namespace` returned by :func:`parse_args`.

    Returns:
        List of verified JSONL log paths processed during execution.

    Raises:
        InvalidConfigError: If CLI flags conflict or log verification fails.
    """
    cfg = load_tournament_config(args.config)
    n_pairs = int(args.n_pairs if args.n_pairs is not None else cfg.n_pairs)
    base_seed = int(args.seed if args.seed is not None else cfg.seed)
    if n_pairs < 1:
        raise InvalidConfigError(f"--n-pairs must be >= 1, got {n_pairs!r}.")
    if base_seed < 0:
        raise InvalidConfigError(f"--seed must be >= 0, got {base_seed!r}.")

    reports_dir: Path | None = None if args.no_reports else Path(args.reports_dir)

    if args.suite:
        if args.agents is not None:
            raise InvalidConfigError("--agents cannot be combined with --suite.")
        if args.verify_only:
            return _verify_suite_logs(args, cfg, n_pairs, base_seed, reports_dir)

        suite_res = run_official_tournament_suite(
            config_path=args.config,
            output_dir=args.output_dir,
            reports_dir=reports_dir,
            base_dir=PROJECT_ROOT,
            n_pairs_override=n_pairs,
            seed_override=base_seed,
            resume=not args.no_resume,
        )
        return list(suite_res.log_paths.values())

    board_cfg = parse_board_preset(args.board) if args.board is not None else cfg.board
    if args.budget_mode not in cfg.budget_modes:
        raise InvalidConfigError(
            f"Budget mode {args.budget_mode!r} not defined in '{args.config}'. "
            f"Available: {sorted(cfg.budget_modes.keys())}."
        )
    mode_cfg = cfg.budget_modes[args.budget_mode]
    seed_everything(base_seed)

    roster = cfg.resolve_agents_for_board(board_cfg)
    if args.agents is not None:
        selected_set = set(args.agents)
        roster = tuple(spec for spec in roster if spec.agent_id in selected_set)
        if len(roster) < 2:
            raise InvalidConfigError(
                f"--agents filter {args.agents!r} matched fewer than 2 agents."
            )

    key = f"{board_cfg.name}_{mode_cfg.name}"
    output_path = (
        args.output
        if args.output is not None
        else Path(args.output_dir) / f"tournament_{key}.jsonl"
    )

    if not args.verify_only:
        runner = TournamentRunner(
            board_config=board_cfg,
            agent_specs=roster,
            n_pairs=n_pairs,
            base_seed=base_seed,
            budget_mode=mode_cfg.name,
            time_budget_override_ms=mode_cfg.time_budget_ms,
            n_simulations_override=mode_cfg.n_simulations,
            base_dir=PROJECT_ROOT,
            tournament_name=f"{cfg.name}_{key}",
        )
        final_path = runner.run(output_path=output_path, resume=not args.no_resume)
    else:
        final_path = Path(output_path)

    audit = verify_tournament_log_integrity(
        final_path,
        expected_agents=[spec.agent_id for spec in roster],
        expected_n_pairs=n_pairs if not args.verify_only or args.n_pairs is not None else None,
        replay_moves=True,
    )
    if not audit.is_valid:
        raise InvalidConfigError(
            f"Tournament integrity check failed for '{final_path}': "
            f"total={audit.total_matches}/{audit.expected_matches}, "
            f"duplicates={audit.duplicate_match_ids}, "
            f"unbalanced={audit.unbalanced_matchups}, "
            f"illegal={audit.corrupt_or_illegal_matches[:5]}."
        )

    if reports_dir is not None:
        summary = summarize_tournament_log(final_path, seed=base_seed)
        export_summary_csv(summary, reports_dir / f"tournament_{key}_metrics.csv")
        save_summary_markdown(summary, reports_dir / f"tournament_{key}_summary.md")

    logging.info(
        "Tournament JSONL verified (%d matches, 0 illegal moves): %s",
        audit.total_matches,
        final_path,
    )
    return [final_path]


def main() -> None:
    """Configure logging and execute the tournament CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
