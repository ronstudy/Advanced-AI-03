"""CLI entrypoint for Task T4.4 Ablation Studies (AB Pruning & RQ1, RQ2, RQ3)."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from ttt_ai.benchmark.ablations import (
    _DEFAULT_FILENAMES,
    AblationRunResult,
    export_ablation_suite_reports,
    run_ablation_alpha_beta,
    run_ablation_rq1_hybrid_mcts,
    run_ablation_rq2_seat_advantage,
    run_ablation_rq3_generalization,
    run_all_ablations,
)
from ttt_ai.engine.exceptions import InvalidConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for running ablation studies.

    Args:
        argv: Optional list of CLI tokens (defaults to ``sys.argv[1:]``).

    Returns:
        Populated :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Run Extended Tic-Tac-Toe scientific ablations (Task T4.4)."
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=("1", "2", "3", "4", "all"),
        help="Ablation scenario selector: 1 (AB), 2 (RQ1), 3 (RQ2), 4 (RQ3), or all.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=("smoke", "full"),
        help="Execution scale: 'smoke' (quick CI sanity check) or 'full' (official data).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base non-negative RNG seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "logs",
        help="Directory where ablation_*.jsonl logs are saved.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Directory where ablation_metrics.csv and ablation_summary.md are written.",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip generating CSV and Markdown summary reports.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def run_from_args(args: argparse.Namespace) -> list[AblationRunResult]:
    """Execute requested ablation scenario(s) and optionally export reports.

    Args:
        args: Parsed :class:`argparse.Namespace` returned by :func:`parse_args`.

    Returns:
        List of :class:`AblationRunResult` objects produced.

    Raises:
        InvalidConfigError: If CLI options are invalid.
    """
    if isinstance(args.seed, bool) or int(args.seed) < 0:
        raise InvalidConfigError(f"--seed must be a non-negative integer, got {args.seed!r}.")

    seed = int(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir: Path | None = None if args.no_reports else Path(args.reports_dir)

    selector = str(args.ablation)
    if selector == "all":
        results = run_all_ablations(
            mode=args.mode,
            seed=seed,
            output_dir=out_dir,
            reports_dir=rep_dir,
            base_dir=PROJECT_ROOT,
        )
        return results

    runners = {
        "1": ("ablation_1", run_ablation_alpha_beta),
        "2": ("ablation_2", run_ablation_rq1_hybrid_mcts),
        "3": ("ablation_3", run_ablation_rq2_seat_advantage),
        "4": ("ablation_4", run_ablation_rq3_generalization),
    }
    ablation_id, fn = runners[selector]
    res = fn(
        mode=args.mode,
        seed=seed,
        output_path=out_dir / _DEFAULT_FILENAMES[ablation_id],
        base_dir=PROJECT_ROOT,
    )
    results = [res]
    if rep_dir is not None:
        export_ablation_suite_reports(results, rep_dir)
    return results


def main() -> None:
    """Configure logging and execute the Task T4.4 ablation CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    results = run_from_args(args)
    for res in results:
        logging.info(
            "Ablation %s [%s]: %d matches logged -> %s",
            res.ablation_id,
            res.mode,
            res.total_matches,
            res.log_path,
        )


if __name__ == "__main__":
    main()
