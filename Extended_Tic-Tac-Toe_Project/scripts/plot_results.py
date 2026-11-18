"""CLI entrypoint to generate Fig 1..5 PNG plots (300 DPI) and empirical reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from ttt_ai.benchmark.visualizer import generate_all_visualizations_and_reports
from ttt_ai.ui import should_use_ascii

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for plotting benchmark figures and exporting reports."""
    parser = argparse.ArgumentParser(
        description="Generate Fig 1..5 (300 DPI PNG) and Markdown/CSV reports from raw logs."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=PROJECT_ROOT / "logs",
        help="Directory containing raw tournament, ablation, and training JSONL logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Root output directory for figures/ and summary reports.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
        help="Directory containing trained checkpoints for failure-case prior inspection.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        choices=("1", "2", "3", "4", "5", "all"),
        help="Figure IDs to render ('1'..'5' or 'all', default: all).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output DPI for PNG figures (default: 300).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Execute figure generation and report export pipeline."""
    args = _parse_args(argv)
    ascii_only = should_use_ascii()
    console = Console(safe_box=ascii_only, highlight=False)

    artifacts = generate_all_visualizations_and_reports(
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
        checkpoints_dir=args.checkpoints_dir,
        figures=args.figures,
        dpi=args.dpi,
    )
    console.print(
        f"[bold green]Exported {len(artifacts)} visualization & report artifacts:[/bold green]"
    )
    for key, path in sorted(artifacts.items()):
        console.print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
