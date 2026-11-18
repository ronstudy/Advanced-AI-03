"""Unit, integration, and artifact verification tests for Task T4.4 Ablation Studies."""

from __future__ import annotations

import ast
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from ttt_ai import (
    AblationRunResult,
    load_ablation_result,
    load_tournament_log,
    run_ablation_alpha_beta,
    run_all_ablations,
)
from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.engine.rules import PLAYER_O, PLAYER_X

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ABLATIONS_SCRIPT = PROJECT_ROOT / "scripts" / "run_ablations.py"
ABLATIONS_MODULE_PATH = PROJECT_ROOT / "src" / "ttt_ai" / "benchmark" / "ablations.py"


def _load_run_ablations_module() -> Any:
    """Dynamically load `scripts/run_ablations.py` for CLI integration testing."""
    spec = importlib.util.spec_from_file_location("run_ablations_cli", RUN_ABLATIONS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestAblationRunnersSmokeAndValidation:
    """Verify end-to-end smoke execution, JSONL schemas, CLI, and validation."""

    def test_run_all_ablations_smoke_and_reports(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        reports_dir = tmp_path / "reports"
        results = run_all_ablations(
            mode="smoke",
            seed=42,
            output_dir=logs_dir,
            reports_dir=reports_dir,
            base_dir=PROJECT_ROOT,
        )
        assert [r.ablation_id for r in results] == [
            "ablation_1",
            "ablation_2",
            "ablation_3",
            "ablation_4",
        ]
        expected_counts = {
            "ablation_1": 20,
            "ablation_2": 184,
            "ablation_3": 104,
            "ablation_4": 180,
        }
        for res in results:
            assert isinstance(res, AblationRunResult)
            assert res.mode == "smoke"
            assert res.log_path.is_file()
            assert res.total_matches == expected_counts[res.ablation_id]
            assert res.to_dict()["total_matches"] == expected_counts[res.ablation_id]

            meta, matches = load_tournament_log(res.log_path)
            assert len(matches) == expected_counts[res.ablation_id]
            assert meta["ablation_id"] == res.ablation_id
            assert "hardware" in meta
            reloaded = load_ablation_result(res.log_path)
            assert reloaded.ablation_id == res.ablation_id
            assert reloaded.total_matches == len(matches)
            for rec in matches:
                assert rec.winner in (PLAYER_X, PLAYER_O, 0)
                assert rec.score_x + rec.score_o == pytest.approx(1.0)
                assert len(rec.moves) == rec.num_moves

        # Ablation 1 assertions: exact root values and pruning savings.
        ab1 = results[0].summary
        assert ab1["probe_by_depth"]["d3"]["all_exact_value_match"] is True
        assert ab1["probe_by_depth"]["d3"]["ab_order_reduction_pct"] > 50.0

        # Ablation 2 (RQ1) assertions: 18 candidates & matched direct H2H rows.
        ab2 = results[1].summary
        assert len(ab2["candidate_metrics"]) == 18
        assert len(ab2["direct_head_to_head"]) == 20

        # Ablation 3 (RQ2) assertions: 3 boards separated cleanly.
        ab3 = results[2].summary["boards"]
        assert set(ab3.keys()) == {"3x3_k3", "4x4_k3", "4x4_k4"}

        # Ablation 4 (RQ3) assertions: 3 seeds reported on both boards.
        ab4 = results[3].summary["boards"]
        assert set(ab4.keys()) == {"4x4_k4", "3x3_k3"}
        assert ab4["4x4_k4"]["rl_seed_stability"]["seeds"] == [42, 123, 999]
        assert ab4["3x3_k3"]["rl_seed_stability"]["seeds"] == [42, 123, 999]

        csv_path = reports_dir / "ablation_metrics.csv"
        md_path = reports_dir / "ablation_summary.md"
        assert csv_path.is_file()
        assert md_path.is_file()
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) > 30
        md_text = md_path.read_text(encoding="utf-8")
        assert "ABLATION_1" in md_text
        assert "ABLATION_4" in md_text

    def test_invalid_arguments_and_cli(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidConfigError):
            run_ablation_alpha_beta(mode="fast", output_path=tmp_path / "x.jsonl")  # type: ignore[arg-type]
        with pytest.raises(InvalidConfigError):
            run_ablation_alpha_beta(mode="smoke", seed=-1, output_path=tmp_path / "x.jsonl")
        with pytest.raises(InvalidConfigError):
            run_ablation_alpha_beta(
                mode="smoke",
                seed=True,
                output_path=tmp_path / "x.jsonl",
            )

        cli = _load_run_ablations_module()
        out_dir = tmp_path / "cli_logs"
        rep_dir = tmp_path / "cli_reports"
        args = cli.parse_args(
            [
                "--ablation",
                "1",
                "--mode",
                "smoke",
                "--seed",
                "42",
                "--output-dir",
                str(out_dir),
                "--reports-dir",
                str(rep_dir),
            ]
        )
        res_list = cli.run_from_args(args)
        assert len(res_list) == 1
        assert res_list[0].ablation_id == "ablation_1"
        assert (out_dir / "ablation_ab_pruning.jsonl").is_file()
        assert (rep_dir / "ablation_metrics.csv").is_file()
        assert (rep_dir / "ablation_summary.md").is_file()

        no_rep_args = cli.parse_args(["--ablation", "3", "--mode", "smoke", "--no-reports"])
        assert no_rep_args.no_reports is True

        with pytest.raises(InvalidConfigError):
            bad_args = cli.parse_args(["--ablation", "1", "--mode", "smoke", "--seed", "-2"])
            cli.run_from_args(bad_args)


class TestOfficialAblationLogsT4_4:
    """Verify completed full-scale T4.4 ablation JSONL logs and reports."""

    @pytest.mark.parametrize(
        ("filename", "expected_id", "expected_matches"),
        [
            ("ablation_ab_pruning.jsonl", "ablation_1", 316),
            ("ablation_rq1_hybrid_mcts.jsonl", "ablation_2", 2760),
            ("ablation_rq2_seat_advantage.jsonl", "ablation_3", 2080),
            ("ablation_rq3_rl_generalization.jsonl", "ablation_4", 3600),
        ],
    )
    def test_official_ablation_jsonl_exists_and_loads_cleanly(
        self,
        filename: str,
        expected_id: str,
        expected_matches: int,
    ) -> None:
        log_path = PROJECT_ROOT / "logs" / filename
        assert log_path.is_file(), f"Missing official ablation log: {log_path}"

        res = load_ablation_result(log_path)
        assert res.ablation_id == expected_id
        assert res.mode == "full"
        assert res.total_matches == expected_matches
        assert res.summary

        meta, matches = load_tournament_log(log_path)
        assert len(matches) == expected_matches
        assert len({m.match_id for m in matches}) == expected_matches
        assert "hardware" in meta

    def test_official_ablation_reports_exist(self) -> None:
        csv_path = PROJECT_ROOT / "reports" / "ablation_metrics.csv"
        md_path = PROJECT_ROOT / "reports" / "ablation_summary.md"
        assert csv_path.is_file()
        assert md_path.is_file()
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 55
        md_text = md_path.read_text(encoding="utf-8")
        for header in ("ABLATION_1", "ABLATION_2", "ABLATION_3", "ABLATION_4"):
            assert header in md_text


class TestAblationsRulesCompliance:
    """AST verification enforcing Planning/RULES.md isolation rules."""

    def test_ablations_module_has_no_print_or_private_engine_access(self) -> None:
        tree = ast.parse(ABLATIONS_MODULE_PATH.read_text(encoding="utf-8"))
        forbidden = {
            "_board",
            "_history",
            "_current_player",
            "_terminal",
            "_winner",
            "_winning_line",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", "Direct print() forbidden in ablations.py"
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden, (
                    f"Forbidden private engine attribute {node.attr} in ablations.py"
                )
