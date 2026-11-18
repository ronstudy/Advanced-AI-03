"""Unit and integration tests for Tasks T4.1 & T4.3 Paired Round-Robin tournaments."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ttt_ai import (
    DEFAULT_OFFICIAL_TOURNAMENT_TARGETS,
    AgentSpec,
    BoardConfig,
    TournamentIntegrityReport,
    TournamentRunner,
    load_tournament_config,
    load_tournament_log,
    parse_board_preset,
    run_official_tournament_suite,
    verify_tournament_log_integrity,
)
from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.engine.rules import PLAYER_O, PLAYER_X

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOURNAMENT_YAML = PROJECT_ROOT / "configs" / "tournament.yaml"
RUN_TOURNAMENT_SCRIPT = PROJECT_ROOT / "scripts" / "run_tournament.py"


def _load_run_tournament_module() -> Any:
    """Dynamically load `scripts/run_tournament.py` for CLI integration tests."""
    spec = importlib.util.spec_from_file_location("run_tournament_cli", RUN_TOURNAMENT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fast_specs_4x4() -> list[AgentSpec]:
    """Return a fast 3-agent roster suitable for deterministic unit tests."""
    return [
        AgentSpec(agent_id="L1_Random", agent_class="RandomAgent", level=1),
        AgentSpec(
            agent_id="L2_RuleBased",
            agent_class="RuleBasedAgent",
            level=2,
            params={"tie_break": "random"},
        ),
        AgentSpec(
            agent_id="L3_AlphaBeta",
            agent_class="AlphaBetaAgent",
            level=3,
            params={"max_depth": 2, "use_alpha_beta": True, "use_move_ordering": True},
        ),
    ]


class TestTournamentConfigAndMetadata:
    """Verify YAML loading, validation, and environment metadata recording."""

    def test_loads_configs_tournament_yaml_and_rosters(self) -> None:
        cfg = load_tournament_config(TOURNAMENT_YAML)
        assert cfg.board == BoardConfig(4, 4, 4)
        assert cfg.n_pairs == 100
        assert "default" in cfg.budget_modes
        assert "fixed_10ms" in cfg.budget_modes
        assert "fixed_50ms" in cfg.budget_modes
        assert cfg.budget_modes["fixed_10ms"].time_budget_ms == 10.0
        assert cfg.budget_modes["fixed_50ms"].time_budget_ms == 50.0

        roster_4x4 = cfg.resolve_agents_for_board(BoardConfig(4, 4, 4))
        assert {spec.level for spec in roster_4x4} == {1, 2, 3, 4, 5, 6, 7}
        roster_3x3 = cfg.resolve_agents_for_board(BoardConfig(3, 3, 3))
        assert {spec.level for spec in roster_3x3} == {1, 2, 3, 4, 5, 6, 7}

    def test_benchmark_modules_have_no_print_or_private_engine_access(self) -> None:
        bmk_dir = PROJECT_ROOT / "src" / "ttt_ai" / "benchmark"
        forbidden = {
            "_board",
            "_history",
            "_current_player",
            "_terminal",
            "_winner",
            "_winning_line",
        }
        for path in sorted(bmk_dir.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "print", f"print() forbidden in {path}"
                if isinstance(node, ast.Attribute):
                    assert node.attr not in forbidden, (
                        f"Forbidden private attribute {node.attr} in {path}"
                    )


class TestTournamentRunnerPairedRoundRobin:
    """Verify 50/50 X/O pairing, JSONL schema, reproducibility, and resume."""

    def test_paired_round_robin_balance_schema_and_reproducibility(
        self,
        tmp_path: Path,
    ) -> None:
        board_cfg = BoardConfig(3, 3, 3)
        specs = _fast_specs_4x4()
        runner_a = TournamentRunner(
            board_config=board_cfg,
            agent_specs=specs,
            n_pairs=4,
            base_seed=100,
            budget_mode="fixed_50ms",
            time_budget_override_ms=50.0,
            tournament_name="unit_test_rr",
        )
        runner_b = TournamentRunner(
            board_config=board_cfg,
            agent_specs=specs,
            n_pairs=4,
            base_seed=100,
            budget_mode="fixed_50ms",
            time_budget_override_ms=50.0,
            tournament_name="unit_test_rr",
        )
        assert runner_a.total_matches == 24  # 3 unordered pairs * 4 pairs * 2 games

        log_a = runner_a.run(tmp_path / "run_a.jsonl", resume=False)
        log_b = runner_b.run(tmp_path / "run_b.jsonl", resume=False)

        meta_a, matches_a = load_tournament_log(log_a)
        meta_b, matches_b = load_tournament_log(log_b)

        assert meta_a["type"] == "meta"
        assert meta_a["board_config"] == {"rows": 3, "cols": 3, "win_k": 3}
        assert meta_a["budget_mode"] == "fixed_50ms"
        assert meta_a["games_per_matchup"] == 8
        assert "hardware" in meta_a
        assert set(meta_a["agents"].keys()) == {"L1_Random", "L2_RuleBased", "L3_AlphaBeta"}
        for agent_cfg in meta_a["agents"].values():
            assert agent_cfg["time_budget_ms"] == 50.0

        assert len(matches_a) == 24
        assert len(matches_b) == 24

        # Check exact 50% X / 50% O balance per agent across all matchups.
        x_counts = {spec.agent_id: 0 for spec in specs}
        o_counts = {spec.agent_id: 0 for spec in specs}
        for rec in matches_a:
            x_counts[rec.agent_x] += 1
            o_counts[rec.agent_o] += 1
            assert rec.winner in (PLAYER_X, PLAYER_O, 0)
            assert rec.score_x + rec.score_o == pytest.approx(1.0)
            assert len(rec.moves) == rec.num_moves
            assert len(rec.latencies_x_ms) + len(rec.latencies_o_ms) == rec.num_moves
            assert len(rec.nodes_x) == len(rec.latencies_x_ms)
            assert len(rec.nodes_o) == len(rec.latencies_o_ms)

        for spec in specs:
            assert x_counts[spec.agent_id] == 8
            assert o_counts[spec.agent_id] == 8

        # Pair games g0 and g1 share the exact same seed and swap colors.
        for idx in range(0, len(matches_a), 2):
            g0, g1 = matches_a[idx], matches_a[idx + 1]
            assert g0.pair_id == g1.pair_id
            assert g0.seed == g1.seed
            assert g0.agent_x == g1.agent_o
            assert g0.agent_o == g1.agent_x

        # Exact move-by-move reproducibility for identical seeds.
        for rec_a, rec_b in zip(matches_a, matches_b, strict=True):
            assert rec_a.match_id == rec_b.match_id
            assert rec_a.winner == rec_b.winner
            assert rec_a.score_x == rec_b.score_x
            assert [mv[0] for mv in rec_a.moves] == [mv[0] for mv in rec_b.moves]

    def test_resume_skips_completed_matches_and_rejects_mismatched_meta(
        self,
        tmp_path: Path,
    ) -> None:
        board_cfg = BoardConfig(4, 4, 4)
        specs = _fast_specs_4x4()
        runner = TournamentRunner(
            board_config=board_cfg,
            agent_specs=specs,
            n_pairs=3,
            base_seed=55,
            budget_mode="default",
        )
        out_path = tmp_path / "resumable.jsonl"
        runner.run(out_path, resume=False)

        all_lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(all_lines) == 1 + 18  # 1 meta + (3 pairs * 3 rounds * 2 games)

        # Simulate interruption after 7 matches (meta + 7 match lines).
        out_path.write_text("\n".join(all_lines[:8]) + "\n", encoding="utf-8")
        runner.run(out_path, resume=True)

        meta, resumed_matches = load_tournament_log(out_path)
        assert meta["total_scheduled_matches"] == 18
        assert len(resumed_matches) == 18
        assert len({rec.match_id for rec in resumed_matches}) == 18

        # Check exact match against uninterrupted output move actions.
        expected_actions = [json.loads(line)["moves"] for line in all_lines[1:]]
        actual_actions = [rec.moves for rec in resumed_matches]
        assert [[mv[0] for mv in m] for m in actual_actions] == [
            [mv[0] for mv in m] for m in expected_actions
        ]

        # Mismatched board or budget_mode must raise InvalidConfigError on resume.
        wrong_board_runner = TournamentRunner(
            board_config=BoardConfig(3, 3, 3),
            agent_specs=specs,
            n_pairs=3,
            base_seed=55,
            budget_mode="default",
        )
        with pytest.raises(InvalidConfigError):
            wrong_board_runner.run(out_path, resume=True)

        wrong_mode_runner = TournamentRunner(
            board_config=board_cfg,
            agent_specs=specs,
            n_pairs=3,
            base_seed=55,
            budget_mode="fixed_10ms",
            time_budget_override_ms=10.0,
        )
        with pytest.raises(InvalidConfigError):
            wrong_mode_runner.run(out_path, resume=True)

    def test_7_level_smoke_tournament_with_real_checkpoints(self, tmp_path: Path) -> None:
        cfg = load_tournament_config(TOURNAMENT_YAML)
        board_cfg = cfg.board
        roster = cfg.resolve_agents_for_board(board_cfg)

        runner = TournamentRunner(
            board_config=board_cfg,
            agent_specs=roster,
            n_pairs=1,
            base_seed=42,
            budget_mode="fixed_10ms",
            time_budget_override_ms=10.0,
            n_simulations_override=15,
            base_dir=PROJECT_ROOT,
            tournament_name="smoke_7lvl_4x4",
        )
        assert runner.total_matches == 42  # C(7, 2) * 1 pair * 2 games
        out_path = runner.run(tmp_path / "smoke_7lvl.jsonl", resume=False)
        meta, matches = load_tournament_log(out_path)
        assert len(matches) == 42
        assert len(meta["agents"]) == 7
        for rec in matches:
            assert rec.num_moves >= 7
            assert rec.winner in (PLAYER_X, PLAYER_O, 0)


class TestIntegrityVerificationAndSuite:
    """Verify log auditing, corruption detection, suite runner, and CLI flags."""

    def test_verify_integrity_passes_and_detects_corruptions(self, tmp_path: Path) -> None:
        board_cfg = BoardConfig(3, 3, 3)
        specs = _fast_specs_4x4()
        runner = TournamentRunner(
            board_config=board_cfg,
            agent_specs=specs,
            n_pairs=2,
            base_seed=77,
            budget_mode="default",
            tournament_name="integrity_test",
        )
        valid_path = runner.run(tmp_path / "valid.jsonl", resume=False)

        report = verify_tournament_log_integrity(
            valid_path,
            expected_agents=[s.agent_id for s in specs],
            expected_n_pairs=2,
            replay_moves=True,
        )
        assert isinstance(report, TournamentIntegrityReport)
        assert report.is_valid is True
        assert report.total_matches == 12
        assert report.expected_matches == 12
        assert not report.duplicate_match_ids
        assert not report.unbalanced_matchups
        assert not report.corrupt_or_illegal_matches
        assert report.to_dict()["is_valid"] is True

        raw_lines = valid_path.read_text(encoding="utf-8").splitlines()

        # 1. Duplicate match ID.
        dup_path = tmp_path / "dup.jsonl"
        dup_path.write_text("\n".join([*raw_lines, raw_lines[1]]) + "\n", encoding="utf-8")
        dup_report = verify_tournament_log_integrity(dup_path, replay_moves=False)
        assert dup_report.is_valid is False
        assert len(dup_report.duplicate_match_ids) == 1

        # 2. Missing game (unbalanced X/O pair) and roster mismatch.
        missing_path = tmp_path / "missing.jsonl"
        missing_path.write_text("\n".join(raw_lines[:-1]) + "\n", encoding="utf-8")
        miss_report = verify_tournament_log_integrity(
            missing_path,
            expected_agents=["L1_Random", "L2_RuleBased"],
            replay_moves=False,
        )
        assert miss_report.is_valid is False
        assert len(miss_report.unbalanced_matchups) >= 2

        # 3. Illegal repeated move in move log.
        illegal_obj = json.loads(raw_lines[1])
        first_action = illegal_obj["moves"][0][0]
        illegal_obj["moves"][1][0] = first_action
        illegal_path = tmp_path / "illegal.jsonl"
        illegal_lines = [raw_lines[0], json.dumps(illegal_obj), *raw_lines[2:]]
        illegal_path.write_text("\n".join(illegal_lines) + "\n", encoding="utf-8")
        illegal_report = verify_tournament_log_integrity(illegal_path, replay_moves=True)
        assert illegal_report.is_valid is False
        assert any("IllegalMoveError" in msg for msg in illegal_report.corrupt_or_illegal_matches)

        # 4. Winner / score mismatch relative to replayed terminal outcome.
        bad_win_obj = json.loads(raw_lines[1])
        bad_win_obj["winner"] = 0 if bad_win_obj["winner"] != 0 else 1
        bad_win_path = tmp_path / "bad_win.jsonl"
        bad_win_lines = [raw_lines[0], json.dumps(bad_win_obj), *raw_lines[2:]]
        bad_win_path.write_text("\n".join(bad_win_lines) + "\n", encoding="utf-8")
        win_report = verify_tournament_log_integrity(bad_win_path, replay_moves=True)
        assert win_report.is_valid is False
        assert any("mismatches replayed winner" in m for m in win_report.corrupt_or_illegal_matches)

    def test_run_official_tournament_suite_smoke_and_validation(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        reports_dir = tmp_path / "reports"
        res = run_official_tournament_suite(
            config_path=TOURNAMENT_YAML,
            output_dir=logs_dir,
            reports_dir=reports_dir,
            base_dir=PROJECT_ROOT,
            n_pairs_override=1,
            seed_override=42,
            targets=(("4x4_k4", "fixed_10ms"), ("3x3_k3", "default")),
            resume=False,
            n_bootstrap=200,
        )
        assert set(res.log_paths.keys()) == {"4x4_k4_fixed_10ms", "3x3_k3_default"}
        for key, path in res.log_paths.items():
            assert path.is_file()
            assert res.integrity_reports[key].is_valid is True
            assert res.integrity_reports[key].total_matches == 42
            assert (reports_dir / f"tournament_{key}_metrics.csv").is_file()
            assert (reports_dir / f"tournament_{key}_summary.md").is_file()

        assert res.suite_csv_path is not None and res.suite_csv_path.is_file()
        assert res.suite_markdown_path is not None and res.suite_markdown_path.is_file()
        with res.suite_csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 14  # 7 agents * 2 target tournaments

        md_content = res.suite_markdown_path.read_text(encoding="utf-8")
        assert "Suite Overview & Integrity Verification" in md_content
        assert "`4x4_k4_fixed_10ms`" in md_content
        assert "`3x3_k3_default`" in md_content

        assert parse_board_preset("4x4_k4") == BoardConfig(4, 4, 4)
        assert parse_board_preset("3x3_k3") == BoardConfig(3, 3, 3)
        with pytest.raises(InvalidConfigError):
            parse_board_preset("6x6_k4")
        with pytest.raises(InvalidConfigError):
            parse_board_preset(123)  # type: ignore[arg-type]
        with pytest.raises(InvalidConfigError):
            run_official_tournament_suite(TOURNAMENT_YAML, logs_dir, targets=())
        with pytest.raises(InvalidConfigError):
            run_official_tournament_suite(
                TOURNAMENT_YAML,
                logs_dir,
                targets=(("4x4_k4", "nonexistent_mode"),),
            )

    def test_cli_parse_args_verify_only_and_validation(self, tmp_path: Path) -> None:
        cli = _load_run_tournament_module()
        out_log = tmp_path / "cli_match.jsonl"
        reports_dir = tmp_path / "cli_reports"

        run_args = cli.parse_args(
            [
                "--board",
                "3x3_k3",
                "--budget-mode",
                "default",
                "--n-pairs",
                "1",
                "--agents",
                "L1_Random",
                "L2_RuleBased",
                "--output",
                str(out_log),
                "--reports-dir",
                str(reports_dir),
                "--no-resume",
            ]
        )
        paths = cli.run_from_args(run_args)
        assert paths == [out_log]
        assert out_log.is_file()
        assert (reports_dir / "tournament_3x3_k3_default_metrics.csv").is_file()

        verify_args = cli.parse_args(
            [
                "--board",
                "3x3_k3",
                "--budget-mode",
                "default",
                "--n-pairs",
                "1",
                "--agents",
                "L1_Random",
                "L2_RuleBased",
                "--output",
                str(out_log),
                "--verify-only",
                "--no-reports",
            ]
        )
        assert cli.run_from_args(verify_args) == [out_log]

        bad_cases = [
            ["--n-pairs", "0"],
            ["--seed", "-1"],
            ["--budget-mode", "bad_mode"],
            ["--suite", "--agents", "L1_Random", "L2_RuleBased"],
        ]
        for bad_argv in bad_cases:
            with pytest.raises(InvalidConfigError):
                cli.run_from_args(cli.parse_args(bad_argv))


class TestOfficialTournamentLogsT4_3:
    """Gate verification on the completed Task T4.3 official JSONL tournament logs."""

    @pytest.mark.parametrize(("board_tag", "mode_name"), DEFAULT_OFFICIAL_TOURNAMENT_TARGETS)
    def test_official_t4_3_log_complete_balanced_and_zero_illegal_moves(
        self,
        board_tag: str,
        mode_name: str,
    ) -> None:
        cfg = load_tournament_config(TOURNAMENT_YAML)
        board_cfg = parse_board_preset(board_tag)
        roster = cfg.resolve_agents_for_board(board_cfg)
        log_path = PROJECT_ROOT / "logs" / f"tournament_{board_tag}_{mode_name}.jsonl"
        assert log_path.is_file(), f"Missing official T4.3 log: {log_path}"

        report = verify_tournament_log_integrity(
            log_path,
            expected_agents=[s.agent_id for s in roster],
            expected_n_pairs=100,
            replay_moves=True,
        )
        assert report.is_valid is True
        assert report.total_matches == 4200
        assert report.expected_matches == 4200
        assert not report.duplicate_match_ids
        assert not report.unbalanced_matchups
        assert not report.corrupt_or_illegal_matches

    def test_official_t4_3_consolidated_reports_exist(self) -> None:
        suite_csv = PROJECT_ROOT / "reports" / "tournament_suite_metrics.csv"
        suite_md = PROJECT_ROOT / "reports" / "tournament_suite_summary.md"
        assert suite_csv.is_file()
        assert suite_md.is_file()
        with suite_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 28  # 7 agents * 4 official tournaments
        md_text = suite_md.read_text(encoding="utf-8")
        for board_tag, mode_name in DEFAULT_OFFICIAL_TOURNAMENT_TARGETS:
            assert f"`{board_tag}_{mode_name}`" in md_text

