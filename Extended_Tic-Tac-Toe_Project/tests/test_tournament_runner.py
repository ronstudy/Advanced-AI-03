"""Unit and integration tests for Task T4.1 Paired Round-Robin TournamentRunner."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from ttt_ai import (
    AgentSpec,
    BoardConfig,
    TournamentRunner,
    load_tournament_config,
    load_tournament_log,
)
from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.engine.rules import PLAYER_O, PLAYER_X

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOURNAMENT_YAML = PROJECT_ROOT / "configs" / "tournament.yaml"


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
