"""Comprehensive unit and property tests for tournament metrics and statistical utilities.

Verifies Task T4.2 per ``Planning/TASKS.md`` and ``Planning/RULES.md``:
- Wilson score 95% confidence intervals (boundaries, symmetry, monotonicity, validations).
- Bootstrap 95% empirical percentile confidence intervals (reproducibility, coverage, diff).
- Agent performance aggregation, move latencies, and search telemetry.
- Head-to-head pairwise cross-matrix calculations.
- Seat advantage (first-mover advantage) analysis.
- Full tournament log summarization, CSV export, and Markdown report generation.
- AST architectural rules (no direct print calls, no private engine attribute accesses).
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import numpy as np
import pytest

from ttt_ai.benchmark.metrics import (
    AgentPerformanceMetrics,
    HeadToHeadMetrics,
    SeatAdvantageSummary,
    TournamentSummary,
    bootstrap_ci,
    bootstrap_diff_ci,
    compute_agent_metrics,
    compute_head_to_head_metrics,
    compute_seat_advantage,
    export_summary_csv,
    format_summary_markdown,
    save_summary_markdown,
    summarize_tournament_log,
    summarize_tournament_records,
    wilson_score_interval,
)
from ttt_ai.benchmark.tournament import MatchRecord

# ---------------------------------------------------------------------------
# 1. Tests for Wilson Score Interval
# ---------------------------------------------------------------------------

class TestWilsonScoreInterval:
    """Mathematical and property tests for Wilson score confidence intervals."""

    def test_known_reference_proportions(self) -> None:
        """Verify standard 95% Wilson intervals against textbook values."""
        lower, upper = wilson_score_interval(50, 100, confidence=0.95)
        assert pytest.approx(0.4038, abs=1e-3) == lower
        assert pytest.approx(0.5962, abs=1e-3) == upper

    def test_boundary_zero_successes(self) -> None:
        """Zero successes must yield non-negative lower bound and non-zero upper bound."""
        lower, upper = wilson_score_interval(0, 10, confidence=0.95)
        assert lower == 0.0
        assert 0.0 < upper <= 1.0
        assert pytest.approx(0.2775, abs=1e-3) == upper

    def test_boundary_all_successes(self) -> None:
        """All successes must yield upper bound equal to 1.0 and lower bound < 1.0."""
        lower, upper = wilson_score_interval(10, 10, confidence=0.95)
        assert upper == 1.0
        assert 0.0 <= lower < 1.0
        assert pytest.approx(0.7225, abs=1e-3) == lower

    def test_zero_total_trials(self) -> None:
        """Zero trials must safely return (0.0, 0.0)."""
        assert wilson_score_interval(0, 0, confidence=0.95) == (0.0, 0.0)

    def test_symmetry_property(self) -> None:
        """For any k and n, Wilson(k, n) must be complementary to Wilson(n-k, n)."""
        n = 80
        for k in (0, 10, 25, 40, 70, 80):
            low_k, high_k = wilson_score_interval(k, n, confidence=0.95)
            low_inv, high_inv = wilson_score_interval(n - k, n, confidence=0.95)
            assert pytest.approx(low_k, abs=1e-5) == (1.0 - high_inv)
            assert pytest.approx(high_k, abs=1e-5) == (1.0 - low_inv)

    def test_confidence_level_monotonicity(self) -> None:
        """Higher confidence levels must yield strictly wider intervals."""
        k, n = 30, 100
        low_90, high_90 = wilson_score_interval(k, n, confidence=0.90)
        low_95, high_95 = wilson_score_interval(k, n, confidence=0.95)
        low_99, high_99 = wilson_score_interval(k, n, confidence=0.99)

        assert (high_90 - low_90) < (high_95 - low_95)
        assert (high_95 - low_95) < (high_99 - low_99)

    def test_sample_size_monotonicity(self) -> None:
        """Larger sample size with same proportion must yield narrower intervals."""
        low_small, high_small = wilson_score_interval(5, 10, confidence=0.95)
        low_large, high_large = wilson_score_interval(500, 1000, confidence=0.95)
        assert (high_large - low_large) < (high_small - low_small)

    def test_invalid_arguments_raise_value_error(self) -> None:
        """Invalid inputs must raise ValueError with clear messages."""
        with pytest.raises(ValueError, match="confidence must be in"):
            wilson_score_interval(5, 10, confidence=0.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            wilson_score_interval(5, 10, confidence=1.0)
        with pytest.raises(ValueError, match="total must be >= 0"):
            wilson_score_interval(0, -1)
        with pytest.raises(ValueError, match="successes must be between"):
            wilson_score_interval(-1, 10)
        with pytest.raises(ValueError, match="successes must be between"):
            wilson_score_interval(11, 10)


# ---------------------------------------------------------------------------
# 2. Tests for Bootstrap Confidence Interval
# ---------------------------------------------------------------------------

class TestBootstrapConfidenceInterval:
    """Statistical and edge-case tests for non-parametric empirical bootstrap."""

    def test_fixed_seed_reproducibility(self) -> None:
        """Bootstrap CI with fixed seed must return identical results across calls."""
        data = [1.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0, 0.5]
        ci1 = bootstrap_ci(data, np.mean, n_resamples=5000, confidence=0.95, seed=42)
        ci2 = bootstrap_ci(data, np.mean, n_resamples=5000, confidence=0.95, seed=42)
        assert ci1 == ci2

    def test_empty_and_singleton_data(self) -> None:
        """Empty sequence returns (0.0, 0.0); singleton returns exact value."""
        assert bootstrap_ci([], np.mean) == (0.0, 0.0)
        assert bootstrap_ci([0.75], np.mean) == (0.75, 0.75)

    def test_constant_data(self) -> None:
        """Constant sequence must yield collapsed interval."""
        data = [0.5] * 20
        assert bootstrap_ci(data, np.mean, n_resamples=1000) == (0.5, 0.5)

    def test_coverage_on_normal_sample(self) -> None:
        """Empirical bootstrap CI of sample from N(0, 1) should bracket true mean (0.0)."""
        rng = np.random.default_rng(12345)
        sample = rng.normal(loc=0.0, scale=1.0, size=200)
        lower, upper = bootstrap_ci(sample, np.mean, n_resamples=5000, confidence=0.95, seed=42)
        assert lower <= 0.0 <= upper

    def test_custom_stat_function(self) -> None:
        """Support arbitrary scalar statistic function."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        lower, upper = bootstrap_ci(data, np.median, n_resamples=2000, confidence=0.90, seed=42)
        assert lower <= np.median(data) <= upper

    def test_bootstrap_diff_ci_separated_distributions(self) -> None:
        """Difference CI between separated distributions must not cover zero."""
        data_a = [10.0, 10.1, 9.9, 10.2, 9.8] * 10
        data_b = [1.0, 1.1, 0.9, 1.2, 0.8] * 10
        lower, upper = bootstrap_diff_ci(
            data_a, data_b, n_resamples=3000, confidence=0.95, seed=42
        )
        assert lower > 8.0
        assert upper < 10.0

    def test_bootstrap_diff_ci_empty_inputs(self) -> None:
        """Empty inputs to difference bootstrap must return (0.0, 0.0)."""
        assert bootstrap_diff_ci([], [1.0, 2.0]) == (0.0, 0.0)
        assert bootstrap_diff_ci([1.0, 2.0], []) == (0.0, 0.0)

    def test_bootstrap_invalid_parameters_raise(self) -> None:
        """Invalid confidence or sample count must raise ValueError."""
        with pytest.raises(ValueError, match="confidence must be in"):
            bootstrap_ci([1.0, 2.0], confidence=-0.1)
        with pytest.raises(ValueError, match="n_resamples must be >= 1"):
            bootstrap_ci([1.0, 2.0], n_resamples=0)
        with pytest.raises(ValueError, match="confidence must be in"):
            bootstrap_diff_ci([1.0], [2.0], confidence=1.5)
        with pytest.raises(ValueError, match="n_resamples must be >= 1"):
            bootstrap_diff_ci([1.0], [2.0], n_resamples=-5)


# ---------------------------------------------------------------------------
# 3. Tests for Agent and Head-to-Head Metrics Aggregation
# ---------------------------------------------------------------------------

def _create_synthetic_match(
    match_id: str,
    agent_x: str,
    agent_o: str,
    winner: int,
    latencies_x: list[float] | None = None,
    latencies_o: list[float] | None = None,
    nodes_x: list[int] | None = None,
    nodes_o: list[int] | None = None,
    sims_x: list[int] | None = None,
    sims_o: list[int] | None = None,
) -> MatchRecord:
    """Helper creating a valid MatchRecord for controlled tests."""
    lx = latencies_x or [1.0, 1.2]
    lo = latencies_o or [0.5, 0.6]
    nx = nodes_x or [10, 12]
    no = nodes_o or [5, 6]
    sx = sims_x or [20, 25]
    so = sims_o or [10, 15]

    score_x = 1.0 if winner == 1 else (0.5 if winner == 0 else 0.0)
    score_o = 1.0 if winner == -1 else (0.5 if winner == 0 else 0.0)

    # Interleave moves: [action, latency, nodes, sims, depth]
    moves: list[list[int | float]] = []
    max_steps = max(len(lx), len(lo))
    for i in range(max_steps):
        if i < len(lx):
            moves.append([i * 2, lx[i], nx[i], sx[i], 3])
        if i < len(lo):
            moves.append([i * 2 + 1, lo[i], no[i], so[i], 2])

    return MatchRecord(
        match_id=match_id,
        pair_id=0,
        game_in_pair=0,
        seed=42,
        board_config={"rows": 4, "cols": 4, "win_k": 4},
        budget_mode="default",
        agent_x=agent_x,
        agent_o=agent_o,
        winner=winner,
        score_x=score_x,
        score_o=score_o,
        num_moves=len(moves),
        latencies_x_ms=lx,
        latencies_o_ms=lo,
        nodes_x=nx,
        nodes_o=no,
        sims_x=sx,
        sims_o=so,
        winning_line=None,
        moves=moves,
    )


class TestMetricsAggregation:
    """Tests for aggregating records into AgentPerformanceMetrics and HeadToHeadMetrics."""

    def test_single_agent_metrics_as_x_and_o(self) -> None:
        """Verify metric calculation when agent plays balanced X and O games."""
        # Match 1: Alpha (X) wins vs Beta (O)
        m1 = _create_synthetic_match("m1", "Alpha", "Beta", winner=1, latencies_x=[10.0])
        # Match 2: Beta (X) vs Alpha (O), draw
        m2 = _create_synthetic_match("m2", "Beta", "Alpha", winner=0, latencies_o=[6.0])

        metrics = compute_agent_metrics([m1, m2], "Alpha", n_bootstrap=1000, seed=42)
        assert isinstance(metrics, AgentPerformanceMetrics)
        assert metrics.agent_id == "Alpha"
        assert metrics.total_games == 2
        assert metrics.wins == 1
        assert metrics.draws == 1
        assert metrics.losses == 0
        assert metrics.score == 0.75  # (1.0 + 0.5) / 2
        assert metrics.games_as_x == 1
        assert metrics.wins_as_x == 1
        assert metrics.score_x == 1.0
        assert metrics.games_as_o == 1
        assert metrics.draws_as_o == 1
        assert metrics.score_o == 0.5
        assert metrics.delta_score == 0.5  # score_x - score_o

        # Move latency: 1 move of 10.0ms and 1 move of 6.0ms -> mean = 8.0ms
        assert metrics.total_moves == 2
        assert metrics.mean_latency_ms == 8.0
        assert metrics.median_latency_ms == 8.0
        assert metrics.max_latency_ms == 10.0

    def test_unplayed_agent_metrics(self) -> None:
        """An agent that participated in no matches returns zeroes without error."""
        m1 = _create_synthetic_match("m1", "Alpha", "Beta", winner=1)
        metrics = compute_agent_metrics([m1], "Gamma", n_bootstrap=500, seed=42)
        assert metrics.agent_id == "Gamma"
        assert metrics.total_games == 0
        assert metrics.score == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.total_moves == 0
        assert metrics.mean_latency_ms == 0.0

    def test_head_to_head_pairwise_symmetry(self) -> None:
        """Head-to-head score between A and B must satisfy score_a + score_b = 1.0."""
        m1 = _create_synthetic_match("m1", "AgentA", "AgentB", winner=1)  # A wins (1-0)
        m2 = _create_synthetic_match("m2", "AgentB", "AgentA", winner=0)  # Draw (0.5-0.5)
        m3 = _create_synthetic_match("m3", "AgentA", "AgentB", winner=-1) # B wins (0-1)
        m4 = _create_synthetic_match("m4", "AgentB", "AgentA", winner=-1) # A wins (1-0)

        h2h = compute_head_to_head_metrics([m1, m2, m3, m4], "AgentA", "AgentB", n_bootstrap=1000)
        assert isinstance(h2h, HeadToHeadMetrics)
        assert h2h.total_games == 4
        assert h2h.wins_a == 2
        assert h2h.draws == 1
        assert h2h.wins_b == 1
        assert h2h.score_a == 0.625  # (2 + 0.5) / 4
        assert h2h.score_b == 0.375  # (1 + 0.5) / 4
        assert pytest.approx(h2h.score_a + h2h.score_b) == 1.0
        assert h2h.delta_score == 0.25

    def test_seat_advantage_aggregation(self) -> None:
        """Verify global first-mover advantage computation."""
        # 2 X wins, 1 O win, 1 Draw -> 4 matches
        m1 = _create_synthetic_match("m1", "A", "B", winner=1)
        m2 = _create_synthetic_match("m2", "B", "A", winner=1)
        m3 = _create_synthetic_match("m3", "A", "B", winner=-1)
        m4 = _create_synthetic_match("m4", "B", "A", winner=0)

        sa = compute_seat_advantage([m1, m2, m3, m4], n_bootstrap=1000, seed=42)
        assert isinstance(sa, SeatAdvantageSummary)
        assert sa.total_games == 4
        assert sa.x_wins == 2
        assert sa.o_wins == 1
        assert sa.draws == 1
        assert sa.x_win_rate == 0.50
        assert sa.o_win_rate == 0.25
        assert sa.draw_rate == 0.25
        assert sa.score_x == 0.625
        assert sa.score_o == 0.375
        assert sa.delta_score == 0.25


# ---------------------------------------------------------------------------
# 4. Tests for Tournament Summarization and Report Generation
# ---------------------------------------------------------------------------

class TestTournamentSummaryAndReports:
    """Integration tests on actual tournament logs and export formats."""

    @pytest.fixture
    def smoke_log_path(self) -> Path:
        """Fixture returning the existing smoke tournament log path."""
        path = Path("logs/smoke_t4_1_4x4_k4_fixed10ms.jsonl")
        if not path.is_file():
            pytest.skip(f"Smoke log '{path}' does not exist.")
        return path

    def test_summarize_real_tournament_log(self, smoke_log_path: Path) -> None:
        """Summarize real 7-level smoke tournament log and verify all constraints."""
        summary = summarize_tournament_log(smoke_log_path, n_bootstrap=2000, seed=42)
        assert isinstance(summary, TournamentSummary)
        assert summary.total_matches == 84
        assert len(summary.participants) == 7
        assert "L1_Random" in summary.participants
        assert "L7_HybridMCTS" in summary.participants

        # Ranking validation: top agent score >= bottom agent score
        rankings = summary.get_rankings()
        assert len(rankings) == 7
        assert rankings[0].score >= rankings[-1].score
        # Random should rank lowest
        assert rankings[-1].agent_id == "L1_Random"

        # Check total games per participant: each agent played 6 opponents * 4 games = 24 games
        for agent_m in rankings:
            assert agent_m.total_games == 24
            assert agent_m.games_as_x == 12
            assert agent_m.games_as_o == 12
            # Confidence interval bounds must encompass the empirical score
            assert agent_m.score_ci[0] <= agent_m.score <= agent_m.score_ci[1]

        # Seat advantage check
        sa = summary.seat_advantage
        assert sa.total_games == 84
        assert sa.x_wins + sa.o_wins + sa.draws == 84
        assert pytest.approx(sa.score_x + sa.score_o) == 1.0

    def test_export_summary_csv(self, smoke_log_path: Path, tmp_path: Path) -> None:
        """Verify summary metrics export cleanly to a valid CSV file."""
        summary = summarize_tournament_log(smoke_log_path, n_bootstrap=1000, seed=42)
        out_csv = tmp_path / "summary_metrics.csv"
        saved_path = export_summary_csv(summary, out_csv)

        assert saved_path.is_file()
        with saved_path.open("r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 7
            first_row = reader[0]
            assert "agent_id" in first_row
            assert "score" in first_row
            assert "score_ci_lower" in first_row
            assert "score_ci_upper" in first_row
            assert "mean_latency_ms" in first_row
            assert "avg_nodes_visited" in first_row
            assert float(first_row["total_games"]) == 24

    def test_format_and_save_summary_markdown(self, smoke_log_path: Path, tmp_path: Path) -> None:
        """Verify markdown report generation and persistence."""
        summary = summarize_tournament_log(smoke_log_path, n_bootstrap=1000, seed=42)
        md_text = format_summary_markdown(summary)

        # Check sections exist
        assert "# Tournament Evaluation Summary:" in md_text
        assert "## 1. Environment & Configuration" in md_text
        assert "## 2. Participant Leaderboard & Overall Metrics" in md_text
        assert "## 3. Head-to-Head Score Matrix" in md_text
        assert "## 4. First-Mover Advantage (Seat Analysis)" in md_text
        assert "L7_HybridMCTS" in md_text
        assert "L1_Random" in md_text

        out_md = tmp_path / "summary_report.md"
        saved = save_summary_markdown(summary, out_md)
        assert saved.is_file()
        assert saved.read_text(encoding="utf-8") == md_text

    def test_empty_tournament_summary(self) -> None:
        """Verify summarizing empty record sequence handles cleanly without crash."""
        summary = summarize_tournament_records([])
        assert summary.total_matches == 0
        assert summary.participants == []
        assert summary.get_rankings() == []
        assert summary.seat_advantage.total_games == 0
        md = format_summary_markdown(summary)
        assert "Total Matches**: 0" in md


# ---------------------------------------------------------------------------
# 5. AST & Architecture Rules Compliance
# ---------------------------------------------------------------------------

class TestRulesCompliance:
    """Verifies strict adherence to Planning/RULES.md."""

    def test_no_print_statements_in_metrics_module(self) -> None:
        """Verify metrics.py contains zero print() calls (RULE 4.4)."""
        metrics_file = Path("src/ttt_ai/benchmark/metrics.py")
        assert metrics_file.is_file(), f"Missing {metrics_file}"

        source = metrics_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(metrics_file))

        print_calls = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "print")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "print")
            )
        ]
        assert not print_calls, f"Direct print() found in metrics.py at lines: {print_calls}"

    def test_no_private_engine_attributes_accessed(self) -> None:
        """Verify metrics.py never accesses private engine attributes (RULE 2.1)."""
        metrics_file = Path("src/ttt_ai/benchmark/metrics.py")
        source = metrics_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(metrics_file))

        forbidden = {"_board", "_history", "_winner", "_current_player"}
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                violations.append(f"Line {node.lineno}: .{node.attr}")

        assert not violations, f"Forbidden private attribute accesses: {violations}"
