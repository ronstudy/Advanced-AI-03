"""Tournament statistical metrics, confidence intervals, and summary exports.

Implements Task T4.2 per ``Planning/TASKS.md``, ``Planning/ARCHITECTURE.md``
Section 5.3, and ``Planning/RULES.md`` Section 3:
- Proportion confidence intervals: Wilson score 95% CI for binomial proportions.
- Non-parametric empirical percentile bootstrap: 95% CI for tournament scores
  and first-mover advantage (delta_score = score_x - score_o).
- Move latency percentiles (mean, median, p95, max) computed over individual moves.
- Search telemetry aggregation (average nodes, simulations, and search depth).
- Head-to-head cross-table statistics and global seat advantage analysis.
- Structured export to CSV and clean GitHub Flavored Markdown summary tables.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as stats

from ttt_ai.benchmark.tournament import MatchRecord, load_tournament_log

logger = logging.getLogger(__name__)


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Computes the asymmetric confidence interval proposed by Edwin Bidwell Wilson (1927).
    Guarantees the bounds are strictly within [0.0, 1.0], avoiding the severe degradation
    of normal approximation (Wald) intervals near boundaries (e.g. when successes == 0
    or successes == total).

    Formula:
        p = successes / total
        z = Phi^-1(1 - alpha / 2)
        denom = 1 + z^2 / total
        center = (p + z^2 / (2 * total)) / denom
        margin = (z / denom) * sqrt(p * (1 - p) / total + z^2 / (4 * total^2))
        interval = [max(0.0, center - margin), min(1.0, center + margin)]

    Args:
        successes: Number of observed successes (integer >= 0).
        total: Total number of trials (integer >= 0).
        confidence: Confidence level in (0.0, 1.0), defaults to 0.95.

    Returns:
        Tuple (lower, upper) representing the confidence interval in [0.0, 1.0].
        If total == 0, returns (0.0, 0.0).

    Raises:
        ValueError: If total < 0, successes < 0, successes > total, or
            confidence is not strictly within (0.0, 1.0).
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total!r}")
    if successes < 0 or successes > total:
        raise ValueError(
            f"successes must be between 0 and total ({total}), got {successes!r}"
        )
    if total == 0:
        return (0.0, 0.0)

    alpha = 1.0 - confidence
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    variance_term = (p * (1.0 - p)) / total + z2 / (4.0 * total * total)
    margin = (z / denom) * math.sqrt(max(0.0, variance_term))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 6), round(upper, 6))


def bootstrap_ci(
    data: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate the empirical percentile bootstrap confidence interval for a statistic.

    Performs non-parametric resampling with replacement B times (default 10,000) using
    a reproducible pseudorandom number generator, computing the target statistic on
    each resample and taking empirical percentiles (alpha/2 and 1 - alpha/2).

    Vectorized along axis=1 when stat_fn is np.mean for sub-50ms performance.

    Args:
        data: Sequence of numeric observations.
        stat_fn: Callable accepting a 1D NumPy array and returning a scalar statistic.
            Defaults to np.mean.
        n_resamples: Number of bootstrap iterations B (default 10,000).
        confidence: Confidence level in (0.0, 1.0), defaults to 0.95.
        seed: Random seed for reproducible resampling.

    Returns:
        Tuple (lower, upper) representing the empirical percentile confidence interval.
        If data is empty, returns (0.0, 0.0). If data has length 1, returns (val, val).

    Raises:
        ValueError: If n_resamples < 1 or confidence is not strictly within (0.0, 1.0).
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples!r}")

    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        val = float(arr[0])
        return (round(val, 6), round(val, 6))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    samples = arr[indices]

    if stat_fn is np.mean or stat_fn == np.mean:
        stats_arr = np.mean(samples, axis=1)
    else:
        stats_arr = np.fromiter(
            (float(stat_fn(sample)) for sample in samples),
            dtype=np.float64,
            count=n_resamples,
        )

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    lower = float(np.percentile(stats_arr, lower_pct))
    upper = float(np.percentile(stats_arr, upper_pct))
    return (round(lower, 6), round(upper, 6))


def bootstrap_diff_ci(
    data_a: Sequence[float],
    data_b: Sequence[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate empirical percentile bootstrap CI for difference of means (mean(A) - mean(B)).

    Resamples sample A and sample B independently with replacement B times, computes
    mean(A*) - mean(B*) for each bootstrap iteration, and extracts empirical percentiles.

    Args:
        data_a: First sequence of numeric observations.
        data_b: Second sequence of numeric observations.
        n_resamples: Number of bootstrap iterations B (default 10,000).
        confidence: Confidence level in (0.0, 1.0), defaults to 0.95.
        seed: Random seed for reproducible resampling.

    Returns:
        Tuple (lower, upper) representing empirical percentile CI of mean(A) - mean(B).
        If either sequence is empty, returns (0.0, 0.0).

    Raises:
        ValueError: If n_resamples < 1 or confidence is not strictly within (0.0, 1.0).
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples!r}")

    arr_a = np.asarray(data_a, dtype=np.float64)
    arr_b = np.asarray(data_b, dtype=np.float64)
    if len(arr_a) == 0 or len(arr_b) == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(arr_a), size=(n_resamples, len(arr_a)))
    idx_b = rng.integers(0, len(arr_b), size=(n_resamples, len(arr_b)))

    means_a = np.mean(arr_a[idx_a], axis=1)
    means_b = np.mean(arr_b[idx_b], axis=1)
    diffs = means_a - means_b

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    lower = float(np.percentile(diffs, lower_pct))
    upper = float(np.percentile(diffs, upper_pct))
    return (round(lower, 6), round(upper, 6))


@dataclass(frozen=True, slots=True)
class AgentPerformanceMetrics:
    """Consolidated tournament performance and search telemetry metrics for an agent."""

    agent_id: str
    total_games: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    win_rate_ci: tuple[float, float]
    score: float
    score_ci: tuple[float, float]
    games_as_x: int
    wins_as_x: int
    draws_as_x: int
    losses_as_x: int
    score_x: float
    games_as_o: int
    wins_as_o: int
    draws_as_o: int
    losses_as_o: int
    score_o: float
    delta_score: float
    delta_score_ci: tuple[float, float]
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    avg_nodes_visited: float
    avg_simulations: float
    avg_depth: float
    total_moves: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a JSON/report-compatible dictionary."""
        return {
            "agent_id": self.agent_id,
            "total_games": self.total_games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "win_rate_ci": [round(self.win_rate_ci[0], 4), round(self.win_rate_ci[1], 4)],
            "score": round(self.score, 4),
            "score_ci": [round(self.score_ci[0], 4), round(self.score_ci[1], 4)],
            "games_as_x": self.games_as_x,
            "wins_as_x": self.wins_as_x,
            "draws_as_x": self.draws_as_x,
            "losses_as_x": self.losses_as_x,
            "score_x": round(self.score_x, 4),
            "games_as_o": self.games_as_o,
            "wins_as_o": self.wins_as_o,
            "draws_as_o": self.draws_as_o,
            "losses_as_o": self.losses_as_o,
            "score_o": round(self.score_o, 4),
            "delta_score": round(self.delta_score, 4),
            "delta_score_ci": [
                round(self.delta_score_ci[0], 4),
                round(self.delta_score_ci[1], 4),
            ],
            "mean_latency_ms": round(self.mean_latency_ms, 4),
            "median_latency_ms": round(self.median_latency_ms, 4),
            "p95_latency_ms": round(self.p95_latency_ms, 4),
            "max_latency_ms": round(self.max_latency_ms, 4),
            "avg_nodes_visited": round(self.avg_nodes_visited, 2),
            "avg_simulations": round(self.avg_simulations, 2),
            "avg_depth": round(self.avg_depth, 2),
            "total_moves": self.total_moves,
        }


@dataclass(frozen=True, slots=True)
class HeadToHeadMetrics:
    """Pairwise matchup metrics between two tournament agents."""

    agent_a: str
    agent_b: str
    total_games: int
    wins_a: int
    draws: int
    wins_b: int
    score_a: float
    score_b: float
    score_a_ci: tuple[float, float]
    score_b_ci: tuple[float, float]
    delta_score: float
    delta_score_ci: tuple[float, float]
    games_a_as_x: int
    wins_a_as_x: int
    draws_a_as_x: int
    losses_a_as_x: int
    games_a_as_o: int
    wins_a_as_o: int
    draws_a_as_o: int
    losses_a_as_o: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize head-to-head metrics to dictionary."""
        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "total_games": self.total_games,
            "wins_a": self.wins_a,
            "draws": self.draws,
            "wins_b": self.wins_b,
            "score_a": round(self.score_a, 4),
            "score_b": round(self.score_b, 4),
            "score_a_ci": [round(self.score_a_ci[0], 4), round(self.score_a_ci[1], 4)],
            "score_b_ci": [round(self.score_b_ci[0], 4), round(self.score_b_ci[1], 4)],
            "delta_score": round(self.delta_score, 4),
            "delta_score_ci": [
                round(self.delta_score_ci[0], 4),
                round(self.delta_score_ci[1], 4),
            ],
            "games_a_as_x": self.games_a_as_x,
            "wins_a_as_x": self.wins_a_as_x,
            "draws_a_as_x": self.draws_a_as_x,
            "losses_a_as_x": self.losses_a_as_x,
            "games_a_as_o": self.games_a_as_o,
            "wins_a_as_o": self.wins_a_as_o,
            "draws_a_as_o": self.draws_a_as_o,
            "losses_a_as_o": self.losses_a_as_o,
        }


@dataclass(frozen=True, slots=True)
class SeatAdvantageSummary:
    """Global seat advantage (first-mover advantage) analysis across tournament matches."""

    total_games: int
    x_wins: int
    o_wins: int
    draws: int
    x_win_rate: float
    x_win_rate_ci: tuple[float, float]
    o_win_rate: float
    o_win_rate_ci: tuple[float, float]
    draw_rate: float
    draw_rate_ci: tuple[float, float]
    score_x: float
    score_o: float
    delta_score: float
    delta_score_ci: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize seat advantage summary to dictionary."""
        return {
            "total_games": self.total_games,
            "x_wins": self.x_wins,
            "o_wins": self.o_wins,
            "draws": self.draws,
            "x_win_rate": round(self.x_win_rate, 4),
            "x_win_rate_ci": [round(self.x_win_rate_ci[0], 4), round(self.x_win_rate_ci[1], 4)],
            "o_win_rate": round(self.o_win_rate, 4),
            "o_win_rate_ci": [round(self.o_win_rate_ci[0], 4), round(self.o_win_rate_ci[1], 4)],
            "draw_rate": round(self.draw_rate, 4),
            "draw_rate_ci": [round(self.draw_rate_ci[0], 4), round(self.draw_rate_ci[1], 4)],
            "score_x": round(self.score_x, 4),
            "score_o": round(self.score_o, 4),
            "delta_score": round(self.delta_score, 4),
            "delta_score_ci": [
                round(self.delta_score_ci[0], 4),
                round(self.delta_score_ci[1], 4),
            ],
        }


@dataclass(frozen=True, slots=True)
class TournamentSummary:
    """Comprehensive statistical aggregation of a tournament run."""

    meta: dict[str, Any]
    total_matches: int
    participants: list[str]
    agent_metrics: dict[str, AgentPerformanceMetrics]
    head_to_head: dict[tuple[str, str], HeadToHeadMetrics]
    seat_advantage: SeatAdvantageSummary

    def get_rankings(self) -> list[AgentPerformanceMetrics]:
        """Return participant metrics sorted by score descending, then wins, then mean latency."""
        return sorted(
            self.agent_metrics.values(),
            key=lambda m: (-m.score, -m.wins, m.mean_latency_ms, m.agent_id),
        )

    def get_head_to_head(self, agent_a: str, agent_b: str) -> HeadToHeadMetrics | None:
        """Return head-to-head metrics for pair (agent_a, agent_b)."""
        return self.head_to_head.get((agent_a, agent_b))

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete tournament summary to dictionary."""
        return {
            "meta": dict(self.meta),
            "total_matches": self.total_matches,
            "participants": list(self.participants),
            "rankings": [m.to_dict() for m in self.get_rankings()],
            "head_to_head": {
                f"{k[0]}__vs__{k[1]}": v.to_dict()
                for k, v in self.head_to_head.items()
            },
            "seat_advantage": self.seat_advantage.to_dict(),
        }


def compute_agent_metrics(
    matches: Sequence[MatchRecord],
    agent_id: str,
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> AgentPerformanceMetrics:
    """Compute complete performance and search telemetry metrics for a single agent.

    Aggregates move latencies, node counts, simulations, and search depths across
    all moves played by `agent_id` across the provided sequence of `matches`.

    Args:
        matches: Sequence of :class:`MatchRecord` instances.
        agent_id: Unique string identifier of the target agent.
        n_bootstrap: Number of bootstrap resamples (default 10,000).
        confidence: Confidence level (default 0.95).
        seed: Random seed for bootstrap reproducibility.

    Returns:
        Populated :class:`AgentPerformanceMetrics` instance.
    """
    scores_as_x: list[float] = []
    scores_as_o: list[float] = []
    wins_as_x = 0
    draws_as_x = 0
    losses_as_x = 0
    wins_as_o = 0
    draws_as_o = 0
    losses_as_o = 0

    all_latencies_ms: list[float] = []
    all_nodes: list[int] = []
    all_sims: list[int] = []
    all_depths: list[int] = []

    for rec in matches:
        if rec.agent_x == agent_id:
            # Agent played as X
            if rec.winner == 1:
                wins_as_x += 1
                scores_as_x.append(1.0)
            elif rec.winner == 0:
                draws_as_x += 1
                scores_as_x.append(0.5)
            else:
                losses_as_x += 1
                scores_as_x.append(0.0)

            all_latencies_ms.extend(rec.latencies_x_ms)
            all_nodes.extend(rec.nodes_x)
            all_sims.extend(rec.sims_x)
            for idx in range(0, len(rec.moves), 2):
                mv = rec.moves[idx]
                if len(mv) >= 5:
                    all_depths.append(int(mv[4]))

        elif rec.agent_o == agent_id:
            # Agent played as O
            if rec.winner == -1:
                wins_as_o += 1
                scores_as_o.append(1.0)
            elif rec.winner == 0:
                draws_as_o += 1
                scores_as_o.append(0.5)
            else:
                losses_as_o += 1
                scores_as_o.append(0.0)

            all_latencies_ms.extend(rec.latencies_o_ms)
            all_nodes.extend(rec.nodes_o)
            all_sims.extend(rec.sims_o)
            for idx in range(1, len(rec.moves), 2):
                mv = rec.moves[idx]
                if len(mv) >= 5:
                    all_depths.append(int(mv[4]))

    games_as_x = len(scores_as_x)
    games_as_o = len(scores_as_o)
    total_games = games_as_x + games_as_o
    wins = wins_as_x + wins_as_o
    draws = draws_as_x + draws_as_o
    losses = losses_as_x + losses_as_o

    win_rate = (wins / total_games) if total_games > 0 else 0.0
    win_rate_ci = wilson_score_interval(wins, total_games, confidence=confidence)

    all_scores = scores_as_x + scores_as_o
    score = float(np.mean(all_scores)) if all_scores else 0.0
    score_ci = bootstrap_ci(
        all_scores,
        np.mean,
        n_resamples=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )

    score_x = float(np.mean(scores_as_x)) if scores_as_x else 0.0
    score_o = float(np.mean(scores_as_o)) if scores_as_o else 0.0
    delta_score = score_x - score_o if (games_as_x > 0 and games_as_o > 0) else 0.0
    delta_score_ci = (
        bootstrap_diff_ci(
            scores_as_x,
            scores_as_o,
            n_resamples=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
        if (games_as_x > 0 and games_as_o > 0)
        else (0.0, 0.0)
    )

    total_moves = len(all_latencies_ms)
    if total_moves > 0:
        mean_latency_ms = float(np.mean(all_latencies_ms))
        median_latency_ms = float(np.median(all_latencies_ms))
        p95_latency_ms = float(np.percentile(all_latencies_ms, 95))
        max_latency_ms = float(np.max(all_latencies_ms))
        avg_nodes_visited = float(np.mean(all_nodes)) if all_nodes else 0.0
        avg_simulations = float(np.mean(all_sims)) if all_sims else 0.0
        avg_depth = float(np.mean(all_depths)) if all_depths else 0.0
    else:
        mean_latency_ms = 0.0
        median_latency_ms = 0.0
        p95_latency_ms = 0.0
        max_latency_ms = 0.0
        avg_nodes_visited = 0.0
        avg_simulations = 0.0
        avg_depth = 0.0

    return AgentPerformanceMetrics(
        agent_id=agent_id,
        total_games=total_games,
        wins=wins,
        draws=draws,
        losses=losses,
        win_rate=win_rate,
        win_rate_ci=win_rate_ci,
        score=score,
        score_ci=score_ci,
        games_as_x=games_as_x,
        wins_as_x=wins_as_x,
        draws_as_x=draws_as_x,
        losses_as_x=losses_as_x,
        score_x=score_x,
        games_as_o=games_as_o,
        wins_as_o=wins_as_o,
        draws_as_o=draws_as_o,
        losses_as_o=losses_as_o,
        score_o=score_o,
        delta_score=delta_score,
        delta_score_ci=delta_score_ci,
        mean_latency_ms=mean_latency_ms,
        median_latency_ms=median_latency_ms,
        p95_latency_ms=p95_latency_ms,
        max_latency_ms=max_latency_ms,
        avg_nodes_visited=avg_nodes_visited,
        avg_simulations=avg_simulations,
        avg_depth=avg_depth,
        total_moves=total_moves,
    )


def compute_head_to_head_metrics(
    matches: Sequence[MatchRecord],
    agent_a: str,
    agent_b: str,
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> HeadToHeadMetrics:
    """Compute pairwise matchup metrics between two specific agents.

    Args:
        matches: Sequence of :class:`MatchRecord` instances.
        agent_a: First agent identifier.
        agent_b: Second agent identifier.
        n_bootstrap: Number of bootstrap iterations (default 10,000).
        confidence: Confidence level (default 0.95).
        seed: Random seed for bootstrap resampling.

    Returns:
        Populated :class:`HeadToHeadMetrics` instance.
    """
    scores_a: list[float] = []
    scores_b: list[float] = []
    wins_a = 0
    draws = 0
    wins_b = 0

    games_a_as_x = 0
    wins_a_as_x = 0
    draws_a_as_x = 0
    losses_a_as_x = 0

    games_a_as_o = 0
    wins_a_as_o = 0
    draws_a_as_o = 0
    losses_a_as_o = 0

    for rec in matches:
        if rec.agent_x == agent_a and rec.agent_o == agent_b:
            games_a_as_x += 1
            if rec.winner == 1:
                wins_a += 1
                wins_a_as_x += 1
                scores_a.append(1.0)
                scores_b.append(0.0)
            elif rec.winner == 0:
                draws += 1
                draws_a_as_x += 1
                scores_a.append(0.5)
                scores_b.append(0.5)
            else:
                wins_b += 1
                losses_a_as_x += 1
                scores_a.append(0.0)
                scores_b.append(1.0)

        elif rec.agent_x == agent_b and rec.agent_o == agent_a:
            games_a_as_o += 1
            if rec.winner == -1:
                wins_a += 1
                wins_a_as_o += 1
                scores_a.append(1.0)
                scores_b.append(0.0)
            elif rec.winner == 0:
                draws += 1
                draws_a_as_o += 1
                scores_a.append(0.5)
                scores_b.append(0.5)
            else:
                wins_b += 1
                losses_a_as_o += 1
                scores_a.append(0.0)
                scores_b.append(1.0)

    total_games = len(scores_a)
    score_a = float(np.mean(scores_a)) if scores_a else 0.0
    score_b = float(np.mean(scores_b)) if scores_b else 0.0
    score_a_ci = bootstrap_ci(
        scores_a,
        np.mean,
        n_resamples=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    score_b_ci = bootstrap_ci(
        scores_b,
        np.mean,
        n_resamples=n_bootstrap,
        confidence=confidence,
        seed=seed + 1,
    )

    diffs = [sa - sb for sa, sb in zip(scores_a, scores_b, strict=False)]
    delta_score = score_a - score_b
    delta_score_ci = (
        bootstrap_ci(
            diffs,
            np.mean,
            n_resamples=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
        if diffs
        else (0.0, 0.0)
    )

    return HeadToHeadMetrics(
        agent_a=agent_a,
        agent_b=agent_b,
        total_games=total_games,
        wins_a=wins_a,
        draws=draws,
        wins_b=wins_b,
        score_a=score_a,
        score_b=score_b,
        score_a_ci=score_a_ci,
        score_b_ci=score_b_ci,
        delta_score=delta_score,
        delta_score_ci=delta_score_ci,
        games_a_as_x=games_a_as_x,
        wins_a_as_x=wins_a_as_x,
        draws_a_as_x=draws_a_as_x,
        losses_a_as_x=losses_a_as_x,
        games_a_as_o=games_a_as_o,
        wins_a_as_o=wins_a_as_o,
        draws_a_as_o=draws_a_as_o,
        losses_a_as_o=losses_a_as_o,
    )


def compute_seat_advantage(
    matches: Sequence[MatchRecord],
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> SeatAdvantageSummary:
    """Compute global first-mover advantage (Player X vs Player O) across tournament matches.

    Args:
        matches: Sequence of :class:`MatchRecord` instances.
        n_bootstrap: Number of bootstrap iterations (default 10,000).
        confidence: Confidence level (default 0.95).
        seed: Random seed for bootstrap resampling.

    Returns:
        Populated :class:`SeatAdvantageSummary` instance.
    """
    total_games = len(matches)
    if total_games == 0:
        return SeatAdvantageSummary(
            total_games=0,
            x_wins=0,
            o_wins=0,
            draws=0,
            x_win_rate=0.0,
            x_win_rate_ci=(0.0, 0.0),
            o_win_rate=0.0,
            o_win_rate_ci=(0.0, 0.0),
            draw_rate=0.0,
            draw_rate_ci=(0.0, 0.0),
            score_x=0.0,
            score_o=0.0,
            delta_score=0.0,
            delta_score_ci=(0.0, 0.0),
        )

    x_wins = sum(1 for m in matches if m.winner == 1)
    o_wins = sum(1 for m in matches if m.winner == -1)
    draws = sum(1 for m in matches if m.winner == 0)

    x_win_rate = x_wins / total_games
    o_win_rate = o_wins / total_games
    draw_rate = draws / total_games

    x_win_rate_ci = wilson_score_interval(x_wins, total_games, confidence=confidence)
    o_win_rate_ci = wilson_score_interval(o_wins, total_games, confidence=confidence)
    draw_rate_ci = wilson_score_interval(draws, total_games, confidence=confidence)

    scores_x = [m.score_x for m in matches]
    scores_o = [m.score_o for m in matches]
    score_x = float(np.mean(scores_x))
    score_o = float(np.mean(scores_o))

    diffs = [m.score_x - m.score_o for m in matches]
    delta_score = score_x - score_o
    delta_score_ci = bootstrap_ci(
        diffs,
        np.mean,
        n_resamples=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )

    return SeatAdvantageSummary(
        total_games=total_games,
        x_wins=x_wins,
        o_wins=o_wins,
        draws=draws,
        x_win_rate=x_win_rate,
        x_win_rate_ci=x_win_rate_ci,
        o_win_rate=o_win_rate,
        o_win_rate_ci=o_win_rate_ci,
        draw_rate=draw_rate,
        draw_rate_ci=draw_rate_ci,
        score_x=score_x,
        score_o=score_o,
        delta_score=delta_score,
        delta_score_ci=delta_score_ci,
    )


def summarize_tournament_records(
    records: Sequence[MatchRecord],
    meta: dict[str, Any] | None = None,
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> TournamentSummary:
    """Aggregate a sequence of match records into a complete :class:`TournamentSummary`.

    Args:
        records: Sequence of :class:`MatchRecord` instances.
        meta: Optional tournament metadata header dictionary.
        n_bootstrap: Number of bootstrap iterations (default 10,000).
        confidence: Confidence level (default 0.95).
        seed: Random seed for bootstrap calculations.

    Returns:
        Fully populated :class:`TournamentSummary`.
    """
    meta_dict = dict(meta) if meta is not None else {}
    participants_set = {rec.agent_x for rec in records} | {rec.agent_o for rec in records}
    if meta_dict and isinstance(meta_dict.get("agents"), dict):
        participants_set |= set(meta_dict["agents"].keys())

    participants = sorted(participants_set)

    agent_metrics: dict[str, AgentPerformanceMetrics] = {}
    for idx, agent_id in enumerate(participants):
        agent_metrics[agent_id] = compute_agent_metrics(
            matches=records,
            agent_id=agent_id,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed + idx * 17,
        )

    head_to_head: dict[tuple[str, str], HeadToHeadMetrics] = {}
    h2h_counter = 0
    for i, a in enumerate(participants):
        for j, b in enumerate(participants):
            if i != j:
                head_to_head[(a, b)] = compute_head_to_head_metrics(
                    matches=records,
                    agent_a=a,
                    agent_b=b,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    seed=seed + 1000 + h2h_counter * 13,
                )
                h2h_counter += 1

    seat_adv = compute_seat_advantage(
        matches=records,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed + 5000,
    )

    return TournamentSummary(
        meta=meta_dict,
        total_matches=len(records),
        participants=participants,
        agent_metrics=agent_metrics,
        head_to_head=head_to_head,
        seat_advantage=seat_adv,
    )


def summarize_tournament_log(
    log_path: str | Path,
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> TournamentSummary:
    """Load a tournament JSONL log from disk and compute complete statistical summaries.

    Args:
        log_path: Path to the JSONL tournament log file.
        n_bootstrap: Number of bootstrap iterations (default 10,000).
        confidence: Confidence level (default 0.95).
        seed: Random seed for bootstrap calculations.

    Returns:
        Fully populated :class:`TournamentSummary`.
    """
    meta_header, records = load_tournament_log(log_path)
    return summarize_tournament_records(
        records=records,
        meta=meta_header,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )


def export_summary_csv(
    summary: TournamentSummary,
    output_csv_path: str | Path,
) -> Path:
    """Export ranked tournament performance metrics to a CSV file.

    Args:
        summary: Computed :class:`TournamentSummary`.
        output_csv_path: Target destination path for the CSV file.

    Returns:
        Resolved Path to the saved CSV file.
    """
    path = Path(output_csv_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "agent_id",
        "total_games",
        "wins",
        "draws",
        "losses",
        "win_rate",
        "win_rate_ci_lower",
        "win_rate_ci_upper",
        "score",
        "score_ci_lower",
        "score_ci_upper",
        "score_x",
        "score_o",
        "delta_score",
        "delta_score_ci_lower",
        "delta_score_ci_upper",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "max_latency_ms",
        "avg_nodes_visited",
        "avg_simulations",
        "avg_depth",
        "total_moves",
    ]

    rankings = summary.get_rankings()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, m in enumerate(rankings, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "agent_id": m.agent_id,
                    "total_games": m.total_games,
                    "wins": m.wins,
                    "draws": m.draws,
                    "losses": m.losses,
                    "win_rate": f"{m.win_rate:.4f}",
                    "win_rate_ci_lower": f"{m.win_rate_ci[0]:.4f}",
                    "win_rate_ci_upper": f"{m.win_rate_ci[1]:.4f}",
                    "score": f"{m.score:.4f}",
                    "score_ci_lower": f"{m.score_ci[0]:.4f}",
                    "score_ci_upper": f"{m.score_ci[1]:.4f}",
                    "score_x": f"{m.score_x:.4f}",
                    "score_o": f"{m.score_o:.4f}",
                    "delta_score": f"{m.delta_score:.4f}",
                    "delta_score_ci_lower": f"{m.delta_score_ci[0]:.4f}",
                    "delta_score_ci_upper": f"{m.delta_score_ci[1]:.4f}",
                    "mean_latency_ms": f"{m.mean_latency_ms:.4f}",
                    "median_latency_ms": f"{m.median_latency_ms:.4f}",
                    "p95_latency_ms": f"{m.p95_latency_ms:.4f}",
                    "max_latency_ms": f"{m.max_latency_ms:.4f}",
                    "avg_nodes_visited": f"{m.avg_nodes_visited:.2f}",
                    "avg_simulations": f"{m.avg_simulations:.2f}",
                    "avg_depth": f"{m.avg_depth:.2f}",
                    "total_moves": m.total_moves,
                }
            )

    logger.info("Exported tournament summary CSV (%d agents) -> %s", len(rankings), path)
    return path


def format_summary_markdown(summary: TournamentSummary) -> str:
    """Format complete tournament performance statistics into GitHub-Flavored Markdown.

    Args:
        summary: Computed :class:`TournamentSummary`.

    Returns:
        Formatted Markdown document string ready for inclusion in reports.
    """
    meta = summary.meta
    t_name = meta.get("tournament_name", "Tournament Evaluation")
    b_cfg = meta.get("board_config", {})
    rows = b_cfg.get("rows", "?")
    cols = b_cfg.get("cols", "?")
    win_k = b_cfg.get("win_k", "?")
    budget_mode = meta.get("budget_mode", "default")
    hw = meta.get("hardware", {})

    lines: list[str] = [
        f"# Tournament Evaluation Summary: {t_name}",
        "",
        "## 1. Environment & Configuration",
        f"- **Board Configuration**: `{rows}x{cols}`, `K={win_k}`",
        f"- **Budget Mode**: `{budget_mode}`",
        f"- **Total Matches**: {summary.total_matches}",
        f"- **Unique Participants**: {len(summary.participants)}",
    ]

    if hw:
        platform_info = hw.get("platform", "Unknown")
        processor_info = hw.get("processor", "Unknown")
        cpu_cores = hw.get("cpu_count", "?")
        ram_gb = hw.get("ram_total_gb", 0.0)
        lines.append(
            f"- **Hardware Environment**: {platform_info} | {processor_info} | "
            f"CPUs: {cpu_cores} | RAM: {ram_gb:.2f} GB"
        )

    lines.extend([
        "",
        "## 2. Participant Leaderboard & Overall Metrics",
        "",
        "| Rank | Agent ID | Games | W | D | L | Win Rate [95% CI] | "
        "Score [95% CI] | Score (X) | Score (O) | ΔScore (X - O) [95% CI] | "
        "Mean Latency (ms) | P95 Latency (ms) | Avg Nodes | Avg Sims | Avg Depth |",
        "|---:|:---|---:|---:|---:|---:|:---:|:---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ])

    rankings = summary.get_rankings()
    for rank, m in enumerate(rankings, start=1):
        wr_str = f"{m.win_rate:.3f} [{m.win_rate_ci[0]:.3f}, {m.win_rate_ci[1]:.3f}]"
        sc_str = f"{m.score:.3f} [{m.score_ci[0]:.3f}, {m.score_ci[1]:.3f}]"
        ds_str = f"{m.delta_score:+.3f} [{m.delta_score_ci[0]:+.3f}, {m.delta_score_ci[1]:+.3f}]"
        lines.append(
            f"| {rank} | **{m.agent_id}** | {m.total_games} | {m.wins} | {m.draws} | {m.losses} | "
            f"{wr_str} | {sc_str} | {m.score_x:.3f} | {m.score_o:.3f} | {ds_str} | "
            f"{m.mean_latency_ms:.2f} | {m.p95_latency_ms:.2f} | "
            f"{m.avg_nodes_visited:.1f} | {m.avg_simulations:.1f} | {m.avg_depth:.1f} |"
        )

    lines.extend([
        "",
        "## 3. Head-to-Head Score Matrix",
        "",
    ])

    # Build cross table
    participants = summary.participants
    header_row = "| Agent | " + " | ".join(participants) + " |"
    separator_row = "|:---|" + "|".join([":---:" for _ in participants]) + "|"
    lines.append(header_row)
    lines.append(separator_row)

    for a in participants:
        row_cells = [f"**{a}**"]
        for b in participants:
            if a == b:
                row_cells.append("-")
            else:
                h2h = summary.get_head_to_head(a, b)
                if h2h and h2h.total_games > 0:
                    row_cells.append(
                        f"{h2h.score_a:.2f} ({h2h.wins_a}-{h2h.draws}-{h2h.wins_b})"
                    )
                else:
                    row_cells.append("N/A")
        lines.append("| " + " | ".join(row_cells) + " |")

    # Seat advantage section
    sa = summary.seat_advantage
    x_pct = (sa.x_win_rate * 100.0) if sa.total_games > 0 else 0.0
    o_pct = (sa.o_win_rate * 100.0) if sa.total_games > 0 else 0.0
    draw_pct = (sa.draw_rate * 100.0) if sa.total_games > 0 else 0.0

    lines.extend([
        "",
        "## 4. First-Mover Advantage (Seat Analysis)",
        "",
        f"- **Total Matches Analyzed**: {sa.total_games}",
        f"- **Player X (First Mover)**: {sa.x_wins} wins ({x_pct:.1f}%), "
        f"Win Rate 95% CI: [{sa.x_win_rate_ci[0]:.3f}, {sa.x_win_rate_ci[1]:.3f}], "
        f"Score: `{sa.score_x:.3f}`",
        f"- **Player O (Second Mover)**: {sa.o_wins} wins ({o_pct:.1f}%), "
        f"Win Rate 95% CI: [{sa.o_win_rate_ci[0]:.3f}, {sa.o_win_rate_ci[1]:.3f}], "
        f"Score: `{sa.score_o:.3f}`",
        f"- **Draws**: {sa.draws} ({draw_pct:.1f}%), "
        f"Draw Rate 95% CI: [{sa.draw_rate_ci[0]:.3f}, {sa.draw_rate_ci[1]:.3f}]",
        f"- **ΔScore (Score_X - Score_O)**: `{sa.delta_score:+.3f}` "
        f"(95% Bootstrap CI: `[{sa.delta_score_ci[0]:+.3f}, {sa.delta_score_ci[1]:+.3f}]`)",
        "",
    ])

    return "\n".join(lines)


def save_summary_markdown(
    summary: TournamentSummary,
    output_md_path: str | Path,
) -> Path:
    """Save the formatted tournament markdown summary report to disk.

    Args:
        summary: Computed :class:`TournamentSummary`.
        output_md_path: Target path for the Markdown file.

    Returns:
        Resolved Path to the saved Markdown file.
    """
    path = Path(output_md_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = format_summary_markdown(summary)
    path.write_text(content, encoding="utf-8")
    logger.info("Saved tournament summary Markdown -> %s", path)
    return path
