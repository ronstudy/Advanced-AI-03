"""Academic visualization and automated empirical report exporter (Task T5.2).

Generates publication-quality 300-DPI PNG figures (Fig 1..5) and consolidated
CSV/Markdown summary reports strictly from raw JSONL tournament, ablation, and
training logs (``Planning/RULES.md`` Zero Fabrication & Game API compliance).
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ttt_ai.agents.policy_agent import PolicyNetworkAgent
from ttt_ai.benchmark.ablations import AblationRunResult, load_ablation_result
from ttt_ai.benchmark.metrics import (
    AgentPerformanceMetrics,
    TournamentSummary,
    summarize_tournament_log,
)
from ttt_ai.benchmark.tournament import MatchRecord, load_tournament_log
from ttt_ai.config import BoardConfig
from ttt_ai.engine.exceptions import CheckpointVerificationError, InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.engine.rules import action_to_coords

logger = logging.getLogger(__name__)

_LEVEL_PATTERN = re.compile(r"^L(\d+)_")

_PALETTE: dict[str, str] = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "amber": "#E69F00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "charcoal": "#333333",
    "gray": "#7F8C8D",
}

_PLOT_RC: dict[str, Any] = {
    "font.size": 9.5,
    "axes.titlesize": 11.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
}


def _validate_dpi(dpi: int) -> int:
    """Validate target raster figure resolution."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 50:
        raise InvalidConfigError(f"dpi must be an integer >= 50, got {dpi!r}.")
    return int(dpi)


def _agent_sort_key(agent_id: str) -> tuple[int, str]:
    """Order agents naturally by hierarchy level (`L1`..`L7`), then alphabetically."""
    match = _LEVEL_PATTERN.match(agent_id)
    if match is not None:
        return (int(match.group(1), 10), agent_id)
    return (99, agent_id)


def _short_agent_label(agent_id: str) -> str:
    """Shorten verbose agent identifiers cleanly for compact chart ticks."""
    replacements = {
        "L1_Random": "L1 Random",
        "L2_RuleBased": "L2 RuleBased",
        "L3_AlphaBeta": "L3 AlphaBeta",
        "L4_MCTS": "L4 MCTS",
        "L5_StrongPolicy": "L5 Policy",
        "L6_DQN": "L6 DQN",
        "L6_TabularQ": "L6 TabularQ",
        "L7_HybridMCTS": "L7 HybridMCTS",
    }
    return replacements.get(agent_id, agent_id)


def _resolve_tournament_summary(
    source: TournamentSummary | str | Path | None,
    *,
    preferred_names: Sequence[str] = (
        "tournament_4x4_k4_default.jsonl",
        "tournament_4x4_k4_fixed_50ms.jsonl",
        "tournament_4x4_k4_fixed_10ms.jsonl",
        "tournament_3x3_k3_default.jsonl",
    ),
) -> TournamentSummary:
    """Load or unwrap a :class:`TournamentSummary` from a summary, file, or logs dir."""
    if isinstance(source, TournamentSummary):
        return source

    if source is None:
        candidate_dir = Path("logs")
    else:
        path = Path(source)
        if path.is_file():
            return summarize_tournament_log(path, n_bootstrap=1000, seed=42)
        if path.is_dir():
            candidate_dir = path
        else:
            raise FileNotFoundError(f"Tournament log or directory not found: {path}")

    for name in preferred_names:
        target = candidate_dir / name
        if target.is_file():
            return summarize_tournament_log(target, n_bootstrap=1000, seed=42)

    fallback = sorted(candidate_dir.glob("tournament_*.jsonl"))
    if fallback:
        return summarize_tournament_log(fallback[0], n_bootstrap=1000, seed=42)
    raise FileNotFoundError(f"No tournament_*.jsonl log found in '{candidate_dir}'.")


def _resolve_ablation_result(
    source: AblationRunResult | str | Path | None,
    *,
    default_filename: str,
) -> AblationRunResult:
    """Load or unwrap an :class:`AblationRunResult` from a result, file, or logs dir."""
    if isinstance(source, AblationRunResult):
        return source

    if source is None:
        target = Path("logs") / default_filename
    else:
        path = Path(source)
        target = (path / default_filename) if path.is_dir() else path

    if not target.is_file():
        raise FileNotFoundError(f"Ablation log not found: {target}")
    return load_ablation_result(target)


def _save_figure(fig: plt.Figure, output_path: str | Path, dpi: int) -> Path:
    """Persist matplotlib Figure as a 300-DPI PNG and close figure memory."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved figure -> %s (dpi=%d)", out, dpi)
    return out


def _board_label_from_meta(meta: dict[str, Any]) -> str:
    """Format board configuration metadata into a compact display tag."""
    b_cfg = meta.get("board_config", {})
    if isinstance(b_cfg, dict) and {"rows", "cols", "win_k"} <= b_cfg.keys():
        return f"{b_cfg['rows']}x{b_cfg['cols']} K={b_cfg['win_k']}"
    return "Board"


def _compute_ci_errors(
    values: Sequence[float],
    lowers: Sequence[float],
    uppers: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Convert `[lower, upper]` bounds into non-negative matplotlib asymmetric errors."""
    err_lo = [max(0.0, float(v) - float(lo)) for v, lo in zip(values, lowers, strict=True)]
    err_hi = [max(0.0, float(hi) - float(v)) for v, hi in zip(values, uppers, strict=True)]
    return err_lo, err_hi


# ---------------------------------------------------------------------------
# Fig 1: Dual Head-to-Head Matrix (As X vs As O)
# ---------------------------------------------------------------------------

def plot_fig1_head_to_head(
    source: TournamentSummary | str | Path | None = None,
    output_path: str | Path = "reports/figures/fig1_head_to_head_matrix.png",
    *,
    dpi: int = 300,
) -> Path:
    """Plot Fig 1: Side-by-side Head-to-Head Score & (W-D-L) matrices for X and O.

    Args:
        source: :class:`TournamentSummary`, path to a tournament JSONL log, or
            directory containing tournament logs. Defaults to ``logs/``.
        output_path: Destination PNG file path.
        dpi: Output resolution in dots per inch (default ``300``).

    Returns:
        Path to the saved PNG figure.
    """
    resolved_dpi = _validate_dpi(dpi)
    summary = _resolve_tournament_summary(source)
    agents = sorted(summary.participants, key=_agent_sort_key)
    n = len(agents)
    if n == 0:
        raise InvalidConfigError("Cannot plot Fig 1 from an empty tournament summary.")

    score_x_mat = np.full((n, n), np.nan, dtype=np.float64)
    score_o_mat = np.full((n, n), np.nan, dtype=np.float64)
    labels_x: list[list[str]] = [["-" for _ in range(n)] for _ in range(n)]
    labels_o: list[list[str]] = [["-" for _ in range(n)] for _ in range(n)]

    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i == j:
                continue
            h2h = summary.get_head_to_head(a, b)
            if h2h is None:
                continue
            if h2h.games_a_as_x > 0:
                sx = (h2h.wins_a_as_x + 0.5 * h2h.draws_a_as_x) / float(h2h.games_a_as_x)
                score_x_mat[i, j] = sx
                labels_x[i][j] = (
                    f"{sx:.2f}\n({h2h.wins_a_as_x}-{h2h.draws_a_as_x}-{h2h.losses_a_as_x})"
                )
            if h2h.games_a_as_o > 0:
                so = (h2h.wins_a_as_o + 0.5 * h2h.draws_a_as_o) / float(h2h.games_a_as_o)
                score_o_mat[i, j] = so
                labels_o[i][j] = (
                    f"{so:.2f}\n({h2h.wins_a_as_o}-{h2h.draws_a_as_o}-{h2h.losses_a_as_o})"
                )

    short_names = [_short_agent_label(a) for a in agents]
    board_lbl = _board_label_from_meta(summary.meta)
    budget_mode = str(summary.meta.get("budget_mode", "default"))

    with plt.rc_context(_PLOT_RC):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(max(12.0, 1.55 * n + 3.0), max(5.6, 0.72 * n + 1.8)),
            constrained_layout=True,
        )
        cmap = plt.get_cmap("RdYlGn").with_extremes(bad="#E8ECEF")

        panels = (
            (axes[0], score_x_mat, labels_x, "Row Agent as X (First Mover) vs Column (O)"),
            (axes[1], score_o_mat, labels_o, "Row Agent as O (Second Mover) vs Column (X)"),
        )
        im = None
        for ax, mat, cell_labels, subtitle in panels:
            im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_title(subtitle, pad=10)
            ax.set_xticks(range(n))
            ax.set_xticklabels(short_names, rotation=32, ha="right")
            ax.set_yticks(range(n))
            ax.set_yticklabels(short_names)
            ax.set_xlabel("Opponent")
            ax.set_ylabel("Evaluated Agent")
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

            for r in range(n):
                for c in range(n):
                    val = mat[r, c]
                    if np.isnan(val):
                        txt_color = "#555555"
                    elif val <= 0.22 or val >= 0.80:
                        txt_color = "white"
                    else:
                        txt_color = "#111111"
                    ax.text(
                        c,
                        r,
                        cell_labels[r][c],
                        ha="center",
                        va="center",
                        fontsize=7.2 if n >= 6 else 8.2,
                        fontweight="bold" if r != c else "normal",
                        color=txt_color,
                    )

        if im is not None:
            cbar = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.02)
            cbar.set_label("Match Score = (W + 0.5·D) / N", rotation=90)

        fig.suptitle(
            f"Fig 1: Dual Head-to-Head Score Matrix — {board_lbl} "
            f"({budget_mode}, N={summary.total_matches} matches)",
            fontsize=12.5,
            fontweight="bold",
        )
        return _save_figure(fig, output_path, resolved_dpi)


# ---------------------------------------------------------------------------
# Fig 2: Inference Latency (Log Scale) & Search Cost (Nodes / Simulations)
# ---------------------------------------------------------------------------

def plot_fig2_latency_and_nodes(
    source: TournamentSummary | str | Path | None = None,
    output_path: str | Path = "reports/figures/fig2_latency_and_nodes.png",
    *,
    dpi: int = 300,
) -> Path:
    """Plot Fig 2: Log-scale move latency (mean & p95) and average search nodes/sims.

    Args:
        source: :class:`TournamentSummary`, path to a tournament JSONL log, or
            directory containing tournament logs.
        output_path: Destination PNG file path.
        dpi: Output resolution in dots per inch (default ``300``).

    Returns:
        Path to the saved PNG figure.
    """
    resolved_dpi = _validate_dpi(dpi)
    summary = _resolve_tournament_summary(
        source,
        preferred_names=(
            "tournament_4x4_k4_default.jsonl",
            "tournament_4x4_k4_fixed_50ms.jsonl",
            "tournament_4x4_k4_fixed_10ms.jsonl",
            "tournament_3x3_k3_default.jsonl",
        ),
    )
    ordered: list[AgentPerformanceMetrics] = sorted(
        summary.agent_metrics.values(),
        key=lambda m: _agent_sort_key(m.agent_id),
    )
    if not ordered:
        raise InvalidConfigError("Cannot plot Fig 2 from an empty tournament summary.")

    labels = [_short_agent_label(m.agent_id) for m in ordered]
    x = np.arange(len(ordered), dtype=np.float64)
    means = [max(m.mean_latency_ms, 1e-3) for m in ordered]
    p95s = [max(m.p95_latency_ms, 1e-3) for m in ordered]
    avg_nodes = [m.avg_nodes_visited for m in ordered]
    avg_sims = [m.avg_simulations for m in ordered]

    board_lbl = _board_label_from_meta(summary.meta)
    budget_mode = str(summary.meta.get("budget_mode", "default"))

    with plt.rc_context(_PLOT_RC):
        fig, (ax_lat, ax_cost) = plt.subplots(
            1,
            2,
            figsize=(13.5, 5.4),
            constrained_layout=True,
        )

        # Left panel: Log-scale mean latency + p95 markers + stems.
        bars = ax_lat.bar(
            x,
            means,
            width=0.52,
            color=_PALETTE["blue"],
            alpha=0.88,
            edgecolor="#1B2631",
            linewidth=0.7,
            label="Mean Latency (ms)",
            zorder=3,
        )
        for xi, mu, p95 in zip(x, means, p95s, strict=True):
            ax_lat.plot([xi, xi], [mu, p95], color=_PALETTE["vermilion"], lw=1.6, zorder=4)
        ax_lat.scatter(
            x,
            p95s,
            color=_PALETTE["vermilion"],
            marker="D",
            s=46,
            zorder=5,
            label="P95 Latency (ms)",
        )
        ax_lat.axhline(
            10.0, color=_PALETTE["amber"], ls="--", lw=1.1, label="10 ms Budget", zorder=2
        )
        ax_lat.axhline(
            50.0, color=_PALETTE["purple"], ls=":", lw=1.3, label="50 ms Budget", zorder=2
        )

        ax_lat.set_yscale("log")
        max_lat = max(max(p95s), 60.0)
        ax_lat.set_ylim(0.01, max_lat * 2.6)
        ax_lat.set_xticks(x)
        ax_lat.set_xticklabels(labels, rotation=25, ha="right")
        ax_lat.set_ylabel("Move Latency (ms, log scale)")
        ax_lat.set_title("(a) Per-Move Latency: Mean vs P95")
        ax_lat.grid(axis="y", alpha=0.25, linestyle="--", which="both")
        ax_lat.legend(loc="upper left", frameon=True, ncol=2)

        for bar, m_obj, p95 in zip(bars, ordered, p95s, strict=True):
            top_y = max(bar.get_height(), p95)
            ax_lat.text(
                bar.get_x() + bar.get_width() / 2.0,
                top_y * 1.18,
                f"{m_obj.mean_latency_ms:.2f}\n(p95:{m_obj.p95_latency_ms:.1f})",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color="#222222",
            )

        # Right panel: Search cost (Avg Nodes Visited & Avg Simulations).
        width = 0.36
        nodes_plot = [max(v, 0.5) if v > 0 else 0.0 for v in avg_nodes]
        sims_plot = [max(v, 0.5) if v > 0 else 0.0 for v in avg_sims]

        b_nodes = ax_cost.bar(
            x - width / 2.0,
            nodes_plot,
            width=width,
            color=_PALETTE["green"],
            alpha=0.90,
            edgecolor="#1B2631",
            linewidth=0.7,
            label="Avg Nodes Visited / Move",
            zorder=3,
        )
        b_sims = ax_cost.bar(
            x + width / 2.0,
            sims_plot,
            width=width,
            color=_PALETTE["amber"],
            alpha=0.90,
            edgecolor="#1B2631",
            linewidth=0.7,
            label="Avg Simulations / Move",
            zorder=3,
        )
        ax_cost.set_yscale("log")
        max_cost = max(max(avg_nodes, default=1.0), max(avg_sims, default=1.0), 10.0)
        ax_cost.set_ylim(0.5, max_cost * 3.0)
        ax_cost.set_xticks(x)
        ax_cost.set_xticklabels(labels, rotation=25, ha="right")
        ax_cost.set_ylabel("Search Operations / Move (log scale)")
        ax_cost.set_title("(b) Average Search Cost: Nodes & Simulations")
        ax_cost.grid(axis="y", alpha=0.25, linestyle="--", which="both")
        ax_cost.legend(loc="upper left", frameon=True)

        for bar, raw_val in zip(b_nodes, avg_nodes, strict=True):
            if raw_val > 0.0:
                ax_cost.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() * 1.15,
                    f"{raw_val:.0f}" if raw_val >= 10 else f"{raw_val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                )
        for bar, raw_val in zip(b_sims, avg_sims, strict=True):
            if raw_val > 0.0:
                ax_cost.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() * 1.15,
                    f"{raw_val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                    color="#7E5109",
                )

        fig.suptitle(
            f"Fig 2: Computational Cost & Inference Latency — {board_lbl} ({budget_mode})",
            fontsize=12.5,
            fontweight="bold",
        )
        return _save_figure(fig, output_path, resolved_dpi)


# ---------------------------------------------------------------------------
# Fig 3: Ablation RQ1 (Policy vs Pure MCTS vs Hybrid MCTS) + Alpha-Beta Pruning
# ---------------------------------------------------------------------------

def plot_fig3_ablation_rq1(
    rq1_source: AblationRunResult | str | Path | None = None,
    ab1_source: AblationRunResult | str | Path | None = None,
    output_path: str | Path = "reports/figures/fig3_ablation_rq1_and_alphabeta.png",
    *,
    dpi: int = 300,
) -> Path:
    """Plot Fig 3: RQ1 Simulation/Time-Budget curves and Alpha-Beta pruning ablation.

    Args:
        rq1_source: ``AblationRunResult`` or path to ``ablation_rq1_hybrid_mcts.jsonl``.
        ab1_source: Optional ``AblationRunResult`` or path to ``ablation_ab_pruning.jsonl``.
        output_path: Destination PNG file path.
        dpi: Output resolution in dots per inch (default ``300``).

    Returns:
        Path to the saved PNG figure.
    """
    resolved_dpi = _validate_dpi(dpi)
    rq1_res = _resolve_ablation_result(
        rq1_source,
        default_filename="ablation_rq1_hybrid_mcts.jsonl",
    )
    cand_metrics = rq1_res.summary.get("candidate_metrics", {})
    if not cand_metrics:
        raise InvalidConfigError("RQ1 ablation log missing 'candidate_metrics' summary.")

    ab1_res: AblationRunResult | None = None
    try:
        ab1_res = _resolve_ablation_result(
            ab1_source,
            default_filename="ablation_ab_pruning.jsonl",
        )
    except FileNotFoundError:
        ab1_res = None

    with plt.rc_context(_PLOT_RC):
        fig, (ax_sim, ax_time, ax_ab) = plt.subplots(
            1,
            3,
            figsize=(16.0, 5.2),
            constrained_layout=True,
        )

        # Panel (a): Line plot over n_simulations with shaded 95% Bootstrap CI.
        families = (
            ("PureMCTS", "Pure MCTS (UCT)", _PALETTE["blue"], "o", "-"),
            ("HybridRollout", "Hybrid MCTS (Rollout)", _PALETTE["amber"], "s", "--"),
            ("HybridMixed", "Hybrid MCTS (Mixed)", _PALETTE["green"], "^", "-"),
        )
        for fam, label, color, marker, ls in families:
            pts = sorted(
                (
                    (int(m["budget_val"]), m)
                    for m in cand_metrics.values()
                    if m.get("family") == fam and m.get("budget_type") == "simulations"
                ),
                key=lambda item: item[0],
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [float(p[1]["score"]) for p in pts]
            lo = [float(p[1]["score_ci"][0]) for p in pts]
            hi = [float(p[1]["score_ci"][1]) for p in pts]
            ax_sim.plot(xs, ys, marker=marker, ls=ls, lw=2.0, color=color, label=label)
            ax_sim.fill_between(xs, lo, hi, color=color, alpha=0.14)

        baseline_refs = (
            ("Policy_Strong", "Policy Strong (0 sims)", _PALETTE["purple"], ":"),
            ("Policy_Fast", "Policy Fast (0 sims)", _PALETTE["sky"], "-."),
            ("Value_Greedy", "Value Greedy (1-ply)", _PALETTE["vermilion"], ":"),
        )
        for cid, lbl, color, ls in baseline_refs:
            if cid in cand_metrics:
                val = float(cand_metrics[cid]["score"])
                ax_sim.axhline(val, color=color, ls=ls, lw=1.4, label=f"{lbl}: {val:.3f}")

        ax_sim.set_title("(a) RQ1: Score vs Simulation Budget")
        ax_sim.set_xlabel("MCTS Simulations per Move")
        ax_sim.set_ylabel("Mean Score vs Opponent Pool (95% CI)")
        ax_sim.set_ylim(0.20, 0.72)
        ax_sim.grid(alpha=0.25, linestyle="--")
        ax_sim.legend(loc="lower right", frameon=True, fontsize=7.5)

        # Panel (b): Fixed Time Budgets (10ms vs 50ms) grouped bars + 95% CI error bars.
        time_budgets = sorted(
            {
                float(m["budget_val"])
                for m in cand_metrics.values()
                if m.get("budget_type") == "time_ms"
            }
        )
        if time_budgets:
            tx = np.arange(len(time_budgets), dtype=np.float64)
            bar_w = 0.24
            for idx, (fam, label, color, _, _) in enumerate(families):
                vals: list[float] = []
                lows: list[float] = []
                highs: list[float] = []
                sims_lbls: list[str] = []
                for tb in time_budgets:
                    matched = [
                        m
                        for m in cand_metrics.values()
                        if m.get("family") == fam
                        and m.get("budget_type") == "time_ms"
                        and abs(float(m.get("budget_val", -1)) - tb) < 1e-6
                    ]
                    if matched:
                        m = matched[0]
                        vals.append(float(m["score"]))
                        lows.append(float(m["score_ci"][0]))
                        highs.append(float(m["score_ci"][1]))
                        sims_lbls.append(f"{float(m.get('avg_simulations', 0)):.0f}s")
                    else:
                        vals.append(0.0)
                        lows.append(0.0)
                        highs.append(0.0)
                        sims_lbls.append("")
                yerr = _compute_ci_errors(vals, lows, highs)
                pos = tx + (idx - 1) * bar_w
                bars = ax_time.bar(
                    pos,
                    vals,
                    width=bar_w,
                    color=color,
                    alpha=0.88,
                    edgecolor="#1B2631",
                    linewidth=0.7,
                    yerr=yerr,
                    capsize=3,
                    label=label,
                    zorder=3,
                )
                for b, v, hi, sl in zip(bars, vals, highs, sims_lbls, strict=True):
                    if v > 0:
                        top_y = max(v, hi) + 0.008
                        ax_time.text(
                            b.get_x() + b.get_width() / 2.0,
                            top_y,
                            f"{v:.3f}\n({sl})",
                            ha="center",
                            va="bottom",
                            fontsize=6.8,
                        )
            ax_time.set_xticks(tx)
            ax_time.set_xticklabels([f"{int(tb)} ms / move" for tb in time_budgets])
            ax_time.set_ylim(0.32, 0.74)
            ax_time.set_ylabel("Mean Score vs Opponent Pool (95% CI)")
            ax_time.set_title("(b) RQ1: Score under Fixed Time Budget")
            ax_time.grid(axis="y", alpha=0.25, linestyle="--")
            ax_time.legend(loc="lower right", frameon=True, fontsize=7.5)
        else:
            ax_time.text(0.5, 0.5, "No time-budget entries found", ha="center", va="center")
            ax_time.set_title("(b) RQ1: Fixed Time Budget")

        # Panel (c): Alpha-Beta Pruning & Move Ordering Node Reduction (log scale).
        probe_by_depth = (
            ab1_res.summary.get("probe_by_depth", {}) if ab1_res is not None else {}
        )
        if probe_by_depth:
            depth_keys = sorted(
                probe_by_depth.keys(),
                key=lambda k: int(probe_by_depth[k]["depth"]),
            )
            dx = np.arange(len(depth_keys), dtype=np.float64)
            bw = 0.25
            mm_vals = [float(probe_by_depth[k]["mm_avg_nodes"]) for k in depth_keys]
            ab_vals = [float(probe_by_depth[k]["ab_no_order_avg_nodes"]) for k in depth_keys]
            ord_vals = [float(probe_by_depth[k]["ab_order_avg_nodes"]) for k in depth_keys]

            ax_ab.bar(
                dx - bw,
                mm_vals,
                width=bw,
                color=_PALETTE["vermilion"],
                edgecolor="#1B2631",
                linewidth=0.7,
                label="Raw Minimax",
            )
            b_ab = ax_ab.bar(
                dx,
                ab_vals,
                width=bw,
                color=_PALETTE["amber"],
                edgecolor="#1B2631",
                linewidth=0.7,
                label="Alpha-Beta (No Order)",
            )
            b_ord = ax_ab.bar(
                dx + bw,
                ord_vals,
                width=bw,
                color=_PALETTE["green"],
                edgecolor="#1B2631",
                linewidth=0.7,
                label="Alpha-Beta + Move Ordering",
            )
            ax_ab.set_yscale("log")
            ax_ab.set_ylim(10.0, max(mm_vals) * 3.5)
            ax_ab.set_xticks(dx)
            ax_ab.set_xticklabels([f"Depth {probe_by_depth[k]['depth']}" for k in depth_keys])
            ax_ab.set_ylabel("Average Visited Nodes (log scale)")
            ax_ab.set_title("(c) Alpha-Beta Pruning & Ordering (4x4 K=4)")
            ax_ab.grid(axis="y", alpha=0.25, linestyle="--", which="both")
            ax_ab.legend(loc="upper left", frameon=True, fontsize=7.5)

            for bar, k in zip(b_ab, depth_keys, strict=True):
                red = float(probe_by_depth[k]["ab_reduction_pct"])
                ax_ab.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() * 1.15,
                    f"-{red:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    color="#7E5109",
                )
            for bar, k in zip(b_ord, depth_keys, strict=True):
                red = float(probe_by_depth[k]["ab_order_reduction_pct"])
                ax_ab.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() * 1.15,
                    f"-{red:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    fontweight="bold",
                    color="#0B5345",
                )
        else:
            ax_ab.text(0.5, 0.5, "Alpha-Beta ablation log not provided", ha="center", va="center")
            ax_ab.set_title("(c) Alpha-Beta Pruning Ablation")

        fig.suptitle(
            "Fig 3: Search Guidance (RQ1) & Alpha-Beta Pruning Efficiency (4x4 K=4)",
            fontsize=12.5,
            fontweight="bold",
        )
        return _save_figure(fig, output_path, resolved_dpi)


# ---------------------------------------------------------------------------
# Fig 4: Ablation RQ2 (First-Mover Advantage across 3x3 K=3, 4x4 K=3, 4x4 K=4)
# ---------------------------------------------------------------------------

def plot_fig4_ablation_rq2(
    rq2_source: AblationRunResult | str | Path | None = None,
    output_path: str | Path = "reports/figures/fig4_ablation_rq2_seat_advantage.png",
    *,
    dpi: int = 300,
) -> Path:
    """Plot Fig 4: First-mover advantage (`delta_score`) across rule variants and AI levels.

    Args:
        rq2_source: ``AblationRunResult`` or path to ``ablation_rq2_seat_advantage.jsonl``.
        output_path: Destination PNG file path.
        dpi: Output resolution in dots per inch (default ``300``).

    Returns:
        Path to the saved PNG figure.
    """
    resolved_dpi = _validate_dpi(dpi)
    rq2_res = _resolve_ablation_result(
        rq2_source,
        default_filename="ablation_rq2_seat_advantage.jsonl",
    )
    boards = rq2_res.summary.get("boards", {})
    if not boards:
        raise InvalidConfigError("RQ2 ablation log missing 'boards' summary.")

    preferred_order = ("4x4_k3", "3x3_k3", "4x4_k4")
    board_keys = [k for k in preferred_order if k in boards] + [
        k for k in sorted(boards.keys()) if k not in preferred_order
    ]
    board_titles = {
        "4x4_k3": "4x4, K=3",
        "3x3_k3": "3x3, K=3",
        "4x4_k4": "4x4, K=4",
    }
    board_colors = {
        "4x4_k3": _PALETTE["vermilion"],
        "3x3_k3": _PALETTE["amber"],
        "4x4_k4": _PALETTE["blue"],
    }

    with plt.rc_context(_PLOT_RC):
        fig, (ax_global, ax_levels) = plt.subplots(
            1,
            2,
            figsize=(14.5, 5.5),
            gridspec_kw={"width_ratios": [1.0, 1.55]},
            constrained_layout=True,
        )

        # Panel (a): Global outcome breakdown + DeltaScore callouts.
        gx = np.arange(len(board_keys), dtype=np.float64)
        x_rates = [100.0 * float(boards[k]["seat_advantage"]["x_win_rate"]) for k in board_keys]
        d_rates = [100.0 * float(boards[k]["seat_advantage"]["draw_rate"]) for k in board_keys]
        o_rates = [100.0 * float(boards[k]["seat_advantage"]["o_win_rate"]) for k in board_keys]

        ax_global.bar(
            gx,
            x_rates,
            width=0.52,
            color=_PALETTE["blue"],
            label="X Wins (1st Mover)",
            edgecolor="#1B2631",
            linewidth=0.7,
        )
        ax_global.bar(
            gx,
            d_rates,
            width=0.52,
            bottom=x_rates,
            color="#D5D8DC",
            label="Draws",
            edgecolor="#1B2631",
            linewidth=0.7,
        )
        bottom_o = [xr + dr for xr, dr in zip(x_rates, d_rates, strict=True)]
        ax_global.bar(
            gx,
            o_rates,
            width=0.52,
            bottom=bottom_o,
            color=_PALETTE["vermilion"],
            label="O Wins (2nd Mover)",
            edgecolor="#1B2631",
            linewidth=0.7,
        )

        for xi, k, xr, dr, orate in zip(
            gx, board_keys, x_rates, d_rates, o_rates, strict=True
        ):
            sa = boards[k]["seat_advantage"]
            ds = float(sa["delta_score"])
            ci = sa["delta_score_ci"]
            if xr >= 9.0:
                ax_global.text(
                    xi, xr / 2.0, f"{xr:.1f}%", ha="center", va="center", color="white",
                    fontweight="bold", fontsize=8.0,
                )
            if dr >= 9.0:
                ax_global.text(
                    xi, xr + dr / 2.0, f"{dr:.1f}%", ha="center", va="center", color="#222222",
                    fontweight="bold", fontsize=8.0,
                )
            if orate >= 9.0:
                ax_global.text(
                    xi, xr + dr + orate / 2.0, f"{orate:.1f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=8.0,
                )
            ax_global.text(
                xi,
                101.8,
                f"ΔScore={ds:+.3f}\n[{ci[0]:+.3f}, {ci[1]:+.3f}]",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold",
                color="#1B2631",
            )

        ax_global.set_xticks(gx)
        ax_global.set_xticklabels([board_titles.get(k, k) for k in board_keys])
        ax_global.set_ylim(0.0, 128.0)
        ax_global.set_ylabel("Match Outcomes (%)")
        ax_global.set_title("(a) Outcome Shift & Global ΔScore (X - O)")
        ax_global.legend(loc="upper center", ncol=3, frameon=True, fontsize=7.2)

        # Panel (b): Per-Level DeltaScore grouped bar chart across the 3 rulesets.
        def _canonical_slot(aid: str) -> str:
            if aid.startswith("L6_"):
                return "L6_RL"
            return aid

        slot_display = {
            "L1_Random": "L1 Random",
            "L2_RuleBased": "L2 RuleBased",
            "L3_AlphaBeta": "L3 AlphaBeta",
            "L4_MCTS": "L4 MCTS",
            "L5_StrongPolicy": "L5 Policy",
            "L6_RL": "L6 RL (Q/DQN)",
            "L7_HybridMCTS": "L7 HybridMCTS",
        }
        slots_set: set[str] = set()
        slot_data: dict[str, dict[str, dict[str, Any]]] = {k: {} for k in board_keys}
        for k in board_keys:
            for aid, m in boards[k].get("agent_metrics", {}).items():
                slot = _canonical_slot(aid)
                slots_set.add(slot)
                slot_data[k][slot] = m

        ordered_slots = sorted(slots_set, key=_agent_sort_key)
        sx = np.arange(len(ordered_slots), dtype=np.float64)
        bw = 0.25 if len(board_keys) == 3 else 0.35

        for idx, k in enumerate(board_keys):
            vals: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            positions: list[float] = []
            offset = (idx - (len(board_keys) - 1) / 2.0) * bw
            for pos_i, slot in enumerate(ordered_slots):
                if slot in slot_data[k]:
                    m = slot_data[k][slot]
                    vals.append(float(m["delta_score"]))
                    lows.append(float(m["delta_score_ci"][0]))
                    highs.append(float(m["delta_score_ci"][1]))
                    positions.append(sx[pos_i] + offset)
            if positions:
                yerr = _compute_ci_errors(vals, lows, highs)
                ax_levels.bar(
                    positions,
                    vals,
                    width=bw,
                    color=board_colors.get(k, _PALETTE["green"]),
                    alpha=0.88,
                    edgecolor="#1B2631",
                    linewidth=0.7,
                    yerr=yerr,
                    capsize=2.5,
                    label=board_titles.get(k, k),
                    zorder=3,
                )

        ax_levels.axhline(0.0, color="#222222", lw=1.0, ls="--", zorder=2)
        ax_levels.set_xticks(sx)
        ax_levels.set_xticklabels(
            [slot_display.get(s, s) for s in ordered_slots],
            rotation=22,
            ha="right",
        )
        ax_levels.set_ylabel("ΔScore = Score(X) - Score(O) [95% CI]")
        ax_levels.set_title("(b) First-Mover Advantage by Agent Level & Win-Length K")
        ax_levels.set_ylim(-0.15, 0.92)
        ax_levels.grid(axis="y", alpha=0.25, linestyle="--")
        ax_levels.legend(loc="upper right", frameon=True)

        fig.suptitle(
            "Fig 4: Ablation RQ2 — First-Mover Advantage Across Rule Variants & AI Levels",
            fontsize=12.5,
            fontweight="bold",
        )
        return _save_figure(fig, output_path, resolved_dpi)


# ---------------------------------------------------------------------------
# Fig 5: Ablation RQ3 (Learning Curves + Unseen Opponent Generalization)
# ---------------------------------------------------------------------------

def _load_supervised_curve(path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    """Read `(epochs, train_losses, val_losses, val_accs)` from a supervised train JSONL."""
    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []
    if not path.is_file():
        return epochs, train_losses, val_losses, val_accs
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") == "epoch":
                epochs.append(int(obj["epoch"]))
                train_losses.append(float(obj["train_loss"]))
                val_losses.append(float(obj["val_loss"]))
                val_accs.append(
                    float(obj["val_accuracy"]) if obj.get("val_accuracy") is not None else np.nan
                )
    return epochs, train_losses, val_losses, val_accs


def _load_rl_multiseed_curves(
    paths: Sequence[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate RL eval scores -> `(progress_pct, rand_mean, rand_std, rule_mean, rule_std)`."""
    per_seed: list[list[tuple[float, float, float]]] = []
    for path in paths:
        if not path.is_file():
            continue
        rows: list[tuple[int, float, float]] = []
        total_eps = 1
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("type") == "meta":
                    total_eps = max(1, int(obj.get("episodes", 1)))
                elif obj.get("type") == "eval":
                    rows.append(
                        (
                            int(obj["episode"]),
                            float(obj["eval_score_vs_random"]),
                            float(obj["eval_score_vs_rule"]),
                        )
                    )
        if rows:
            max_ep = max(total_eps, rows[-1][0])
            per_seed.append([(100.0 * ep / max_ep, sr, su) for ep, sr, su in rows])

    if not per_seed:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty, empty

    min_len = min(len(s) for s in per_seed)
    trimmed = np.asarray([s[:min_len] for s in per_seed], dtype=np.float64)
    pct = trimmed[0, :, 0]
    rand_mean = np.mean(trimmed[:, :, 1], axis=0)
    rand_std = np.std(trimmed[:, :, 1], axis=0)
    rule_mean = np.mean(trimmed[:, :, 2], axis=0)
    rule_std = np.std(trimmed[:, :, 2], axis=0)
    return pct, rand_mean, rand_std, rule_mean, rule_std


def _plot_rq3_board_panel(
    ax: plt.Axes,
    board_data: dict[str, Any],
    *,
    title: str,
) -> None:
    """Render grouped generalization bar chart for one RQ3 board configuration."""
    cands = board_data.get("candidates", {})
    if not cands:
        ax.text(0.5, 0.5, "No RQ3 candidate data", ha="center", va="center")
        ax.set_title(title)
        return

    sl_keys = [k for k in ("SL_FastPolicy", "SL_StrongPolicy", "SL_ValueGreedy") if k in cands]
    rl_keys = sorted(k for k in cands if k.startswith("RL_"))

    eval_axes = (
        ("vs_unseen_mcts", "Unseen MCTS"),
        ("vs_unseen_all", "Unseen All\n(MCTS + AB)"),
        ("vs_baseline", "Baseline\n(Rand + Rule)"),
        ("overall", "Overall\n(All Opps)"),
    )

    series: list[tuple[str, str, list[float], list[float], list[float]]] = []
    sl_styles = {
        "SL_FastPolicy": ("SL FastPolicy", _PALETTE["sky"]),
        "SL_StrongPolicy": ("SL StrongPolicy", _PALETTE["purple"]),
        "SL_ValueGreedy": ("SL ValueGreedy", _PALETTE["amber"]),
    }
    for sl_id in sl_keys:
        m = cands[sl_id]
        vals, lows, highs = [], [], []
        for metric_key, _ in eval_axes:
            src = m if metric_key == "overall" else m[metric_key]
            vals.append(float(src["score"]))
            lows.append(float(src["score_ci"][0]))
            highs.append(float(src["score_ci"][1]))
        lbl, col = sl_styles.get(sl_id, (sl_id, _PALETTE["gray"]))
        series.append((lbl, col, vals, lows, highs))

    if rl_keys:
        rl_vals, rl_lows, rl_highs = [], [], []
        for metric_key, _ in eval_axes:
            seed_scores = [
                float(
                    cands[rk]["score"]
                    if metric_key == "overall"
                    else cands[rk][metric_key]["score"]
                )
                for rk in rl_keys
            ]
            mu = float(np.mean(seed_scores))
            sd = float(np.std(seed_scores))
            rl_vals.append(mu)
            rl_lows.append(max(0.0, mu - sd))
            rl_highs.append(min(1.0, mu + sd))
        rl_lbl = (
            f"RL DQN ({len(rl_keys)}-seed μ±σ)"
            if any("DQN" in k for k in rl_keys)
            else f"RL TabQ ({len(rl_keys)}-seed μ±σ)"
        )
        series.append((rl_lbl, _PALETTE["green"], rl_vals, rl_lows, rl_highs))

    gx = np.arange(len(eval_axes), dtype=np.float64)
    n_series = max(1, len(series))
    bw = min(0.20, 0.78 / n_series)

    for idx, (lbl, col, vals, lows, highs) in enumerate(series):
        offset = (idx - (n_series - 1) / 2.0) * bw
        yerr = _compute_ci_errors(vals, lows, highs)
        bars = ax.bar(
            gx + offset,
            vals,
            width=bw,
            color=col,
            alpha=0.90,
            edgecolor="#1B2631",
            linewidth=0.7,
            yerr=yerr,
            capsize=2.5,
            label=lbl,
            zorder=3,
        )
        for b, v, hi in zip(bars, vals, highs, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                max(v, hi) + 0.015,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                rotation=90 if n_series >= 4 else 0,
            )

    ax.axhline(0.50, color="#666666", ls="--", lw=0.9, zorder=2)
    ax.set_xticks(gx)
    ax.set_xticklabels([lbl for _, lbl in eval_axes])
    ax.set_ylim(0.0, 0.96)
    ax.set_ylabel("Match Score = (W + 0.5·D) / N")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", frameon=True, fontsize=7.5, ncol=2)


def plot_fig5_ablation_rq3(
    rq3_source: AblationRunResult | str | Path | None = None,
    logs_dir: str | Path = "logs",
    output_path: str | Path = "reports/figures/fig5_ablation_rq3_learning_and_generalization.png",
    *,
    dpi: int = 300,
) -> Path:
    """Plot Fig 5: Supervised/RL multi-seed learning curves and Unseen Opponents (RQ3).

    Args:
        rq3_source: ``AblationRunResult`` or path to
            ``ablation_rq3_rl_generalization.jsonl``.
        logs_dir: Directory containing ``train_*.jsonl`` learning curve logs.
        output_path: Destination PNG file path.
        dpi: Output resolution in dots per inch (default ``300``).

    Returns:
        Path to the saved PNG figure.
    """
    resolved_dpi = _validate_dpi(dpi)
    rq3_res = _resolve_ablation_result(
        rq3_source,
        default_filename="ablation_rq3_rl_generalization.jsonl",
    )
    boards = rq3_res.summary.get("boards", {})
    if not boards:
        raise InvalidConfigError("RQ3 ablation log missing 'boards' summary.")

    resolved_logs = Path(logs_dir)

    with plt.rc_context(_PLOT_RC):
        fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.6), constrained_layout=True)
        ax_sl, ax_rl, ax_4x4, ax_3x3 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        # Panel (a): Supervised Learning Curves (4x4 K=4).
        sl_specs = (
            ("train_strongpolicynet_4x4_k4.jsonl", "StrongPolicyNet", _PALETTE["purple"]),
            ("train_fastpolicynet_4x4_k4.jsonl", "FastPolicyNet", _PALETTE["blue"]),
            ("train_valuenet_4x4_k4.jsonl", "ValueNet (MSE)", _PALETTE["amber"]),
        )
        has_sl = False
        ax_acc = ax_sl.twinx()
        ax_acc.spines["top"].set_visible(False)
        for fname, label, color in sl_specs:
            ep, tr_loss, val_loss, val_acc = _load_supervised_curve(resolved_logs / fname)
            if not ep:
                continue
            has_sl = True
            ax_sl.plot(ep, val_loss, color=color, lw=2.0, label=f"{label} Val Loss")
            ax_sl.plot(ep, tr_loss, color=color, lw=1.2, ls="--", alpha=0.65)
            valid_acc = [
                (e, a * 100.0)
                for e, a in zip(ep, val_acc, strict=True)
                if not np.isnan(a)
            ]
            if valid_acc:
                ax_acc.plot(
                    [p[0] for p in valid_acc],
                    [p[1] for p in valid_acc],
                    color=color,
                    ls=":",
                    lw=1.7,
                    label=f"{label} Val Acc (%)",
                )
        if has_sl:
            ax_sl.set_xlabel("Training Epoch")
            ax_sl.set_ylabel("Loss (Solid: Val | Dashed: Train)")
            ax_acc.set_ylabel("Validation Accuracy (%, Dotted)")
            ax_acc.set_ylim(25.0, 95.0)
            ax_sl.grid(alpha=0.25, linestyle="--")
            h1, l1 = ax_sl.get_legend_handles_labels()
            h2, l2 = ax_acc.get_legend_handles_labels()
            ax_sl.legend(h1 + h2, l1 + l2, loc="center right", frameon=True, fontsize=7.0)
        else:
            ax_acc.set_visible(False)
            ax_sl.text(0.5, 0.5, "No supervised training logs found", ha="center", va="center")
        ax_sl.set_title("(a) Supervised Learning Curves (4x4 K=4)")

        # Panel (b): RL Multi-Seed Learning Curves (mean +/- std across seeds 42, 123, 999).
        dqn_paths = [resolved_logs / f"train_dqn_4x4_k4_seed{s}.jsonl" for s in (42, 123, 999)]
        tabq_paths = [
            resolved_logs / f"train_tabular_q_3x3_k3_seed{s}.jsonl" for s in (42, 123, 999)
        ]
        pct_d, r_mu_d, r_sd_d, u_mu_d, u_sd_d = _load_rl_multiseed_curves(dqn_paths)
        pct_t, r_mu_t, r_sd_t, u_mu_t, u_sd_t = _load_rl_multiseed_curves(tabq_paths)

        has_rl = False
        if len(pct_d) > 0:
            has_rl = True
            ax_rl.plot(
                pct_d, r_mu_d, color=_PALETTE["green"], lw=2.0, marker="o", ms=4,
                label="DQN (4x4 K=4) vs Random (3-seed)",
            )
            ax_rl.fill_between(
                pct_d, r_mu_d - r_sd_d, np.minimum(1.0, r_mu_d + r_sd_d),
                color=_PALETTE["green"], alpha=0.18,
            )
            ax_rl.plot(
                pct_d, u_mu_d, color=_PALETTE["blue"], lw=1.8, ls="--", marker="s", ms=3.5,
                label="DQN (4x4 K=4) vs RuleBased",
            )
            ax_rl.fill_between(
                pct_d, u_mu_d - u_sd_d, u_mu_d + u_sd_d, color=_PALETTE["blue"], alpha=0.14
            )
        if len(pct_t) > 0:
            has_rl = True
            ax_rl.plot(
                pct_t, r_mu_t, color=_PALETTE["vermilion"], lw=2.0, marker="^", ms=4,
                label="TabularQ (3x3 K=3) vs Random (3-seed)",
            )
            ax_rl.fill_between(
                pct_t, r_mu_t - r_sd_t, np.minimum(1.0, r_mu_t + r_sd_t),
                color=_PALETTE["vermilion"], alpha=0.16,
            )

        if has_rl:
            ax_rl.axhline(0.50, color="#666666", ls=":", lw=1.0)
            ax_rl.set_xlabel("Training Progress (% Episodes: 20k DQN / 60k TabQ)")
            ax_rl.set_ylabel("Evaluation Score (Mean ± Std)")
            ax_rl.set_ylim(0.38, 1.03)
            ax_rl.grid(alpha=0.25, linestyle="--")
            ax_rl.legend(loc="lower right", frameon=True, fontsize=7.4)
        else:
            ax_rl.text(0.5, 0.5, "No RL multi-seed training logs found", ha="center", va="center")
        ax_rl.set_title("(b) RL Multi-Seed Learning Curves (Seeds 42, 123, 999)")

        # Panels (c) & (d): RQ3 Generalization vs Unseen Opponents on 4x4 K=4 and 3x3 K=3.
        if "4x4_k4" in boards:
            _plot_rq3_board_panel(
                ax_4x4,
                boards["4x4_k4"],
                title="(c) RQ3 Generalization on 4x4 K=4 (SL vs 3-Seed DQN)",
            )
        else:
            first_key = sorted(boards.keys())[0]
            _plot_rq3_board_panel(
                ax_4x4,
                boards[first_key],
                title=f"(c) RQ3 Generalization on {first_key}",
            )

        if "3x3_k3" in boards:
            _plot_rq3_board_panel(
                ax_3x3,
                boards["3x3_k3"],
                title="(d) RQ3 Generalization on 3x3 K=3 (SL vs 3-Seed TabularQ)",
            )
        else:
            ax_3x3.text(0.5, 0.5, "3x3_k3 RQ3 data not in log", ha="center", va="center")
            ax_3x3.set_title("(d) RQ3 Generalization on 3x3 K=3")

        fig.suptitle(
            "Fig 5: Ablation RQ3 — Learning Curves & Generalization Against Unseen Opponents",
            fontsize=13.0,
            fontweight="bold",
        )
        return _save_figure(fig, output_path, resolved_dpi)


# ---------------------------------------------------------------------------
# Automated Empirical Report & Failure-Mode Excerpt Generators
# ---------------------------------------------------------------------------

def _replay_match_record(rec: MatchRecord) -> tuple[ExtendedTicTacToe, list[str]]:
    """Replay a :class:`MatchRecord` via public `ExtendedTicTacToe` APIs."""
    b_cfg = rec.board_config
    game = ExtendedTicTacToe(int(b_cfg["rows"]), int(b_cfg["cols"]), int(b_cfg["win_k"]))
    trace: list[str] = []
    for ply_idx, mv in enumerate(rec.moves, start=1):
        action = int(mv[0])
        player = game.current_player
        r, c = action_to_coords(action, game.cols)
        trace.append(f"{ply_idx}.{player.symbol}({r},{c})[#{action}]")
        game.step(action)
    return game, trace


def _inspect_policy_prior_on_state(
    board_cfg: BoardConfig,
    prefix_actions: Sequence[int],
    actual_action: int,
    checkpoints_dir: Path,
) -> str | None:
    """Evaluate `StrongPolicyNet` prior on a historical board state using public APIs."""
    ckpt = checkpoints_dir / f"strongpolicynet_{board_cfg.name}.pt"
    if not ckpt.is_file():
        return None
    try:
        agent = PolicyNetworkAgent(
            checkpoint_path=ckpt,
            board_config=board_cfg,
            sample=False,
            device="cpu",
            name="PriorInspector",
        )
    except (CheckpointVerificationError, InvalidConfigError, FileNotFoundError):
        return None

    sim = ExtendedTicTacToe(board_cfg.rows, board_cfg.cols, board_cfg.win_k)
    for act in prefix_actions:
        sim.step(act)
    if sim.is_terminal():
        return None

    decision = agent.select_action(sim)
    top3 = decision.top_actions(k=3)
    actual_prob = decision.action_scores.get(actual_action, 0.0)
    top_str = ", ".join(
        f"({action_to_coords(a, board_cfg.cols)[0]},{action_to_coords(a, board_cfg.cols)[1]})"
        f"[#{a}]: {p * 100.0:.1f}%"
        for a, p in top3
    )
    act_r, act_c = action_to_coords(actual_action, board_cfg.cols)
    return (
        f"StrongPolicyNet Top-3 Priors before move ({act_r},{act_c})[#{actual_action}]: "
        f"{top_str} | Played move prior P(s,a) = {actual_prob * 100.0:.1f}%"
    )


def _format_case_markdown(
    *,
    title: str,
    context: str,
    rec: MatchRecord,
    checkpoints_dir: Path,
    inspect_ply_for_loser: bool = True,
) -> list[str]:
    """Render one replayed failure/tactical case into Markdown lines."""
    game, trace = _replay_match_record(rec)
    winner_str = (
        f"{rec.agent_x} (X)"
        if rec.winner == 1
        else (f"{rec.agent_o} (O)" if rec.winner == -1 else "Draw")
    )
    lines: list[str] = [
        f"### {title}",
        f"- **Context**: {context}",
        f"- **Match ID**: `{rec.match_id}` (`seed={rec.seed}`, `budget={rec.budget_mode}`)",
        f"- **Matchup**: `X = {rec.agent_x}` vs `O = {rec.agent_o}`",
        f"- **Outcome**: **{winner_str}** in `{rec.num_moves}` plies "
        f"(winning line: `{rec.winning_line}`)",
        f"- **Move Sequence**: {' -> '.join(trace)}",
    ]

    b_cfg = BoardConfig(
        rows=int(rec.board_config["rows"]),
        cols=int(rec.board_config["cols"]),
        win_k=int(rec.board_config["win_k"]),
    )
    if len(rec.moves) >= 2 and rec.winner != 0:
        # Inspect the loser's final move immediately before the opponent's winning ply.
        target_idx = len(rec.moves) - 2 if inspect_ply_for_loser else len(rec.moves) - 1
        prefix = [int(mv[0]) for mv in rec.moves[:target_idx]]
        played_action = int(rec.moves[target_idx][0])
        prior_note = _inspect_policy_prior_on_state(
            board_cfg=b_cfg,
            prefix_actions=prefix,
            actual_action=played_action,
            checkpoints_dir=checkpoints_dir,
        )
        if prior_note:
            lines.append(f"- **Policy Prior Analysis**: {prior_note}")

    lines.extend(["", "```text", game.render(), "```", ""])
    return lines


def _extract_failure_cases_markdown(
    *,
    logs_dir: Path,
    checkpoints_dir: Path,
) -> str:
    """Build `reports/sample_failure_cases.md` automatically from real raw JSONL logs."""
    lines: list[str] = [
        "# Sample Failure Modes & Tactical Analysis (Automated from Raw Logs)",
        "",
        "All boards and move sequences below are replayed verbatim via "
        "`ExtendedTicTacToe` public APIs from raw `logs/*.jsonl` records.",
        "",
    ]
    extracted_count = 0

    def _find_match(
        log_path: Path,
        predicate: Any,
    ) -> MatchRecord | None:
        if not log_path.is_file():
            return None
        _, recs = load_tournament_log(log_path)
        for rec in recs:
            if predicate(rec):
                return rec
        return None

    # Case 1: L5_StrongPolicy loses to L4_MCTS on 4x4_k4 (no lookahead failure).
    case1 = _find_match(
        logs_dir / "tournament_4x4_k4_default.jsonl",
        lambda r: (
            (r.agent_x == "L5_StrongPolicy" and r.agent_o == "L4_MCTS" and r.winner == -1)
            or (r.agent_o == "L5_StrongPolicy" and r.agent_x == "L4_MCTS" and r.winner == 1)
        ),
    )
    if case1 is not None:
        extracted_count += 1
        lines.extend(
            _format_case_markdown(
                title=(
                    "Case 1 (RQ1 / RQ3): Supervised Policy (`L5_StrongPolicy`) "
                    "Defeated by Lookahead (`L4_MCTS`, `4x4 K=4`)"
                ),
                context=(
                    "Illustrates how zero-search policy imitation misses tactical "
                    "traps against unseen tree search."
                ),
                rec=case1,
                checkpoints_dir=checkpoints_dir,
            )
        )

    # Case 2: L6_DQN loses to L7_HybridMCTS on 4x4_k4.
    case2 = _find_match(
        logs_dir / "tournament_4x4_k4_default.jsonl",
        lambda r: (
            (r.agent_x == "L6_DQN" and r.agent_o == "L7_HybridMCTS" and r.winner == -1)
            or (r.agent_o == "L6_DQN" and r.agent_x == "L7_HybridMCTS" and r.winner == 1)
        ),
    )
    if case2 is not None:
        extracted_count += 1
        lines.extend(
            _format_case_markdown(
                title="Case 2 (RQ1): `L6_DQN` vs `L7_HybridMCTS` (`4x4 K=4`)",
                context=(
                    "Demonstrates how PUCT search refines learned priors into a "
                    "tactical victory."
                ),
                rec=case2,
                checkpoints_dir=checkpoints_dir,
            )
        )

    # Case 3: RQ2 First-Mover Dominance on 4x4_k3.
    case3 = _find_match(
        logs_dir / "ablation_rq2_seat_advantage.jsonl",
        lambda r: (
            r.board_config.get("rows") == 4
            and r.board_config.get("win_k") == 3
            and r.winner == 1
        ),
    )
    if case3 is not None:
        extracted_count += 1
        lines.extend(
            _format_case_markdown(
                title="Case 3 (RQ2): Amplified First-Mover Fork Victory on `4x4, K=3`",
                context=(
                    "Illustrates rapid first-mover (`X`) fork conversion when `K=3` "
                    "on a `4x4` board."
                ),
                rec=case3,
                checkpoints_dir=checkpoints_dir,
            )
        )

    # Case 4: 3x3_k3 control: L5_StrongPolicy loses to L7_HybridMCTS.
    case4 = _find_match(
        logs_dir / "tournament_3x3_k3_default.jsonl",
        lambda r: (
            (r.agent_x == "L5_StrongPolicy" and r.agent_o == "L7_HybridMCTS" and r.winner == -1)
            or (r.agent_o == "L5_StrongPolicy" and r.agent_x == "L7_HybridMCTS" and r.winner == 1)
        ),
    )
    if case4 is not None:
        extracted_count += 1
        lines.extend(
            _format_case_markdown(
                title="Case 4 (3x3 K=3 Control): `L5_StrongPolicy` Outplayed by `L7_HybridMCTS`",
                context="Compares raw Policy prior against Hybrid PUCT outcome on `3x3 K=3`.",
                rec=case4,
                checkpoints_dir=checkpoints_dir,
            )
        )

    # Fallback for custom/minimal log directories in tests.
    if extracted_count == 0:
        for t_path in sorted(logs_dir.glob("*.jsonl")):
            if t_path.name.startswith("train_"):
                continue
            try:
                _, recs = load_tournament_log(t_path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            decisive = next((r for r in recs if r.winner != 0), None)
            if decisive is not None:
                extracted_count += 1
                lines.extend(
                    _format_case_markdown(
                        title=f"Case 1: Decisive Match Excerpt from `{t_path.name}`",
                        context="Automatically extracted decisive match from available JSONL log.",
                        rec=decisive,
                        checkpoints_dir=checkpoints_dir,
                    )
                )
                break

    if extracted_count == 0:
        lines.append("No match records available in the provided logs directory.")

    return "\n".join(lines)


def _export_consolidated_csv(
    *,
    tournament_logs: Sequence[tuple[Path, TournamentSummary]],
    ablation_results: Sequence[AblationRunResult],
    output_csv: Path,
) -> Path:
    """Write `reports/summary_metrics.csv` combining tournament and ablation metrics."""
    fieldnames = [
        "source_type",
        "experiment_id",
        "board",
        "budget_or_group",
        "agent_or_candidate",
        "total_games",
        "wins",
        "draws",
        "losses",
        "win_rate",
        "score",
        "score_ci_lower",
        "score_ci_upper",
        "score_x",
        "score_o",
        "delta_score",
        "delta_score_ci_lower",
        "delta_score_ci_upper",
        "mean_latency_ms",
        "p95_latency_ms",
        "avg_nodes_visited",
        "avg_simulations",
        "avg_depth",
        "notes",
    ]
    rows: list[dict[str, Any]] = []

    for log_path, summary in tournament_logs:
        board_tag = _board_label_from_meta(summary.meta).replace(" ", "_").replace("=", "")
        b_mode = str(summary.meta.get("budget_mode", "default"))
        for rank, m in enumerate(summary.get_rankings(), start=1):
            rows.append(
                {
                    "source_type": "tournament",
                    "experiment_id": log_path.stem,
                    "board": board_tag,
                    "budget_or_group": b_mode,
                    "agent_or_candidate": m.agent_id,
                    "total_games": m.total_games,
                    "wins": m.wins,
                    "draws": m.draws,
                    "losses": m.losses,
                    "win_rate": f"{m.win_rate:.4f}",
                    "score": f"{m.score:.4f}",
                    "score_ci_lower": f"{m.score_ci[0]:.4f}",
                    "score_ci_upper": f"{m.score_ci[1]:.4f}",
                    "score_x": f"{m.score_x:.4f}",
                    "score_o": f"{m.score_o:.4f}",
                    "delta_score": f"{m.delta_score:.4f}",
                    "delta_score_ci_lower": f"{m.delta_score_ci[0]:.4f}",
                    "delta_score_ci_upper": f"{m.delta_score_ci[1]:.4f}",
                    "mean_latency_ms": f"{m.mean_latency_ms:.4f}",
                    "p95_latency_ms": f"{m.p95_latency_ms:.4f}",
                    "avg_nodes_visited": f"{m.avg_nodes_visited:.2f}",
                    "avg_simulations": f"{m.avg_simulations:.2f}",
                    "avg_depth": f"{m.avg_depth:.2f}",
                    "notes": f"rank={rank}",
                }
            )

    def _ab_row(
        exp_id: str,
        board: str,
        group: str,
        cand: str,
        m: dict[str, Any],
        notes: str,
    ) -> dict[str, Any]:
        sc_ci = m.get("score_ci", [0.0, 0.0])
        ds_ci = m.get("delta_score_ci", [0.0, 0.0])
        return {
            "source_type": "ablation",
            "experiment_id": exp_id,
            "board": board,
            "budget_or_group": group,
            "agent_or_candidate": cand,
            "total_games": m.get("total_games", 0),
            "wins": m.get("wins", 0),
            "draws": m.get("draws", 0),
            "losses": m.get("losses", 0),
            "win_rate": f"{m.get('win_rate', 0.0):.4f}",
            "score": f"{m.get('score', 0.0):.4f}",
            "score_ci_lower": f"{sc_ci[0]:.4f}",
            "score_ci_upper": f"{sc_ci[1]:.4f}",
            "score_x": f"{m.get('score_x', 0.0):.4f}",
            "score_o": f"{m.get('score_o', 0.0):.4f}",
            "delta_score": f"{m.get('delta_score', 0.0):.4f}",
            "delta_score_ci_lower": f"{ds_ci[0]:.4f}",
            "delta_score_ci_upper": f"{ds_ci[1]:.4f}",
            "mean_latency_ms": f"{m.get('mean_latency_ms', 0.0):.4f}",
            "p95_latency_ms": f"{m.get('p95_latency_ms', 0.0):.4f}",
            "avg_nodes_visited": f"{m.get('avg_nodes_visited', 0.0):.2f}",
            "avg_simulations": f"{m.get('avg_simulations', 0.0):.2f}",
            "avg_depth": f"{m.get('avg_depth', 0.0):.2f}",
            "notes": notes,
        }

    for res in ablation_results:
        if res.ablation_id == "ablation_1":
            for cid, m in res.summary.get("match_metrics", {}).items():
                rows.append(_ab_row("ablation_1", "4x4_K4", f"d{m.get('depth')}", cid, m, ""))
        elif res.ablation_id == "ablation_2":
            for cid, m in res.summary.get("candidate_metrics", {}).items():
                note = f"{m.get('budget_type')}={m.get('budget_val')}"
                rows.append(
                    _ab_row("ablation_2_rq1", "4x4_K4", str(m.get("family", "")), cid, m, note)
                )
        elif res.ablation_id == "ablation_3":
            for btag, bdata in res.summary.get("boards", {}).items():
                for aid, m in bdata.get("agent_metrics", {}).items():
                    rows.append(_ab_row("ablation_3_rq2", btag, btag, aid, m, ""))
        elif res.ablation_id == "ablation_4":
            for btag, bdata in res.summary.get("boards", {}).items():
                for cid, m in bdata.get("candidates", {}).items():
                    note = (
                        f"unseen_mcts={m['vs_unseen_mcts']['score']:.4f};"
                        f"unseen_all={m['vs_unseen_all']['score']:.4f}"
                    )
                    rows.append(
                        _ab_row(
                            "ablation_4_rq3",
                            btag,
                            str(m.get("group", "")),
                            cid,
                            m,
                            note,
                        )
                    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def _format_summary_tables_markdown(
    *,
    tournament_logs: Sequence[tuple[Path, TournamentSummary]],
    ablation_results: Sequence[AblationRunResult],
) -> str:
    """Build consolidated `reports/summary_tables.md` directly from raw logs."""
    lines: list[str] = [
        "# Consolidated Empirical Summary Tables (Extended Tic-Tac-Toe)",
        "",
        "Generated directly from raw `logs/*.jsonl` files without manual edits.",
        "",
        "## Figures Reference",
        "- **Fig 1:** `figures/fig1_head_to_head_matrix.png` — Dual X/O Head-to-Head Matrix",
        (
            "- **Fig 2:** `figures/fig2_latency_and_nodes.png` — "
            "Log Latency (Mean & P95) & Search Cost"
        ),
        (
            "- **Fig 3:** `figures/fig3_ablation_rq1_and_alphabeta.png` — "
            "RQ1 Hybrid MCTS & Alpha-Beta Pruning"
        ),
        (
            "- **Fig 4:** `figures/fig4_ablation_rq2_seat_advantage.png` — "
            "RQ2 First-Mover Advantage"
        ),
        (
            "- **Fig 5:** `figures/fig5_ablation_rq3_learning_and_generalization.png` — "
            "RQ3 Learning Curves & Unseen Opponents"
        ),
        "",
        "## 1. Official Tournament Summaries",
        "",
    ]

    if tournament_logs:
        lines.extend([
            (
                "| Log File | Board | Mode | Matches | Top Agent | "
                "Top Score [95% CI] | Seat ΔScore [95% CI] |"
            ),
            "|:---|:---:|:---|---:|:---|:---:|:---:|",
        ])
        for log_path, summary in tournament_logs:
            rankings = summary.get_rankings()
            top = rankings[0] if rankings else None
            sa = summary.seat_advantage
            board_lbl = _board_label_from_meta(summary.meta)
            b_mode = str(summary.meta.get("budget_mode", "default"))
            top_name = top.agent_id if top else "N/A"
            top_sc = (
                f"{top.score:.3f} [{top.score_ci[0]:.3f}, {top.score_ci[1]:.3f}]"
                if top
                else "N/A"
            )
            ds = (
                f"{sa.delta_score:+.3f} "
                f"[{sa.delta_score_ci[0]:+.3f}, {sa.delta_score_ci[1]:+.3f}]"
            )
            lines.append(
                f"| `{log_path.name}` | `{board_lbl}` | `{b_mode}` | {summary.total_matches} | "
                f"**{top_name}** | {top_sc} | {ds} |"
            )
        lines.append("")

        for log_path, summary in tournament_logs:
            board_lbl = _board_label_from_meta(summary.meta)
            b_mode = str(summary.meta.get("budget_mode", "default"))
            lines.extend([
                f"### Leaderboard — `{log_path.name}` (`{board_lbl}`, `{b_mode}`)",
                "",
                (
                    "| Rank | Agent | W-D-L | Score [95% CI] | Score(X) | Score(O) | "
                    "ΔScore | Mean (ms) | P95 (ms) | Avg Nodes | Avg Sims |"
                ),
                "|---:|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for rank, m in enumerate(summary.get_rankings(), start=1):
                lines.append(
                    f"| {rank} | **{m.agent_id}** | {m.wins}-{m.draws}-{m.losses} | "
                    f"{m.score:.3f} [{m.score_ci[0]:.3f}, {m.score_ci[1]:.3f}] | "
                    f"{m.score_x:.3f} | {m.score_o:.3f} | {m.delta_score:+.3f} | "
                    f"{m.mean_latency_ms:.2f} | {m.p95_latency_ms:.2f} | "
                    f"{m.avg_nodes_visited:.1f} | {m.avg_simulations:.1f} |"
                )
            lines.append("")

    if ablation_results:
        lines.extend(["## 2. Ablation Studies Summary (Alpha-Beta, RQ1, RQ2, RQ3)", ""])
        for res in ablation_results:
            lines.append(
                f"### {res.ablation_id.upper()} "
                f"(`{res.log_path.name}`, matches={res.total_matches})"
            )
            lines.append("")
            if res.ablation_id == "ablation_1":
                lines.append(
                    "| Depth | MM Nodes | AB No-Order Nodes | "
                    "AB + Order Nodes | AB+Order Reduction |"
                )
                lines.append("|:---:|---:|---:|---:|---:|")
                for row in res.summary.get("probe_by_depth", {}).values():
                    lines.append(
                        f"| d={row['depth']} | {row['mm_avg_nodes']:.1f} | "
                        f"{row['ab_no_order_avg_nodes']:.1f} | {row['ab_order_avg_nodes']:.1f} | "
                        f"-{row['ab_order_reduction_pct']:.2f}% |"
                    )
            elif res.ablation_id == "ablation_2":
                lines.append(
                    "| Candidate | Budget | Score [95% CI] | "
                    "Mean (ms) | Avg Sims | Cache Hit |"
                )
                lines.append("|:---|:---:|:---:|---:|---:|---:|")
                for cid, m in res.summary.get("candidate_metrics", {}).items():
                    ci = m["score_ci"]
                    hit = f"{m['cache_hit_rate']:.1%}" if "cache_hit_rate" in m else "-"
                    lines.append(
                        f"| `{cid}` | `{m['budget_type']}={m['budget_val']}` | "
                        f"{m['score']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                        f"{m['mean_latency_ms']:.2f} | {m['avg_simulations']:.1f} | {hit} |"
                    )
            elif res.ablation_id == "ablation_3":
                lines.append(
                    "| Board | Matches | X Win% | Draw% | O Win% | ΔScore (X-O) [95% CI] |"
                )
                lines.append("|:---|---:|---:|---:|---:|:---:|")
                for btag, bdata in res.summary.get("boards", {}).items():
                    sa = bdata["seat_advantage"]
                    ci = sa["delta_score_ci"]
                    lines.append(
                        f"| `{btag}` | {sa['total_games']} | {sa['x_win_rate']:.1%} | "
                        f"{sa['draw_rate']:.1%} | {sa['o_win_rate']:.1%} | "
                        f"{sa['delta_score']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |"
                    )
            elif res.ablation_id == "ablation_4":
                lines.append(
                    "| Board | Candidate | Group | Overall Score | "
                    "Unseen MCTS | Unseen All | Mean (ms) |"
                )
                lines.append("|:---|:---|:---:|---:|---:|---:|---:|")
                for btag, bdata in res.summary.get("boards", {}).items():
                    for cid, m in bdata.get("candidates", {}).items():
                        mcts_sc = m["vs_unseen_mcts"]["score"]
                        all_sc = m["vs_unseen_all"]["score"]
                        mean_lat = m.get("mean_latency_ms")
                        lat_str = f"{float(mean_lat):.2f}" if mean_lat is not None else "-"
                        lines.append(
                            f"| `{btag}` | `{cid}` | {m['group']} | {m['score']:.3f} | "
                            f"{mcts_sc:.3f} | {all_sc:.3f} | "
                            f"{lat_str} |"
                        )
            lines.append("")

    return "\n".join(lines)


def generate_summary_reports(
    *,
    logs_dir: str | Path = "logs",
    output_dir: str | Path = "reports",
    checkpoints_dir: str | Path = "checkpoints",
    n_bootstrap: int = 2000,
) -> dict[str, Path]:
    """Generate `summary_metrics.csv`, `summary_tables.md`, and `sample_failure_cases.md`.

    Args:
        logs_dir: Directory containing raw tournament and ablation JSONL logs.
        output_dir: Target directory where CSV and Markdown reports are written.
        checkpoints_dir: Optional directory containing trained checkpoints for
            Policy prior comparison in `sample_failure_cases.md`.
        n_bootstrap: Bootstrap resamples for tournament summary CI calculation.

    Returns:
        Dictionary mapping report keys (``"summary_csv"``, ``"summary_tables"``,
        ``"sample_failure_cases"``) to generated file paths.
    """
    resolved_logs = Path(logs_dir)
    resolved_out = Path(output_dir)
    resolved_ckpts = Path(checkpoints_dir)
    resolved_out.mkdir(parents=True, exist_ok=True)

    tournament_logs: list[tuple[Path, TournamentSummary]] = []
    for p in sorted(resolved_logs.glob("tournament_*.jsonl")):
        tournament_logs.append((p, summarize_tournament_log(p, n_bootstrap=n_bootstrap, seed=42)))

    ablation_order = (
        "ablation_ab_pruning.jsonl",
        "ablation_rq1_hybrid_mcts.jsonl",
        "ablation_rq2_seat_advantage.jsonl",
        "ablation_rq3_rl_generalization.jsonl",
    )
    ablation_results: list[AblationRunResult] = []
    seen_ab: set[Path] = set()
    for fname in ablation_order:
        p = resolved_logs / fname
        if p.is_file():
            ablation_results.append(load_ablation_result(p))
            seen_ab.add(p)
    for p in sorted(resolved_logs.glob("ablation_*.jsonl")):
        if p not in seen_ab:
            ablation_results.append(load_ablation_result(p))

    csv_path = _export_consolidated_csv(
        tournament_logs=tournament_logs,
        ablation_results=ablation_results,
        output_csv=resolved_out / "summary_metrics.csv",
    )
    tables_path = resolved_out / "summary_tables.md"
    tables_path.write_text(
        _format_summary_tables_markdown(
            tournament_logs=tournament_logs,
            ablation_results=ablation_results,
        ),
        encoding="utf-8",
    )
    failures_path = resolved_out / "sample_failure_cases.md"
    failures_path.write_text(
        _extract_failure_cases_markdown(
            logs_dir=resolved_logs,
            checkpoints_dir=resolved_ckpts,
        ),
        encoding="utf-8",
    )

    logger.info("Exported summary reports -> %s, %s, %s", csv_path, tables_path, failures_path)
    return {
        "summary_csv": csv_path,
        "summary_tables": tables_path,
        "sample_failure_cases": failures_path,
    }


def generate_all_visualizations_and_reports(
    *,
    logs_dir: str | Path = "logs",
    output_dir: str | Path = "reports",
    checkpoints_dir: str | Path = "checkpoints",
    figures: Sequence[str] = ("1", "2", "3", "4", "5"),
    dpi: int = 300,
) -> dict[str, Path]:
    """Generate requested figures (Fig 1..5) and empirical Markdown/CSV reports.

    Args:
        logs_dir: Directory containing raw JSONL logs.
        output_dir: Root reports directory (figures saved under ``output_dir/figures/``).
        checkpoints_dir: Directory containing trained model checkpoints.
        figures: Iterable of figure IDs among ``"1".."5"`` or ``"all"``.
        dpi: Output PNG DPI resolution (default ``300``).

    Returns:
        Mapping of artifact labels to saved file :class:`Path` instances.
    """
    resolved_dpi = _validate_dpi(dpi)
    resolved_logs = Path(logs_dir)
    resolved_out = Path(output_dir)
    fig_dir = resolved_out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        ["1", "2", "3", "4", "5"]
        if any(str(f).strip().lower() == "all" for f in figures)
        else [str(f).strip() for f in figures]
    )
    valid_ids = {"1", "2", "3", "4", "5"}
    unknown = [f for f in selected if f not in valid_ids]
    if unknown:
        raise InvalidConfigError(
            f"Unsupported figure identifier(s) {unknown!r}; expected values in {sorted(valid_ids)}."
        )

    artifacts: dict[str, Path] = {}
    if "1" in selected:
        artifacts["fig1"] = plot_fig1_head_to_head(
            resolved_logs,
            fig_dir / "fig1_head_to_head_matrix.png",
            dpi=resolved_dpi,
        )
    if "2" in selected:
        artifacts["fig2"] = plot_fig2_latency_and_nodes(
            resolved_logs,
            fig_dir / "fig2_latency_and_nodes.png",
            dpi=resolved_dpi,
        )
    if "3" in selected:
        artifacts["fig3"] = plot_fig3_ablation_rq1(
            resolved_logs,
            resolved_logs,
            fig_dir / "fig3_ablation_rq1_and_alphabeta.png",
            dpi=resolved_dpi,
        )
    if "4" in selected:
        artifacts["fig4"] = plot_fig4_ablation_rq2(
            resolved_logs,
            fig_dir / "fig4_ablation_rq2_seat_advantage.png",
            dpi=resolved_dpi,
        )
    if "5" in selected:
        artifacts["fig5"] = plot_fig5_ablation_rq3(
            resolved_logs,
            logs_dir=resolved_logs,
            output_path=fig_dir / "fig5_ablation_rq3_learning_and_generalization.png",
            dpi=resolved_dpi,
        )

    artifacts.update(
        generate_summary_reports(
            logs_dir=resolved_logs,
            output_dir=resolved_out,
            checkpoints_dir=checkpoints_dir,
        )
    )
    return artifacts
