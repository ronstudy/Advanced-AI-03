"""Scientific ablation studies for Alpha-Beta pruning and Research Questions RQ1, RQ2, RQ3.

Implements Task T4.4 per ``Planning/TASKS.md``, ``Planning/PRD.md`` Sections 1.3 & 4.3,
and ``Planning/RULES.md`` Sections 1 & 3:
1. Ablation 1 (`run_ablation_alpha_beta`): Alpha-Beta pruning & move ordering node/latency
   reduction on `4x4, K=4` (controlled state probes + paired matches).
2. Ablation 2 (`run_ablation_rq1_hybrid_mcts`): RQ1 comparison across Policy-only,
   Value-only, Pure UCT MCTS, and Hybrid MCTS (`rollout` & `mixed`) over simulation
   budgets `{50, 150, 400}` and fixed time budgets `{10.0, 50.0} ms` on `4x4, K=4`.
3. Ablation 3 (`run_ablation_rq2_seat_advantage`): RQ2 first-mover advantage
   (`delta_score = score_x - score_o` with 95% Bootstrap CI) across `3x3 K=3`,
   `4x4 K=3`, and `4x4 K=4`.
4. Ablation 4 (`run_ablation_rq3_generalization`): RQ3 generalization and multi-seed
   stability (`seeds = 42, 123, 999`) comparing Supervised networks vs RL self-play
   (`DQN` on `4x4`, `TabularQ` on `3x3`) against unseen & baseline opponents.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ttt_ai.agents.base import BaseAgent
from ttt_ai.agents.hybrid_mcts_agent import HybridMCTSAgent
from ttt_ai.agents.mcts_agent import MCTSAgent
from ttt_ai.agents.minimax_agent import MinimaxAgent
from ttt_ai.agents.policy_agent import PolicyNetworkAgent, ValueGreedyAgent
from ttt_ai.agents.random_agent import RandomAgent
from ttt_ai.agents.rl_agent import DQNAgent, TabularQAgent
from ttt_ai.agents.rule_based_agent import RuleBasedAgent
from ttt_ai.benchmark.metrics import (
    bootstrap_ci,
    compute_agent_metrics,
    compute_head_to_head_metrics,
    compute_seat_advantage,
    wilson_score_interval,
)
from ttt_ai.benchmark.tournament import (
    MatchRecord,
    collect_tournament_environment_meta,
    load_tournament_log,
    play_single_match,
)
from ttt_ai.config import BoardConfig, seed_everything
from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe

logger = logging.getLogger(__name__)

_RL_SEEDS: tuple[int, ...] = (42, 123, 999)
_VALID_MODES: tuple[str, str] = ("smoke", "full")

_DEFAULT_FILENAMES: dict[str, str] = {
    "ablation_1": "ablation_ab_pruning.jsonl",
    "ablation_2": "ablation_rq1_hybrid_mcts.jsonl",
    "ablation_3": "ablation_rq2_seat_advantage.jsonl",
    "ablation_4": "ablation_rq3_rl_generalization.jsonl",
}


@dataclass(frozen=True, slots=True)
class AblationRunResult:
    """Structured output returned by each ablation study runner."""

    ablation_id: str
    mode: str
    seed: int
    log_path: Path
    total_matches: int
    meta: dict[str, Any]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize run result metadata and summary to a plain dictionary."""
        return {
            "ablation_id": self.ablation_id,
            "mode": self.mode,
            "seed": self.seed,
            "log_path": str(self.log_path),
            "total_matches": self.total_matches,
            "summary": dict(self.summary),
        }


def load_ablation_result(log_path: str | Path) -> AblationRunResult:
    """Reconstruct an :class:`AblationRunResult` from a completed ablation JSONL log."""
    path = Path(log_path)
    meta, records = load_tournament_log(path)
    return AblationRunResult(
        ablation_id=str(meta.get("ablation_id", path.stem)),
        mode=str(meta.get("mode", "full")),
        seed=int(meta.get("seed", 42)),
        log_path=path,
        total_matches=len(records),
        meta=meta,
        summary=dict(meta.get("summary", {})),
    )


def _resolve_base_dir(base_dir: str | Path | None) -> Path:
    """Resolve project root used to locate checkpoints and default log paths."""
    if base_dir is not None:
        return Path(base_dir).resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "checkpoints").is_dir() and (cwd / "configs").is_dir():
        return cwd
    return Path(__file__).resolve().parents[3]


def _resolve_mode(mode: str) -> Literal["smoke", "full"]:
    """Validate execution mode label (`smoke` or `full`)."""
    if not isinstance(mode, str) or mode not in _VALID_MODES:
        raise InvalidConfigError(
            f"mode must be one of {_VALID_MODES}, got {mode!r}."
        )
    return mode  # type: ignore[return-value]


def _validate_seed(seed: int) -> int:
    """Validate non-negative integer RNG seed."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise InvalidConfigError(f"seed must be a non-negative integer, got {seed!r}.")
    return int(seed)


def _ckpt(resolved_base: Path, filename: str) -> Path:
    """Resolve a checkpoint file under `checkpoints/` and ensure it exists."""
    path = resolved_base / "checkpoints" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required trained checkpoint not found: {path}")
    return path


def _default_output_path(
    output_path: str | Path | None,
    ablation_id: str,
    resolved_base: Path,
) -> Path:
    """Resolve output JSONL file path and create parent directories."""
    if output_path is not None:
        path = Path(output_path)
    else:
        path = resolved_base / "logs" / _DEFAULT_FILENAMES[ablation_id]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _build_ablation_meta(
    *,
    ablation_id: str,
    ablation_name: str,
    mode: str,
    seed: int,
    board_cfg: BoardConfig,
    resolved_base: Path,
    agents_config: dict[str, dict[str, Any]],
    total_matches: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Construct standardized `type='meta'` JSONL header for an ablation log."""
    return {
        "type": "meta",
        "ablation_id": ablation_id,
        "ablation_name": ablation_name,
        "mode": mode,
        "seed": seed,
        "board_config": board_cfg.to_dict(),
        "total_scheduled_matches": total_matches,
        "agents": agents_config,
        "hardware": collect_tournament_environment_meta(resolved_base),
        "summary": summary,
    }


def _write_ablation_jsonl(
    path: Path,
    meta_header: dict[str, Any],
    records: Sequence[MatchRecord],
) -> Path:
    """Write line-buffered `meta` header and `match` records atomically to JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta_header) + "\n")
        for rec in records:
            handle.write(json.dumps(rec.to_dict()) + "\n")
        handle.flush()
    return path


def _play_paired_matchup(
    *,
    game: ExtendedTicTacToe,
    agent_a_id: str,
    agent_a: BaseAgent,
    agent_b_id: str,
    agent_b: BaseAgent,
    n_pairs: int,
    base_seed: int,
    budget_mode: str,
    prefix: str,
) -> list[MatchRecord]:
    """Execute `n_pairs` color-swapped (`X/O`) matches between two agents."""
    matches: list[MatchRecord] = []
    for pair_id in range(n_pairs):
        pair_seed = base_seed + pair_id
        m0 = play_single_match(
            game=game,
            agent_x=agent_a,
            agent_o=agent_b,
            match_id=f"{prefix}__{agent_a_id}__vs__{agent_b_id}__p{pair_id:04d}__g0",
            pair_id=pair_id,
            game_in_pair=0,
            seed=pair_seed,
            budget_mode=budget_mode,
            agent_x_id=agent_a_id,
            agent_o_id=agent_b_id,
        )
        m1 = play_single_match(
            game=game,
            agent_x=agent_b,
            agent_o=agent_a,
            match_id=f"{prefix}__{agent_a_id}__vs__{agent_b_id}__p{pair_id:04d}__g1",
            pair_id=pair_id,
            game_in_pair=1,
            seed=pair_seed,
            budget_mode=budget_mode,
            agent_x_id=agent_b_id,
            agent_o_id=agent_a_id,
        )
        matches.extend((m0, m1))
    return matches


# ---------------------------------------------------------------------------
# Ablation 1: Alpha-Beta Pruning & Move Ordering
# ---------------------------------------------------------------------------

def run_ablation_alpha_beta(
    *,
    mode: Literal["smoke", "full"] = "full",
    seed: int = 42,
    output_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> AblationRunResult:
    """Execute Ablation 1 comparing Minimax, Alpha-Beta (no ordering), and AB + Ordering.

    Evaluates:
    1. Controlled state probes at plies `{0, 2, 4, 6}` on `4x4, K=4` across depths
       to measure exact root-value equivalence, node reduction `%`, and speedup.
    2. Paired 50/50 `X/O` matches against `Ref_Random` and `Ref_RuleBased`.
    """
    resolved_mode = _resolve_mode(mode)
    resolved_seed = _validate_seed(seed)
    resolved_base = _resolve_base_dir(base_dir)
    out_path = _default_output_path(output_path, "ablation_1", resolved_base)
    seed_everything(resolved_seed)

    board_cfg = BoardConfig(rows=4, cols=4, win_k=4)
    n_boot = 500 if resolved_mode == "smoke" else 10000
    probe_depths = (2, 3) if resolved_mode == "smoke" else (2, 3, 4)

    probe_defs: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("ply0_empty", ()),
        ("ply2_center", (5, 6)),
        ("ply4_mid", (5, 6, 10, 0)),
        ("ply6_tactical", (5, 6, 10, 0, 9, 15)),
    )
    probe_configs: tuple[tuple[str, bool, bool], ...] = (
        ("MM_Raw", False, False),
        ("AB_NoOrder", True, False),
        ("AB_Order", True, True),
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        state_probes: list[dict[str, Any]] = []
        probe_game = ExtendedTicTacToe(4, 4, 4)

        for depth in probe_depths:
            for state_name, prefix_moves in probe_defs:
                probe_game.reset()
                for mv in prefix_moves:
                    probe_game.step(mv)

                probe_row: dict[str, Any] = {
                    "depth": depth,
                    "state": state_name,
                    "ply": len(prefix_moves),
                    "variants": {},
                }
                root_values: list[float] = []
                for variant_name, use_ab, use_order in probe_configs:
                    agent = MinimaxAgent(
                        name=f"{variant_name}_d{depth}",
                        max_depth=depth,
                        use_alpha_beta=use_ab,
                        use_move_ordering=use_order,
                        iterative_deepening=False,
                        seed=resolved_seed,
                        time_budget_ms=None,
                    )
                    decision = agent.select_action(probe_game)
                    val = float(decision.value_estimate or 0.0)
                    root_values.append(val)
                    probe_row["variants"][variant_name] = {
                        "action": decision.action,
                        "nodes_visited": decision.nodes_visited,
                        "latency_ms": round(decision.latency_ms, 4),
                        "value_estimate": round(val, 6),
                    }

                raw_v = probe_row["variants"]["MM_Raw"]
                ab_v = probe_row["variants"]["AB_NoOrder"]
                ord_v = probe_row["variants"]["AB_Order"]
                raw_nodes = max(1, raw_v["nodes_visited"])
                ab_nodes = max(1, ab_v["nodes_visited"])
                raw_lat = max(1e-6, raw_v["latency_ms"])
                probe_row["exact_value_match"] = max(root_values) - min(root_values) <= 1e-6
                probe_row["ab_node_reduction_pct"] = round(
                    100.0 * (1.0 - ab_v["nodes_visited"] / raw_nodes), 2
                )
                probe_row["order_total_node_reduction_pct"] = round(
                    100.0 * (1.0 - ord_v["nodes_visited"] / raw_nodes), 2
                )
                probe_row["order_over_ab_reduction_pct"] = round(
                    100.0 * (1.0 - ord_v["nodes_visited"] / ab_nodes), 2
                )
                probe_row["speedup_vs_mm"] = round(raw_lat / max(1e-6, ord_v["latency_ms"]), 2)
                state_probes.append(probe_row)

        # Aggregate probe reductions by depth.
        probe_by_depth: dict[str, dict[str, Any]] = {}
        for depth in probe_depths:
            sub = [p for p in state_probes if p["depth"] == depth]
            mm_nodes = float(np.mean([p["variants"]["MM_Raw"]["nodes_visited"] for p in sub]))
            ab_nodes = float(np.mean([p["variants"]["AB_NoOrder"]["nodes_visited"] for p in sub]))
            ord_nodes = float(np.mean([p["variants"]["AB_Order"]["nodes_visited"] for p in sub]))
            mm_ms = float(np.mean([p["variants"]["MM_Raw"]["latency_ms"] for p in sub]))
            ab_ms = float(np.mean([p["variants"]["AB_NoOrder"]["latency_ms"] for p in sub]))
            ord_ms = float(np.mean([p["variants"]["AB_Order"]["latency_ms"] for p in sub]))
            probe_by_depth[f"d{depth}"] = {
                "depth": depth,
                "mm_avg_nodes": round(mm_nodes, 2),
                "ab_no_order_avg_nodes": round(ab_nodes, 2),
                "ab_order_avg_nodes": round(ord_nodes, 2),
                "ab_reduction_pct": round(100.0 * (1.0 - ab_nodes / max(1.0, mm_nodes)), 2),
                "ab_order_reduction_pct": round(
                    100.0 * (1.0 - ord_nodes / max(1.0, mm_nodes)), 2
                ),
                "mm_avg_ms": round(mm_ms, 3),
                "ab_no_order_avg_ms": round(ab_ms, 3),
                "ab_order_avg_ms": round(ord_ms, 3),
                "all_exact_value_match": all(p["exact_value_match"] for p in sub),
            }

        # Paired gameplay matches against Random and RuleBased references.
        if resolved_mode == "smoke":
            match_specs: tuple[tuple[str, int, bool, bool, int], ...] = (
                ("MM_Raw_d3", 3, False, False, 1),
                ("AB_NoOrder_d3", 3, True, False, 1),
                ("AB_Order_d3", 3, True, True, 1),
                ("AB_NoOrder_d4", 4, True, False, 1),
                ("AB_Order_d4", 4, True, True, 1),
            )
        else:
            match_specs = (
                ("MM_Raw_d3", 3, False, False, 15),
                ("AB_NoOrder_d3", 3, True, False, 15),
                ("AB_Order_d3", 3, True, True, 15),
                ("MM_Raw_d4", 4, False, False, 4),
                ("AB_NoOrder_d4", 4, True, False, 15),
                ("AB_Order_d4", 4, True, True, 15),
            )

        candidates: dict[str, BaseAgent] = {
            cid: MinimaxAgent(
                name=cid,
                max_depth=depth,
                use_alpha_beta=use_ab,
                use_move_ordering=use_order,
                iterative_deepening=False,
                seed=resolved_seed,
                time_budget_ms=None,
            )
            for cid, depth, use_ab, use_order, _ in match_specs
        }
        ref_opponents: dict[str, BaseAgent] = {
            "Ref_Random": RandomAgent(name="Ref_Random", seed=resolved_seed + 101),
            "Ref_RuleBased": RuleBasedAgent(
                name="Ref_RuleBased", tie_break="random", seed=resolved_seed + 102
            ),
        }

        game = ExtendedTicTacToe(4, 4, 4)
        records: list[MatchRecord] = []
        for idx, (cid, depth, _, _, n_pairs) in enumerate(match_specs):
            cand = candidates[cid]
            for jdx, (opp_id, opp) in enumerate(ref_opponents.items()):
                pair_records = _play_paired_matchup(
                    game=game,
                    agent_a_id=cid,
                    agent_a=cand,
                    agent_b_id=opp_id,
                    agent_b=opp,
                    n_pairs=n_pairs,
                    base_seed=resolved_seed + idx * 100 + jdx * 50,
                    budget_mode=f"ab_depth_{depth}",
                    prefix="ab1",
                )
                records.extend(pair_records)

        match_metrics: dict[str, dict[str, Any]] = {}
        for cid, depth, _, _, _ in match_specs:
            m = compute_agent_metrics(records, cid, n_bootstrap=n_boot, seed=resolved_seed)
            d_dict = m.to_dict()
            d_dict["depth"] = depth
            match_metrics[cid] = d_dict

        for depth_val in (3, 4):
            base_id = f"MM_Raw_d{depth_val}"
            if base_id in match_metrics:
                base_nodes = max(1e-6, match_metrics[base_id]["avg_nodes_visited"])
                for cid in match_metrics:
                    if match_metrics[cid]["depth"] == depth_val:
                        nodes = match_metrics[cid]["avg_nodes_visited"]
                        match_metrics[cid]["node_reduction_vs_raw_pct"] = round(
                            100.0 * (1.0 - nodes / base_nodes), 2
                        )

        summary = {
            "probe_by_depth": probe_by_depth,
            "state_probes": state_probes,
            "match_metrics": match_metrics,
        }
        agents_cfg = {
            cid: agent.get_config() for cid, agent in {**candidates, **ref_opponents}.items()
        }
        meta_header = _build_ablation_meta(
            ablation_id="ablation_1",
            ablation_name="Alpha-Beta Pruning & Move Ordering (4x4 K=4)",
            mode=resolved_mode,
            seed=resolved_seed,
            board_cfg=board_cfg,
            resolved_base=resolved_base,
            agents_config=agents_cfg,
            total_matches=len(records),
            summary=summary,
        )
        _write_ablation_jsonl(out_path, meta_header, records)
    finally:
        torch.set_num_threads(previous_threads)

    logger.info(
        "Completed Ablation 1 (%s, %d matches, %d probes) -> %s",
        resolved_mode,
        len(records),
        len(state_probes),
        out_path,
    )
    return AblationRunResult(
        ablation_id="ablation_1",
        mode=resolved_mode,
        seed=resolved_seed,
        log_path=out_path,
        total_matches=len(records),
        meta=meta_header,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Ablation 2: RQ1 — Policy-only vs Pure MCTS vs Hybrid MCTS
# ---------------------------------------------------------------------------

def _build_rq1_candidates(
    resolved_base: Path,
    board_cfg: BoardConfig,
    seed: int,
) -> tuple[dict[str, BaseAgent], dict[str, dict[str, Any]]]:
    """Instantiate all 18 candidate agents across zero-search, sim-budget, and time-budget."""
    fast_ckpt = _ckpt(resolved_base, "fastpolicynet_4x4_k4.pt")
    strong_ckpt = _ckpt(resolved_base, "strongpolicynet_4x4_k4.pt")
    value_ckpt = _ckpt(resolved_base, "valuenet_4x4_k4.pt")

    candidates: dict[str, BaseAgent] = {}
    meta_specs: dict[str, dict[str, Any]] = {}

    candidates["Policy_Fast"] = PolicyNetworkAgent(
        checkpoint_path=fast_ckpt,
        board_config=board_cfg,
        sample=False,
        device="cpu",
        name="Policy_Fast",
        seed=seed,
    )
    meta_specs["Policy_Fast"] = {"family": "Policy_Fast", "budget_type": "none", "budget_val": 0}

    candidates["Policy_Strong"] = PolicyNetworkAgent(
        checkpoint_path=strong_ckpt,
        board_config=board_cfg,
        sample=False,
        device="cpu",
        name="Policy_Strong",
        seed=seed,
    )
    meta_specs["Policy_Strong"] = {
        "family": "Policy_Strong",
        "budget_type": "none",
        "budget_val": 0,
    }

    candidates["Value_Greedy"] = ValueGreedyAgent(
        checkpoint_path=value_ckpt,
        board_config=board_cfg,
        device="cpu",
        name="Value_Greedy",
        seed=seed,
    )
    meta_specs["Value_Greedy"] = {
        "family": "Value_Greedy",
        "budget_type": "none",
        "budget_val": 0,
    }

    sim_budgets = (50, 150, 400)
    for n_sims in sim_budgets:
        tag = f"sim{n_sims}"
        p_id = f"PureMCTS_{tag}"
        candidates[p_id] = MCTSAgent(
            n_simulations=n_sims,
            c_uct=math.sqrt(2.0),
            name=p_id,
            seed=seed,
            time_budget_ms=None,
        )
        meta_specs[p_id] = {
            "family": "PureMCTS",
            "budget_type": "simulations",
            "budget_val": n_sims,
        }

        hr_id = f"HybridRollout_{tag}"
        candidates[hr_id] = HybridMCTSAgent(
            policy_checkpoint_path=strong_ckpt,
            value_checkpoint_path=None,
            board_config=board_cfg,
            n_simulations=n_sims,
            c_puct=1.5,
            leaf_eval_mode="rollout",
            max_cache_size=4096,
            device="cpu",
            name=hr_id,
            seed=seed,
            time_budget_ms=None,
        )
        meta_specs[hr_id] = {
            "family": "HybridRollout",
            "budget_type": "simulations",
            "budget_val": n_sims,
        }

        hm_id = f"HybridMixed_{tag}"
        candidates[hm_id] = HybridMCTSAgent(
            policy_checkpoint_path=strong_ckpt,
            value_checkpoint_path=value_ckpt,
            board_config=board_cfg,
            n_simulations=n_sims,
            c_puct=1.5,
            leaf_eval_mode="mixed",
            value_weight=0.5,
            max_cache_size=4096,
            device="cpu",
            name=hm_id,
            seed=seed,
            time_budget_ms=None,
        )
        meta_specs[hm_id] = {
            "family": "HybridMixed",
            "budget_type": "simulations",
            "budget_val": n_sims,
        }

    time_budgets = (10.0, 50.0)
    for t_ms in time_budgets:
        tag = f"time{int(t_ms)}ms"
        p_id = f"PureMCTS_{tag}"
        candidates[p_id] = MCTSAgent(
            n_simulations=400,
            c_uct=math.sqrt(2.0),
            name=p_id,
            seed=seed,
            time_budget_ms=t_ms,
        )
        meta_specs[p_id] = {"family": "PureMCTS", "budget_type": "time_ms", "budget_val": t_ms}

        hr_id = f"HybridRollout_{tag}"
        candidates[hr_id] = HybridMCTSAgent(
            policy_checkpoint_path=strong_ckpt,
            value_checkpoint_path=None,
            board_config=board_cfg,
            n_simulations=400,
            c_puct=1.5,
            leaf_eval_mode="rollout",
            max_cache_size=4096,
            device="cpu",
            name=hr_id,
            seed=seed,
            time_budget_ms=t_ms,
        )
        meta_specs[hr_id] = {
            "family": "HybridRollout",
            "budget_type": "time_ms",
            "budget_val": t_ms,
        }

        hm_id = f"HybridMixed_{tag}"
        candidates[hm_id] = HybridMCTSAgent(
            policy_checkpoint_path=strong_ckpt,
            value_checkpoint_path=value_ckpt,
            board_config=board_cfg,
            n_simulations=400,
            c_puct=1.5,
            leaf_eval_mode="mixed",
            value_weight=0.5,
            max_cache_size=4096,
            device="cpu",
            name=hm_id,
            seed=seed,
            time_budget_ms=t_ms,
        )
        meta_specs[hm_id] = {"family": "HybridMixed", "budget_type": "time_ms", "budget_val": t_ms}

    return candidates, meta_specs


def run_ablation_rq1_hybrid_mcts(
    *,
    mode: Literal["smoke", "full"] = "full",
    seed: int = 42,
    output_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> AblationRunResult:
    """Execute Ablation 2 answering RQ1: Policy-only vs Pure MCTS vs Hybrid MCTS.

    Compares `FastPolicyNet`, `StrongPolicyNet`, `ValueGreedy`, `PureMCTS`,
    `HybridMCTS(rollout)`, and `HybridMCTS(mixed)` across simulation budgets
    `{50, 150, 400}` and fixed time budgets `{10.0, 50.0} ms` on `4x4, K=4`.
    """
    resolved_mode = _resolve_mode(mode)
    resolved_seed = _validate_seed(seed)
    resolved_base = _resolve_base_dir(base_dir)
    out_path = _default_output_path(output_path, "ablation_2", resolved_base)
    seed_everything(resolved_seed)

    board_cfg = BoardConfig(rows=4, cols=4, win_k=4)
    n_pairs_gauntlet = 1 if resolved_mode == "smoke" else 15
    n_pairs_h2h = 1 if resolved_mode == "smoke" else 15
    n_boot = 500 if resolved_mode == "smoke" else 10000

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        candidates, cand_tags = _build_rq1_candidates(resolved_base, board_cfg, resolved_seed)
        opponents: dict[str, BaseAgent] = {
            "Opp_Random": RandomAgent(name="Opp_Random", seed=resolved_seed + 501),
            "Opp_RuleBased": RuleBasedAgent(
                name="Opp_RuleBased", tie_break="random", seed=resolved_seed + 502
            ),
            "Opp_AlphaBeta_d3": MinimaxAgent(
                name="Opp_AlphaBeta_d3",
                max_depth=3,
                use_alpha_beta=True,
                use_move_ordering=True,
                seed=resolved_seed + 503,
            ),
            "Opp_PureMCTS_150": MCTSAgent(
                n_simulations=150,
                c_uct=math.sqrt(2.0),
                name="Opp_PureMCTS_150",
                seed=resolved_seed + 504,
            ),
        }

        game = ExtendedTicTacToe(4, 4, 4)
        records: list[MatchRecord] = []
        gauntlet_records: list[MatchRecord] = []
        h2h_records: list[MatchRecord] = []

        opp_list = list(opponents.items())
        for c_idx, (cid, cand) in enumerate(candidates.items()):
            for o_idx, (opp_id, opp) in enumerate(opp_list):
                pair_recs = _play_paired_matchup(
                    game=game,
                    agent_a_id=cid,
                    agent_a=cand,
                    agent_b_id=opp_id,
                    agent_b=opp,
                    n_pairs=n_pairs_gauntlet,
                    base_seed=resolved_seed + c_idx * 100 + o_idx * 30,
                    budget_mode=f"gauntlet_{cand_tags[cid]['budget_type']}",
                    prefix="rq1g",
                )
                gauntlet_records.extend(pair_recs)
                records.extend(pair_recs)

        budget_tags = ("sim50", "sim150", "sim400", "time10ms", "time50ms")
        h2h_pairs_summary: list[dict[str, Any]] = []
        for b_idx, tag in enumerate(budget_tags):
            hm_id = f"HybridMixed_{tag}"
            hr_id = f"HybridRollout_{tag}"
            pm_id = f"PureMCTS_{tag}"
            direct_matchups = (
                (hm_id, pm_id),
                (hr_id, pm_id),
                (hm_id, hr_id),
                (hm_id, "Policy_Strong"),
            )
            for m_idx, (a_id, b_id) in enumerate(direct_matchups):
                pair_recs = _play_paired_matchup(
                    game=game,
                    agent_a_id=a_id,
                    agent_a=candidates[a_id],
                    agent_b_id=b_id,
                    agent_b=candidates[b_id],
                    n_pairs=n_pairs_h2h,
                    base_seed=resolved_seed + 5000 + b_idx * 200 + m_idx * 40,
                    budget_mode=f"h2h_{tag}",
                    prefix="rq1d",
                )
                h2h_records.extend(pair_recs)
                records.extend(pair_recs)
                h2h_stat = compute_head_to_head_metrics(
                    pair_recs, a_id, b_id, n_bootstrap=n_boot, seed=resolved_seed
                )
                h2h_dict = h2h_stat.to_dict()
                h2h_dict["budget_tag"] = tag
                h2h_pairs_summary.append(h2h_dict)

        candidate_metrics: dict[str, dict[str, Any]] = {}
        for cid, cand in candidates.items():
            # Compute gauntlet-only score on common opponent pool for fair comparison.
            m = compute_agent_metrics(
                gauntlet_records, cid, n_bootstrap=n_boot, seed=resolved_seed
            )
            d_dict = m.to_dict()
            d_dict.update(cand_tags[cid])
            if isinstance(cand, HybridMCTSAgent):
                d_dict["cache_hits"] = cand.cache_hits
                d_dict["cache_misses"] = cand.cache_misses
                d_dict["cache_hit_rate"] = round(cand.cache_hit_rate, 4)
            candidate_metrics[cid] = d_dict

        summary = {
            "candidate_metrics": candidate_metrics,
            "direct_head_to_head": h2h_pairs_summary,
        }
        agents_cfg = {
            cid: agent.get_config() for cid, agent in {**candidates, **opponents}.items()
        }
        meta_header = _build_ablation_meta(
            ablation_id="ablation_2",
            ablation_name="RQ1 Hybrid MCTS vs Pure MCTS vs Policy-only (4x4 K=4)",
            mode=resolved_mode,
            seed=resolved_seed,
            board_cfg=board_cfg,
            resolved_base=resolved_base,
            agents_config=agents_cfg,
            total_matches=len(records),
            summary=summary,
        )
        _write_ablation_jsonl(out_path, meta_header, records)
    finally:
        torch.set_num_threads(previous_threads)

    logger.info(
        "Completed Ablation 2 RQ1 (%s, %d matches) -> %s",
        resolved_mode,
        len(records),
        out_path,
    )
    return AblationRunResult(
        ablation_id="ablation_2",
        mode=resolved_mode,
        seed=resolved_seed,
        log_path=out_path,
        total_matches=len(records),
        meta=meta_header,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Ablation 3: RQ2 — First-Mover Advantage Across 3x3 K=3, 4x4 K=3, 4x4 K=4
# ---------------------------------------------------------------------------

def _build_rq2_roster(
    board_tag: str,
    board_cfg: BoardConfig,
    resolved_base: Path,
    seed: int,
) -> dict[str, BaseAgent]:
    """Instantiate verified AI roster available for a given RQ2 board rule variant."""
    roster: dict[str, BaseAgent] = {
        "L1_Random": RandomAgent(name="L1_Random", seed=seed),
        "L2_RuleBased": RuleBasedAgent(name="L2_RuleBased", tie_break="random", seed=seed),
    }
    if board_tag == "3x3_k3":
        roster["L3_AlphaBeta"] = MinimaxAgent(
            name="L3_AlphaBeta",
            max_depth=6,
            use_alpha_beta=True,
            use_move_ordering=True,
            seed=seed,
        )
        roster["L4_MCTS"] = MCTSAgent(
            n_simulations=150, c_uct=math.sqrt(2.0), name="L4_MCTS", seed=seed
        )
        strong_3x3 = _ckpt(resolved_base, "strongpolicynet_3x3_k3.pt")
        value_3x3 = _ckpt(resolved_base, "valuenet_3x3_k3.pt")
        tabq_3x3 = _ckpt(resolved_base, "tabular_q_3x3_k3_seed42.pkl")
        roster["L5_StrongPolicy"] = PolicyNetworkAgent(
            checkpoint_path=strong_3x3,
            board_config=board_cfg,
            sample=False,
            device="cpu",
            name="L5_StrongPolicy",
            seed=seed,
        )
        roster["L6_TabularQ"] = TabularQAgent(
            checkpoint_path=tabq_3x3,
            board_config=board_cfg,
            epsilon=0.0,
            name="L6_TabularQ",
            seed=seed,
        )
        roster["L7_HybridMCTS"] = HybridMCTSAgent(
            policy_checkpoint_path=strong_3x3,
            value_checkpoint_path=value_3x3,
            board_config=board_cfg,
            n_simulations=150,
            c_puct=1.5,
            leaf_eval_mode="mixed",
            value_weight=0.5,
            max_cache_size=4096,
            device="cpu",
            name="L7_HybridMCTS",
            seed=seed,
        )
    elif board_tag == "4x4_k3":
        roster["L3_AlphaBeta"] = MinimaxAgent(
            name="L3_AlphaBeta",
            max_depth=4,
            use_alpha_beta=True,
            use_move_ordering=True,
            seed=seed,
        )
        roster["L4_MCTS"] = MCTSAgent(
            n_simulations=150, c_uct=math.sqrt(2.0), name="L4_MCTS", seed=seed
        )
        dqn_4x4_k3 = _ckpt(resolved_base, "dqnnet_4x4_k3_seed42.pt")
        roster["L6_DQN"] = DQNAgent(
            checkpoint_path=dqn_4x4_k3,
            board_config=board_cfg,
            epsilon=0.0,
            device="cpu",
            name="L6_DQN",
            seed=seed,
        )
    elif board_tag == "4x4_k4":
        roster["L3_AlphaBeta"] = MinimaxAgent(
            name="L3_AlphaBeta",
            max_depth=4,
            use_alpha_beta=True,
            use_move_ordering=True,
            seed=seed,
        )
        roster["L4_MCTS"] = MCTSAgent(
            n_simulations=150, c_uct=math.sqrt(2.0), name="L4_MCTS", seed=seed
        )
        strong_4x4 = _ckpt(resolved_base, "strongpolicynet_4x4_k4.pt")
        value_4x4 = _ckpt(resolved_base, "valuenet_4x4_k4.pt")
        dqn_4x4_k4 = _ckpt(resolved_base, "dqnnet_4x4_k4_seed42.pt")
        roster["L5_StrongPolicy"] = PolicyNetworkAgent(
            checkpoint_path=strong_4x4,
            board_config=board_cfg,
            sample=False,
            device="cpu",
            name="L5_StrongPolicy",
            seed=seed,
        )
        roster["L6_DQN"] = DQNAgent(
            checkpoint_path=dqn_4x4_k4,
            board_config=board_cfg,
            epsilon=0.0,
            device="cpu",
            name="L6_DQN",
            seed=seed,
        )
        roster["L7_HybridMCTS"] = HybridMCTSAgent(
            policy_checkpoint_path=strong_4x4,
            value_checkpoint_path=value_4x4,
            board_config=board_cfg,
            n_simulations=150,
            c_puct=1.5,
            leaf_eval_mode="mixed",
            value_weight=0.5,
            max_cache_size=4096,
            device="cpu",
            name="L7_HybridMCTS",
            seed=seed,
        )
    return roster


def run_ablation_rq2_seat_advantage(
    *,
    mode: Literal["smoke", "full"] = "full",
    seed: int = 42,
    output_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> AblationRunResult:
    """Execute Ablation 3 answering RQ2: First-Mover Advantage (`delta_score`) across rule variants.

    Runs Paired Round-Robin tournaments across `3x3 K=3`, `4x4 K=3`, and `4x4 K=4`,
    separating global `X` vs `O` outcomes and per-agent `score_x`, `score_o`, and
    `delta_score = score_x - score_o` with 95% Bootstrap confidence intervals.
    """
    resolved_mode = _resolve_mode(mode)
    resolved_seed = _validate_seed(seed)
    resolved_base = _resolve_base_dir(base_dir)
    out_path = _default_output_path(output_path, "ablation_3", resolved_base)
    seed_everything(resolved_seed)

    n_pairs = 1 if resolved_mode == "smoke" else 20
    n_boot = 500 if resolved_mode == "smoke" else 10000
    board_variants: tuple[tuple[str, BoardConfig], ...] = (
        ("3x3_k3", BoardConfig(3, 3, 3)),
        ("4x4_k3", BoardConfig(4, 4, 3)),
        ("4x4_k4", BoardConfig(4, 4, 4)),
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        all_records: list[MatchRecord] = []
        board_summaries: dict[str, dict[str, Any]] = {}
        agents_cfg: dict[str, dict[str, Any]] = {}

        for b_idx, (board_tag, board_cfg) in enumerate(board_variants):
            roster = _build_rq2_roster(board_tag, board_cfg, resolved_base, resolved_seed)
            for aid, ag in roster.items():
                agents_cfg[f"{board_tag}:{aid}"] = ag.get_config()

            game = ExtendedTicTacToe(board_cfg.rows, board_cfg.cols, board_cfg.win_k)
            board_records: list[MatchRecord] = []
            ids = list(roster.keys())
            pair_counter = 0
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pair_recs = _play_paired_matchup(
                        game=game,
                        agent_a_id=ids[i],
                        agent_a=roster[ids[i]],
                        agent_b_id=ids[j],
                        agent_b=roster[ids[j]],
                        n_pairs=n_pairs,
                        base_seed=resolved_seed + b_idx * 1000 + pair_counter * 50,
                        budget_mode=f"rq2_{board_tag}",
                        prefix=f"rq2_{board_tag}",
                    )
                    board_records.extend(pair_recs)
                    all_records.extend(pair_recs)
                    pair_counter += 1

            seat_summary = compute_seat_advantage(
                board_records, n_bootstrap=n_boot, seed=resolved_seed + b_idx
            )
            per_agent_seat: dict[str, dict[str, Any]] = {}
            for aid in ids:
                m = compute_agent_metrics(
                    board_records, aid, n_bootstrap=n_boot, seed=resolved_seed + b_idx
                )
                per_agent_seat[aid] = m.to_dict()

            board_summaries[board_tag] = {
                "board_config": board_cfg.to_dict(),
                "n_participants": len(ids),
                "total_matches": len(board_records),
                "seat_advantage": seat_summary.to_dict(),
                "agent_metrics": per_agent_seat,
            }

        summary = {"boards": board_summaries}
        meta_header = _build_ablation_meta(
            ablation_id="ablation_3",
            ablation_name="RQ2 First-Mover Advantage Across 3x3 K=3, 4x4 K=3, 4x4 K=4",
            mode=resolved_mode,
            seed=resolved_seed,
            board_cfg=BoardConfig(4, 4, 4),
            resolved_base=resolved_base,
            agents_config=agents_cfg,
            total_matches=len(all_records),
            summary=summary,
        )
        _write_ablation_jsonl(out_path, meta_header, all_records)
    finally:
        torch.set_num_threads(previous_threads)

    logger.info(
        "Completed Ablation 3 RQ2 (%s, %d matches) -> %s",
        resolved_mode,
        len(all_records),
        out_path,
    )
    return AblationRunResult(
        ablation_id="ablation_3",
        mode=resolved_mode,
        seed=resolved_seed,
        log_path=out_path,
        total_matches=len(all_records),
        meta=meta_header,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Ablation 4: RQ3 — Supervised vs Multi-Seed RL Self-Play Generalization
# ---------------------------------------------------------------------------

def _score_subset(
    records: Sequence[MatchRecord],
    candidate_id: str,
    opponent_ids: set[str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Compute filtered win-rate, score, and 95% CIs against a subset of opponents."""
    sub = [
        r
        for r in records
        if (r.agent_x == candidate_id and r.agent_o in opponent_ids)
        or (r.agent_o == candidate_id and r.agent_x in opponent_ids)
    ]
    if not sub:
        return {
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "score": 0.0,
            "score_ci": [0.0, 0.0],
            "win_rate_ci": [0.0, 0.0],
        }
    wins = 0
    draws = 0
    losses = 0
    scores: list[float] = []
    for r in sub:
        s = r.score_x if r.agent_x == candidate_id else r.score_o
        scores.append(s)
        if s == 1.0:
            wins += 1
        elif s == 0.5:
            draws += 1
        else:
            losses += 1
    sc_ci = bootstrap_ci(scores, np.mean, n_resamples=n_bootstrap, seed=seed)
    wr_ci = wilson_score_interval(wins, len(sub))
    return {
        "games": len(sub),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round(float(np.mean(scores)), 4),
        "score_ci": [round(sc_ci[0], 4), round(sc_ci[1], 4)],
        "win_rate_ci": [round(wr_ci[0], 4), round(wr_ci[1], 4)],
    }


def _evaluate_rq3_board(
    *,
    board_tag: str,
    board_cfg: BoardConfig,
    resolved_base: Path,
    n_pairs: int,
    n_boot: int,
    base_seed: int,
) -> tuple[list[MatchRecord], dict[str, Any], dict[str, dict[str, Any]]]:
    """Run RQ3 SL-vs-RL evaluation across 3 RL seeds on one board configuration."""
    if board_tag == "4x4_k4":
        sl_agents: dict[str, BaseAgent] = {
            "SL_FastPolicy": PolicyNetworkAgent(
                checkpoint_path=_ckpt(resolved_base, "fastpolicynet_4x4_k4.pt"),
                board_config=board_cfg,
                sample=False,
                device="cpu",
                name="SL_FastPolicy",
                seed=base_seed,
            ),
            "SL_StrongPolicy": PolicyNetworkAgent(
                checkpoint_path=_ckpt(resolved_base, "strongpolicynet_4x4_k4.pt"),
                board_config=board_cfg,
                sample=False,
                device="cpu",
                name="SL_StrongPolicy",
                seed=base_seed,
            ),
            "SL_ValueGreedy": ValueGreedyAgent(
                checkpoint_path=_ckpt(resolved_base, "valuenet_4x4_k4.pt"),
                board_config=board_cfg,
                device="cpu",
                name="SL_ValueGreedy",
                seed=base_seed,
            ),
        }
        rl_agents: dict[str, BaseAgent] = {
            f"RL_DQN_seed{s}": DQNAgent(
                checkpoint_path=_ckpt(resolved_base, f"dqnnet_4x4_k4_seed{s}.pt"),
                board_config=board_cfg,
                epsilon=0.0,
                device="cpu",
                name=f"RL_DQN_seed{s}",
                seed=base_seed + s,
            )
            for s in _RL_SEEDS
        }
        unseen_mcts: dict[str, BaseAgent] = {
            "Unseen_MCTS_50": MCTSAgent(n_simulations=50, name="Unseen_MCTS_50", seed=base_seed),
            "Unseen_MCTS_150": MCTSAgent(
                n_simulations=150, name="Unseen_MCTS_150", seed=base_seed
            ),
        }
        unseen_ab: dict[str, BaseAgent] = {
            "Unseen_AB_d2": MinimaxAgent(
                name="Unseen_AB_d2",
                max_depth=2,
                use_alpha_beta=True,
                use_move_ordering=True,
                seed=base_seed,
            ),
            "Unseen_AB_d4": MinimaxAgent(
                name="Unseen_AB_d4",
                max_depth=4,
                use_alpha_beta=True,
                use_move_ordering=True,
                seed=base_seed,
            ),
        }
    else:
        sl_agents = {
            "SL_FastPolicy": PolicyNetworkAgent(
                checkpoint_path=_ckpt(resolved_base, "fastpolicynet_3x3_k3.pt"),
                board_config=board_cfg,
                sample=False,
                device="cpu",
                name="SL_FastPolicy",
                seed=base_seed,
            ),
            "SL_StrongPolicy": PolicyNetworkAgent(
                checkpoint_path=_ckpt(resolved_base, "strongpolicynet_3x3_k3.pt"),
                board_config=board_cfg,
                sample=False,
                device="cpu",
                name="SL_StrongPolicy",
                seed=base_seed,
            ),
            "SL_ValueGreedy": ValueGreedyAgent(
                checkpoint_path=_ckpt(resolved_base, "valuenet_3x3_k3.pt"),
                board_config=board_cfg,
                device="cpu",
                name="SL_ValueGreedy",
                seed=base_seed,
            ),
        }
        rl_agents = {
            f"RL_TabQ_seed{s}": TabularQAgent(
                checkpoint_path=_ckpt(resolved_base, f"tabular_q_3x3_k3_seed{s}.pkl"),
                board_config=board_cfg,
                epsilon=0.0,
                name=f"RL_TabQ_seed{s}",
                seed=base_seed + s,
            )
            for s in _RL_SEEDS
        }
        unseen_mcts = {
            "Unseen_MCTS_50": MCTSAgent(n_simulations=50, name="Unseen_MCTS_50", seed=base_seed),
            "Unseen_MCTS_150": MCTSAgent(
                n_simulations=150, name="Unseen_MCTS_150", seed=base_seed
            ),
        }
        unseen_ab = {
            "Unseen_AB_d3": MinimaxAgent(
                name="Unseen_AB_d3",
                max_depth=3,
                use_alpha_beta=True,
                use_move_ordering=True,
                seed=base_seed,
            ),
            "Unseen_AB_d6": MinimaxAgent(
                name="Unseen_AB_d6",
                max_depth=6,
                use_alpha_beta=True,
                use_move_ordering=True,
                seed=base_seed,
            ),
        }

    baseline_opps: dict[str, BaseAgent] = {
        "Base_Random": RandomAgent(name="Base_Random", seed=base_seed + 801),
        "Base_RuleBased": RuleBasedAgent(
            name="Base_RuleBased", tie_break="random", seed=base_seed + 802
        ),
    }

    candidates = {**sl_agents, **rl_agents}
    opponents = {**unseen_mcts, **unseen_ab, **baseline_opps}
    unseen_all_ids = set(unseen_mcts.keys()) | set(unseen_ab.keys())
    unseen_mcts_ids = set(unseen_mcts.keys())
    baseline_ids = set(baseline_opps.keys())

    game = ExtendedTicTacToe(board_cfg.rows, board_cfg.cols, board_cfg.win_k)
    records: list[MatchRecord] = []

    for c_idx, (cid, cand) in enumerate(candidates.items()):
        for o_idx, (oid, opp) in enumerate(opponents.items()):
            records.extend(
                _play_paired_matchup(
                    game=game,
                    agent_a_id=cid,
                    agent_a=cand,
                    agent_b_id=oid,
                    agent_b=opp,
                    n_pairs=n_pairs,
                    base_seed=base_seed + c_idx * 100 + o_idx * 25,
                    budget_mode=f"rq3_{board_tag}_eval",
                    prefix=f"rq3_{board_tag}",
                )
            )

    direct_h2h: list[dict[str, Any]] = []
    for s_idx, (sl_id, sl_ag) in enumerate(sl_agents.items()):
        for r_idx, (rl_id, rl_ag) in enumerate(rl_agents.items()):
            pair_recs = _play_paired_matchup(
                game=game,
                agent_a_id=rl_id,
                agent_a=rl_ag,
                agent_b_id=sl_id,
                agent_b=sl_ag,
                n_pairs=n_pairs,
                base_seed=base_seed + 3000 + s_idx * 100 + r_idx * 30,
                budget_mode=f"rq3_{board_tag}_h2h",
                prefix=f"rq3_{board_tag}_h2h",
            )
            records.extend(pair_recs)
            h2h_m = compute_head_to_head_metrics(
                pair_recs, rl_id, sl_id, n_bootstrap=n_boot, seed=base_seed
            )
            direct_h2h.append(h2h_m.to_dict())

    candidate_summary: dict[str, dict[str, Any]] = {}
    for cid in candidates:
        overall = compute_agent_metrics(records, cid, n_bootstrap=n_boot, seed=base_seed).to_dict()
        overall["vs_unseen_mcts"] = _score_subset(
            records, cid, unseen_mcts_ids, n_boot, base_seed
        )
        overall["vs_unseen_all"] = _score_subset(
            records, cid, unseen_all_ids, n_boot, base_seed
        )
        overall["vs_baseline"] = _score_subset(records, cid, baseline_ids, n_boot, base_seed)
        overall["group"] = "Supervised" if cid.startswith("SL_") else "RL"
        candidate_summary[cid] = overall

    rl_scores_all = [candidate_summary[rid]["score"] for rid in rl_agents]
    rl_scores_unseen = [candidate_summary[rid]["vs_unseen_all"]["score"] for rid in rl_agents]
    rl_scores_mcts = [candidate_summary[rid]["vs_unseen_mcts"]["score"] for rid in rl_agents]
    rl_stability = {
        "seeds": list(_RL_SEEDS),
        "overall_mean": round(float(np.mean(rl_scores_all)), 4),
        "overall_std": round(float(np.std(rl_scores_all)), 4),
        "unseen_all_mean": round(float(np.mean(rl_scores_unseen)), 4),
        "unseen_all_std": round(float(np.std(rl_scores_unseen)), 4),
        "unseen_mcts_mean": round(float(np.mean(rl_scores_mcts)), 4),
        "unseen_mcts_std": round(float(np.std(rl_scores_mcts)), 4),
    }

    board_summary = {
        "board_config": board_cfg.to_dict(),
        "total_matches": len(records),
        "candidates": candidate_summary,
        "rl_seed_stability": rl_stability,
        "direct_rl_vs_sl": direct_h2h,
    }
    agents_cfg = {
        f"{board_tag}:{k}": ag.get_config() for k, ag in {**candidates, **opponents}.items()
    }
    return records, board_summary, agents_cfg


def run_ablation_rq3_generalization(
    *,
    mode: Literal["smoke", "full"] = "full",
    seed: int = 42,
    output_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> AblationRunResult:
    """Execute Ablation 4 answering RQ3: Supervised vs Multi-Seed RL Generalization.

    Compares `FastPolicyNet`, `StrongPolicyNet`, and `ValueGreedy` (Supervised) against
    3-seed RL Self-Play (`DQN` seeds `{42, 123, 999}` on `4x4 K=4`; `TabularQ` seeds
    `{42, 123, 999}` on `3x3 K=3`) against strictly unseen MCTS opponents, bounded
    AlphaBeta opponents, baseline opponents, and direct SL-vs-RL matches.
    """
    resolved_mode = _resolve_mode(mode)
    resolved_seed = _validate_seed(seed)
    resolved_base = _resolve_base_dir(base_dir)
    out_path = _default_output_path(output_path, "ablation_4", resolved_base)
    seed_everything(resolved_seed)

    n_pairs = 1 if resolved_mode == "smoke" else 20
    n_boot = 500 if resolved_mode == "smoke" else 10000

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        all_records: list[MatchRecord] = []
        board_summaries: dict[str, Any] = {}
        agents_cfg: dict[str, dict[str, Any]] = {}

        for b_idx, (board_tag, board_cfg) in enumerate(
            (("4x4_k4", BoardConfig(4, 4, 4)), ("3x3_k3", BoardConfig(3, 3, 3)))
        ):
            recs, b_sum, b_agents = _evaluate_rq3_board(
                board_tag=board_tag,
                board_cfg=board_cfg,
                resolved_base=resolved_base,
                n_pairs=n_pairs,
                n_boot=n_boot,
                base_seed=resolved_seed + b_idx * 10000,
            )
            all_records.extend(recs)
            board_summaries[board_tag] = b_sum
            agents_cfg.update(b_agents)

        summary = {"boards": board_summaries}
        meta_header = _build_ablation_meta(
            ablation_id="ablation_4",
            ablation_name="RQ3 Supervised vs Multi-Seed RL Generalization (4x4 K=4 & 3x3 K=3)",
            mode=resolved_mode,
            seed=resolved_seed,
            board_cfg=BoardConfig(4, 4, 4),
            resolved_base=resolved_base,
            agents_config=agents_cfg,
            total_matches=len(all_records),
            summary=summary,
        )
        _write_ablation_jsonl(out_path, meta_header, all_records)
    finally:
        torch.set_num_threads(previous_threads)

    logger.info(
        "Completed Ablation 4 RQ3 (%s, %d matches) -> %s",
        resolved_mode,
        len(all_records),
        out_path,
    )
    return AblationRunResult(
        ablation_id="ablation_4",
        mode=resolved_mode,
        seed=resolved_seed,
        log_path=out_path,
        total_matches=len(all_records),
        meta=meta_header,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Suite Orchestration & Report Exporters
# ---------------------------------------------------------------------------

def format_ablation_suite_markdown(results: Sequence[AblationRunResult]) -> str:
    """Render consolidated Markdown tables covering all executed ablation runs."""
    lines: list[str] = [
        "# Ablation & Research Question Empirical Summary (Task T4.4)",
        "",
    ]
    for res in results:
        lines.append(
            f"## {res.ablation_id.upper()} (`mode={res.mode}`, `matches={res.total_matches}`, "
            f"`log={res.log_path.name}`)"
        )
        lines.append("")
        if res.ablation_id == "ablation_1":
            lines.append("### Controlled State Probes (`4x4, K=4`)")
            lines.append(
                "| Depth | MM Raw Avg Nodes | AB (No Order) Avg Nodes | AB + Order Avg Nodes | "
                "AB Reduction (%) | AB + Order Reduction (%) | Exact Root Value |"
            )
            lines.append("|:---:|---:|---:|---:|---:|---:|:---:|")
            for row in res.summary.get("probe_by_depth", {}).values():
                lines.append(
                    f"| d={row['depth']} | {row['mm_avg_nodes']:.1f} | "
                    f"{row['ab_no_order_avg_nodes']:.1f} | {row['ab_order_avg_nodes']:.1f} | "
                    f"{row['ab_reduction_pct']:.2f}% | {row['ab_order_reduction_pct']:.2f}% | "
                    f"{'YES' if row['all_exact_value_match'] else 'NO'} |"
                )
            lines.extend(["", "### Paired Gameplay Comparison"])
            lines.append(
                "| Variant | Games | Score [95% CI] | Mean Latency (ms) | "
                "P95 Latency (ms) | Avg Nodes |"
            )
            lines.append("|:---|---:|:---:|---:|---:|---:|")
            for cid, m in res.summary.get("match_metrics", {}).items():
                ci = m["score_ci"]
                lines.append(
                    f"| `{cid}` | {m['total_games']} | {m['score']:.3f} "
                    f"[{ci[0]:.3f}, {ci[1]:.3f}] | {m['mean_latency_ms']:.2f} | "
                    f"{m['p95_latency_ms']:.2f} | {m['avg_nodes_visited']:.1f} |"
                )
        elif res.ablation_id == "ablation_2":
            lines.append(
                "| Candidate | Budget | Games vs Pool | Score [95% CI] | "
                "Mean (ms) | P95 (ms) | Avg Sims | Cache Hit |"
            )
            lines.append("|:---|:---:|---:|:---:|---:|---:|---:|---:|")
            for cid, m in res.summary.get("candidate_metrics", {}).items():
                ci = m["score_ci"]
                b_str = f"{m['budget_type']}={m['budget_val']}"
                hit_str = f"{m['cache_hit_rate']:.1%}" if "cache_hit_rate" in m else "-"
                lines.append(
                    f"| `{cid}` | `{b_str}` | {m['total_games']} | {m['score']:.3f} "
                    f"[{ci[0]:.3f}, {ci[1]:.3f}] | {m['mean_latency_ms']:.2f} | "
                    f"{m['p95_latency_ms']:.2f} | {m['avg_simulations']:.1f} | {hit_str} |"
                )
        elif res.ablation_id == "ablation_3":
            lines.append(
                "| Board Variant | Matches | X Win% | O Win% | Draw% | Score(X) | Score(O) | "
                "ΔScore (X-O) [95% CI] |"
            )
            lines.append("|:---|---:|---:|---:|---:|---:|---:|:---:|")
            for btag, bdata in res.summary.get("boards", {}).items():
                sa = bdata["seat_advantage"]
                ci = sa["delta_score_ci"]
                lines.append(
                    f"| `{btag}` | {sa['total_games']} | {sa['x_win_rate']:.1%} | "
                    f"{sa['o_win_rate']:.1%} | {sa['draw_rate']:.1%} | {sa['score_x']:.3f} | "
                    f"{sa['score_o']:.3f} | {sa['delta_score']:+.3f} "
                    f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |"
                )
        elif res.ablation_id == "ablation_4":
            for btag, bdata in res.summary.get("boards", {}).items():
                lines.append(f"### Board `{btag}`")
                lines.append(
                    "| Candidate | Group | Overall Score [95% CI] | Unseen MCTS Score | "
                    "Unseen All Score | Baseline Score | Mean (ms) |"
                )
                lines.append("|:---|:---:|:---:|---:|---:|---:|---:|")
                for cid, m in bdata.get("candidates", {}).items():
                    ci = m["score_ci"]
                    lines.append(
                        f"| `{cid}` | {m['group']} | {m['score']:.3f} "
                        f"[{ci[0]:.3f}, {ci[1]:.3f}] | {m['vs_unseen_mcts']['score']:.3f} | "
                        f"{m['vs_unseen_all']['score']:.3f} | {m['vs_baseline']['score']:.3f} | "
                        f"{m['mean_latency_ms']:.2f} |"
                    )
                stab = bdata.get("rl_seed_stability", {})
                lines.append("")
                lines.append(
                    f"- **RL 3-Seed Stability (`{stab.get('seeds', [])}`)**: "
                    f"Overall `{stab.get('overall_mean', 0.0):.3f} ± "
                    f"{stab.get('overall_std', 0.0):.3f}`, Unseen MCTS `"
                    f"{stab.get('unseen_mcts_mean', 0.0):.3f} ± "
                    f"{stab.get('unseen_mcts_std', 0.0):.3f}`"
                )
                lines.append("")
        lines.append("")
    return "\n".join(lines)


def export_ablation_suite_reports(
    results: Sequence[AblationRunResult],
    reports_dir: str | Path,
) -> tuple[Path, Path]:
    """Export unified CSV metrics and Markdown summary for executed ablations."""
    rep_dir = Path(reports_dir).resolve()
    rep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = rep_dir / "ablation_metrics.csv"
    md_path = rep_dir / "ablation_summary.md"

    fieldnames = [
        "ablation_id",
        "group",
        "candidate",
        "total_games",
        "wins",
        "draws",
        "losses",
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
    for res in results:
        if res.ablation_id == "ablation_1":
            for cid, m in res.summary.get("match_metrics", {}).items():
                rows.append(_agent_row(res.ablation_id, f"d{m.get('depth')}", cid, m, ""))
        elif res.ablation_id == "ablation_2":
            for cid, m in res.summary.get("candidate_metrics", {}).items():
                note = f"{m.get('budget_type')}={m.get('budget_val')}"
                rows.append(_agent_row(res.ablation_id, m.get("family", ""), cid, m, note))
        elif res.ablation_id == "ablation_3":
            for btag, bdata in res.summary.get("boards", {}).items():
                for aid, m in bdata.get("agent_metrics", {}).items():
                    rows.append(_agent_row(res.ablation_id, btag, aid, m, ""))
        elif res.ablation_id == "ablation_4":
            for btag, bdata in res.summary.get("boards", {}).items():
                for cid, m in bdata.get("candidates", {}).items():
                    note = (
                        f"unseen_mcts={m['vs_unseen_mcts']['score']:.4f};"
                        f"unseen_all={m['vs_unseen_all']['score']:.4f}"
                    )
                    rows.append(
                        _agent_row(res.ablation_id, f"{btag}:{m.get('group')}", cid, m, note)
                    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path.write_text(format_ablation_suite_markdown(results), encoding="utf-8")
    logger.info("Exported ablation reports -> %s, %s", csv_path, md_path)
    return csv_path, md_path


def _agent_row(
    ablation_id: str,
    group: str,
    candidate: str,
    m: dict[str, Any],
    notes: str,
) -> dict[str, Any]:
    """Flatten an `AgentPerformanceMetrics.to_dict()` mapping into a CSV row."""
    sc_ci = m.get("score_ci", [0.0, 0.0])
    ds_ci = m.get("delta_score_ci", [0.0, 0.0])
    return {
        "ablation_id": ablation_id,
        "group": group,
        "candidate": candidate,
        "total_games": m.get("total_games", 0),
        "wins": m.get("wins", 0),
        "draws": m.get("draws", 0),
        "losses": m.get("losses", 0),
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


def run_all_ablations(
    *,
    mode: Literal["smoke", "full"] = "full",
    seed: int = 42,
    output_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> list[AblationRunResult]:
    """Run all 4 ablation scenarios sequentially and optionally export unified reports."""
    resolved_base = _resolve_base_dir(base_dir)
    out_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (resolved_base / "logs")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    results = [
        run_ablation_alpha_beta(
            mode=mode,
            seed=seed,
            output_path=out_root / _DEFAULT_FILENAMES["ablation_1"],
            base_dir=resolved_base,
        ),
        run_ablation_rq1_hybrid_mcts(
            mode=mode,
            seed=seed,
            output_path=out_root / _DEFAULT_FILENAMES["ablation_2"],
            base_dir=resolved_base,
        ),
        run_ablation_rq2_seat_advantage(
            mode=mode,
            seed=seed,
            output_path=out_root / _DEFAULT_FILENAMES["ablation_3"],
            base_dir=resolved_base,
        ),
        run_ablation_rq3_generalization(
            mode=mode,
            seed=seed,
            output_path=out_root / _DEFAULT_FILENAMES["ablation_4"],
            base_dir=resolved_base,
        ),
    ]
    if reports_dir is not None:
        export_ablation_suite_reports(results, reports_dir)
    return results
