"""Unit and integration tests for Task T5.2 visualization and report pipeline."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from ttt_ai import (
    generate_all_visualizations_and_reports,
    generate_summary_reports,
    plot_fig1_head_to_head,
    plot_fig2_latency_and_nodes,
    plot_fig3_ablation_rq1,
    plot_fig4_ablation_rq2,
    plot_fig5_ablation_rq3,
)
from ttt_ai.engine.exceptions import InvalidConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_plot_script() -> ModuleType:
    """Dynamically load `scripts/plot_results.py` for CLI flag and main() testing."""
    script_path = PROJECT_ROOT / "scripts" / "plot_results.py"
    spec = importlib.util.spec_from_file_location("ttt_plot_results_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_valid_png(path: Path, *, min_bytes: int = 10_000) -> None:
    """Verify `path` exists, starts with the PNG signature, and exceeds `min_bytes`."""
    assert path.is_file(), f"Missing PNG output: {path}"
    data = path.read_bytes()
    assert data.startswith(PNG_MAGIC), f"{path} is not a valid PNG file."
    assert len(data) >= min_bytes, f"{path} is unexpectedly small ({len(data)} bytes)."


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a list of JSON objects to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _build_synthetic_logs(log_dir: Path) -> None:
    """Populate `log_dir` with minimal valid tournament, ablation, and train JSONL logs."""
    board_4x4 = {"rows": 4, "cols": 4, "win_k": 4}
    agents_meta = {
        "L1_Random": {"level": 1},
        "L2_RuleBased": {"level": 2},
        "L4_MCTS": {"level": 4},
    }
    matches = [
        {
            "type": "match",
            "match_id": "m0",
            "pair_id": 0,
            "game_in_pair": 0,
            "seed": 1,
            "board_config": board_4x4,
            "budget_mode": "default",
            "agent_x": "L2_RuleBased",
            "agent_o": "L1_Random",
            "winner": 1,
            "score_x": 1.0,
            "score_o": 0.0,
            "num_moves": 7,
            "latencies_x_ms": [0.1, 0.1, 0.1, 0.1],
            "latencies_o_ms": [0.03, 0.03, 0.03],
            "nodes_x": [20, 20, 20, 20],
            "nodes_o": [1, 1, 1],
            "sims_x": [0, 0, 0, 0],
            "sims_o": [0, 0, 0],
            "winning_line": [[0, 0], [0, 1], [0, 2], [0, 3]],
            "moves": [
                [0, 0.1, 20, 0, 1],
                [4, 0.03, 1, 0, 0],
                [1, 0.1, 20, 0, 1],
                [5, 0.03, 1, 0, 0],
                [2, 0.1, 20, 0, 1],
                [6, 0.03, 1, 0, 0],
                [3, 0.1, 20, 0, 1],
            ],
        },
        {
            "type": "match",
            "match_id": "m1",
            "pair_id": 0,
            "game_in_pair": 1,
            "seed": 1,
            "board_config": board_4x4,
            "budget_mode": "default",
            "agent_x": "L1_Random",
            "agent_o": "L2_RuleBased",
            "winner": -1,
            "score_x": 0.0,
            "score_o": 1.0,
            "num_moves": 8,
            "latencies_x_ms": [0.03, 0.03, 0.03, 0.03],
            "latencies_o_ms": [0.1, 0.1, 0.1, 0.1],
            "nodes_x": [1, 1, 1, 1],
            "nodes_o": [20, 20, 20, 20],
            "sims_x": [0, 0, 0, 0],
            "sims_o": [0, 0, 0, 0],
            "winning_line": [[1, 0], [1, 1], [1, 2], [1, 3]],
            "moves": [
                [0, 0.03, 1, 0, 0],
                [4, 0.1, 20, 0, 1],
                [1, 0.03, 1, 0, 0],
                [5, 0.1, 20, 0, 1],
                [8, 0.03, 1, 0, 0],
                [6, 0.1, 20, 0, 1],
                [9, 0.03, 1, 0, 0],
                [7, 0.1, 20, 0, 1],
            ],
        },
        {
            "type": "match",
            "match_id": "m2",
            "pair_id": 1,
            "game_in_pair": 0,
            "seed": 2,
            "board_config": board_4x4,
            "budget_mode": "default",
            "agent_x": "L4_MCTS",
            "agent_o": "L2_RuleBased",
            "winner": 0,
            "score_x": 0.5,
            "score_o": 0.5,
            "num_moves": 2,
            "latencies_x_ms": [12.0],
            "latencies_o_ms": [0.12],
            "nodes_x": [110],
            "nodes_o": [22],
            "sims_x": [200],
            "sims_o": [0],
            "winning_line": None,
            "moves": [[5, 12.0, 110, 200, 3], [6, 0.12, 22, 0, 1]],
        },
        {
            "type": "match",
            "match_id": "m3",
            "pair_id": 1,
            "game_in_pair": 1,
            "seed": 2,
            "board_config": board_4x4,
            "budget_mode": "default",
            "agent_x": "L2_RuleBased",
            "agent_o": "L4_MCTS",
            "winner": 0,
            "score_x": 0.5,
            "score_o": 0.5,
            "num_moves": 2,
            "latencies_x_ms": [0.11],
            "latencies_o_ms": [11.5],
            "nodes_x": [22],
            "nodes_o": [105],
            "sims_x": [0],
            "sims_o": [200],
            "winning_line": None,
            "moves": [[5, 0.11, 22, 0, 1], [6, 11.5, 105, 200, 3]],
        },
    ]
    _write_jsonl(
        log_dir / "tournament_4x4_k4_default.jsonl",
        [
            {
                "type": "meta",
                "tournament_name": "synthetic_rr",
                "board_config": board_4x4,
                "budget_mode": "default",
                "agents": agents_meta,
            },
            *matches,
        ],
    )

    _write_jsonl(
        log_dir / "ablation_ab_pruning.jsonl",
        [
            {
                "type": "meta",
                "ablation_id": "ablation_1",
                "mode": "smoke",
                "seed": 42,
                "summary": {
                    "probe_by_depth": {
                        "d2": {
                            "depth": 2,
                            "mm_avg_nodes": 175.0,
                            "ab_no_order_avg_nodes": 60.0,
                            "ab_order_avg_nodes": 54.0,
                            "ab_reduction_pct": 65.7,
                            "ab_order_reduction_pct": 69.1,
                        },
                        "d3": {
                            "depth": 3,
                            "mm_avg_nodes": 2071.0,
                            "ab_no_order_avg_nodes": 353.0,
                            "ab_order_avg_nodes": 322.0,
                            "ab_reduction_pct": 83.0,
                            "ab_order_reduction_pct": 84.5,
                        },
                    },
                    "match_metrics": {},
                },
            }
        ],
    )

    _write_jsonl(
        log_dir / "ablation_rq1_hybrid_mcts.jsonl",
        [
            {
                "type": "meta",
                "ablation_id": "ablation_2",
                "mode": "smoke",
                "seed": 42,
                "summary": {
                    "candidate_metrics": {
                        "Policy_Strong": {
                            "family": "Policy_Strong",
                            "budget_type": "none",
                            "budget_val": 0,
                            "score": 0.50,
                            "score_ci": [0.45, 0.55],
                            "mean_latency_ms": 0.35,
                            "avg_simulations": 0.0,
                        },
                        "PureMCTS_sim50": {
                            "family": "PureMCTS",
                            "budget_type": "simulations",
                            "budget_val": 50,
                            "score": 0.52,
                            "score_ci": [0.47, 0.57],
                            "mean_latency_ms": 3.5,
                            "avg_simulations": 50.0,
                        },
                        "PureMCTS_sim150": {
                            "family": "PureMCTS",
                            "budget_type": "simulations",
                            "budget_val": 150,
                            "score": 0.58,
                            "score_ci": [0.53, 0.63],
                            "mean_latency_ms": 10.0,
                            "avg_simulations": 150.0,
                        },
                        "HybridMixed_sim50": {
                            "family": "HybridMixed",
                            "budget_type": "simulations",
                            "budget_val": 50,
                            "score": 0.58,
                            "score_ci": [0.53, 0.63],
                            "mean_latency_ms": 8.0,
                            "avg_simulations": 50.0,
                            "cache_hit_rate": 0.60,
                        },
                        "HybridMixed_sim150": {
                            "family": "HybridMixed",
                            "budget_type": "simulations",
                            "budget_val": 150,
                            "score": 0.60,
                            "score_ci": [0.55, 0.65],
                            "mean_latency_ms": 20.0,
                            "avg_simulations": 150.0,
                            "cache_hit_rate": 0.64,
                        },
                        "PureMCTS_time10ms": {
                            "family": "PureMCTS",
                            "budget_type": "time_ms",
                            "budget_val": 10.0,
                            "score": 0.59,
                            "score_ci": [0.54, 0.64],
                            "mean_latency_ms": 9.2,
                            "avg_simulations": 180.0,
                        },
                        "HybridMixed_time10ms": {
                            "family": "HybridMixed",
                            "budget_type": "time_ms",
                            "budget_val": 10.0,
                            "score": 0.57,
                            "score_ci": [0.52, 0.62],
                            "mean_latency_ms": 9.4,
                            "avg_simulations": 140.0,
                            "cache_hit_rate": 0.65,
                        },
                    }
                },
            }
        ],
    )

    _write_jsonl(
        log_dir / "ablation_rq2_seat_advantage.jsonl",
        [
            {
                "type": "meta",
                "ablation_id": "ablation_3",
                "mode": "smoke",
                "seed": 42,
                "summary": {
                    "boards": {
                        "4x4_k3": {
                            "seat_advantage": {
                                "total_games": 40,
                                "x_win_rate": 0.75,
                                "draw_rate": 0.0,
                                "o_win_rate": 0.25,
                                "score_x": 0.75,
                                "score_o": 0.25,
                                "delta_score": 0.50,
                                "delta_score_ci": [0.40, 0.60],
                            },
                            "agent_metrics": {
                                "L2_RuleBased": {
                                    "total_games": 20,
                                    "wins": 14,
                                    "draws": 0,
                                    "losses": 6,
                                    "score": 0.70,
                                    "score_ci": [0.55, 0.85],
                                    "score_x": 1.0,
                                    "score_o": 0.4,
                                    "delta_score": 0.60,
                                    "delta_score_ci": [0.45, 0.75],
                                }
                            },
                        },
                        "4x4_k4": {
                            "seat_advantage": {
                                "total_games": 40,
                                "x_win_rate": 0.15,
                                "draw_rate": 0.75,
                                "o_win_rate": 0.10,
                                "score_x": 0.525,
                                "score_o": 0.475,
                                "delta_score": 0.05,
                                "delta_score_ci": [-0.02, 0.12],
                            },
                            "agent_metrics": {
                                "L2_RuleBased": {
                                    "total_games": 20,
                                    "wins": 4,
                                    "draws": 16,
                                    "losses": 0,
                                    "score": 0.60,
                                    "score_ci": [0.52, 0.68],
                                    "score_x": 0.62,
                                    "score_o": 0.58,
                                    "delta_score": 0.04,
                                    "delta_score_ci": [-0.04, 0.12],
                                }
                            },
                        },
                    }
                },
            }
        ],
    )

    _write_jsonl(
        log_dir / "ablation_rq3_rl_generalization.jsonl",
        [
            {
                "type": "meta",
                "ablation_id": "ablation_4",
                "mode": "smoke",
                "seed": 42,
                "summary": {
                    "boards": {
                        "4x4_k4": {
                            "candidates": {
                                "SL_StrongPolicy": {
                                    "group": "Supervised",
                                    "total_games": 40,
                                    "wins": 4,
                                    "draws": 30,
                                    "losses": 6,
                                    "score": 0.475,
                                    "score_ci": [0.40, 0.55],
                                    "vs_unseen_mcts": {"score": 0.35, "score_ci": [0.25, 0.45]},
                                    "vs_unseen_all": {"score": 0.42, "score_ci": [0.33, 0.51]},
                                    "vs_baseline": {"score": 0.65, "score_ci": [0.55, 0.75]},
                                },
                                "RL_DQN_seed42": {
                                    "group": "RL",
                                    "total_games": 40,
                                    "wins": 10,
                                    "draws": 29,
                                    "losses": 1,
                                    "score": 0.612,
                                    "score_ci": [0.54, 0.68],
                                    "vs_unseen_mcts": {"score": 0.55, "score_ci": [0.45, 0.65]},
                                    "vs_unseen_all": {"score": 0.53, "score_ci": [0.44, 0.62]},
                                    "vs_baseline": {"score": 0.72, "score_ci": [0.62, 0.82]},
                                },
                            }
                        }
                    }
                },
            }
        ],
    )


class TestVisualizerUnitSynthetic:
    """Verify plotting and reporting helpers on controlled synthetic JSONL logs."""

    def test_synthetic_end_to_end_generation_and_validation(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        out_dir = tmp_path / "reports"
        _build_synthetic_logs(log_dir)

        artifacts = generate_all_visualizations_and_reports(
            logs_dir=log_dir,
            output_dir=out_dir,
            checkpoints_dir=tmp_path / "missing_ckpts",
            figures=("all",),
            dpi=150,
        )
        expected_keys = {
            "fig1",
            "fig2",
            "fig3",
            "fig4",
            "fig5",
            "summary_csv",
            "summary_tables",
            "sample_failure_cases",
        }
        assert expected_keys <= artifacts.keys()
        for key in ("fig1", "fig2", "fig3", "fig4", "fig5"):
            _assert_valid_png(artifacts[key])

        csv_rows = list(csv.DictReader(artifacts["summary_csv"].open("r", encoding="utf-8")))
        assert len(csv_rows) >= 5
        t_rows = {r["agent_or_candidate"]: r for r in csv_rows if r["source_type"] == "tournament"}
        assert t_rows["L2_RuleBased"]["wins"] == "2"
        assert t_rows["L2_RuleBased"]["draws"] == "2"
        assert t_rows["L1_Random"]["losses"] == "2"

        failures_text = artifacts["sample_failure_cases"].read_text(encoding="utf-8")
        assert "Decisive Match Excerpt" in failures_text or "Case 1" in failures_text

    def test_error_handling_for_invalid_dpi_or_missing_logs(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidConfigError):
            plot_fig1_head_to_head(LOGS_DIR, tmp_path / "bad.png", dpi=10)
        with pytest.raises(InvalidConfigError):
            generate_all_visualizations_and_reports(
                logs_dir=LOGS_DIR,
                output_dir=tmp_path / "out",
                figures=("9",),
            )
        with pytest.raises(FileNotFoundError):
            plot_fig3_ablation_rq1(tmp_path / "nonexistent.jsonl", output_path=tmp_path / "f3.png")

    def test_visualizer_module_ast_compliance(self) -> None:
        viz_path = PROJECT_ROOT / "src" / "ttt_ai" / "benchmark" / "visualizer.py"
        tree = ast.parse(viz_path.read_text(encoding="utf-8"))
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
                assert node.func.id != "print", "Forbidden bare print() in visualizer.py"
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden, f"Forbidden private access {node.attr}"


class TestVisualizerOfficialLogsAndCLI:
    """Verify full figure generation and CLI execution against real Sprint 4 logs."""

    def test_official_logs_generate_all_five_300dpi_figures_and_reports(
        self, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "official_reports"
        play_viz = _load_plot_script()
        args = play_viz._parse_args(
            [
                "--logs-dir",
                str(LOGS_DIR),
                "--output-dir",
                str(out_dir),
                "--checkpoints-dir",
                str(CKPT_DIR),
                "--figures",
                "all",
                "--dpi",
                "300",
            ]
        )
        assert args.dpi == 300
        assert args.figures == ["all"]

        play_viz.main(
            [
                "--logs-dir",
                str(LOGS_DIR),
                "--output-dir",
                str(out_dir),
                "--checkpoints-dir",
                str(CKPT_DIR),
                "--figures",
                "all",
                "--dpi",
                "300",
            ]
        )

        fig_dir = out_dir / "figures"
        expected_figs = (
            "fig1_head_to_head_matrix.png",
            "fig2_latency_and_nodes.png",
            "fig3_ablation_rq1_and_alphabeta.png",
            "fig4_ablation_rq2_seat_advantage.png",
            "fig5_ablation_rq3_learning_and_generalization.png",
        )
        for fname in expected_figs:
            _assert_valid_png(fig_dir / fname, min_bytes=40_000)

        assert (out_dir / "summary_metrics.csv").is_file()
        tables_md = (out_dir / "summary_tables.md").read_text(encoding="utf-8")
        failures_md = (out_dir / "sample_failure_cases.md").read_text(encoding="utf-8")
        assert "tournament_4x4_k4_default.jsonl" in tables_md
        assert "AB_Order" in tables_md or "ablation_ab_pruning" in tables_md
        assert "StrongPolicyNet Top-3 Priors" in failures_md
        assert "Case 1" in failures_md
        assert "Case 3" in failures_md

    def test_individual_plot_entrypoints_subset(self, tmp_path: Path) -> None:
        f1 = plot_fig1_head_to_head(
            LOGS_DIR / "tournament_4x4_k4_fixed_50ms.jsonl",
            tmp_path / "f1.png",
            dpi=150,
        )
        f2 = plot_fig2_latency_and_nodes(
            LOGS_DIR / "tournament_4x4_k4_fixed_50ms.jsonl",
            tmp_path / "f2.png",
            dpi=150,
        )
        f3 = plot_fig3_ablation_rq1(
            LOGS_DIR / "ablation_rq1_hybrid_mcts.jsonl",
            LOGS_DIR / "ablation_ab_pruning.jsonl",
            tmp_path / "f3.png",
            dpi=150,
        )
        f4 = plot_fig4_ablation_rq2(
            LOGS_DIR / "ablation_rq2_seat_advantage.jsonl",
            tmp_path / "f4.png",
            dpi=150,
        )
        f5 = plot_fig5_ablation_rq3(
            LOGS_DIR / "ablation_rq3_rl_generalization.jsonl",
            logs_dir=LOGS_DIR,
            output_path=tmp_path / "f5.png",
            dpi=150,
        )
        reports = generate_summary_reports(
            logs_dir=LOGS_DIR,
            output_dir=tmp_path / "rep",
            checkpoints_dir=CKPT_DIR,
            n_bootstrap=500,
        )
        for p in (f1, f2, f3, f4, f5):
            _assert_valid_png(p)
        assert all(p.is_file() for p in reports.values())
