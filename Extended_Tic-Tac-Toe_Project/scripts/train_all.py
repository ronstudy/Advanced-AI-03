"""CLI entrypoint to train multi-seed Level 6 RL agents (TabularQ & DQN Self-Play)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ttt_ai.agents import DQNAgent, RandomAgent, RuleBasedAgent, TabularQAgent
from ttt_ai.config import BoardConfig, load_yaml_dict, seed_everything
from ttt_ai.training import (
    DQNTrainConfig,
    TabularQTrainConfig,
    evaluate_agent_paired,
    train_dqn_multiseed,
    train_tabular_q_multiseed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for multi-seed RL training."""
    parser = argparse.ArgumentParser(
        description="Train Level 6 TabularQ (3x3) and DQN Self-Play (4x4) across 3 seeds."
    )
    parser.add_argument(
        "--rl-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_rl.yaml",
        help="Path to configs/train_rl.yaml.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=("all", "tabular", "dqn"),
        help="Which RL pipeline(s) to run.",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="all",
        choices=("all", "3x3_k3", "4x4_k4", "4x4_k3"),
        help="Target board configuration to train.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of RNG seeds overriding config seeds.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional episode count override.",
    )
    parser.add_argument(
        "--benchmark-games",
        type=int,
        default=100,
        help="Paired games vs RandomAgent and RuleBasedAgent after training each seed.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute multi-seed RL training and post-training paired evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    cfg = load_yaml_dict(args.rl_config)
    seeds = [int(s) for s in (args.seeds if args.seeds is not None else cfg["seeds"])]

    if args.mode in ("all", "tabular"):
        tab_boards = cfg["tabular_q"]["boards"]
        target_keys = (
            list(tab_boards.keys())
            if args.board == "all"
            else ([args.board] if args.board in tab_boards else [])
        )
        for key in target_keys:
            entry = tab_boards[key]
            board_cfg = BoardConfig(
                rows=int(entry["rows"]),
                cols=int(entry["cols"]),
                win_k=int(entry["win_k"]),
            )
            episodes = int(args.episodes if args.episodes is not None else entry["episodes"])
            decay_eps = min(episodes, int(entry["epsilon_decay_episodes"]))
            base_cfg = TabularQTrainConfig(
                board=board_cfg,
                episodes=episodes,
                alpha=float(entry["alpha"]),
                alpha_min=float(entry["alpha_min"]),
                gamma=float(entry["gamma"]),
                epsilon_start=float(entry["epsilon_start"]),
                epsilon_end=float(entry["epsilon_end"]),
                epsilon_decay_episodes=decay_eps,
                self_play_ratio=float(entry["self_play_ratio"]),
                eval_every_episodes=min(episodes, int(entry["eval_every_episodes"])),
                eval_games=int(entry["eval_games"]),
                seed=seeds[0],
            )
            results = train_tabular_q_multiseed(
                base_cfg,
                seeds=seeds,
                checkpoint_dir=PROJECT_ROOT / str(entry.get("checkpoint_dir", "checkpoints")),
                log_dir=PROJECT_ROOT / str(entry.get("log_dir", "logs")),
            )
            for seed, res in results.items():
                assert res.checkpoint_path is not None
                seed_everything(seed)
                agent = TabularQAgent(res.checkpoint_path)
                score_rand, w_r, d_r, l_r, ms_r = evaluate_agent_paired(
                    agent,
                    RandomAgent(seed=seed + 9000),
                    board_cfg,
                    n_games=args.benchmark_games,
                    base_seed=100,
                )
                score_rule, w_l, d_l, l_l, ms_l = evaluate_agent_paired(
                    agent,
                    RuleBasedAgent(seed=seed + 9000),
                    board_cfg,
                    n_games=args.benchmark_games,
                    base_seed=200,
                )
                logging.info(
                    "TabularQ (%s, seed=%d) | states=%d | vs Random: %d-%d-%d (%.3f, %.3f ms) "
                    "| vs Rule: %d-%d-%d (%.3f, %.3f ms)",
                    key,
                    seed,
                    res.num_states,
                    w_r,
                    d_r,
                    l_r,
                    score_rand,
                    ms_r,
                    w_l,
                    d_l,
                    l_l,
                    score_rule,
                    ms_l,
                )

    if args.mode in ("all", "dqn"):
        dqn_boards = cfg["dqn"]["boards"]
        target_keys = (
            list(dqn_boards.keys())
            if args.board == "all"
            else ([args.board] if args.board in dqn_boards else [])
        )
        for key in target_keys:
            entry = dqn_boards[key]
            board_cfg = BoardConfig(
                rows=int(entry["rows"]),
                cols=int(entry["cols"]),
                win_k=int(entry["win_k"]),
            )
            episodes = int(args.episodes if args.episodes is not None else entry["episodes"])
            decay_eps = min(episodes, int(entry["epsilon_decay_episodes"]))
            base_cfg = DQNTrainConfig(
                board=board_cfg,
                episodes=episodes,
                replay_capacity=int(entry["replay_capacity"]),
                batch_size=int(entry["batch_size"]),
                min_replay_size=int(entry["min_replay_size"]),
                train_every_steps=int(entry["train_every_steps"]),
                target_update_steps=int(entry["target_update_steps"]),
                learning_rate=float(entry["learning_rate"]),
                weight_decay=float(entry["weight_decay"]),
                gamma=float(entry["gamma"]),
                epsilon_start=float(entry["epsilon_start"]),
                epsilon_end=float(entry["epsilon_end"]),
                epsilon_decay_episodes=decay_eps,
                self_play_ratio=float(entry["self_play_ratio"]),
                opponent_pool_size=int(entry["opponent_pool_size"]),
                snapshot_every_episodes=int(entry["snapshot_every_episodes"]),
                eval_every_episodes=min(episodes, int(entry["eval_every_episodes"])),
                eval_games=int(entry["eval_games"]),
                double_dqn=bool(entry["double_dqn"]),
                conv_channels=int(entry["conv_channels"]),
                hidden_dim=int(entry["hidden_dim"]),
                seed=seeds[0],
                device=str(entry.get("device", "cpu")),
            )
            results = train_dqn_multiseed(
                base_cfg,
                seeds=seeds,
                checkpoint_dir=PROJECT_ROOT / str(entry.get("checkpoint_dir", "checkpoints")),
                log_dir=PROJECT_ROOT / str(entry.get("log_dir", "logs")),
            )
            for seed, res in results.items():
                assert res.checkpoint_path is not None
                seed_everything(seed)
                agent = DQNAgent(res.checkpoint_path)
                score_rand, w_r, d_r, l_r, ms_r = evaluate_agent_paired(
                    agent,
                    RandomAgent(seed=seed + 9000),
                    board_cfg,
                    n_games=args.benchmark_games,
                    base_seed=100,
                )
                score_rule, w_l, d_l, l_l, ms_l = evaluate_agent_paired(
                    agent,
                    RuleBasedAgent(seed=seed + 9000),
                    board_cfg,
                    n_games=args.benchmark_games,
                    base_seed=200,
                )
                logging.info(
                    "DQN (%s, seed=%d) | vs Random: %d-%d-%d (%.3f, %.3f ms) "
                    "| vs Rule: %d-%d-%d (%.3f, %.3f ms)",
                    key,
                    seed,
                    w_r,
                    d_r,
                    l_r,
                    score_rand,
                    ms_r,
                    w_l,
                    d_l,
                    l_l,
                    score_rule,
                    ms_l,
                )


if __name__ == "__main__":
    main()
