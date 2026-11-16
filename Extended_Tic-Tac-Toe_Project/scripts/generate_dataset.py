"""CLI script to generate expert supervised datasets and optionally train Level 5 models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ttt_ai.config import BoardConfig, load_yaml_dict, seed_everything
from ttt_ai.training import (
    DatasetConfig,
    SupervisedTrainConfig,
    generate_expert_dataset,
    save_supervised_dataset,
    train_all_supervised_models,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    """Parse command-line flags for dataset generation and optional training."""
    parser = argparse.ArgumentParser(
        description="Generate game-split expert datasets (and optionally train Level 5 networks)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_supervised.yaml",
        help="Path to train_supervised.yaml configuration file.",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="all",
        choices=("all", "4x4_k4", "3x3_k3"),
        help="Target board entry from configs/train_supervised.yaml.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=None,
        help="Optional override for number of games generated per board.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training epochs when --train is specified.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed override.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Also train FastPolicyNet, StrongPolicyNet, and ValueNet after generation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run supervised dataset generation (and optional Level 5 training)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    cfg = load_yaml_dict(args.config)

    global_seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    seed_everything(global_seed)

    dataset_section = cfg["dataset"]
    train_section = cfg["training"]
    board_entries = dataset_section["boards"]
    selected_keys = list(board_entries.keys()) if args.board == "all" else [args.board]

    ratios = tuple(float(v) for v in dataset_section.get("split_ratios", [0.70, 0.15, 0.15]))
    split_ratios = (ratios[0], ratios[1], ratios[2])

    for key in selected_keys:
        entry = board_entries[key]
        board_cfg = BoardConfig(
            rows=int(entry["rows"]),
            cols=int(entry["cols"]),
            win_k=int(entry["win_k"]),
        )
        num_games = int(args.num_games if args.num_games is not None else entry["num_games"])
        ds_cfg = DatasetConfig(
            board=board_cfg,
            num_games=num_games,
            opening_random_moves=int(entry.get("opening_random_moves", 2)),
            explore_prob=float(entry.get("explore_prob", 0.08)),
            teacher_max_depth=int(entry.get("teacher_max_depth", 4)),
            teacher_mode=str(dataset_section.get("teacher_mode", "alphabeta")),  # type: ignore[arg-type]
            mix_rule_based_opponent=bool(dataset_section.get("mix_rule_based_opponent", True)),
            split_ratios=split_ratios,
            deduplicate_within_split=bool(dataset_section.get("deduplicate_within_split", True)),
            seed=global_seed,
        )

        dataset = generate_expert_dataset(ds_cfg)
        out_path = PROJECT_ROOT / str(
            entry.get("output_path", f"logs/datasets/supervised_{board_cfg.name}.npz")
        )
        saved_path, file_hash = save_supervised_dataset(dataset, out_path)
        logging.info(
            "Saved %s -> %s (%s) | summary=%s",
            key,
            saved_path,
            file_hash,
            dataset.summary(),
        )

        if args.train:
            epochs = int(
                args.epochs if args.epochs is not None else train_section.get("epochs", 40)
            )
            base_train_cfg = SupervisedTrainConfig(
                model_name="StrongPolicyNet",
                epochs=epochs,
                batch_size=int(train_section.get("batch_size", 128)),
                learning_rate=float(train_section.get("learning_rate", 1e-3)),
                weight_decay=float(train_section.get("weight_decay", 1e-4)),
                conv_channels=int(train_section.get("conv_channels", 128)),
                hidden_dim=int(train_section.get("hidden_dim", 64)),
                seed=global_seed,
                device=str(train_section.get("device", "cpu")),
            )
            ckpt_dir = PROJECT_ROOT / str(train_section.get("checkpoint_dir", "checkpoints"))
            log_dir = PROJECT_ROOT / str(train_section.get("log_dir", "logs"))
            model_names = tuple(
                train_section.get("models", ["FastPolicyNet", "StrongPolicyNet", "ValueNet"])
            )
            results = train_all_supervised_models(
                dataset,
                base_config=base_train_cfg,
                checkpoint_dir=ckpt_dir,
                log_dir=log_dir,
                model_names=model_names,
            )
            for model_name, res in results.items():
                logging.info(
                    "Trained %s (%s): best_epoch=%d val_loss=%.4f test_loss=%.4f "
                    "test_acc=%s ckpt=%s",
                    model_name,
                    board_cfg.name,
                    res.best_epoch,
                    res.best_val_loss,
                    res.test_loss,
                    f"{res.test_accuracy:.4f}" if res.test_accuracy is not None else "n/a",
                    res.checkpoint_path,
                )


if __name__ == "__main__":
    main()
