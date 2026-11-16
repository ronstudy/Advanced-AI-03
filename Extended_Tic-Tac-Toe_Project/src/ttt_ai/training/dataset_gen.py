"""Expert supervised dataset generation and leak-free game-level splitting.

Generates ``(canonical_board, action_label, value_target, game_id)`` tuples from
expert self-play/mixed matches (``AlphaBetaAgent`` and ``RuleBasedAgent`` with
controlled opening perturbation), enforces strict ``game_id``-level splitting
across ``train / val / test`` (70/15/15 by default), tracks deduplication stats,
and persists compressed ``.npz`` archives with SHA-256 verification digests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ttt_ai.agents.base import BaseAgent, _is_finite_real, _is_strict_int
from ttt_ai.agents.minimax_agent import AlphaBetaAgent
from ttt_ai.agents.rule_based_agent import RuleBasedAgent
from ttt_ai.config import BoardConfig
from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.engine.rules import Player
from ttt_ai.training.checkpoint_io import compute_file_sha256

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
DEFAULT_SPLIT_POLICY: str = "game_id_70_15_15"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Validated configuration for supervised expert dataset generation.

    Attributes:
        board: Target :class:`BoardConfig` (e.g., ``4x4 K=4`` or ``3x3 K=3``).
        num_games: Total expert games to generate (must be ``>= 3``).
        opening_random_moves: Maximum random opening plies sampled per game
            in ``[0, opening_random_moves]`` to diversify board trajectories.
        explore_prob: Probability ``[0.0, 1.0)`` of stepping a random legal move
            *after* recording the expert teacher's label at a non-terminal state.
        teacher_max_depth: Search depth for the :class:`AlphaBetaAgent` teacher.
        teacher_mode: Primary teacher algorithm (``"alphabeta"`` or ``"rule_based"``).
        mix_rule_based_opponent: When ``True``, alternates 50% games against
            ``RuleBasedAgent`` to broaden tactical and decisive game outcomes.
        split_ratios: ``(train, val, test)`` game-count proportions summing to 1.0.
        deduplicate_within_split: If ``True``, removes duplicate canonical boards
            inside each split independently while keeping ``game_id`` disjointness.
        seed: Non-negative RNG seed governing opening randomness and splitting.
    """

    board: BoardConfig
    num_games: int = 1000
    opening_random_moves: int = 2
    explore_prob: float = 0.08
    teacher_max_depth: int = 4
    teacher_mode: Literal["alphabeta", "rule_based"] = "alphabeta"
    mix_rule_based_opponent: bool = True
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS
    deduplicate_within_split: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate dataset generation and splitting parameters."""
        if not isinstance(self.board, BoardConfig):
            raise InvalidConfigError(
                f"DatasetConfig.board must be a BoardConfig, got {type(self.board).__name__}."
            )
        if not _is_strict_int(self.num_games) or int(self.num_games) < 3:
            raise InvalidConfigError(
                f"num_games must be an integer >= 3 for 3-way split, got {self.num_games!r}."
            )
        object.__setattr__(self, "num_games", int(self.num_games))

        if not _is_strict_int(self.opening_random_moves) or int(self.opening_random_moves) < 0:
            raise InvalidConfigError(
                "opening_random_moves must be an integer >= 0, "
                f"got {self.opening_random_moves!r}."
            )
        object.__setattr__(self, "opening_random_moves", int(self.opening_random_moves))

        if (
            not _is_finite_real(self.explore_prob)
            or not (0.0 <= float(self.explore_prob) < 1.0)
        ):
            raise InvalidConfigError(
                f"explore_prob must lie in [0.0, 1.0), got {self.explore_prob!r}."
            )
        object.__setattr__(self, "explore_prob", float(self.explore_prob))

        if not _is_strict_int(self.teacher_max_depth) or int(self.teacher_max_depth) < 1:
            raise InvalidConfigError(
                f"teacher_max_depth must be an integer >= 1, got {self.teacher_max_depth!r}."
            )
        object.__setattr__(self, "teacher_max_depth", int(self.teacher_max_depth))

        if self.teacher_mode not in ("alphabeta", "rule_based"):
            raise InvalidConfigError(
                "teacher_mode must be 'alphabeta' or 'rule_based', "
                f"got {self.teacher_mode!r}."
            )
        if not isinstance(self.mix_rule_based_opponent, bool):
            raise InvalidConfigError(
                "mix_rule_based_opponent must be a bool, "
                f"got {type(self.mix_rule_based_opponent).__name__}."
            )
        if not isinstance(self.deduplicate_within_split, bool):
            raise InvalidConfigError(
                "deduplicate_within_split must be a bool, "
                f"got {type(self.deduplicate_within_split).__name__}."
            )

        if not isinstance(self.split_ratios, Sequence) or len(self.split_ratios) != 3:
            raise InvalidConfigError(
                f"split_ratios must be a 3-tuple (train, val, test), got {self.split_ratios!r}."
            )
        ratios: list[float] = []
        for ratio in self.split_ratios:
            if not _is_finite_real(ratio) or float(ratio) <= 0.0:
                raise InvalidConfigError(
                    f"Each split ratio must be finite > 0.0, got {self.split_ratios!r}."
                )
            ratios.append(float(ratio))
        if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
            raise InvalidConfigError(
                f"split_ratios must sum to 1.0, got {sum(ratios):.6f} ({self.split_ratios!r})."
            )
        object.__setattr__(self, "split_ratios", (ratios[0], ratios[1], ratios[2]))

        if not _is_strict_int(self.seed) or int(self.seed) < 0:
            raise InvalidConfigError(
                f"seed must be a non-negative integer, got {self.seed!r}."
            )
        object.__setattr__(self, "seed", int(self.seed))

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataset configuration to a plain JSON-safe dictionary."""
        return {
            "board": self.board.to_dict(),
            "num_games": self.num_games,
            "opening_random_moves": self.opening_random_moves,
            "explore_prob": self.explore_prob,
            "teacher_max_depth": self.teacher_max_depth,
            "teacher_mode": self.teacher_mode,
            "mix_rule_based_opponent": self.mix_rule_based_opponent,
            "split_ratios": list(self.split_ratios),
            "deduplicate_within_split": self.deduplicate_within_split,
            "seed": self.seed,
        }


@dataclass(slots=True)
class SupervisedSplit:
    """In-memory supervised split holding canonical boards, labels, masks, and game IDs."""

    boards: np.ndarray
    actions: np.ndarray
    values: np.ndarray
    legal_masks: np.ndarray
    game_ids: np.ndarray
    raw_count: int = 0
    deduplicated_count: int = 0

    def __post_init__(self) -> None:
        """Normalize dtypes and validate consistent leading dimensions."""
        self.boards = np.asarray(self.boards, dtype=np.int8)
        self.actions = np.asarray(self.actions, dtype=np.int64)
        self.values = np.asarray(self.values, dtype=np.float32)
        self.legal_masks = np.asarray(self.legal_masks, dtype=np.bool_)
        self.game_ids = np.asarray(self.game_ids, dtype=np.int64)

        count = int(self.boards.shape[0])
        if self.boards.ndim != 3:
            raise InvalidConfigError(
                f"SupervisedSplit.boards must be 3D (N, M, N), got {self.boards.shape}."
            )
        num_cells = int(self.boards.shape[1] * self.boards.shape[2])
        if self.actions.shape != (count,):
            raise InvalidConfigError(
                f"SupervisedSplit.actions shape {self.actions.shape} != ({count},)."
            )
        if self.values.shape != (count,):
            raise InvalidConfigError(
                f"SupervisedSplit.values shape {self.values.shape} != ({count},)."
            )
        if self.legal_masks.shape != (count, num_cells):
            raise InvalidConfigError(
                f"SupervisedSplit.legal_masks shape {self.legal_masks.shape} != "
                f"({count}, {num_cells})."
            )
        if self.game_ids.shape != (count,):
            raise InvalidConfigError(
                f"SupervisedSplit.game_ids shape {self.game_ids.shape} != ({count},)."
            )
        if self.raw_count == 0:
            self.raw_count = count
        if self.deduplicated_count == 0:
            self.deduplicated_count = count

    def __len__(self) -> int:
        """Return number of samples in this split."""
        return int(self.boards.shape[0])

    @property
    def unique_game_ids(self) -> set[int]:
        """Set of distinct ``game_id`` values present in this split."""
        return {int(gid) for gid in np.unique(self.game_ids)}

    @property
    def duplicate_ratio(self) -> float:
        """Fraction of raw samples removed by deduplication in this split."""
        if self.raw_count <= 0:
            return 0.0
        removed = max(0, self.raw_count - self.deduplicated_count)
        return float(removed) / float(self.raw_count)


@dataclass(slots=True)
class SupervisedDataset:
    """Complete ``train / val / test`` supervised dataset split strictly by ``game_id``."""

    board_config: BoardConfig
    train: SupervisedSplit
    val: SupervisedSplit
    test: SupervisedSplit
    split_policy: str = DEFAULT_SPLIT_POLICY
    dataset_hash: str = ""
    raw_samples: int = 0
    deduplicated_samples: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Verify disjoint ``game_id`` sets across train, val, and test splits."""
        verify_no_game_leakage(self.train, self.val, self.test)
        if self.raw_samples == 0:
            self.raw_samples = (
                self.train.raw_count + self.val.raw_count + self.test.raw_count
            )
        if self.deduplicated_samples == 0:
            self.deduplicated_samples = len(self.train) + len(self.val) + len(self.test)
        if not self.dataset_hash:
            self.dataset_hash = compute_dataset_hash(self)

    @property
    def duplicate_ratio(self) -> float:
        """Overall duplicate removal rate across all three splits."""
        if self.raw_samples <= 0:
            return 0.0
        removed = max(0, self.raw_samples - self.deduplicated_samples)
        return float(removed) / float(self.raw_samples)

    def summary(self) -> dict[str, Any]:
        """Return compact dictionary of split sizes, game counts, and hash."""
        return {
            "board": self.board_config.name,
            "split_policy": self.split_policy,
            "dataset_hash": self.dataset_hash,
            "raw_samples": self.raw_samples,
            "deduplicated_samples": self.deduplicated_samples,
            "duplicate_ratio": round(self.duplicate_ratio, 6),
            "train_samples": len(self.train),
            "val_samples": len(self.val),
            "test_samples": len(self.test),
            "train_games": len(self.train.unique_game_ids),
            "val_games": len(self.val.unique_game_ids),
            "test_games": len(self.test.unique_game_ids),
        }


def verify_no_game_leakage(
    train: SupervisedSplit,
    val: SupervisedSplit,
    test: SupervisedSplit,
) -> None:
    """Assert that ``game_id`` sets across train, val, and test are pairwise disjoint.

    Raises:
        InvalidConfigError: If any ``game_id`` appears in more than one split.
    """
    train_ids = train.unique_game_ids
    val_ids = val.unique_game_ids
    test_ids = test.unique_game_ids

    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    if overlap_tv or overlap_tt or overlap_vt:
        raise InvalidConfigError(
            "Data leakage detected across game_id splits: "
            f"train&val={len(overlap_tv)}, train&test={len(overlap_tt)}, "
            f"val&test={len(overlap_vt)}."
        )


def compute_dataset_hash(dataset: SupervisedDataset) -> str:
    """Compute deterministic SHA-256 digest over split contents for checkpoint metadata."""
    hasher = hashlib.sha256()
    hasher.update(dataset.board_config.name.encode("utf-8"))
    for split_name, split in (
        ("train", dataset.train),
        ("val", dataset.val),
        ("test", dataset.test),
    ):
        hasher.update(split_name.encode("utf-8"))
        hasher.update(np.ascontiguousarray(split.boards).tobytes())
        hasher.update(np.ascontiguousarray(split.actions).tobytes())
        hasher.update(np.ascontiguousarray(split.values).tobytes())
        hasher.update(np.ascontiguousarray(split.game_ids).tobytes())
    return f"sha256:{hasher.hexdigest().lower()}"


def _deduplicate_split(
    boards: np.ndarray,
    actions: np.ndarray,
    values: np.ndarray,
    legal_masks: np.ndarray,
    game_ids: np.ndarray,
    *,
    deduplicate: bool,
) -> SupervisedSplit:
    """Build a :class:`SupervisedSplit`, optionally filtering repeated canonical boards."""
    raw_count = int(boards.shape[0])
    if not deduplicate or raw_count == 0:
        return SupervisedSplit(
            boards=boards,
            actions=actions,
            values=values,
            legal_masks=legal_masks,
            game_ids=game_ids,
            raw_count=raw_count,
            deduplicated_count=raw_count,
        )

    seen_keys: set[bytes] = set()
    keep_indices: list[int] = []
    for idx in range(raw_count):
        key = boards[idx].tobytes()
        if key not in seen_keys:
            seen_keys.add(key)
            keep_indices.append(idx)

    idx_arr = np.asarray(keep_indices, dtype=np.int64)
    return SupervisedSplit(
        boards=boards[idx_arr],
        actions=actions[idx_arr],
        values=values[idx_arr],
        legal_masks=legal_masks[idx_arr],
        game_ids=game_ids[idx_arr],
        raw_count=raw_count,
        deduplicated_count=len(keep_indices),
    )


def split_by_game_id(
    boards: np.ndarray,
    actions: np.ndarray,
    values: np.ndarray,
    legal_masks: np.ndarray,
    game_ids: np.ndarray,
    board_config: BoardConfig,
    *,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 42,
    deduplicate_within_split: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> SupervisedDataset:
    """Split generated samples into disjoint ``train / val / test`` sets by ``game_id``.

    Splitting at the game level ensures states from the same game trajectory
    never appear in both training and evaluation splits (``RULES.md`` Sec 1.4).

    Args:
        boards: Array of canonical boards ``(N, M, N)``.
        actions: Expert action labels ``(N,)``.
        values: Terminal outcomes ``(N,)`` from the acting player's perspective.
        legal_masks: Boolean legal-action masks ``(N, M * N)``.
        game_ids: Source game identifier for each state ``(N,)``.
        board_config: Validated :class:`BoardConfig`.
        split_ratios: ``(train, val, test)`` proportions summing to ``1.0``.
        seed: RNG seed used to shuffle unique ``game_id`` values before splitting.
        deduplicate_within_split: Whether to drop duplicate canonical states
            inside each split independently.
        metadata: Optional extra dataset metadata to preserve.

    Returns:
        Validated :class:`SupervisedDataset` with disjoint splits.

    Raises:
        InvalidConfigError: If shapes are inconsistent, fewer than 3 unique games
            exist, or split ratios are invalid.
    """
    boards_arr = np.asarray(boards, dtype=np.int8)
    actions_arr = np.asarray(actions, dtype=np.int64)
    values_arr = np.asarray(values, dtype=np.float32)
    masks_arr = np.asarray(legal_masks, dtype=np.bool_)
    gids_arr = np.asarray(game_ids, dtype=np.int64)

    n_samples = int(boards_arr.shape[0])
    if n_samples < 3:
        raise InvalidConfigError(
            f"At least 3 samples across 3 distinct games are required, got {n_samples}."
        )
    if boards_arr.shape[1:] != board_config.shape:
        raise InvalidConfigError(
            f"Board shape {boards_arr.shape[1:]} does not match {board_config.shape}."
        )
    if (
        actions_arr.shape != (n_samples,)
        or values_arr.shape != (n_samples,)
        or masks_arr.shape != (n_samples, board_config.num_cells)
        or gids_arr.shape != (n_samples,)
    ):
        raise InvalidConfigError("Input arrays to split_by_game_id() have mismatched lengths.")

    if not isinstance(split_ratios, Sequence) or len(split_ratios) != 3:
        raise InvalidConfigError(f"split_ratios must have length 3, got {split_ratios!r}.")
    r_train, r_val, r_test = (float(x) for x in split_ratios)
    if (
        min(r_train, r_val, r_test) <= 0.0
        or not math.isclose(r_train + r_val + r_test, 1.0, abs_tol=1e-6)
    ):
        raise InvalidConfigError(f"Invalid split_ratios: {split_ratios!r}.")

    unique_gids = np.unique(gids_arr)
    n_games = int(unique_gids.shape[0])
    if n_games < 3:
        raise InvalidConfigError(
            f"Need at least 3 distinct games to split train/val/test, found {n_games}."
        )

    rng = np.random.default_rng(seed)
    shuffled_gids = unique_gids.copy()
    rng.shuffle(shuffled_gids)

    n_train = max(1, int(round(n_games * r_train)))
    n_val = max(1, int(round(n_games * r_val)))
    # Guarantee at least 1 game stays in test and 1 in val even for small n_games.
    if n_train + n_val >= n_games:
        excess = (n_train + n_val) - (n_games - 1)
        reduce_train = min(excess, n_train - 1)
        n_train -= reduce_train
        excess -= reduce_train
        if excess > 0:
            n_val = max(1, n_val - excess)

    train_gid_set = shuffled_gids[:n_train]
    val_gid_set = shuffled_gids[n_train : n_train + n_val]
    test_gid_set = shuffled_gids[n_train + n_val :]

    train_mask = np.isin(gids_arr, train_gid_set)
    val_mask = np.isin(gids_arr, val_gid_set)
    test_mask = np.isin(gids_arr, test_gid_set)

    train_split = _deduplicate_split(
        boards_arr[train_mask],
        actions_arr[train_mask],
        values_arr[train_mask],
        masks_arr[train_mask],
        gids_arr[train_mask],
        deduplicate=deduplicate_within_split,
    )
    val_split = _deduplicate_split(
        boards_arr[val_mask],
        actions_arr[val_mask],
        values_arr[val_mask],
        masks_arr[val_mask],
        gids_arr[val_mask],
        deduplicate=deduplicate_within_split,
    )
    test_split = _deduplicate_split(
        boards_arr[test_mask],
        actions_arr[test_mask],
        values_arr[test_mask],
        masks_arr[test_mask],
        gids_arr[test_mask],
        deduplicate=deduplicate_within_split,
    )

    split_pct = (
        f"game_id_{int(round(r_train * 100))}_{int(round(r_val * 100))}"
        f"_{int(round(r_test * 100))}"
    )
    return SupervisedDataset(
        board_config=board_config,
        train=train_split,
        val=val_split,
        test=test_split,
        split_policy=split_pct,
        metadata=dict(metadata) if metadata is not None else {},
    )


def _query_teacher_action(
    game: ExtendedTicTacToe,
    teacher: BaseAgent,
    cache: dict[bytes, int] | None,
) -> int:
    """Query the expert teacher with optional canonical-board memoization."""
    if cache is None:
        return teacher.select_action(game).action

    key = game.canonical_board().tobytes()
    cached = cache.get(key)
    if cached is not None:
        return cached
    action = teacher.select_action(game).action
    cache[key] = action
    return action


def generate_expert_dataset(
    config: DatasetConfig,
    *,
    teacher: BaseAgent | None = None,
    opponent: BaseAgent | None = None,
) -> SupervisedDataset:
    """Generate an expert-labeled dataset and split strictly by ``game_id``.

    Each game samples ``k_random`` random opening moves in
    ``[0, config.opening_random_moves]`` so trajectories cover diverse board
    configurations. From ``ply >= k_random`` onward, every state is labeled by
    the expert teacher (``AlphaBetaAgent`` by default) and paired with the
    eventual game outcome ``value_target`` from the acting player's perspective.

    Args:
        config: Validated :class:`DatasetConfig`.
        teacher: Optional custom expert teacher agent. When omitted, builds
            :class:`AlphaBetaAgent` or :class:`RuleBasedAgent` per `config`.
        opponent: Optional opponent agent for mixed matches. When omitted and
            ``config.mix_rule_based_opponent`` is ``True``, uses
            :class:`RuleBasedAgent`.

    Returns:
        Disjoint :class:`SupervisedDataset` ready for supervised training.
    """
    if not isinstance(config, DatasetConfig):
        raise InvalidConfigError(
            f"config must be a DatasetConfig, got {type(config).__name__}."
        )

    rng = np.random.default_rng(config.seed)
    board_cfg = config.board
    num_cells = board_cfg.num_cells

    if teacher is not None:
        resolved_teacher = teacher
        use_cache = False
    elif config.teacher_mode == "alphabeta":
        resolved_teacher = AlphaBetaAgent(
            name="DatasetTeacherAlphaBeta",
            max_depth=config.teacher_max_depth,
            use_alpha_beta=True,
            use_move_ordering=True,
            iterative_deepening=True,
            time_budget_ms=None,
            seed=config.seed,
        )
        use_cache = True
    else:
        resolved_teacher = RuleBasedAgent(
            name="DatasetTeacherRuleBased",
            tie_break="deterministic",
            seed=config.seed,
        )
        use_cache = True

    resolved_opponent: BaseAgent | None = opponent
    if resolved_opponent is None and config.mix_rule_based_opponent:
        resolved_opponent = RuleBasedAgent(
            name="DatasetMixedRuleOpponent",
            tie_break="random",
            seed=config.seed + 1,
        )

    teacher_cache: dict[bytes, int] | None = {} if use_cache else None
    boards_list: list[np.ndarray] = []
    actions_list: list[int] = []
    values_list: list[float] = []
    masks_list: list[np.ndarray] = []
    game_ids_list: list[int] = []

    max_opening = min(config.opening_random_moves, max(0, num_cells - 2))
    game = ExtendedTicTacToe(
        rows=board_cfg.rows,
        cols=board_cfg.cols,
        win_k=board_cfg.win_k,
    )

    for game_id in range(config.num_games):
        game.reset()
        resolved_teacher.reset()
        if resolved_opponent is not None:
            resolved_opponent.reset()

        opening_plies = int(rng.integers(0, max_opening + 1)) if max_opening > 0 else 0
        # Alternate pure expert self-play vs mixed expert-vs-RuleBased matches
        # so ValueNet observes decisive win/loss targets alongside expert draws.
        use_mixed_opp = resolved_opponent is not None and (game_id % 2 == 1)
        opp_player = Player.O if (game_id // 2) % 2 == 0 else Player.X

        game_records: list[tuple[np.ndarray, int, np.ndarray, Player]] = []
        ply = 0

        while not game.is_terminal():
            legal = game.legal_actions()
            if ply < opening_plies:
                chosen_action = int(legal[int(rng.integers(0, len(legal)))])
                game.step(chosen_action)
                ply += 1
                continue

            canonical = game.canonical_board()
            legal_mask = np.zeros(num_cells, dtype=np.bool_)
            legal_mask[legal] = True
            acting_player = game.current_player

            teacher_action = _query_teacher_action(game, resolved_teacher, teacher_cache)
            game_records.append((canonical, teacher_action, legal_mask, acting_player))

            if config.explore_prob > 0.0 and float(rng.random()) < config.explore_prob:
                played_action = int(legal[int(rng.integers(0, len(legal)))])
            elif use_mixed_opp and resolved_opponent is not None and acting_player == opp_player:
                played_action = resolved_opponent.select_action(game).action
            else:
                played_action = teacher_action

            game.step(played_action)
            ply += 1

        if not game_records:
            continue

        for canonical, teacher_action, legal_mask, acting_player in game_records:
            outcome = game.result(int(acting_player))
            assert outcome is not None
            boards_list.append(canonical)
            actions_list.append(teacher_action)
            values_list.append(float(outcome))
            masks_list.append(legal_mask)
            game_ids_list.append(game_id)

    boards_arr = np.stack(boards_list, axis=0).astype(np.int8, copy=False)
    actions_arr = np.asarray(actions_list, dtype=np.int64)
    values_arr = np.asarray(values_list, dtype=np.float32)
    masks_arr = np.stack(masks_list, axis=0).astype(np.bool_, copy=False)
    game_ids_arr = np.asarray(game_ids_list, dtype=np.int64)

    dataset_meta: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "teacher_class": resolved_teacher.__class__.__name__,
        "opponent_class": resolved_opponent.__class__.__name__ if resolved_opponent else None,
        "teacher_cache_size": len(teacher_cache) if teacher_cache is not None else 0,
    }
    dataset = split_by_game_id(
        boards_arr,
        actions_arr,
        values_arr,
        masks_arr,
        game_ids_arr,
        board_cfg,
        split_ratios=config.split_ratios,
        seed=config.seed,
        deduplicate_within_split=config.deduplicate_within_split,
        metadata=dataset_meta,
    )
    logger.info("Generated supervised dataset: %s", dataset.summary())
    return dataset


def save_supervised_dataset(dataset: SupervisedDataset, path: str | Path) -> tuple[Path, str]:
    """Save a :class:`SupervisedDataset` to a compressed ``.npz`` file.

    Args:
        dataset: Validated :class:`SupervisedDataset` instance.
        path: Destination ``.npz`` file path.

    Returns:
        Tuple ``(saved_path, file_sha256_prefixed)``.
    """
    if not isinstance(dataset, SupervisedDataset):
        raise InvalidConfigError(
            f"dataset must be a SupervisedDataset, got {type(dataset).__name__}."
        )
    target = Path(path)
    if target.suffix != ".npz":
        target = target.with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "board_config": dataset.board_config.to_dict(),
        "split_policy": dataset.split_policy,
        "dataset_hash": dataset.dataset_hash,
        "raw_samples": dataset.raw_samples,
        "deduplicated_samples": dataset.deduplicated_samples,
        "train_raw_count": dataset.train.raw_count,
        "val_raw_count": dataset.val.raw_count,
        "test_raw_count": dataset.test.raw_count,
        "metadata": dataset.metadata,
    }
    np.savez_compressed(
        target,
        metadata_json=np.array(json.dumps(header), dtype=np.str_),
        train_boards=dataset.train.boards,
        train_actions=dataset.train.actions,
        train_values=dataset.train.values,
        train_legal_masks=dataset.train.legal_masks,
        train_game_ids=dataset.train.game_ids,
        val_boards=dataset.val.boards,
        val_actions=dataset.val.actions,
        val_values=dataset.val.values,
        val_legal_masks=dataset.val.legal_masks,
        val_game_ids=dataset.val.game_ids,
        test_boards=dataset.test.boards,
        test_actions=dataset.test.actions,
        test_values=dataset.test.values,
        test_legal_masks=dataset.test.legal_masks,
        test_game_ids=dataset.test.game_ids,
    )
    file_hash = f"sha256:{compute_file_sha256(target)}"
    return target, file_hash


def load_supervised_dataset(path: str | Path) -> SupervisedDataset:
    """Load and validate a compressed ``.npz`` :class:`SupervisedDataset`.

    Args:
        path: Path to a ``.npz`` file created by :func:`save_supervised_dataset`.

    Returns:
        Verified :class:`SupervisedDataset` with disjoint splits.

    Raises:
        FileNotFoundError: If `path` does not exist.
        InvalidConfigError: If archive structure or metadata is malformed.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Supervised dataset file not found: {target}")

    try:
        with np.load(target, allow_pickle=False) as archive:
            header = json.loads(str(archive["metadata_json"].item()))
            board_cfg = BoardConfig.from_dict(header["board_config"])
            train_split = SupervisedSplit(
                boards=archive["train_boards"],
                actions=archive["train_actions"],
                values=archive["train_values"],
                legal_masks=archive["train_legal_masks"],
                game_ids=archive["train_game_ids"],
                raw_count=int(header.get("train_raw_count", len(archive["train_boards"]))),
                deduplicated_count=len(archive["train_boards"]),
            )
            val_split = SupervisedSplit(
                boards=archive["val_boards"],
                actions=archive["val_actions"],
                values=archive["val_values"],
                legal_masks=archive["val_legal_masks"],
                game_ids=archive["val_game_ids"],
                raw_count=int(header.get("val_raw_count", len(archive["val_boards"]))),
                deduplicated_count=len(archive["val_boards"]),
            )
            test_split = SupervisedSplit(
                boards=archive["test_boards"],
                actions=archive["test_actions"],
                values=archive["test_values"],
                legal_masks=archive["test_legal_masks"],
                game_ids=archive["test_game_ids"],
                raw_count=int(header.get("test_raw_count", len(archive["test_boards"]))),
                deduplicated_count=len(archive["test_boards"]),
            )
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise InvalidConfigError(
            f"Failed to load supervised dataset '{target}': {exc}"
        ) from exc

    return SupervisedDataset(
        board_config=board_cfg,
        train=train_split,
        val=val_split,
        test=test_split,
        split_policy=str(header.get("split_policy", DEFAULT_SPLIT_POLICY)),
        dataset_hash=str(header.get("dataset_hash", "")),
        raw_samples=int(header.get("raw_samples", 0)),
        deduplicated_samples=int(header.get("deduplicated_samples", 0)),
        metadata=dict(header.get("metadata", {})),
    )
