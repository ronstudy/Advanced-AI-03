"""DQN Self-Play training with Experience Replay, Target Net, and Mixed Opponent Pool.

Implements Level 6 Deep Q-Learning for ``4x4 K=4``, ``4x4 K=3``, and general
``M x N, K`` boards using masked zero-sum Bellman targets, frozen past
snapshots (70% self-play pool), baseline opponents (30% Random/RuleBased), and
multi-seed evaluation (``RQ3``).
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ttt_ai.agents.base import BaseAgent, _is_finite_real, _is_strict_int
from ttt_ai.agents.nn_models import (
    DEFAULT_CONV_CHANNELS,
    DEFAULT_HIDDEN_DIM,
    DQNNet,
    apply_action_mask,
    board_to_tensor,
)
from ttt_ai.agents.random_agent import RandomAgent
from ttt_ai.agents.rl_agent import DQNAgent
from ttt_ai.agents.rule_based_agent import RuleBasedAgent
from ttt_ai.config import BoardConfig, seed_everything
from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.engine.rules import Player
from ttt_ai.training.checkpoint_io import CheckpointMetadata, save_torch_checkpoint
from ttt_ai.training.train_q_tabular import (
    RLCurvePoint,
    _linear_schedule,
    _write_rl_curve_jsonl,
    evaluate_agent_paired,
)

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Fixed-capacity ring buffer storing ``(s, a, r, s_next, done, legal_mask_next)``."""

    def __init__(self, capacity: int, board_config: BoardConfig) -> None:
        """Allocate compact NumPy storage for canonical board transitions."""
        if not _is_strict_int(capacity) or int(capacity) < 1:
            raise InvalidConfigError(f"capacity must be >= 1, got {capacity!r}.")
        if not isinstance(board_config, BoardConfig):
            raise InvalidConfigError(
                f"board_config must be a BoardConfig, got {type(board_config).__name__}."
            )

        self._capacity: int = int(capacity)
        self._rows: int = board_config.rows
        self._cols: int = board_config.cols
        self._num_cells: int = board_config.num_cells
        self._pos: int = 0
        self._size: int = 0

        self._states = np.zeros((self._capacity, self._rows, self._cols), dtype=np.int8)
        self._actions = np.zeros(self._capacity, dtype=np.int64)
        self._rewards = np.zeros(self._capacity, dtype=np.float32)
        self._next_states = np.zeros((self._capacity, self._rows, self._cols), dtype=np.int8)
        self._dones = np.zeros(self._capacity, dtype=np.bool_)
        self._next_legal_masks = np.zeros((self._capacity, self._num_cells), dtype=np.bool_)

    def __len__(self) -> int:
        """Number of transitions currently stored in the buffer."""
        return self._size

    @property
    def capacity(self) -> int:
        """Maximum transition capacity before oldest entries are overwritten."""
        return self._capacity

    def push(
        self,
        state_canon: np.ndarray,
        action: int,
        reward_terminal: float,
        next_state_canon: np.ndarray,
        done: bool,
        legal_mask_next: np.ndarray,
    ) -> None:
        """Append one transition tuple into the circular buffer."""
        idx = self._pos
        self._states[idx] = state_canon
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward_terminal)
        self._next_states[idx] = next_state_canon
        self._dones[idx] = bool(done)
        self._next_legal_masks[idx] = legal_mask_next

        self._pos = (self._pos + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a uniform random minibatch as contiguous PyTorch tensors."""
        if not _is_strict_int(batch_size) or int(batch_size) < 1:
            raise InvalidConfigError(f"batch_size must be >= 1, got {batch_size!r}.")
        if self._size < int(batch_size):
            raise InvalidConfigError(
                f"Cannot sample batch_size={batch_size} from buffer of size {self._size}."
            )

        indices = rng.integers(0, self._size, size=int(batch_size))
        target_dev = torch.device(device)
        states_t = (
            torch.from_numpy(self._states[indices].copy())
            .to(device=target_dev, dtype=torch.float32)
            .unsqueeze(1)
        )
        actions_t = torch.from_numpy(self._actions[indices].copy()).to(
            device=target_dev, dtype=torch.long
        )
        rewards_t = torch.from_numpy(self._rewards[indices].copy()).to(
            device=target_dev, dtype=torch.float32
        )
        next_states_t = (
            torch.from_numpy(self._next_states[indices].copy())
            .to(device=target_dev, dtype=torch.float32)
            .unsqueeze(1)
        )
        dones_t = torch.from_numpy(self._dones[indices].copy()).to(
            device=target_dev, dtype=torch.bool
        )
        next_masks_t = torch.from_numpy(self._next_legal_masks[indices].copy()).to(
            device=target_dev, dtype=torch.bool
        )
        return states_t, actions_t, rewards_t, next_states_t, dones_t, next_masks_t


def compute_zero_sum_bellman_targets(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    next_legal_masks: torch.Tensor,
    *,
    target_net: DQNNet,
    gamma: float,
    online_net: DQNNet | None = None,
    clamp_range: tuple[float, float] | None = (-1.0, 1.0),
) -> torch.Tensor:
    """Compute masked zero-sum Bellman targets for alternating-turn self-play.

    For terminal transitions (``done=True``), ``y = r_terminal``.
    For non-terminal transitions (``done=False``), the successor state ``s'`` is
    encoded from the opponent's perspective, so the target negates the maximum
    legal Q-value of ``s'``: ``y = -gamma * max_{a' in legal(s')} Q_target(s', a')``
    (or uses Double-DQN action selection via ``online_net`` when supplied).
    """
    with torch.no_grad():
        target_q_raw = target_net(next_states)
        target_q_masked = apply_action_mask(
            target_q_raw, next_legal_masks, allow_empty=True
        )
        if online_net is not None:
            online_q_masked = apply_action_mask(
                online_net(next_states), next_legal_masks, allow_empty=True
            )
            best_next_actions = online_q_masked.argmax(dim=-1, keepdim=True)
            next_values = target_q_masked.gather(1, best_next_actions).squeeze(1)
        else:
            next_values = target_q_masked.max(dim=-1).values

        targets = torch.where(dones, rewards, -float(gamma) * next_values)
        if clamp_range is not None:
            targets = targets.clamp(min=float(clamp_range[0]), max=float(clamp_range[1]))
        return targets


@dataclass(frozen=True, slots=True)
class DQNTrainConfig:
    """Validated hyperparameters for DQN Self-Play + Mixed Opponent Pool training."""

    board: BoardConfig = field(default_factory=lambda: BoardConfig(4, 4, 4))
    episodes: int = 20000
    replay_capacity: int = 50000
    batch_size: int = 64
    min_replay_size: int = 512
    train_every_steps: int = 4
    target_update_steps: int = 500
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 15000
    self_play_ratio: float = 0.70
    opponent_pool_size: int = 5
    snapshot_every_episodes: int = 1000
    eval_every_episodes: int = 2000
    eval_games: int = 40
    double_dqn: bool = True
    conv_channels: int = DEFAULT_CONV_CHANNELS
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate all DQN self-play hyperparameters."""
        if not isinstance(self.board, BoardConfig):
            raise InvalidConfigError(
                f"board must be a BoardConfig, got {type(self.board).__name__}."
            )
        for int_name, min_val in (
            ("episodes", 1),
            ("replay_capacity", 1),
            ("batch_size", 1),
            ("min_replay_size", 1),
            ("train_every_steps", 1),
            ("target_update_steps", 1),
            ("epsilon_decay_episodes", 1),
            ("opponent_pool_size", 1),
            ("snapshot_every_episodes", 1),
            ("eval_every_episodes", 1),
            ("eval_games", 2),
            ("conv_channels", 1),
            ("hidden_dim", 1),
            ("seed", 0),
        ):
            val = getattr(self, int_name)
            if not _is_strict_int(val) or int(val) < min_val:
                raise InvalidConfigError(
                    f"{int_name} must be an integer >= {min_val}, got {val!r}."
                )
            object.__setattr__(self, int_name, int(val))

        if self.min_replay_size < self.batch_size:
            object.__setattr__(self, "min_replay_size", self.batch_size)
        if self.replay_capacity < self.batch_size:
            raise InvalidConfigError(
                f"replay_capacity ({self.replay_capacity}) must be >= "
                f"batch_size ({self.batch_size})."
            )

        for float_name, lo, hi in (
            ("gamma", 0.0, 1.0),
            ("epsilon_start", 0.0, 1.0),
            ("epsilon_end", 0.0, 1.0),
            ("self_play_ratio", 0.0, 1.0),
        ):
            val = getattr(self, float_name)
            if not _is_finite_real(val) or not (lo <= float(val) <= hi):
                raise InvalidConfigError(
                    f"{float_name} must lie in [{lo}, {hi}], got {val!r}."
                )
            object.__setattr__(self, float_name, float(val))

        if not _is_finite_real(self.learning_rate) or float(self.learning_rate) <= 0.0:
            raise InvalidConfigError(
                f"learning_rate must be finite > 0.0, got {self.learning_rate!r}."
            )
        object.__setattr__(self, "learning_rate", float(self.learning_rate))

        if not _is_finite_real(self.weight_decay) or float(self.weight_decay) < 0.0:
            raise InvalidConfigError(
                f"weight_decay must be finite >= 0.0, got {self.weight_decay!r}."
            )
        object.__setattr__(self, "weight_decay", float(self.weight_decay))

        if self.epsilon_end > self.epsilon_start:
            raise InvalidConfigError(
                f"epsilon_end ({self.epsilon_end}) cannot exceed "
                f"epsilon_start ({self.epsilon_start})."
            )
        if not isinstance(self.double_dqn, bool):
            raise InvalidConfigError(
                f"double_dqn must be a bool, got {type(self.double_dqn).__name__}."
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise InvalidConfigError(f"device must be a non-empty string, got {self.device!r}.")
        object.__setattr__(self, "device", self.device.strip())


@dataclass(slots=True)
class DQNTrainingResult:
    """Summary returned after training a Level 6 DQN self-play agent."""

    board_config: BoardConfig
    seed: int
    episodes_trained: int
    total_env_steps: int
    optimizer_steps: int
    best_eval_score_vs_random: float
    best_eval_score_vs_rule: float
    training_time_sec: float
    model: DQNNet
    history: list[RLCurvePoint] = field(default_factory=list)
    checkpoint_path: Path | None = None
    meta_path: Path | None = None
    log_path: Path | None = None
    metadata: CheckpointMetadata | None = None


def _pick_dqn_action(
    model: DQNNet,
    canonical_board: np.ndarray,
    legal_actions: list[int],
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    """Select an epsilon-greedy action from `DQNNet` with legal-action masking."""
    if float(rng.random()) < epsilon:
        return int(legal_actions[int(rng.integers(0, len(legal_actions)))])

    x_tensor = board_to_tensor(canonical_board, device=device)
    model.eval()
    with torch.no_grad():
        q_vals = model.predict_masked_q(x_tensor, legal_actions)[0].detach().cpu().numpy()
    legal_q = q_vals[legal_actions]
    best_idx = int(np.argmax(legal_q))
    return int(legal_actions[best_idx])


def train_dqn_selfplay(
    config: DQNTrainConfig,
    *,
    checkpoint_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> DQNTrainingResult:
    """Train ``DQNNet`` via self-play snapshot pool (70%) and baselines (30%)."""
    if not isinstance(config, DQNTrainConfig):
        raise InvalidConfigError(
            f"config must be a DQNTrainConfig, got {type(config).__name__}."
        )

    rng = seed_everything(config.seed)
    board_cfg = config.board
    num_cells = board_cfg.num_cells
    device = torch.device(config.device)

    online_net = DQNNet(
        board_config=board_cfg,
        conv_channels=config.conv_channels,
        hidden_dim=config.hidden_dim,
    ).to(device)
    target_net = DQNNet(
        board_config=board_cfg,
        conv_channels=config.conv_channels,
        hidden_dim=config.hidden_dim,
    ).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    pool_net = DQNNet(
        board_config=board_cfg,
        conv_channels=config.conv_channels,
        hidden_dim=config.hidden_dim,
    ).to(device)
    pool_net.eval()

    opponent_pool: list[dict[str, torch.Tensor]] = [
        copy.deepcopy(online_net.state_dict())
    ]

    optimizer = torch.optim.Adam(
        online_net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(config.replay_capacity, board_cfg)

    random_opp = RandomAgent(seed=config.seed + 101)
    rule_opp = RuleBasedAgent(tie_break="random", seed=config.seed + 202)
    eval_rand_opp = RandomAgent(seed=config.seed + 50_001)
    eval_rule_opp = RuleBasedAgent(tie_break="random", seed=config.seed + 50_002)

    game = ExtendedTicTacToe(
        rows=board_cfg.rows,
        cols=board_cfg.cols,
        win_k=board_cfg.win_k,
    )

    history: list[RLCurvePoint] = []
    recent_losses: list[float] = []
    total_env_steps = 0
    optimizer_steps = 0
    best_composite = -1.0
    best_score_rand = 0.0
    best_score_rule = 0.0
    best_state_dict = copy.deepcopy(online_net.state_dict())

    train_start = time.perf_counter()
    for episode in range(1, config.episodes + 1):
        epsilon = _linear_schedule(
            config.epsilon_start,
            config.epsilon_end,
            episode - 1,
            config.epsilon_decay_episodes,
        )

        use_self_play_pool = float(rng.random()) < config.self_play_ratio
        baseline_opp: BaseAgent | None = None
        use_snapshot = False
        if use_self_play_pool:
            # Mix pure current-self play (50%) and frozen historical snapshot pool (50%).
            if len(opponent_pool) > 1 and float(rng.random()) < 0.5:
                snap_idx = int(rng.integers(0, len(opponent_pool)))
                pool_net.load_state_dict(opponent_pool[snap_idx])
                use_snapshot = True
        else:
            baseline_opp = random_opp if (episode % 2 == 0) else rule_opp
            baseline_opp.reset()

        opp_mark = Player.O if (episode % 2 == 0) else Player.X
        game.reset()

        while not game.is_terminal():
            legal = game.legal_actions()
            s_canon = game.canonical_board()
            acting_player = game.current_player

            if not use_self_play_pool and baseline_opp is not None and acting_player == opp_mark:
                action = baseline_opp.select_action(game).action
            elif use_snapshot and acting_player == opp_mark:
                action = _pick_dqn_action(
                    pool_net, s_canon, legal, 0.05, rng, device
                )
            else:
                action = _pick_dqn_action(
                    online_net, s_canon, legal, epsilon, rng, device
                )

            _, reward, done, _ = game.step(action)
            total_env_steps += 1

            if done:
                s_next_canon = np.zeros((board_cfg.rows, board_cfg.cols), dtype=np.int8)
                next_mask = np.zeros(num_cells, dtype=np.bool_)
            else:
                s_next_canon = game.canonical_board()
                next_legal = game.legal_actions()
                next_mask = np.zeros(num_cells, dtype=np.bool_)
                next_mask[next_legal] = True

            replay.push(s_canon, action, float(reward), s_next_canon, done, next_mask)

            if (
                len(replay) >= config.min_replay_size
                and total_env_steps % config.train_every_steps == 0
            ):
                s_b, a_b, r_b, sn_b, d_b, nm_b = replay.sample(
                    config.batch_size, rng, device=device
                )
                online_net.train()
                q_pred = online_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                targets = compute_zero_sum_bellman_targets(
                    r_b,
                    sn_b,
                    d_b,
                    nm_b,
                    target_net=target_net,
                    gamma=config.gamma,
                    online_net=online_net if config.double_dqn else None,
                )
                loss = loss_fn(q_pred, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), 5.0)
                optimizer.step()

                optimizer_steps += 1
                recent_losses.append(float(loss.item()))

                if optimizer_steps % config.target_update_steps == 0:
                    target_net.load_state_dict(online_net.state_dict())

        if episode % config.snapshot_every_episodes == 0:
            opponent_pool.append(copy.deepcopy(online_net.state_dict()))
            if len(opponent_pool) > config.opponent_pool_size:
                opponent_pool.pop(0)

        if episode % config.eval_every_episodes == 0 or episode == config.episodes:
            online_net.eval()
            eval_agent = DQNAgent(
                model=online_net,
                epsilon=0.0,
                device=device,
                allow_unverified_model=True,
                seed=config.seed,
            )
            score_rand, _, _, _, _ = evaluate_agent_paired(
                eval_agent,
                eval_rand_opp,
                board_cfg,
                n_games=config.eval_games,
                base_seed=config.seed + 60_000,
            )
            score_rule, _, _, _, _ = evaluate_agent_paired(
                eval_agent,
                eval_rule_opp,
                board_cfg,
                n_games=config.eval_games,
                base_seed=config.seed + 70_000,
            )
            composite = 0.65 * score_rand + 0.35 * score_rule
            if composite >= best_composite:
                best_composite = composite
                best_score_rand = score_rand
                best_score_rule = score_rule
                best_state_dict = copy.deepcopy(online_net.state_dict())

            mean_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
            recent_losses.clear()
            history.append(
                RLCurvePoint(
                    episode=episode,
                    epsilon=epsilon,
                    mean_loss=mean_loss,
                    eval_score_vs_random=score_rand,
                    eval_score_vs_rule=score_rule,
                    aux_count=len(replay),
                    elapsed_sec=time.perf_counter() - train_start,
                )
            )

    training_time = time.perf_counter() - train_start
    online_net.load_state_dict(best_state_dict)
    online_net.eval()

    saved_ckpt: Path | None = None
    saved_meta_path: Path | None = None
    saved_meta: CheckpointMetadata | None = None
    if checkpoint_path is not None:
        meta = CheckpointMetadata(
            agent_class="DQNNet",
            board_config=board_cfg,
            seed=config.seed,
            training_type="rl_dqn",
            episodes_trained=config.episodes,
            final_eval_score=best_score_rand,
            reward_summary={
                "best_score_vs_random": best_score_rand,
                "best_score_vs_rule": best_score_rule,
                "optimizer_steps": optimizer_steps,
                "total_env_steps": total_env_steps,
            },
            training_time_sec=training_time,
            extra={
                "replay_capacity": config.replay_capacity,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "gamma": config.gamma,
                "self_play_ratio": config.self_play_ratio,
                "double_dqn": config.double_dqn,
            },
        )
        saved_ckpt, saved_meta_path, saved_meta = save_torch_checkpoint(
            online_net, checkpoint_path, meta
        )

    resolved_log = Path(log_path) if log_path is not None else None
    if resolved_log is not None:
        _write_rl_curve_jsonl(
            resolved_log,
            algorithm="DQNNet",
            board_config=board_cfg,
            seed=config.seed,
            episodes=config.episodes,
            history=history,
            summary={
                "best_score_vs_random": best_score_rand,
                "best_score_vs_rule": best_score_rule,
                "total_env_steps": total_env_steps,
                "optimizer_steps": optimizer_steps,
                "training_time_sec": training_time,
            },
        )

    logger.info(
        "Trained DQNAgent (%s, seed=%d): steps=%d opt_steps=%d "
        "score_rand=%.3f score_rule=%.3f (%.2fs)",
        board_cfg.name,
        config.seed,
        total_env_steps,
        optimizer_steps,
        best_score_rand,
        best_score_rule,
        training_time,
    )
    return DQNTrainingResult(
        board_config=board_cfg,
        seed=config.seed,
        episodes_trained=config.episodes,
        total_env_steps=total_env_steps,
        optimizer_steps=optimizer_steps,
        best_eval_score_vs_random=best_score_rand,
        best_eval_score_vs_rule=best_score_rule,
        training_time_sec=training_time,
        model=online_net,
        history=history,
        checkpoint_path=saved_ckpt,
        meta_path=saved_meta_path,
        log_path=resolved_log,
        metadata=saved_meta,
    )


def train_dqn_multiseed(
    base_config: DQNTrainConfig,
    *,
    seeds: Sequence[int] = (42, 123, 999),
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> dict[int, DQNTrainingResult]:
    """Run DQN self-play training across multiple independent random seeds."""
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    log_root = Path(log_dir) if log_dir is not None else None
    results: dict[int, DQNTrainingResult] = {}

    for seed in seeds:
        per_seed_cfg = DQNTrainConfig(
            board=base_config.board,
            episodes=base_config.episodes,
            replay_capacity=base_config.replay_capacity,
            batch_size=base_config.batch_size,
            min_replay_size=base_config.min_replay_size,
            train_every_steps=base_config.train_every_steps,
            target_update_steps=base_config.target_update_steps,
            learning_rate=base_config.learning_rate,
            weight_decay=base_config.weight_decay,
            gamma=base_config.gamma,
            epsilon_start=base_config.epsilon_start,
            epsilon_end=base_config.epsilon_end,
            epsilon_decay_episodes=base_config.epsilon_decay_episodes,
            self_play_ratio=base_config.self_play_ratio,
            opponent_pool_size=base_config.opponent_pool_size,
            snapshot_every_episodes=base_config.snapshot_every_episodes,
            eval_every_episodes=base_config.eval_every_episodes,
            eval_games=base_config.eval_games,
            double_dqn=base_config.double_dqn,
            conv_channels=base_config.conv_channels,
            hidden_dim=base_config.hidden_dim,
            seed=int(seed),
            device=base_config.device,
        )
        tag = f"{base_config.board.name}_seed{int(seed)}"
        ckpt_file = ckpt_root / f"dqnnet_{tag}.pt" if ckpt_root is not None else None
        log_file = log_root / f"train_dqn_{tag}.jsonl" if log_root is not None else None
        results[int(seed)] = train_dqn_selfplay(
            per_seed_cfg, checkpoint_path=ckpt_file, log_path=log_file
        )
    return results
