"""Tabular Q-learning training pipeline for `3x3, K=3` (and small boards).

Implements zero-sum alternating-turn Q-learning over canonical board byte
keys: ``Q(s, a) <- (1 - alpha) Q(s, a) + alpha * [r - gamma * max_{a'} Q(s', a')]``,
mixing self-play with baseline opponents across multiple random seeds.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ttt_ai.agents.base import BaseAgent, _is_finite_real, _is_strict_int
from ttt_ai.agents.random_agent import RandomAgent
from ttt_ai.agents.rl_agent import TabularQAgent
from ttt_ai.agents.rule_based_agent import RuleBasedAgent
from ttt_ai.config import BoardConfig, seed_everything
from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.engine.rules import Player
from ttt_ai.training.checkpoint_io import CheckpointMetadata, save_pickle_checkpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RLCurvePoint:
    """Single evaluation checkpoint recorded along an RL training curve."""

    episode: int
    epsilon: float
    mean_loss: float
    eval_score_vs_random: float
    eval_score_vs_rule: float
    aux_count: int
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize curve point into a JSON-safe dictionary."""
        return {
            "episode": self.episode,
            "epsilon": self.epsilon,
            "mean_loss": self.mean_loss,
            "eval_score_vs_random": self.eval_score_vs_random,
            "eval_score_vs_rule": self.eval_score_vs_rule,
            "aux_count": self.aux_count,
            "elapsed_sec": self.elapsed_sec,
        }


@dataclass(frozen=True, slots=True)
class TabularQTrainConfig:
    """Validated hyperparameters for Tabular Q-learning."""

    board: BoardConfig = field(default_factory=lambda: BoardConfig(3, 3, 3))
    episodes: int = 60000
    alpha: float = 0.15
    alpha_min: float = 0.03
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 45000
    self_play_ratio: float = 0.70
    eval_every_episodes: int = 5000
    eval_games: int = 60
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate tabular RL hyperparameters."""
        if not isinstance(self.board, BoardConfig):
            raise InvalidConfigError(
                f"board must be a BoardConfig, got {type(self.board).__name__}."
            )
        if not _is_strict_int(self.episodes) or int(self.episodes) < 1:
            raise InvalidConfigError(f"episodes must be >= 1, got {self.episodes!r}.")
        object.__setattr__(self, "episodes", int(self.episodes))

        for name, lo, hi in (
            ("alpha", 0.0, 1.0),
            ("alpha_min", 0.0, 1.0),
            ("gamma", 0.0, 1.0),
            ("epsilon_start", 0.0, 1.0),
            ("epsilon_end", 0.0, 1.0),
            ("self_play_ratio", 0.0, 1.0),
        ):
            val = getattr(self, name)
            if not _is_finite_real(val) or not (lo <= float(val) <= hi):
                raise InvalidConfigError(
                    f"{name} must lie in [{lo}, {hi}], got {val!r}."
                )
            object.__setattr__(self, name, float(val))

        if self.alpha <= 0.0:
            raise InvalidConfigError(f"alpha must be > 0.0, got {self.alpha!r}.")
        if self.alpha_min > self.alpha:
            raise InvalidConfigError(
                f"alpha_min ({self.alpha_min}) cannot exceed alpha ({self.alpha})."
            )
        if self.epsilon_end > self.epsilon_start:
            raise InvalidConfigError(
                f"epsilon_end ({self.epsilon_end}) cannot exceed "
                f"epsilon_start ({self.epsilon_start})."
            )
        if not _is_strict_int(self.epsilon_decay_episodes) or int(self.epsilon_decay_episodes) < 1:
            raise InvalidConfigError(
                f"epsilon_decay_episodes must be >= 1, got {self.epsilon_decay_episodes!r}."
            )
        object.__setattr__(self, "epsilon_decay_episodes", int(self.epsilon_decay_episodes))

        if not _is_strict_int(self.eval_every_episodes) or int(self.eval_every_episodes) < 1:
            raise InvalidConfigError(
                f"eval_every_episodes must be >= 1, got {self.eval_every_episodes!r}."
            )
        object.__setattr__(self, "eval_every_episodes", int(self.eval_every_episodes))

        if not _is_strict_int(self.eval_games) or int(self.eval_games) < 2:
            raise InvalidConfigError(f"eval_games must be >= 2, got {self.eval_games!r}.")
        object.__setattr__(self, "eval_games", int(self.eval_games))

        if not _is_strict_int(self.seed) or int(self.seed) < 0:
            raise InvalidConfigError(f"seed must be >= 0, got {self.seed!r}.")
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(slots=True)
class TabularQTrainingResult:
    """Summary returned after training a Tabular Q-learning agent."""

    board_config: BoardConfig
    seed: int
    episodes_trained: int
    num_states: int
    best_eval_score: float
    final_eval_score_vs_random: float
    final_eval_score_vs_rule: float
    training_time_sec: float
    q_table: dict[bytes, np.ndarray]
    history: list[RLCurvePoint] = field(default_factory=list)
    checkpoint_path: Path | None = None
    meta_path: Path | None = None
    log_path: Path | None = None
    metadata: CheckpointMetadata | None = None


def evaluate_agent_paired(
    agent: BaseAgent,
    opponent: BaseAgent,
    board_config: BoardConfig,
    *,
    n_games: int,
    base_seed: int,
) -> tuple[float, int, int, int, float]:
    """Evaluate `agent` vs `opponent` across paired X/O games.

    Returns:
        Tuple ``(score, wins, draws, losses, mean_agent_latency_ms)``.
    """
    pairs = max(1, n_games // 2)
    wins = 0
    draws = 0
    losses = 0
    latencies: list[float] = []
    game = ExtendedTicTacToe(
        rows=board_config.rows,
        cols=board_config.cols,
        win_k=board_config.win_k,
    )

    for pair_idx in range(pairs):
        for swap in (False, True):
            game.reset()
            agent_x, agent_o = (opponent, agent) if swap else (agent, opponent)
            agent_x.reset(seed=base_seed + pair_idx)
            agent_o.reset(seed=base_seed + 10_000 + pair_idx)

            while not game.is_terminal():
                actor = agent_x if game.current_player == Player.X else agent_o
                decision = actor.select_action(game)
                if actor is agent:
                    latencies.append(decision.latency_ms)
                game.step(decision.action)

            agent_mark = int(Player.O if swap else Player.X)
            payoff = game.result(agent_mark)
            assert payoff is not None
            if payoff > 0.0:
                wins += 1
            elif payoff < 0.0:
                losses += 1
            else:
                draws += 1

    total = wins + draws + losses
    score = (float(wins) + 0.5 * float(draws)) / float(max(1, total))
    mean_ms = float(np.mean(latencies)) if latencies else 0.0
    return score, wins, draws, losses, mean_ms


def _linear_schedule(start: float, end: float, step: int, decay_steps: int) -> float:
    """Compute linearly annealed parameter value clamped at `end`."""
    frac = min(1.0, float(max(0, step)) / float(max(1, decay_steps)))
    return start + (end - start) * frac


def _select_tabular_action(
    q_table: dict[bytes, np.ndarray],
    state_key: bytes,
    legal_actions: list[int],
    num_cells: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Select an epsilon-greedy legal action with random tie-breaking during training."""
    if float(rng.random()) < epsilon:
        return int(legal_actions[int(rng.integers(0, len(legal_actions)))])

    q_vec = q_table.get(state_key)
    if q_vec is None:
        return int(legal_actions[int(rng.integers(0, len(legal_actions)))])

    legal_vals = q_vec[legal_actions]
    max_val = float(np.max(legal_vals))
    best_candidates = [
        a for a, v in zip(legal_actions, legal_vals, strict=True) if float(v) == max_val
    ]
    if len(best_candidates) == 1:
        return int(best_candidates[0])
    return int(best_candidates[int(rng.integers(0, len(best_candidates)))])


def _write_rl_curve_jsonl(
    log_path: Path,
    *,
    algorithm: str,
    board_config: BoardConfig,
    seed: int,
    episodes: int,
    history: Sequence[RLCurvePoint],
    summary: dict[str, Any],
) -> None:
    """Write RL learning curve points and final summary to JSONL."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "type": "meta",
                "algorithm": algorithm,
                "board_config": board_config.to_dict(),
                "seed": seed,
                "episodes": episodes,
            }
        )
    ]
    for pt in history:
        row = {"type": "eval"}
        row.update(pt.to_dict())
        lines.append(json.dumps(row))
    summary_row = {"type": "summary"}
    summary_row.update(summary)
    lines.append(json.dumps(summary_row))
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_tabular_q(
    config: TabularQTrainConfig,
    *,
    checkpoint_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> TabularQTrainingResult:
    """Train a Tabular Q-learning agent via self-play and mixed baseline opponents."""
    if not isinstance(config, TabularQTrainConfig):
        raise InvalidConfigError(
            f"config must be a TabularQTrainConfig, got {type(config).__name__}."
        )

    rng = seed_everything(config.seed)
    board_cfg = config.board
    num_cells = board_cfg.num_cells
    q_table: dict[bytes, np.ndarray] = {}

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
    best_combined = -1.0
    best_q_table: dict[bytes, np.ndarray] = {}
    best_score_rand = 0.0
    best_score_rule = 0.0
    td_errors: list[float] = []

    train_start = time.perf_counter()
    for episode in range(1, config.episodes + 1):
        epsilon = _linear_schedule(
            config.epsilon_start,
            config.epsilon_end,
            episode - 1,
            config.epsilon_decay_episodes,
        )
        alpha = _linear_schedule(
            config.alpha,
            config.alpha_min,
            episode - 1,
            config.episodes,
        )

        use_self_play = float(rng.random()) < config.self_play_ratio
        opp_agent: BaseAgent | None = None
        if not use_self_play:
            opp_agent = random_opp if (episode % 2 == 0) else rule_opp
            opp_agent.reset()
        opp_mark = Player.O if (episode % 2 == 0) else Player.X

        game.reset()
        while not game.is_terminal():
            legal = game.legal_actions()
            s_key = game.canonical_board().tobytes()
            acting_player = game.current_player

            if not use_self_play and opp_agent is not None and acting_player == opp_mark:
                action = opp_agent.select_action(game).action
            else:
                action = _select_tabular_action(
                    q_table, s_key, legal, num_cells, epsilon, rng
                )

            _, reward, done, _ = game.step(action)
            if s_key not in q_table:
                q_table[s_key] = np.zeros(num_cells, dtype=np.float32)

            if done:
                target = float(reward)
            else:
                next_legal = game.legal_actions()
                next_key = game.canonical_board().tobytes()
                q_next = q_table.get(next_key)
                next_max = float(np.max(q_next[next_legal])) if q_next is not None else 0.0
                # Negate next-state value because the opponent moves next (zero-sum).
                target = -config.gamma * next_max

            q_sa = float(q_table[s_key][action])
            td_err = target - q_sa
            q_table[s_key][action] = np.float32(q_sa + alpha * td_err)
            td_errors.append(abs(td_err))

        if episode % config.eval_every_episodes == 0 or episode == config.episodes:
            eval_agent = TabularQAgent(
                q_table=q_table,
                board_config=board_cfg,
                epsilon=0.0,
                allow_unverified_table=True,
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
            combined = 0.6 * score_rand + 0.4 * score_rule
            if combined >= best_combined:
                best_combined = combined
                best_score_rand = score_rand
                best_score_rule = score_rule
                best_q_table = copy.deepcopy(q_table)

            mean_td = float(np.mean(td_errors)) if td_errors else 0.0
            td_errors.clear()
            history.append(
                RLCurvePoint(
                    episode=episode,
                    epsilon=epsilon,
                    mean_loss=mean_td,
                    eval_score_vs_random=score_rand,
                    eval_score_vs_rule=score_rule,
                    aux_count=len(q_table),
                    elapsed_sec=time.perf_counter() - train_start,
                )
            )

    training_time = time.perf_counter() - train_start
    final_table = best_q_table if best_q_table else q_table

    saved_ckpt: Path | None = None
    saved_meta_path: Path | None = None
    saved_meta: CheckpointMetadata | None = None
    if checkpoint_path is not None:
        meta = CheckpointMetadata(
            agent_class="TabularQAgent",
            board_config=board_cfg,
            seed=config.seed,
            training_type="rl_tabular",
            architecture=f"TabularQ(states={len(final_table)}, board={board_cfg.name})",
            episodes_trained=config.episodes,
            final_eval_score=best_score_rand,
            reward_summary={
                "best_score_vs_random": best_score_rand,
                "best_score_vs_rule": best_score_rule,
                "num_states": len(final_table),
            },
            training_time_sec=training_time,
            extra={
                "alpha": config.alpha,
                "alpha_min": config.alpha_min,
                "gamma": config.gamma,
                "self_play_ratio": config.self_play_ratio,
            },
        )
        saved_ckpt, saved_meta_path, saved_meta = save_pickle_checkpoint(
            final_table, checkpoint_path, meta
        )

    resolved_log = Path(log_path) if log_path is not None else None
    if resolved_log is not None:
        _write_rl_curve_jsonl(
            resolved_log,
            algorithm="TabularQAgent",
            board_config=board_cfg,
            seed=config.seed,
            episodes=config.episodes,
            history=history,
            summary={
                "num_states": len(final_table),
                "best_score_vs_random": best_score_rand,
                "best_score_vs_rule": best_score_rule,
                "training_time_sec": training_time,
            },
        )

    logger.info(
        "Trained TabularQAgent (%s, seed=%d): states=%d score_rand=%.3f score_rule=%.3f (%.2fs)",
        board_cfg.name,
        config.seed,
        len(final_table),
        best_score_rand,
        best_score_rule,
        training_time,
    )
    return TabularQTrainingResult(
        board_config=board_cfg,
        seed=config.seed,
        episodes_trained=config.episodes,
        num_states=len(final_table),
        best_eval_score=best_score_rand,
        final_eval_score_vs_random=best_score_rand,
        final_eval_score_vs_rule=best_score_rule,
        training_time_sec=training_time,
        q_table=final_table,
        history=history,
        checkpoint_path=saved_ckpt,
        meta_path=saved_meta_path,
        log_path=resolved_log,
        metadata=saved_meta,
    )


def train_tabular_q_multiseed(
    base_config: TabularQTrainConfig,
    *,
    seeds: Sequence[int] = (42, 123, 999),
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> dict[int, TabularQTrainingResult]:
    """Run Tabular Q-learning across multiple independent random seeds."""
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    log_root = Path(log_dir) if log_dir is not None else None
    results: dict[int, TabularQTrainingResult] = {}

    for seed in seeds:
        per_seed_cfg = TabularQTrainConfig(
            board=base_config.board,
            episodes=base_config.episodes,
            alpha=base_config.alpha,
            alpha_min=base_config.alpha_min,
            gamma=base_config.gamma,
            epsilon_start=base_config.epsilon_start,
            epsilon_end=base_config.epsilon_end,
            epsilon_decay_episodes=base_config.epsilon_decay_episodes,
            self_play_ratio=base_config.self_play_ratio,
            eval_every_episodes=base_config.eval_every_episodes,
            eval_games=base_config.eval_games,
            seed=int(seed),
        )
        tag = f"{base_config.board.name}_seed{int(seed)}"
        ckpt_file = ckpt_root / f"tabular_q_{tag}.pkl" if ckpt_root is not None else None
        log_file = log_root / f"train_tabular_q_{tag}.jsonl" if log_root is not None else None
        results[int(seed)] = train_tabular_q(
            per_seed_cfg, checkpoint_path=ckpt_file, log_path=log_file
        )
    return results
