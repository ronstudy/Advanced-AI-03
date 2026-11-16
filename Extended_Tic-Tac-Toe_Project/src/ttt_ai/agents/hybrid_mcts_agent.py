"""Level 7 AlphaGo-inspired hybrid agent: PUCT MCTS with Policy/Value networks and LRU cache."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ttt_ai.agents.base import (
    AgentDecision,
    BaseAgent,
    SearchTimeoutError,
    _is_finite_real,
    _is_strict_int,
)
from ttt_ai.agents.nn_models import (
    FastPolicyNet,
    StrongPolicyNet,
    ValueNet,
    board_to_tensor,
    masked_softmax,
)
from ttt_ai.agents.policy_agent import (
    _resolve_verified_torch_agent_model,
    _validate_game_matches_board,
)
from ttt_ai.config import BoardConfig
from ttt_ai.engine.exceptions import CheckpointVerificationError, InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.engine.rules import Player
from ttt_ai.training.checkpoint_io import CheckpointMetadata

_DEFAULT_C_PUCT: float = 1.5
_DEFAULT_CACHE_SIZE: int = 4096


class _PUCTNode:
    """Search-tree node storing empirical statistics and Policy network priors.

    Statistics (`visits`, `value_sum`, `q_value`) are stored from the perspective
    of ``just_moved_player`` so ``player_to_move`` directly maximizes child
    ``PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))``.
    """

    __slots__ = (
        "__weakref__",
        "parent",
        "action_from_parent",
        "just_moved_player",
        "player_to_move",
        "depth",
        "prior_prob",
        "visits",
        "value_sum",
        "children",
        "child_priors",
        "legal_actions",
        "is_expanded",
        "is_terminal",
        "terminal_value_for_just_moved",
    )

    def __init__(
        self,
        parent: _PUCTNode | None,
        action_from_parent: int | None,
        just_moved_player: Player,
        player_to_move: Player,
        depth: int,
        prior_prob: float = 1.0,
        is_terminal: bool = False,
        terminal_value_for_just_moved: float = 0.0,
    ) -> None:
        """Initialize a PUCT node prior to network expansion."""
        self.parent: _PUCTNode | None = parent
        self.action_from_parent: int | None = action_from_parent
        self.just_moved_player: Player = just_moved_player
        self.player_to_move: Player = player_to_move
        self.depth: int = depth
        self.prior_prob: float = prior_prob
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: dict[int, _PUCTNode] = {}
        self.child_priors: dict[int, float] = {}
        self.legal_actions: list[int] = []
        self.is_expanded: bool = False
        self.is_terminal: bool = is_terminal
        self.terminal_value_for_just_moved: float = terminal_value_for_just_moved

    @property
    def q_value(self) -> float:
        """Mean empirical return in ``[-1.0, +1.0]`` for ``just_moved_player``."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / float(self.visits)


def _release_puct_tree(root: _PUCTNode | None) -> int:
    """Iteratively break parent-child cycles to reclaim PUCT search trees immediately."""
    if root is None:
        return 0

    released = 0
    stack: list[_PUCTNode] = [root]
    while stack:
        node = stack.pop()
        if node.children:
            stack.extend(node.children.values())
            node.children.clear()
        node.parent = None
        # Rebind rather than mutating shared dicts in case an external/cached ref existed.
        node.child_priors = {}
        node.legal_actions = []
        released += 1
    return released


class HybridMCTSAgent(BaseAgent):
    """Level 7 AlphaGo-inspired hybrid agent combining PUCT MCTS with Policy/Value nets.

    Uses a trained Policy network (``StrongPolicyNet`` or ``FastPolicyNet``) to
    supply legal-masked prior probabilities ``P(s, a)`` in PUCT selection,
    memoizes network evaluations in a bounded LRU cache keyed by canonical board
    bytes, and evaluates expanded leaves via either:
    - ``"rollout"``: Uniform random playout to terminal outcome.
    - ``"mixed"``: Convex blend ``value_weight * (-V(s_leaf)) + (1 - value_weight) * z_rollout``.
    """

    def __init__(
        self,
        policy_checkpoint_path: str | Path | None = None,
        value_checkpoint_path: str | Path | None = None,
        *,
        policy_model: FastPolicyNet | StrongPolicyNet | None = None,
        value_model: ValueNet | None = None,
        policy_metadata: CheckpointMetadata | None = None,
        value_metadata: CheckpointMetadata | None = None,
        board_config: BoardConfig | None = None,
        n_simulations: int = 200,
        c_puct: float = _DEFAULT_C_PUCT,
        leaf_eval_mode: Literal["rollout", "mixed"] = "rollout",
        value_weight: float = 0.5,
        max_cache_size: int = _DEFAULT_CACHE_SIZE,
        device: str | torch.device = "cpu",
        allow_unverified_model: bool = False,
        name: str | None = None,
        seed: int | None = None,
        time_budget_ms: float | None = None,
    ) -> None:
        """Initialize a Level 7 Hybrid PUCT-MCTS agent."""
        if leaf_eval_mode not in ("rollout", "mixed"):
            raise InvalidConfigError(
                f"leaf_eval_mode must be 'rollout' or 'mixed', got {leaf_eval_mode!r}."
            )
        if not _is_finite_real(value_weight) or not (0.0 <= float(value_weight) <= 1.0):
            raise InvalidConfigError(
                f"value_weight must be a finite number in [0.0, 1.0], got {value_weight!r}."
            )
        if not _is_strict_int(max_cache_size) or int(max_cache_size) < 0:
            raise InvalidConfigError(
                f"max_cache_size must be an integer >= 0, got {max_cache_size!r}."
            )

        resolved_policy, resolved_p_meta, resolved_p_ckpt = _resolve_verified_torch_agent_model(
            checkpoint_path=policy_checkpoint_path,
            model=policy_model,
            metadata=policy_metadata,
            board_config=board_config,
            expected_classes=("StrongPolicyNet", "FastPolicyNet"),
            expected_types=(StrongPolicyNet, FastPolicyNet),
            device=device,
            allow_unverified_model=allow_unverified_model,
        )
        resolved_board = resolved_policy.board_config

        resolved_value: ValueNet | None = None
        resolved_v_meta: CheckpointMetadata | None = None
        resolved_v_ckpt: str | None = None
        if value_checkpoint_path is not None or value_model is not None:
            resolved_value, resolved_v_meta, resolved_v_ckpt = (
                _resolve_verified_torch_agent_model(
                    checkpoint_path=value_checkpoint_path,
                    model=value_model,
                    metadata=value_metadata,
                    board_config=resolved_board,
                    expected_classes=("ValueNet",),
                    expected_types=(ValueNet,),
                    device=device,
                    allow_unverified_model=allow_unverified_model,
                )
            )
        elif leaf_eval_mode == "mixed":
            raise CheckpointVerificationError(
                "HybridMCTSAgent with leaf_eval_mode='mixed' requires a verified "
                "value_checkpoint_path or ValueNet instance."
            )

        resolved_name = name if name is not None else f"HybridMCTSAgent({leaf_eval_mode})"
        super().__init__(
            name=resolved_name,
            level=7,
            seed=seed,
            time_budget_ms=time_budget_ms,
        )

        self._policy_model: FastPolicyNet | StrongPolicyNet = resolved_policy
        self._value_model: ValueNet | None = resolved_value
        self._policy_metadata: CheckpointMetadata | None = resolved_p_meta
        self._value_metadata: CheckpointMetadata | None = resolved_v_meta
        self._policy_checkpoint_path: str | None = resolved_p_ckpt
        self._value_checkpoint_path: str | None = resolved_v_ckpt
        self._board_config: BoardConfig = resolved_board
        self._device: torch.device = torch.device(device)

        self._n_simulations: int = 200
        self._c_puct: float = _DEFAULT_C_PUCT
        self._leaf_eval_mode: Literal["rollout", "mixed"] = leaf_eval_mode
        self._value_weight: float = float(value_weight)
        self._max_cache_size: int = int(max_cache_size)
        self._eval_cache: OrderedDict[bytes, tuple[dict[int, float], float | None]] = (
            OrderedDict()
        )
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._last_tree_nodes_released: int = 0

        self.set_n_simulations(n_simulations)
        self.set_c_puct(c_puct)

    @property
    def board_config(self) -> BoardConfig:
        """Validated :class:`BoardConfig` supported by the loaded networks."""
        return self._board_config

    @property
    def n_simulations(self) -> int:
        """Configured maximum number of PUCT simulations per move."""
        return self._n_simulations

    @property
    def c_puct(self) -> float:
        """Configured PUCT exploration constant."""
        return self._c_puct

    @property
    def leaf_eval_mode(self) -> Literal["rollout", "mixed"]:
        """Configured leaf evaluation strategy (``'rollout'`` or ``'mixed'``)."""
        return self._leaf_eval_mode

    @property
    def value_weight(self) -> float:
        """Blend factor ``lambda`` on ``ValueNet`` when ``leaf_eval_mode='mixed'``."""
        return self._value_weight

    @property
    def max_cache_size(self) -> int:
        """Maximum number of canonical states cached in the LRU table."""
        return self._max_cache_size

    @property
    def cache_hits(self) -> int:
        """Cumulative LRU cache hits across network evaluation calls."""
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        """Cumulative LRU cache misses across network evaluation calls."""
        return self._cache_misses

    @property
    def cache_size(self) -> int:
        """Current number of entries stored in the LRU evaluation cache."""
        return len(self._eval_cache)

    @property
    def cache_hit_rate(self) -> float:
        """Ratio of cache hits over total network evaluation requests."""
        total = self._cache_hits + self._cache_misses
        return float(self._cache_hits) / float(total) if total > 0 else 0.0

    @property
    def last_tree_nodes_released(self) -> int:
        """Number of PUCT tree nodes freed after the most recent move."""
        return self._last_tree_nodes_released

    def set_n_simulations(self, n_simulations: int) -> None:
        """Validate and update per-move PUCT simulation budget."""
        if not _is_strict_int(n_simulations) or int(n_simulations) < 1:
            raise InvalidConfigError(
                f"n_simulations must be an integer >= 1, got {n_simulations!r}."
            )
        self._n_simulations = int(n_simulations)

    def set_c_puct(self, c_puct: float) -> None:
        """Validate and update PUCT exploration constant."""
        if not _is_finite_real(c_puct) or float(c_puct) < 0.0:
            raise InvalidConfigError(
                f"c_puct must be a finite number >= 0.0, got {c_puct!r}."
            )
        self._c_puct = float(c_puct)

    def clear_cache(self) -> None:
        """Clear the LRU board evaluation cache and reset hit/miss counters."""
        self._eval_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_config(self) -> dict[str, Any]:
        """Return serializable metadata and Hybrid MCTS hyperparameters."""
        cfg = super().get_config()
        cfg.update(
            {
                "board_config": self._board_config.to_dict(),
                "n_simulations": self._n_simulations,
                "c_puct": self._c_puct,
                "leaf_eval_mode": self._leaf_eval_mode,
                "value_weight": self._value_weight,
                "max_cache_size": self._max_cache_size,
                "policy_model_class": self._policy_model.__class__.__name__,
                "value_model_class": (
                    self._value_model.__class__.__name__
                    if self._value_model is not None
                    else None
                ),
                "policy_checkpoint_path": self._policy_checkpoint_path,
                "value_checkpoint_path": self._value_checkpoint_path,
                "device": str(self._device),
            }
        )
        return cfg

    def _evaluate_state_cached(
        self,
        canonical_board: np.ndarray,
        legal_actions: list[int],
    ) -> tuple[dict[int, float], float | None]:
        """Evaluate Policy prior ``P(s, .)`` (and Value ``V(s)`` if needed) with LRU cache."""
        key = canonical_board.tobytes()
        if self._max_cache_size > 0 and key in self._eval_cache:
            self._eval_cache.move_to_end(key)
            self._cache_hits += 1
            cached_priors, cached_value = self._eval_cache[key]
            if self._leaf_eval_mode != "mixed" or cached_value is not None:
                return cached_priors, cached_value
            # Cache hit had no value entry yet (e.g. if value_model was added); fall through.
            self._cache_hits -= 1

        self._cache_misses += 1
        x_tensor = board_to_tensor(canonical_board, device=self._device)
        with torch.no_grad():
            logits = self._policy_model(x_tensor)
            probs = masked_softmax(logits, legal_actions)[0].detach().cpu().numpy()
            state_value: float | None = None
            if self._leaf_eval_mode == "mixed" and self._value_model is not None:
                state_value = float(self._value_model(x_tensor).item())

        priors = {action: float(probs[action]) for action in legal_actions}
        if self._max_cache_size > 0:
            self._eval_cache[key] = (priors, state_value)
            if len(self._eval_cache) > self._max_cache_size:
                self._eval_cache.popitem(last=False)
        return priors, state_value

    def _expand_node(
        self,
        node: _PUCTNode,
        sim: ExtendedTicTacToe,
    ) -> float | None:
        """Expand a non-terminal node with masked Policy priors and return ``V(s)``."""
        legal = sim.legal_actions()
        canonical = sim.canonical_board()
        priors, state_value = self._evaluate_state_cached(canonical, legal)
        node.legal_actions = list(legal)
        node.child_priors = dict(priors)
        node.is_expanded = True
        return state_value

    def _select_puct_action(self, node: _PUCTNode) -> int:
        """Pick the legal action maximizing AlphaGo PUCT at an expanded node."""
        total_child_visits = sum(child.visits for child in node.children.values())
        sqrt_parent = math.sqrt(float(max(1, total_child_visits)))

        best_action = node.legal_actions[0]
        best_score = -math.inf

        for action in node.legal_actions:
            prior = node.child_priors[action]
            child = node.children.get(action)
            if child is None or child.visits == 0:
                q_val = 0.0
                visits = 0
            else:
                q_val = child.value_sum / float(child.visits)
                visits = child.visits

            puct_score = q_val + self._c_puct * prior * (sqrt_parent / float(1 + visits))
            if puct_score > best_score:
                best_score = puct_score
                best_action = action

        return best_action

    def _run_random_rollout(
        self,
        sim: ExtendedTicTacToe,
        perspective_player: Player,
    ) -> tuple[float, int]:
        """Roll out uniformly at random until terminal and return ``(reward, steps)``."""
        rollout_steps = 0
        while not sim.is_terminal():
            legal = sim.legal_actions()
            chosen_idx = int(self.rng.integers(0, len(legal)))
            sim.step(legal[chosen_idx])
            rollout_steps += 1

        outcome = sim.result(int(perspective_player))
        reward = 0.0 if outcome is None else float(outcome)
        return reward, rollout_steps

    @staticmethod
    def _backpropagate(leaf: _PUCTNode, reward_for_leaf_mover: float) -> None:
        """Backpropagate alternating-sign zero-sum value from ``leaf`` up to ``root``."""
        node: _PUCTNode | None = leaf
        reward = reward_for_leaf_mover
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            reward = -reward
            node = node.parent

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        """Run PUCT MCTS with Policy prior + LRU cache and return search telemetry."""
        _validate_game_matches_board(game, self._board_config, self._name)
        deadline = self._compute_deadline()
        sim = game.clone()
        legal_actions = game.legal_actions()
        root_player = game.current_player

        root = _PUCTNode(
            parent=None,
            action_from_parent=None,
            just_moved_player=root_player.opponent,
            player_to_move=root_player,
            depth=0,
            prior_prob=1.0,
            is_terminal=False,
            terminal_value_for_just_moved=0.0,
        )
        self._expand_node(root, sim)
        root_priors = dict(root.child_priors)

        nodes_created = 1
        sims_done = 0
        max_depth = 0

        try:
            while sims_done < self._n_simulations:
                undo_count = 0
                node = root
                try:
                    # Phase 1 — Selection: descend expanded nodes via PUCT until reaching
                    # a terminal state or an unexpanded/unvisited child action.
                    while not node.is_terminal and node.is_expanded:
                        chosen_action = self._select_puct_action(node)
                        child = node.children.get(chosen_action)
                        if child is not None:
                            sim.step(chosen_action)
                            undo_count += 1
                            node = child
                            continue

                        # Phase 2 — Expansion: instantiate the newly selected child node.
                        acting_player = node.player_to_move
                        _, _, done, _ = sim.step(chosen_action)
                        undo_count += 1
                        if done:
                            outcome = sim.result(int(acting_player))
                            terminal_val = 0.0 if outcome is None else float(outcome)
                        else:
                            terminal_val = 0.0

                        child = _PUCTNode(
                            parent=node,
                            action_from_parent=chosen_action,
                            just_moved_player=acting_player,
                            player_to_move=acting_player.opponent,
                            depth=node.depth + 1,
                            prior_prob=node.child_priors[chosen_action],
                            is_terminal=bool(done),
                            terminal_value_for_just_moved=terminal_val,
                        )
                        node.children[chosen_action] = child
                        nodes_created += 1
                        node = child
                        break

                    if node.depth > max_depth:
                        max_depth = node.depth

                    # Phase 3 — Leaf Evaluation (rollout or ValueNet-mixed).
                    if node.is_terminal:
                        reward = node.terminal_value_for_just_moved
                    else:
                        leaf_opp_value = self._expand_node(node, sim)
                        # Negate opponent-perspective ValueNet prediction to match
                        # `node.just_moved_player` in this zero-sum game.
                        if (
                            self._leaf_eval_mode == "mixed"
                            and leaf_opp_value is not None
                        ):
                            leaf_self_value = -float(leaf_opp_value)
                            if self._value_weight >= 1.0:
                                reward = leaf_self_value
                            else:
                                z_rollout, rollout_steps = self._run_random_rollout(
                                    sim=sim,
                                    perspective_player=node.just_moved_player,
                                )
                                undo_count += rollout_steps
                                reward = (
                                    self._value_weight * leaf_self_value
                                    + (1.0 - self._value_weight) * z_rollout
                                )
                        else:
                            reward, rollout_steps = self._run_random_rollout(
                                sim=sim,
                                perspective_player=node.just_moved_player,
                            )
                            undo_count += rollout_steps
                finally:
                    for _ in range(undo_count):
                        sim.undo()

                # Phase 4 — Backpropagation.
                self._backpropagate(node, reward)
                sims_done += 1

                try:
                    self._check_timeout(deadline)
                except SearchTimeoutError:
                    break

            total_root_visits = sum(
                child.visits for child in root.children.values()
            )
            action_scores: dict[int, float] = {
                action: (
                    float(root.children[action].visits) / float(total_root_visits)
                    if action in root.children and total_root_visits > 0
                    else 0.0
                )
                for action in legal_actions
            }

            best_action, best_child = max(
                root.children.items(),
                key=lambda item: (
                    item[1].visits,
                    item[1].q_value,
                    item[1].prior_prob,
                    -item[0],
                ),
            )

            return AgentDecision(
                action=int(best_action),
                nodes_visited=nodes_created,
                simulations=sims_done,
                depth_reached=max_depth,
                value_estimate=float(best_child.q_value),
                action_scores=action_scores,
                prior_probs=root_priors,
            )
        finally:
            self._last_tree_nodes_released = _release_puct_tree(root)
