"""Level 4 search agent: 4-phase UCT Monte Carlo Tree Search (MCTS)."""

from __future__ import annotations

import math
from typing import Any

from ttt_ai.agents.base import (
    AgentDecision,
    BaseAgent,
    SearchTimeoutError,
    _is_finite_real,
    _is_strict_int,
)
from ttt_ai.engine.exceptions import InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.engine.rules import Player


class _MCTSNode:
    """Compact search-tree node for 4-phase UCT Monte Carlo Tree Search.

    Statistics are stored from the perspective of ``just_moved_player`` (the
    player whose move created this node) so a parent node's ``player_to_move``
    can directly maximize child ``Q(s, a) = value_sum / visits`` during UCT
    selection across alternating turns.
    """

    __slots__ = (
        "__weakref__",
        "parent",
        "action_from_parent",
        "just_moved_player",
        "player_to_move",
        "depth",
        "visits",
        "value_sum",
        "children",
        "untried_actions",
        "is_terminal",
        "terminal_value_for_just_moved",
    )

    def __init__(
        self,
        parent: _MCTSNode | None,
        action_from_parent: int | None,
        just_moved_player: Player,
        player_to_move: Player,
        depth: int,
        untried_actions: list[int],
        is_terminal: bool = False,
        terminal_value_for_just_moved: float = 0.0,
    ) -> None:
        """Initialize an MCTS search-tree node with zero visit counts."""
        self.parent: _MCTSNode | None = parent
        self.action_from_parent: int | None = action_from_parent
        self.just_moved_player: Player = just_moved_player
        self.player_to_move: Player = player_to_move
        self.depth: int = depth
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: dict[int, _MCTSNode] = {}
        self.untried_actions: list[int] = untried_actions
        self.is_terminal: bool = is_terminal
        self.terminal_value_for_just_moved: float = terminal_value_for_just_moved

    @property
    def q_value(self) -> float:
        """Mean empirical return in `[-1.0, +1.0]` for `just_moved_player`."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / float(self.visits)


def _release_tree(root: _MCTSNode | None) -> int:
    """Explicitly break parent/child reference cycles across the MCTS tree.

    Because child nodes reference their parent and parents reference their
    children, Python reference counting cannot immediately reclaim completed
    search trees. Iterative cleanup prevents cyclic-GC latency spikes and
    memory growth across tournaments of thousands of games.

    Args:
        root: Root node of the completed search tree, or ``None``.

    Returns:
        Total number of tree nodes cleared and detached.
    """
    if root is None:
        return 0

    released = 0
    stack: list[_MCTSNode] = [root]
    while stack:
        node = stack.pop()
        if node.children:
            stack.extend(node.children.values())
            node.children.clear()
        node.parent = None
        node.untried_actions.clear()
        released += 1
    return released


class MCTSAgent(BaseAgent):
    """Level 4 AI agent implementing 4-phase UCT Monte Carlo Tree Search.

    Executes up to ``n_simulations`` iterations (or until ``time_budget_ms``
    expires after completing at least one simulation) across Selection (UCT),
    Expansion (shuffled untried action), Simulation (uniform random rollout),
    and Backpropagation (alternating zero-sum return), then picks the robust
    root child with the highest visit count and releases the entire search tree.
    """

    def __init__(
        self,
        n_simulations: int = 200,
        c_uct: float = math.sqrt(2.0),
        name: str | None = "MCTSAgent",
        seed: int | None = None,
        time_budget_ms: float | None = None,
    ) -> None:
        """Initialize a Level 4 UCT MCTS agent.

        Args:
            n_simulations: Maximum MCTS simulations per move (`>= 1`).
            c_uct: UCT exploration constant (`>= 0.0`, default `sqrt(2)`).
            name: Human-readable display name. Defaults to `"MCTSAgent"`.
            seed: Optional non-negative RNG seed for reproducible rollouts.
            time_budget_ms: Optional positive per-move compute budget in ms.

        Raises:
            InvalidConfigError: If any hyperparameter or base config is invalid.
        """
        super().__init__(name=name, level=4, seed=seed, time_budget_ms=time_budget_ms)
        self._n_simulations: int = 200
        self._c_uct: float = math.sqrt(2.0)
        self._last_tree_nodes_released: int = 0

        self.set_n_simulations(n_simulations)
        self.set_c_uct(c_uct)

    @property
    def n_simulations(self) -> int:
        """Configured maximum number of MCTS simulations per move."""
        return self._n_simulations

    @property
    def c_uct(self) -> float:
        """Configured UCT exploration constant."""
        return self._c_uct

    @property
    def last_tree_nodes_released(self) -> int:
        """Number of search-tree nodes freed by the most recent move cleanup."""
        return self._last_tree_nodes_released

    def set_n_simulations(self, n_simulations: int) -> None:
        """Validate and update the simulation budget per move.

        Args:
            n_simulations: Integer count `>= 1`.

        Raises:
            InvalidConfigError: If `n_simulations` is not an integer `>= 1`.
        """
        if not _is_strict_int(n_simulations) or int(n_simulations) < 1:
            raise InvalidConfigError(
                f"n_simulations must be an integer >= 1, got {n_simulations!r}."
            )
        self._n_simulations = int(n_simulations)

    def set_c_uct(self, c_uct: float) -> None:
        """Validate and update the UCT exploration constant.

        Args:
            c_uct: Finite non-negative float exploration constant.

        Raises:
            InvalidConfigError: If `c_uct` is negative, non-finite, or invalid.
        """
        if not _is_finite_real(c_uct) or float(c_uct) < 0.0:
            raise InvalidConfigError(
                f"c_uct must be a finite number >= 0.0, got {c_uct!r}."
            )
        self._c_uct = float(c_uct)

    def get_config(self) -> dict[str, Any]:
        """Return serializable metadata and MCTS hyperparameters."""
        config = super().get_config()
        config.update(
            {
                "n_simulations": self._n_simulations,
                "c_uct": self._c_uct,
            }
        )
        return config

    def _shuffled_actions(self, actions: list[int]) -> list[int]:
        """Return a randomized copy of `actions` using the agent's local RNG."""
        if len(actions) <= 1:
            return list(actions)
        permutation = self.rng.permutation(len(actions))
        return [actions[int(idx)] for idx in permutation]

    def _select_uct_child(self, node: _MCTSNode) -> _MCTSNode:
        """Pick the child maximizing `Q(s, a) + c_uct * sqrt(ln N(s) / N(s, a))`."""
        log_parent_visits = math.log(max(1, node.visits))
        best_child: _MCTSNode | None = None
        best_score = -math.inf

        for child in node.children.values():
            if child.visits == 0:
                return child
            exploitation = child.value_sum / float(child.visits)
            exploration = self._c_uct * math.sqrt(
                log_parent_visits / float(child.visits)
            )
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise RuntimeError("Cannot select UCT child from a node with no children.")
        return best_child

    def _run_random_rollout(
        self,
        sim: ExtendedTicTacToe,
        perspective_player: Player,
    ) -> tuple[float, int]:
        """Simulate uniformly random moves until terminal and return `(reward, plies)`."""
        rollout_steps = 0
        while not sim.is_terminal():
            legal = sim.legal_actions()
            chosen_idx = int(self.rng.integers(0, len(legal)))
            sim.step(legal[chosen_idx])
            rollout_steps += 1

        # Request result explicitly for `perspective_player` because
        # `ExtendedTicTacToe.step()` preserves winner/mover as `current_player`
        # on terminal steps instead of flipping turns.
        outcome = sim.result(int(perspective_player))
        reward = 0.0 if outcome is None else float(outcome)
        return reward, rollout_steps

    @staticmethod
    def _backpropagate(leaf: _MCTSNode, reward_for_leaf_mover: float) -> None:
        """Backpropagate alternating-sign zero-sum outcome from `leaf` to `root`."""
        node: _MCTSNode | None = leaf
        reward = reward_for_leaf_mover
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            # Parent node's `just_moved_player` is the opponent of `node`'s
            # `just_moved_player`, so zero-sum value flips sign at every ply.
            reward = -reward
            node = node.parent

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        """Run 4-phase UCT MCTS and pick the robust child by visit count."""
        deadline = self._compute_deadline()
        sim = game.clone()
        legal_actions = game.legal_actions()
        root_player = game.current_player

        root = _MCTSNode(
            parent=None,
            action_from_parent=None,
            just_moved_player=root_player.opponent,
            player_to_move=root_player,
            depth=0,
            untried_actions=self._shuffled_actions(legal_actions),
            is_terminal=False,
            terminal_value_for_just_moved=0.0,
        )

        nodes_created = 1
        sims_done = 0
        max_depth = 0

        try:
            while sims_done < self._n_simulations:
                undo_count = 0
                node = root
                try:
                    # Phase 1 — Selection: descend fully expanded UCT nodes.
                    while (
                        not node.is_terminal
                        and not node.untried_actions
                        and node.children
                    ):
                        node = self._select_uct_child(node)
                        sim.step(int(node.action_from_parent))
                        undo_count += 1

                    # Phase 2 — Expansion: expand one random untried legal move.
                    if not node.is_terminal and node.untried_actions:
                        action = node.untried_actions.pop()
                        acting_player = node.player_to_move
                        _, _, done, _ = sim.step(action)
                        undo_count += 1

                        if done:
                            outcome = sim.result(int(acting_player))
                            terminal_val = 0.0 if outcome is None else float(outcome)
                            child_untried: list[int] = []
                        else:
                            terminal_val = 0.0
                            child_untried = self._shuffled_actions(sim.legal_actions())

                        child = _MCTSNode(
                            parent=node,
                            action_from_parent=action,
                            just_moved_player=acting_player,
                            player_to_move=acting_player.opponent,
                            depth=node.depth + 1,
                            untried_actions=child_untried,
                            is_terminal=bool(done),
                            terminal_value_for_just_moved=terminal_val,
                        )
                        node.children[action] = child
                        nodes_created += 1
                        node = child

                    if node.depth > max_depth:
                        max_depth = node.depth

                    # Phase 3 — Simulation: random playout from expanded leaf.
                    if node.is_terminal:
                        reward = node.terminal_value_for_just_moved
                    else:
                        reward, rollout_steps = self._run_random_rollout(
                            sim=sim,
                            perspective_player=node.just_moved_player,
                        )
                        undo_count += rollout_steps
                finally:
                    # Unwind every simulation move in reverse so `sim` returns
                    # identically to the root state before the next playout.
                    for _ in range(undo_count):
                        sim.undo()

                # Phase 4 — Backpropagation.
                self._backpropagate(node, reward)
                sims_done += 1

                # Enforce optional time budget only after finishing >= 1
                # simulation so the agent always produces a valid visit tree.
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

            # Robust child selection: highest visit count, breaking exact visit
            # ties by highest empirical `Q(s, a)` in root player's perspective.
            best_action, best_child = max(
                root.children.items(),
                key=lambda item: (item[1].visits, item[1].q_value),
            )

            return AgentDecision(
                action=int(best_action),
                nodes_visited=nodes_created,
                simulations=sims_done,
                depth_reached=max_depth,
                value_estimate=float(best_child.q_value),
                action_scores=action_scores,
            )
        finally:
            self._last_tree_nodes_released = _release_tree(root)
