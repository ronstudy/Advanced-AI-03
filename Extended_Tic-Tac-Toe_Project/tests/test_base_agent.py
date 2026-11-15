"""Unit tests for Task T2.1: BaseAgent contract, AgentDecision, and SearchTimeoutError."""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pytest

from ttt_ai import AgentDecision, BaseAgent, ExtendedTicTacToe, SearchTimeoutError
from ttt_ai.agents import uniform_action_scores
from ttt_ai.engine import IllegalMoveError, InvalidConfigError, Player, TicTacToeError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = PROJECT_ROOT / "src" / "ttt_ai" / "agents"


class _DeterministicTestAgent(BaseAgent):
    """Concrete agent subclass that exercises clone/step/undo via public Game APIs."""

    def __init__(
        self,
        name: str = "DeterministicTest",
        level: int = 1,
        seed: int | None = 42,
        time_budget_ms: float | None = None,
        use_clone: bool = True,
        busy_wait_ms: float = 0.0,
    ) -> None:
        super().__init__(name=name, level=level, seed=seed, time_budget_ms=time_budget_ms)
        self.use_clone = use_clone
        self.busy_wait_ms = busy_wait_ms

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        if self.busy_wait_ms > 0.0:
            target = time.perf_counter() + (self.busy_wait_ms / 1000.0)
            while time.perf_counter() < target:
                pass

        legal = game.legal_actions()
        scores = uniform_action_scores(legal)

        # Probe one step with either clone() or balanced step()/undo() to verify
        # BaseAgent accepts clean search routines without flagging state mutation.
        probe_action = legal[0]
        if self.use_clone:
            sim = game.clone()
            sim.step(probe_action)
        else:
            game.step(probe_action)
            game.undo()

        chosen_idx = int(self.rng.integers(0, len(legal)))
        chosen_action = legal[chosen_idx]
        return AgentDecision(
            action=chosen_action,
            nodes_visited=len(legal),
            simulations=1,
            depth_reached=1,
            value_estimate=0.0,
            action_scores=scores,
            prior_probs=dict(scores),
        )


class _IntReturnAgent(BaseAgent):
    """Subclass returning a raw integer to verify normalization into AgentDecision."""

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        return game.legal_actions()[0]  # type: ignore[return-value]


class _MutatingAgent(BaseAgent):
    """Faulty subclass that mutates the source game without calling undo()."""

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        legal = game.legal_actions()
        game.step(legal[0])
        return AgentDecision(action=legal[0])


class _IllegalActionAgent(BaseAgent):
    """Faulty subclass returning a fixed illegal action index."""

    def __init__(self, forced_action: int) -> None:
        super().__init__(name="IllegalActionAgent", level=1)
        self.forced_action = forced_action

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        return AgentDecision(action=self.forced_action)


class _BadReturnTypeAgent(BaseAgent):
    """Faulty subclass returning an unsupported type from _select_action."""

    def _select_action(self, game: ExtendedTicTacToe) -> AgentDecision:
        return "invalid_return"  # type: ignore[return-value]


def test_agent_decision_validation_normalization_and_helpers() -> None:
    """Verify AgentDecision normalizes NumPy types, ranks top_actions, and validates fields."""
    decision = AgentDecision(
        action=np.int64(5),  # type: ignore[arg-type]
        latency_ms=np.float32(1.25),  # type: ignore[arg-type]
        nodes_visited=np.int32(12),  # type: ignore[arg-type]
        simulations=np.int64(8),  # type: ignore[arg-type]
        depth_reached=np.int16(3),  # type: ignore[arg-type]
        value_estimate=np.float64(-0.5),  # type: ignore[arg-type]
        action_scores={np.int64(5): np.float32(0.5), 2: 0.5, 7: 0.25},  # type: ignore[dict-item]
        prior_probs={5: 0.5, 2: 0.3, 7: 0.2},
    )

    assert type(decision.action) is int and decision.action == 5
    assert type(decision.latency_ms) is float and decision.latency_ms == pytest.approx(1.25)
    assert type(decision.nodes_visited) is int and decision.nodes_visited == 12
    assert type(decision.simulations) is int and decision.simulations == 8
    assert type(decision.depth_reached) is int and decision.depth_reached == 3
    assert type(decision.value_estimate) is float and decision.value_estimate == pytest.approx(-0.5)

    # Ties on 0.5 must break deterministically by ascending action index: 2 before 5.
    assert decision.top_actions(k=2) == [(2, 0.5), (5, 0.5)]
    assert decision.top_actions(k=0) == []

    serialized = decision.to_dict()
    assert serialized["action"] == 5
    assert serialized["action_scores"] == {5: 0.5, 2: 0.5, 7: 0.25}
    # Mutating serialized dict must not mutate the decision instance.
    serialized["action_scores"][5] = 0.99
    assert decision.action_scores[5] == 0.5

    with pytest.raises(InvalidConfigError):
        decision.top_actions(k=-1)
    with pytest.raises(InvalidConfigError):
        decision.top_actions(k=True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "exc_type"),
    [
        ({"action": -1}, IllegalMoveError),
        ({"action": True}, IllegalMoveError),
        ({"action": 1.5}, IllegalMoveError),
        ({"action": 0, "latency_ms": -0.1}, InvalidConfigError),
        ({"action": 0, "latency_ms": float("nan")}, InvalidConfigError),
        ({"action": 0, "latency_ms": float("inf")}, InvalidConfigError),
        ({"action": 0, "latency_ms": True}, InvalidConfigError),
        ({"action": 0, "nodes_visited": -1}, InvalidConfigError),
        ({"action": 0, "simulations": True}, InvalidConfigError),
        ({"action": 0, "depth_reached": -2}, InvalidConfigError),
        ({"action": 0, "value_estimate": float("nan")}, InvalidConfigError),
        ({"action": 0, "value_estimate": False}, InvalidConfigError),
        ({"action": 0, "action_scores": {-1: 0.5}}, IllegalMoveError),
        ({"action": 0, "action_scores": {0: float("inf")}}, InvalidConfigError),
        ({"action": 0, "prior_probs": "bad"}, InvalidConfigError),
    ],
)
def test_agent_decision_rejects_invalid_inputs(
    kwargs: dict[str, object],
    exc_type: type[Exception],
) -> None:
    """Ensure AgentDecision rejects malformed actions, metrics, and score maps."""
    with pytest.raises(exc_type):
        AgentDecision(**kwargs)  # type: ignore[arg-type]


def test_uniform_action_scores_behavior_and_validation() -> None:
    """Verify uniform_action_scores handles empty, standard, and invalid sequences."""
    assert uniform_action_scores([]) == {}

    scores = uniform_action_scores([0, 3, 5, 7])
    assert len(scores) == 4
    assert sum(scores.values()) == pytest.approx(1.0)
    for action, prob in scores.items():
        assert type(action) is int
        assert prob == pytest.approx(0.25)

    for bad in ([-1], [True], [1.2]):
        with pytest.raises(IllegalMoveError):
            uniform_action_scores(bad)  # type: ignore[arg-type]


def test_base_agent_abstract_and_config_validation() -> None:
    """Verify BaseAgent cannot be instantiated directly and validates config fields."""
    assert issubclass(SearchTimeoutError, TicTacToeError)

    with pytest.raises(TypeError):
        BaseAgent()  # type: ignore[abstract]

    agent = _DeterministicTestAgent(name="  MyAgent  ", level=2, seed=7, time_budget_ms=25)
    assert agent.name == "MyAgent"
    assert agent.level == 2
    assert agent.seed == 7
    assert agent.time_budget_ms == 25.0
    assert isinstance(agent.time_budget_ms, float)
    assert agent.get_config() == {
        "agent_class": "_DeterministicTestAgent",
        "name": "MyAgent",
        "level": 2,
        "seed": 7,
        "time_budget_ms": 25.0,
    }

    default_name_agent = _IntReturnAgent()
    assert default_name_agent.name == "_IntReturnAgent"
    assert default_name_agent.level == 0
    assert default_name_agent.seed is None
    assert default_name_agent.time_budget_ms is None

    for bad_name in ("", "   ", 123):
        with pytest.raises(InvalidConfigError):
            _DeterministicTestAgent(name=bad_name)  # type: ignore[arg-type]
    for bad_level in (-1, True, 1.5):
        with pytest.raises(InvalidConfigError):
            _DeterministicTestAgent(level=bad_level)  # type: ignore[arg-type]
    for bad_seed in (-1, True, 3.14):
        with pytest.raises(InvalidConfigError):
            _DeterministicTestAgent(seed=bad_seed)  # type: ignore[arg-type]
    for bad_budget in (0, -5.0, float("nan"), float("inf"), False):
        with pytest.raises(InvalidConfigError):
            _DeterministicTestAgent(time_budget_ms=bad_budget)  # type: ignore[arg-type]


def test_base_agent_seed_reproducibility_and_timeout_helpers() -> None:
    """Verify isolated RNG reproducibility via reset/set_seed and timeout enforcement."""
    agent_a = _DeterministicTestAgent(seed=99, time_budget_ms=10.0)
    agent_b = _DeterministicTestAgent(seed=99, time_budget_ms=10.0)
    game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)

    seq_a = [agent_a.select_action(game).action for _ in range(5)]
    seq_b = [agent_b.select_action(game).action for _ in range(5)]
    assert seq_a == seq_b

    agent_a.reset(seed=99)
    seq_a_reseeded = [agent_a.select_action(game).action for _ in range(5)]
    assert seq_a_reseeded == seq_a

    # Timeout helpers: unbounded returns None; past deadline raises SearchTimeoutError.
    unbounded = _DeterministicTestAgent(time_budget_ms=None)
    assert unbounded._compute_deadline() is None
    unbounded._check_timeout(None)

    past_origin = time.perf_counter() - 1.0
    past_deadline = agent_a._compute_deadline(start_time=past_origin)
    assert past_deadline is not None
    with pytest.raises(SearchTimeoutError):
        agent_a._check_timeout(past_deadline)

    future_deadline = agent_a._compute_deadline()
    assert future_deadline is not None and future_deadline > time.perf_counter()
    agent_a._check_timeout(future_deadline)


@pytest.mark.parametrize(
    ("rows", "cols", "win_k"),
    [(3, 3, 3), (4, 4, 3), (4, 4, 4), (5, 5, 4)],
)
def test_base_agent_full_game_rollout_and_state_preservation(
    rows: int,
    cols: int,
    win_k: int,
) -> None:
    """Play full matches across all board sizes and verify legal moves and immutability."""
    agent_x = _DeterministicTestAgent(name="X_Agent", level=1, seed=101, use_clone=True)
    agent_o = _DeterministicTestAgent(name="O_Agent", level=1, seed=202, use_clone=False)
    game = ExtendedTicTacToe(rows=rows, cols=cols, win_k=win_k)

    while not game.is_terminal():
        active_agent = agent_x if game.current_player is Player.X else agent_o
        legal_before = game.legal_actions()
        board_before = game.board
        history_before = game.move_history

        decision = active_agent.select_action(game)

        # Source game state must remain completely untouched by select_action().
        np.testing.assert_array_equal(game.board, board_before)
        assert game.move_history == history_before
        assert game.legal_actions() == legal_before

        assert isinstance(decision, AgentDecision)
        assert type(decision.action) is int
        assert decision.action in legal_before
        assert type(decision.latency_ms) is float and decision.latency_ms >= 0.0
        assert set(decision.action_scores.keys()) == set(legal_before)

        game.step(decision.action)

    assert game.is_terminal()
    assert game.legal_actions() == []


def test_latency_measurement_wraps_entire_select_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify latency_ms uses time.perf_counter() spanning the full select_action call."""
    # Deterministic clock check via monkeypatched perf_counter: 100.0 -> 100.0125 s = 12.5 ms.
    timestamps = iter([100.0, 100.0125])
    monkeypatch.setattr("ttt_ai.agents.base.time.perf_counter", lambda: next(timestamps))

    agent = _IntReturnAgent()
    game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    decision = agent.select_action(game)

    assert isinstance(decision, AgentDecision)
    assert decision.action == 0
    assert decision.latency_ms == pytest.approx(12.5)

    monkeypatch.undo()

    # Wall-clock sanity check with a short real busy wait.
    timed_agent = _DeterministicTestAgent(seed=1, busy_wait_ms=4.0)
    real_decision = timed_agent.select_action(game)
    assert real_decision.latency_ms >= 3.0


def test_edge_cases_empty_near_full_and_terminal_boards() -> None:
    """Verify empty board, single-remaining-cell board, and terminal/contract rejections."""
    agent = _DeterministicTestAgent(seed=7)

    # 1. Empty board: all M*N cells must be scored and legal.
    empty_game = ExtendedTicTacToe(rows=4, cols=4, win_k=4)
    empty_decision = agent.select_action(empty_game)
    assert empty_decision.action in range(16)
    assert len(empty_decision.action_scores) == 16

    # 2. Near-full 3x3 board with only cell 8 left (draw sequence: 0,1,2,4,3,5,7,6,8).
    near_full_draw = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    for move in [0, 1, 2, 4, 3, 5, 7, 6]:
        near_full_draw.step(move)
    assert near_full_draw.legal_actions() == [8]
    last_decision = agent.select_action(near_full_draw)
    assert last_decision.action == 8
    assert last_decision.action_scores == {8: pytest.approx(1.0)}
    near_full_draw.step(last_decision.action)
    assert near_full_draw.is_terminal()

    # 3. Terminal draw board must raise IllegalMoveError.
    with pytest.raises(IllegalMoveError):
        agent.select_action(near_full_draw)

    # 4. Terminal win board must raise IllegalMoveError.
    win_game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    for move in [0, 3, 1, 4, 2]:
        win_game.step(move)
    assert win_game.is_terminal()
    with pytest.raises(IllegalMoveError):
        agent.select_action(win_game)


def test_contract_enforcement_rejects_bad_game_mutation_and_illegal_outputs() -> None:
    """Verify BaseAgent catches non-Engine inputs, source mutation, and illegal moves."""
    agent = _DeterministicTestAgent(seed=1)
    with pytest.raises(TypeError):
        agent.select_action("not_a_game")  # type: ignore[arg-type]

    game = ExtendedTicTacToe(rows=3, cols=3, win_k=3)
    game.step(4)

    # Mutating source game without undo() must raise IllegalMoveError.
    mutator = _MutatingAgent(name="Mutator", level=1)
    with pytest.raises(IllegalMoveError):
        mutator.select_action(game)

    # Returning an already occupied cell (4) or out-of-bounds cell (99) must raise.
    with pytest.raises(IllegalMoveError):
        _IllegalActionAgent(forced_action=4).select_action(game)
    with pytest.raises(IllegalMoveError):
        _IllegalActionAgent(forced_action=99).select_action(game)

    # Returning unsupported type from _select_action must raise TypeError.
    with pytest.raises(TypeError):
        _BadReturnTypeAgent(name="BadReturn", level=1).select_action(game)


def test_base_agent_module_has_no_print_statements() -> None:
    """Enforce RULES.md Section 4.4: agents modules must never call print()."""
    for filename in ("__init__.py", "base.py"):
        source = (AGENTS_DIR / filename).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", f"Forbidden print() found in agents/{filename}"
