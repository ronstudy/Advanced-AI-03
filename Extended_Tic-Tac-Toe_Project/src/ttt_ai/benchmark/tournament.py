"""Paired Round-Robin tournament runner, integrity verification, and official suite orchestration.

Implements Tasks T4.1 & T4.3 per ``Planning/ARCHITECTURE.md`` Section 5.3 and
``Planning/RULES.md`` Section 3:
- Balanced 50% ``X`` / 50% ``O`` paired scheduling on identical match seeds.
- Line-buffered JSONL output with a ``type="meta"`` environment/config header
  followed by per-match ``type="match"`` records containing full move telemetry.
- Reproducible seeding and resumable execution from partial JSONL logs.
- Independent move-by-move log replay verification (`verify_tournament_log_integrity`)
  and full T4.3 suite execution with automated CSV/Markdown reporting.
"""

from __future__ import annotations

import csv
import ctypes
import json
import logging
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ttt_ai.agents.base import BaseAgent, _is_finite_real, _is_strict_int
from ttt_ai.agents.hybrid_mcts_agent import HybridMCTSAgent
from ttt_ai.agents.mcts_agent import MCTSAgent
from ttt_ai.agents.minimax_agent import AlphaBetaAgent, MinimaxAgent
from ttt_ai.agents.policy_agent import PolicyNetworkAgent, ValueGreedyAgent
from ttt_ai.agents.random_agent import RandomAgent
from ttt_ai.agents.rl_agent import DQNAgent, TabularQAgent
from ttt_ai.agents.rule_based_agent import RuleBasedAgent
from ttt_ai.config import BoardConfig, load_yaml_dict, seed_everything
from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError
from ttt_ai.engine.game import ExtendedTicTacToe
from ttt_ai.engine.rules import PLAYER_X, Player
from ttt_ai.training.checkpoint_io import collect_hardware_info

logger = logging.getLogger(__name__)

_BOARD_PRESETS: dict[str, tuple[int, int, int]] = {
    "3x3_k3": (3, 3, 3),
    "4x4_k3": (4, 4, 3),
    "4x4_k4": (4, 4, 4),
    "5x5_k4": (5, 5, 4),
}

DEFAULT_OFFICIAL_TOURNAMENT_TARGETS: tuple[tuple[str, str], ...] = (
    ("4x4_k4", "fixed_10ms"),
    ("4x4_k4", "fixed_50ms"),
    ("4x4_k4", "default"),
    ("3x3_k3", "default"),
)


def parse_board_preset(board_tag: str) -> BoardConfig:
    """Convert a standardized CLI/config board tag into a validated :class:`BoardConfig`.

    Args:
        board_tag: Preset tag such as ``"4x4_k4"``, ``"4x4_k3"``, ``"3x3_k3"``, or ``"5x5_k4"``.

    Returns:
        Validated :class:`BoardConfig`.

    Raises:
        InvalidConfigError: If `board_tag` is unknown.
    """
    if not isinstance(board_tag, str):
        raise InvalidConfigError(
            f"board_tag must be a string, got {type(board_tag).__name__}."
        )
    key = board_tag.strip().lower()
    if key not in _BOARD_PRESETS:
        raise InvalidConfigError(
            f"Unknown board preset {board_tag!r}. Choose from {sorted(_BOARD_PRESETS.keys())}."
        )
    rows, cols, win_k = _BOARD_PRESETS[key]
    return BoardConfig(rows=rows, cols=cols, win_k=win_k)

_AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "RandomAgent": RandomAgent,
    "RuleBasedAgent": RuleBasedAgent,
    "MinimaxAgent": MinimaxAgent,
    "AlphaBetaAgent": AlphaBetaAgent,
    "MCTSAgent": MCTSAgent,
    "PolicyNetworkAgent": PolicyNetworkAgent,
    "ValueGreedyAgent": ValueGreedyAgent,
    "TabularQAgent": TabularQAgent,
    "DQNAgent": DQNAgent,
    "HybridMCTSAgent": HybridMCTSAgent,
}

_PATH_PARAM_KEYS: tuple[str, ...] = (
    "checkpoint_path",
    "policy_checkpoint_path",
    "value_checkpoint_path",
)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Serializable specification describing a tournament participant."""

    agent_id: str
    agent_class: str
    level: int
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate participant identifier, class name, level, and parameter dict."""
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise InvalidConfigError(
                f"AgentSpec.agent_id must be a non-empty string, got {self.agent_id!r}."
            )
        object.__setattr__(self, "agent_id", self.agent_id.strip())

        if not isinstance(self.agent_class, str) or self.agent_class not in _AGENT_REGISTRY:
            raise InvalidConfigError(
                f"Unknown agent_class {self.agent_class!r}. "
                f"Supported classes: {sorted(_AGENT_REGISTRY.keys())}."
            )
        if not _is_strict_int(self.level) or not (0 <= int(self.level) <= 7):
            raise InvalidConfigError(
                f"AgentSpec.level must be an integer in 0..7, got {self.level!r}."
            )
        object.__setattr__(self, "level", int(self.level))

        if not isinstance(self.params, Mapping):
            raise InvalidConfigError(
                f"AgentSpec.params must be a mapping, got {type(self.params).__name__}."
            )
        object.__setattr__(self, "params", dict(self.params))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentSpec:
        """Construct an :class:`AgentSpec` from parsed YAML/JSON mapping."""
        if not isinstance(data, Mapping):
            raise InvalidConfigError(
                f"AgentSpec entry must be a mapping, got {type(data).__name__}."
            )
        for key in ("agent_id", "agent_class", "level"):
            if key not in data:
                raise InvalidConfigError(f"AgentSpec entry missing required key '{key}'.")
        return cls(
            agent_id=data["agent_id"],
            agent_class=data["agent_class"],
            level=data["level"],
            params=data.get("params") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize specification to a plain JSON-safe dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_class": self.agent_class,
            "level": self.level,
            "params": dict(self.params),
        }


@dataclass(frozen=True, slots=True)
class BudgetModeConfig:
    """Compute budget override applied to tournament agents for a given mode."""

    name: str
    time_budget_ms: float | None = None
    n_simulations: int | None = None

    def __post_init__(self) -> None:
        """Validate mode identifier and optional budget constraints."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidConfigError(
                f"BudgetModeConfig.name must be non-empty, got {self.name!r}."
            )
        object.__setattr__(self, "name", self.name.strip())

        if self.time_budget_ms is not None:
            if (
                not _is_finite_real(self.time_budget_ms)
                or float(self.time_budget_ms) <= 0.0
            ):
                raise InvalidConfigError(
                    "time_budget_ms must be finite > 0.0 or None, "
                    f"got {self.time_budget_ms!r}."
                )
            object.__setattr__(self, "time_budget_ms", float(self.time_budget_ms))

        if self.n_simulations is not None:
            if not _is_strict_int(self.n_simulations) or int(self.n_simulations) < 1:
                raise InvalidConfigError(
                    "n_simulations must be an integer >= 1 or None, "
                    f"got {self.n_simulations!r}."
                )
            object.__setattr__(self, "n_simulations", int(self.n_simulations))

    def to_dict(self) -> dict[str, Any]:
        """Serialize budget mode configuration to JSON-compatible dictionary."""
        return {
            "name": self.name,
            "time_budget_ms": self.time_budget_ms,
            "n_simulations": self.n_simulations,
        }


@dataclass(frozen=True, slots=True)
class MoveRecord:
    """Compact per-move telemetry recorded inside a :class:`MatchRecord`."""

    action: int
    latency_ms: float
    nodes_visited: int = 0
    simulations: int = 0
    depth_reached: int = 0

    def to_compact_list(self) -> list[int | float]:
        """Serialize move telemetry as `[action, latency_ms, nodes, sims, depth]`."""
        return [
            int(self.action),
            round(float(self.latency_ms), 4),
            int(self.nodes_visited),
            int(self.simulations),
            int(self.depth_reached),
        ]

    @classmethod
    def from_compact_list(cls, raw: Sequence[int | float]) -> MoveRecord:
        """Reconstruct a :class:`MoveRecord` from a 5-element JSON list."""
        if len(raw) != 5:
            raise InvalidConfigError(
                f"Compact move record must have 5 elements, got {len(raw)}."
            )
        return cls(
            action=int(raw[0]),
            latency_ms=float(raw[1]),
            nodes_visited=int(raw[2]),
            simulations=int(raw[3]),
            depth_reached=int(raw[4]),
        )


@dataclass(slots=True)
class MatchRecord:
    """Complete per-match record stored as one JSONL line in ``logs/*.jsonl``."""

    match_id: str
    pair_id: int
    game_in_pair: int
    seed: int
    board_config: dict[str, int]
    budget_mode: str
    agent_x: str
    agent_o: str
    winner: int
    score_x: float
    score_o: float
    num_moves: int
    latencies_x_ms: list[float]
    latencies_o_ms: list[float]
    nodes_x: list[int]
    nodes_o: list[int]
    sims_x: list[int]
    sims_o: list[int]
    winning_line: list[list[int]] | None
    moves: list[list[int | float]]
    type: str = "match"

    def to_dict(self) -> dict[str, Any]:
        """Serialize match record to a JSON-compatible dictionary."""
        return {
            "type": self.type,
            "match_id": self.match_id,
            "pair_id": self.pair_id,
            "game_in_pair": self.game_in_pair,
            "seed": self.seed,
            "board_config": dict(self.board_config),
            "budget_mode": self.budget_mode,
            "agent_x": self.agent_x,
            "agent_o": self.agent_o,
            "winner": self.winner,
            "score_x": self.score_x,
            "score_o": self.score_o,
            "num_moves": self.num_moves,
            "latencies_x_ms": list(self.latencies_x_ms),
            "latencies_o_ms": list(self.latencies_o_ms),
            "nodes_x": list(self.nodes_x),
            "nodes_o": list(self.nodes_o),
            "sims_x": list(self.sims_x),
            "sims_o": list(self.sims_o),
            "winning_line": (
                [list(coord) for coord in self.winning_line]
                if self.winning_line is not None
                else None
            ),
            "moves": [list(mv) for mv in self.moves],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MatchRecord:
        """Deserialize a JSONL match record dictionary into :class:`MatchRecord`."""
        return cls(
            type=str(data.get("type", "match")),
            match_id=str(data["match_id"]),
            pair_id=int(data["pair_id"]),
            game_in_pair=int(data["game_in_pair"]),
            seed=int(data["seed"]),
            board_config={str(k): int(v) for k, v in data["board_config"].items()},
            budget_mode=str(data["budget_mode"]),
            agent_x=str(data["agent_x"]),
            agent_o=str(data["agent_o"]),
            winner=int(data["winner"]),
            score_x=float(data["score_x"]),
            score_o=float(data["score_o"]),
            num_moves=int(data["num_moves"]),
            latencies_x_ms=[float(v) for v in data["latencies_x_ms"]],
            latencies_o_ms=[float(v) for v in data["latencies_o_ms"]],
            nodes_x=[int(v) for v in data["nodes_x"]],
            nodes_o=[int(v) for v in data["nodes_o"]],
            sims_x=[int(v) for v in data["sims_x"]],
            sims_o=[int(v) for v in data["sims_o"]],
            winning_line=(
                [[int(c) for c in coord] for coord in data["winning_line"]]
                if data.get("winning_line") is not None
                else None
            ),
            moves=[
                [int(mv[0]), float(mv[1]), int(mv[2]), int(mv[3]), int(mv[4])]
                for mv in data["moves"]
            ],
        )


@dataclass(frozen=True, slots=True)
class TournamentConfig:
    """Validated tournament configuration loaded from ``configs/tournament.yaml``."""

    name: str
    board: BoardConfig
    seed: int
    n_pairs: int
    budget_modes: dict[str, BudgetModeConfig]
    agents: tuple[AgentSpec, ...]
    agents_3x3_k3: tuple[AgentSpec, ...] = ()

    def __post_init__(self) -> None:
        """Validate tournament settings and participant uniqueness."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidConfigError(
                f"TournamentConfig.name must be non-empty, got {self.name!r}."
            )
        if not isinstance(self.board, BoardConfig):
            raise InvalidConfigError(
                f"TournamentConfig.board must be a BoardConfig, got {type(self.board).__name__}."
            )
        if not _is_strict_int(self.seed) or int(self.seed) < 0:
            raise InvalidConfigError(f"seed must be >= 0, got {self.seed!r}.")
        if not _is_strict_int(self.n_pairs) or int(self.n_pairs) < 1:
            raise InvalidConfigError(f"n_pairs must be >= 1, got {self.n_pairs!r}.")
        if len(self.agents) < 2:
            raise InvalidConfigError("TournamentConfig requires at least 2 agents.")

        _validate_unique_agent_ids(self.agents, "agents")
        if self.agents_3x3_k3:
            _validate_unique_agent_ids(self.agents_3x3_k3, "agents_3x3_k3")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TournamentConfig:
        """Construct a :class:`TournamentConfig` from parsed YAML data."""
        if not isinstance(data, Mapping):
            raise InvalidConfigError(
                f"Tournament config root must be a mapping, got {type(data).__name__}."
            )
        for key in ("name", "board", "agents"):
            if key not in data:
                raise InvalidConfigError(f"Tournament config missing required key '{key}'.")

        board = BoardConfig.from_dict(data["board"])
        raw_modes = data.get("budget_modes") or {"default": {}}
        if not isinstance(raw_modes, Mapping):
            raise InvalidConfigError("budget_modes must be a mapping.")

        budget_modes: dict[str, BudgetModeConfig] = {}
        for mode_name, mode_cfg in raw_modes.items():
            cfg_map = mode_cfg if isinstance(mode_cfg, Mapping) else {}
            budget_modes[str(mode_name)] = BudgetModeConfig(
                name=str(mode_name),
                time_budget_ms=cfg_map.get("time_budget_ms"),
                n_simulations=cfg_map.get("n_simulations"),
            )

        agents = tuple(AgentSpec.from_dict(item) for item in data["agents"])
        raw_3x3 = data.get("agents_3x3_k3") or ()
        agents_3x3 = tuple(AgentSpec.from_dict(item) for item in raw_3x3)

        return cls(
            name=str(data["name"]),
            board=board,
            seed=int(data.get("seed", 42)),
            n_pairs=int(data.get("n_pairs", 100)),
            budget_modes=budget_modes,
            agents=agents,
            agents_3x3_k3=agents_3x3,
        )

    def resolve_agents_for_board(self, board_config: BoardConfig) -> tuple[AgentSpec, ...]:
        """Return the configured agent roster appropriate for `board_config`."""
        if board_config == BoardConfig(3, 3, 3) and self.agents_3x3_k3:
            return self.agents_3x3_k3
        return self.agents


def _validate_unique_agent_ids(specs: Sequence[AgentSpec], field_name: str) -> None:
    """Verify all `agent_id` strings within a roster are distinct."""
    seen: set[str] = set()
    for spec in specs:
        if spec.agent_id in seen:
            raise InvalidConfigError(
                f"Duplicate agent_id {spec.agent_id!r} in {field_name}."
            )
        seen.add(spec.agent_id)


def load_tournament_config(path: str | Path) -> TournamentConfig:
    """Load and validate a :class:`TournamentConfig` from a YAML file.

    Args:
        path: Path to ``configs/tournament.yaml``.

    Returns:
        Validated :class:`TournamentConfig`.
    """
    raw = load_yaml_dict(path)
    return TournamentConfig.from_dict(raw)


def _query_total_ram_gb() -> float | None:
    """Detect total physical RAM in gigabytes cleanly without external dependencies."""
    try:
        if platform.system() == "Windows":
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
                ctypes.byref(stat)
            )
            if ok:
                return round(stat.ullTotalPhys / (1024.0**3), 2)
        elif hasattr(os, "sysconf"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            return round((page_size * phys_pages) / (1024.0**3), 2)
    except Exception:
        return None
    return None


def _read_git_commit_hash(start_dir: str | Path | None = None) -> str | None:
    """Read current git commit hash from `.git/HEAD` without spawning subprocesses."""
    origin = Path(start_dir).resolve() if start_dir is not None else Path.cwd().resolve()
    for candidate in (origin, *origin.parents):
        git_dir = candidate / ".git"
        head_file = git_dir / "HEAD"
        if not head_file.is_file():
            continue
        try:
            head_text = head_file.read_text(encoding="utf-8").strip()
            if head_text.startswith("ref:"):
                ref_rel = head_text.split(":", 1)[1].strip()
                ref_file = git_dir / ref_rel
                if ref_file.is_file():
                    return ref_file.read_text(encoding="utf-8").strip()
                packed_refs = git_dir / "packed-refs"
                if packed_refs.is_file():
                    for line in packed_refs.read_text(encoding="utf-8").splitlines():
                        if not line or line.startswith(("#", "^")):
                            continue
                        parts = line.split()
                        if len(parts) == 2 and parts[1] == ref_rel:
                            return parts[0]
            elif len(head_text) >= 7:
                return head_text
        except OSError:
            return None
    return None


def collect_tournament_environment_meta(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Collect hardware, runtime library, RAM, and git metadata for tournament logs."""
    info = collect_hardware_info()
    info["ram_total_gb"] = _query_total_ram_gb()
    info["git_commit"] = _read_git_commit_hash(project_root)
    info["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    return info


def _resolve_param_path(raw_path: str | Path, base_dir: Path | None) -> str:
    """Resolve relative checkpoint paths against `base_dir` when needed."""
    path_obj = Path(raw_path)
    if path_obj.is_absolute() or path_obj.exists() or base_dir is None:
        return str(path_obj)
    candidate = base_dir / path_obj
    return str(candidate) if candidate.exists() else str(path_obj)


def instantiate_agent(
    spec: AgentSpec,
    board_config: BoardConfig,
    *,
    seed: int,
    time_budget_override_ms: float | None = None,
    n_simulations_override: int | None = None,
    base_dir: str | Path | None = None,
) -> BaseAgent:
    """Instantiate a Level 1-7 :class:`BaseAgent` from an :class:`AgentSpec`.

    Args:
        spec: Agent specification defining class, id, and hyperparameters.
        board_config: Target board dimensions passed to verified NN/RL agents.
        seed: Initial RNG seed assigned to the constructed agent.
        time_budget_override_ms: Optional compute budget overriding ``spec.params``.
        n_simulations_override: Optional MCTS simulation count override.
        base_dir: Optional base directory used to resolve relative checkpoint paths.

    Returns:
        Constructed :class:`BaseAgent` ready for tournament matches.
    """
    cls = _AGENT_REGISTRY[spec.agent_class]
    kwargs: dict[str, Any] = dict(spec.params)
    resolved_base = Path(base_dir) if base_dir is not None else None

    for key in _PATH_PARAM_KEYS:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = _resolve_param_path(kwargs[key], resolved_base)

    kwargs["name"] = spec.agent_id
    kwargs["seed"] = seed

    if time_budget_override_ms is not None:
        kwargs["time_budget_ms"] = float(time_budget_override_ms)

    if (
        n_simulations_override is not None
        and spec.agent_class in ("MCTSAgent", "HybridMCTSAgent")
    ):
        kwargs["n_simulations"] = int(n_simulations_override)

    if spec.agent_class in (
        "PolicyNetworkAgent",
        "ValueGreedyAgent",
        "TabularQAgent",
        "DQNAgent",
        "HybridMCTSAgent",
    ):
        kwargs.setdefault("board_config", board_config)

    return cls(**kwargs)


def play_single_match(
    game: ExtendedTicTacToe,
    agent_x: BaseAgent,
    agent_o: BaseAgent,
    *,
    match_id: str,
    pair_id: int,
    game_in_pair: int,
    seed: int,
    budget_mode: str = "default",
    agent_x_id: str | None = None,
    agent_o_id: str | None = None,
) -> MatchRecord:
    """Execute one complete match between `agent_x` and `agent_o` and capture telemetry.

    Args:
        game: Reusable :class:`ExtendedTicTacToe` environment (reset before play).
        agent_x: Agent playing ``Player.X`` (first mover).
        agent_o: Agent playing ``Player.O`` (second mover).
        match_id: Unique match identifier stored in the JSONL record.
        pair_id: Zero-based paired matchup index (`0..n_pairs - 1`).
        game_in_pair: ``0`` for first game of pair, ``1`` for color-swapped second game.
        seed: Base match seed; derives independent deterministic seeds for `X` and `O`.
        budget_mode: Budget mode label stored in the output record.
        agent_x_id: Optional identifier override for `agent_x`.
        agent_o_id: Optional identifier override for `agent_o`.

    Returns:
        Populated :class:`MatchRecord` containing outcome, payoffs, and move telemetry.
    """
    game.reset()
    # Derive disjoint deterministic seeds per seat so two stochastic agents do not
    # draw identical random streams within the same match while staying 100% repeatable.
    agent_x.reset(seed=seed * 2 + 1)
    agent_o.reset(seed=seed * 2 + 2)

    latencies_x: list[float] = []
    latencies_o: list[float] = []
    nodes_x: list[int] = []
    nodes_o: list[int] = []
    sims_x: list[int] = []
    sims_o: list[int] = []
    compact_moves: list[list[int | float]] = []

    while not game.is_terminal():
        acting_player = game.current_player
        actor = agent_x if acting_player is Player.X else agent_o
        decision = actor.select_action(game)
        game.step(decision.action)

        lat_rounded = round(float(decision.latency_ms), 4)
        move_rec = MoveRecord(
            action=decision.action,
            latency_ms=decision.latency_ms,
            nodes_visited=decision.nodes_visited,
            simulations=decision.simulations,
            depth_reached=decision.depth_reached,
        )
        compact_moves.append(move_rec.to_compact_list())

        if acting_player is Player.X:
            latencies_x.append(lat_rounded)
            nodes_x.append(decision.nodes_visited)
            sims_x.append(decision.simulations)
        else:
            latencies_o.append(lat_rounded)
            nodes_o.append(decision.nodes_visited)
            sims_o.append(decision.simulations)

    winner = int(game.winner) if game.winner is not None else 0
    if winner == PLAYER_X:
        score_x, score_o = 1.0, 0.0
    elif winner == 0:
        score_x, score_o = 0.5, 0.5
    else:
        score_x, score_o = 0.0, 1.0

    winning_line = (
        [[int(r), int(c)] for r, c in game.winning_line]
        if game.winning_line is not None
        else None
    )

    return MatchRecord(
        match_id=match_id,
        pair_id=pair_id,
        game_in_pair=game_in_pair,
        seed=seed,
        board_config={
            "rows": game.rows,
            "cols": game.cols,
            "win_k": game.win_k,
        },
        budget_mode=budget_mode,
        agent_x=agent_x_id or agent_x.name,
        agent_o=agent_o_id or agent_o.name,
        winner=winner,
        score_x=score_x,
        score_o=score_o,
        num_moves=game.move_count,
        latencies_x_ms=latencies_x,
        latencies_o_ms=latencies_o,
        nodes_x=nodes_x,
        nodes_o=nodes_o,
        sims_x=sims_x,
        sims_o=sims_o,
        winning_line=winning_line,
        moves=compact_moves,
    )


def load_tournament_log(log_path: str | Path) -> tuple[dict[str, Any], list[MatchRecord]]:
    """Load and validate a tournament JSONL log file into `(meta_header, matches)`.

    Args:
        log_path: Path to a ``.jsonl`` file produced by :class:`TournamentRunner`.

    Returns:
        Tuple ``(meta_dict, match_records)``.

    Raises:
        FileNotFoundError: If `log_path` does not exist.
        InvalidConfigError: If the file is empty, lacks a ``type="meta"`` header,
            or contains malformed JSON lines.
    """
    path = Path(log_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tournament log file not found: {path}")

    meta_header: dict[str, Any] | None = None
    records: list[MatchRecord] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise InvalidConfigError(
                    f"Malformed JSON on line {line_num} of '{path}': {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise InvalidConfigError(
                    f"JSONL line {line_num} of '{path}' must be a JSON object."
                )
            rec_type = obj.get("type")
            if line_num == 1 or meta_header is None:
                if rec_type != "meta":
                    raise InvalidConfigError(
                        f"First record in '{path}' must have type='meta', got {rec_type!r}."
                    )
                meta_header = obj
            elif rec_type == "match":
                records.append(MatchRecord.from_dict(obj))
            else:
                raise InvalidConfigError(
                    f"Unexpected record type {rec_type!r} on line {line_num} of '{path}'."
                )

    if meta_header is None:
        raise InvalidConfigError(f"Tournament log file '{path}' is empty.")
    return meta_header, records


class TournamentRunner:
    """Executes Paired Round-Robin tournaments across AI rosters with JSONL logging."""

    def __init__(
        self,
        board_config: BoardConfig,
        agent_specs: Sequence[AgentSpec],
        *,
        n_pairs: int = 100,
        base_seed: int = 42,
        budget_mode: str = "default",
        time_budget_override_ms: float | None = None,
        n_simulations_override: int | None = None,
        base_dir: str | Path | None = None,
        tournament_name: str = "tournament",
    ) -> None:
        """Initialize the Paired Round-Robin tournament runner.

        Args:
            board_config: Validated :class:`BoardConfig` for all matches.
            agent_specs: Sequence of at least 2 distinct :class:`AgentSpec` items.
            n_pairs: Number of paired X/O seed rounds per unordered matchup
                (total games per matchup = ``2 * n_pairs``).
            base_seed: Starting integer seed for pair ``0`` (`seed = base_seed + pair_id`).
            budget_mode: Descriptive budget mode label (e.g. ``"default"``, ``"fixed_50ms"``).
            time_budget_override_ms: Optional compute budget overriding all agents.
            n_simulations_override: Optional MCTS simulation count override.
            base_dir: Base directory used to resolve relative checkpoint paths.
            tournament_name: Logical tournament identifier written to log metadata.

        Raises:
            InvalidConfigError: If any argument fails validation or duplicate IDs exist.
        """
        if not isinstance(board_config, BoardConfig):
            raise InvalidConfigError(
                f"board_config must be a BoardConfig, got {type(board_config).__name__}."
            )
        specs = tuple(agent_specs)
        if len(specs) < 2:
            raise InvalidConfigError("TournamentRunner requires at least 2 agents.")
        _validate_unique_agent_ids(specs, "agent_specs")

        if not _is_strict_int(n_pairs) or int(n_pairs) < 1:
            raise InvalidConfigError(f"n_pairs must be >= 1, got {n_pairs!r}.")
        if not _is_strict_int(base_seed) or int(base_seed) < 0:
            raise InvalidConfigError(f"base_seed must be >= 0, got {base_seed!r}.")
        if not isinstance(budget_mode, str) or not budget_mode.strip():
            raise InvalidConfigError(
                f"budget_mode must be a non-empty string, got {budget_mode!r}."
            )
        if time_budget_override_ms is not None and (
            not _is_finite_real(time_budget_override_ms)
            or float(time_budget_override_ms) <= 0.0
        ):
            raise InvalidConfigError(
                "time_budget_override_ms must be finite > 0.0 or None, "
                f"got {time_budget_override_ms!r}."
            )
        if n_simulations_override is not None and (
            not _is_strict_int(n_simulations_override) or int(n_simulations_override) < 1
        ):
            raise InvalidConfigError(
                "n_simulations_override must be >= 1 or None, "
                f"got {n_simulations_override!r}."
            )

        self._board_config: BoardConfig = board_config
        self._agent_specs: tuple[AgentSpec, ...] = specs
        self._n_pairs: int = int(n_pairs)
        self._base_seed: int = int(base_seed)
        self._budget_mode: str = budget_mode.strip()
        self._time_budget_override_ms: float | None = (
            float(time_budget_override_ms) if time_budget_override_ms is not None else None
        )
        self._n_simulations_override: int | None = (
            int(n_simulations_override) if n_simulations_override is not None else None
        )
        self._base_dir: Path | None = Path(base_dir) if base_dir is not None else None
        self._tournament_name: str = tournament_name.strip() or "tournament"

    @property
    def total_matches(self) -> int:
        """Total number of individual games scheduled across all unordered pairs."""
        num_agents = len(self._agent_specs)
        num_pairs_of_agents = (num_agents * (num_agents - 1)) // 2
        return num_pairs_of_agents * self._n_pairs * 2

    def _build_agents(self) -> dict[str, BaseAgent]:
        """Instantiate and cache one :class:`BaseAgent` per participant spec."""
        roster: dict[str, BaseAgent] = {}
        for spec in self._agent_specs:
            roster[spec.agent_id] = instantiate_agent(
                spec,
                self._board_config,
                seed=self._base_seed,
                time_budget_override_ms=self._time_budget_override_ms,
                n_simulations_override=self._n_simulations_override,
                base_dir=self._base_dir,
            )
        return roster

    def _build_meta_header(self, roster: Mapping[str, BaseAgent]) -> dict[str, Any]:
        """Build the line-1 ``type="meta"`` dictionary for the JSONL tournament log."""
        return {
            "type": "meta",
            "tournament_name": self._tournament_name,
            "board_config": self._board_config.to_dict(),
            "budget_mode": self._budget_mode,
            "time_budget_override_ms": self._time_budget_override_ms,
            "n_simulations_override": self._n_simulations_override,
            "base_seed": self._base_seed,
            "n_pairs": self._n_pairs,
            "games_per_matchup": self._n_pairs * 2,
            "total_scheduled_matches": self.total_matches,
            "agents": {
                spec.agent_id: roster[spec.agent_id].get_config()
                for spec in self._agent_specs
            },
            "hardware": collect_tournament_environment_meta(self._base_dir),
        }

    def _iter_scheduled_matches(self) -> list[tuple[str, str, str, int, int, int]]:
        """Generate deterministic Paired Round-Robin schedule across unordered pairs."""
        schedule: list[tuple[str, str, str, int, int, int]] = []
        num_agents = len(self._agent_specs)
        for i in range(num_agents):
            spec_a = self._agent_specs[i]
            for j in range(i + 1, num_agents):
                spec_b = self._agent_specs[j]
                for pair_id in range(self._n_pairs):
                    pair_seed = self._base_seed + pair_id
                    # Game 0: A plays X (first), B plays O (second).
                    match_id_0 = (
                        f"{spec_a.agent_id}__vs__{spec_b.agent_id}"
                        f"__p{pair_id:04d}__g0"
                    )
                    schedule.append(
                        (match_id_0, spec_a.agent_id, spec_b.agent_id, pair_id, 0, pair_seed)
                    )
                    # Game 1: B plays X (first), A plays O (second) on exact same seed.
                    match_id_1 = (
                        f"{spec_a.agent_id}__vs__{spec_b.agent_id}"
                        f"__p{pair_id:04d}__g1"
                    )
                    schedule.append(
                        (match_id_1, spec_b.agent_id, spec_a.agent_id, pair_id, 1, pair_seed)
                    )
        return schedule

    def run(
        self,
        output_path: str | Path,
        *,
        resume: bool = True,
    ) -> Path:
        """Execute the tournament and stream results to `output_path` as JSONL.

        Args:
            output_path: Destination ``.jsonl`` file path.
            resume: When ``True`` and `output_path` already exists with a valid
                matching ``meta`` header, skips already logged ``match_id`` entries
                and appends only remaining matches.

        Returns:
            Resolved :class:`pathlib.Path` to the completed JSONL log file.

        Raises:
            InvalidConfigError: If resuming against a log whose `board_config` or
                `budget_mode` does not match this runner.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        roster = self._build_agents()
        completed_ids: set[str] = set()
        file_mode = "w"

        if resume and out_path.is_file() and out_path.stat().st_size > 0:
            existing_meta, existing_matches = load_tournament_log(out_path)
            if existing_meta.get("board_config") != self._board_config.to_dict():
                raise InvalidConfigError(
                    f"Cannot resume '{out_path}': existing board_config "
                    f"{existing_meta.get('board_config')} mismatches "
                    f"{self._board_config.to_dict()}."
                )
            if existing_meta.get("budget_mode") != self._budget_mode:
                raise InvalidConfigError(
                    f"Cannot resume '{out_path}': existing budget_mode "
                    f"{existing_meta.get('budget_mode')!r} mismatches {self._budget_mode!r}."
                )
            completed_ids = {rec.match_id for rec in existing_matches}
            file_mode = "a"
            logger.info(
                "Resuming tournament '%s' from '%s': %d/%d matches already recorded.",
                self._tournament_name,
                out_path,
                len(completed_ids),
                self.total_matches,
            )

        game = ExtendedTicTacToe(
            rows=self._board_config.rows,
            cols=self._board_config.cols,
            win_k=self._board_config.win_k,
        )
        schedule = self._iter_scheduled_matches()

        # Restrict PyTorch to 1 intra-op thread during tournament evaluation to eliminate
        # multi-thread barrier overhead on tiny (1, 1, M, N) tensors and keep latency clean.
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with out_path.open(file_mode, encoding="utf-8") as handle:
                if file_mode == "w":
                    meta_header = self._build_meta_header(roster)
                    handle.write(json.dumps(meta_header) + "\n")
                    handle.flush()

                executed_now = 0
                for match_id, id_x, id_o, pair_id, game_in_pair, match_seed in schedule:
                    if match_id in completed_ids:
                        continue
                    record = play_single_match(
                        game=game,
                        agent_x=roster[id_x],
                        agent_o=roster[id_o],
                        match_id=match_id,
                        pair_id=pair_id,
                        game_in_pair=game_in_pair,
                        seed=match_seed,
                        budget_mode=self._budget_mode,
                        agent_x_id=id_x,
                        agent_o_id=id_o,
                    )
                    handle.write(json.dumps(record.to_dict()) + "\n")
                    handle.flush()
                    executed_now += 1
        finally:
            torch.set_num_threads(previous_threads)

        logger.info(
            "Tournament '%s' complete: wrote %d new matches (total scheduled=%d) -> %s",
            self._tournament_name,
            executed_now,
            self.total_matches,
            out_path,
        )
        return out_path


@dataclass(frozen=True, slots=True)
class TournamentIntegrityReport:
    """Audit report verifying completeness, 50/50 X/O balance, and move legality."""

    log_path: str
    tournament_name: str
    board_config: dict[str, int]
    budget_mode: str
    participants: list[str]
    n_pairs: int
    total_matches: int
    expected_matches: int
    duplicate_match_ids: list[str]
    unbalanced_matchups: list[str]
    corrupt_or_illegal_matches: list[str]
    is_valid: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize integrity audit results to a JSON-compatible dictionary."""
        return {
            "log_path": self.log_path,
            "tournament_name": self.tournament_name,
            "board_config": dict(self.board_config),
            "budget_mode": self.budget_mode,
            "participants": list(self.participants),
            "n_pairs": self.n_pairs,
            "total_matches": self.total_matches,
            "expected_matches": self.expected_matches,
            "duplicate_match_ids": list(self.duplicate_match_ids),
            "unbalanced_matchups": list(self.unbalanced_matchups),
            "corrupt_or_illegal_matches": list(self.corrupt_or_illegal_matches),
            "is_valid": self.is_valid,
        }


def _replay_match_record(game: ExtendedTicTacToe, rec: MatchRecord) -> str | None:
    """Replay all logged moves on `game` and return an error reason if inconsistent."""
    if rec.num_moves != len(rec.moves):
        return f"{rec.match_id}: num_moves={rec.num_moves} != len(moves)={len(rec.moves)}"

    expected_x_moves = (len(rec.moves) + 1) // 2
    expected_o_moves = len(rec.moves) // 2
    if (
        len(rec.latencies_x_ms) != expected_x_moves
        or len(rec.latencies_o_ms) != expected_o_moves
    ):
        return (
            f"{rec.match_id}: seat latency lengths ({len(rec.latencies_x_ms)}, "
            f"{len(rec.latencies_o_ms)}) mismatch move counts ({expected_x_moves}, "
            f"{expected_o_moves})"
        )

    game.reset()
    for step_idx, move_entry in enumerate(rec.moves):
        action = int(move_entry[0])
        try:
            game.step(action)
        except IllegalMoveError as exc:
            return f"{rec.match_id}: IllegalMoveError at ply {step_idx}: {exc}"

    if not game.is_terminal():
        return f"{rec.match_id}: game did not reach terminal state after {len(rec.moves)} moves"

    actual_winner = int(game.winner) if game.winner is not None else 0
    if actual_winner != rec.winner:
        return f"{rec.match_id}: winner={rec.winner} mismatches replayed winner={actual_winner}"

    exp_sx = 1.0 if actual_winner == PLAYER_X else (0.5 if actual_winner == 0 else 0.0)
    exp_so = 1.0 - exp_sx
    if abs(rec.score_x - exp_sx) > 1e-6 or abs(rec.score_o - exp_so) > 1e-6:
        return (
            f"{rec.match_id}: scores ({rec.score_x}, {rec.score_o}) mismatch "
            f"expected ({exp_sx}, {exp_so})"
        )
    return None


def verify_tournament_log_integrity(
    log_path: str | Path,
    *,
    expected_agents: Sequence[str] | None = None,
    expected_n_pairs: int | None = None,
    replay_moves: bool = True,
) -> TournamentIntegrityReport:
    """Verify a tournament JSONL log for completeness, 50/50 balance, and zero illegal moves.

    Args:
        log_path: Path to the tournament ``.jsonl`` file.
        expected_agents: Optional expected agent identifier roster.
        expected_n_pairs: Optional expected number of paired seed rounds per matchup.
        replay_moves: When ``True``, replays every move of every match through
            :class:`ExtendedTicTacToe` to prove zero ``IllegalMoveError`` and exact
            winner/score consistency.

    Returns:
        :class:`TournamentIntegrityReport` summarizing all checks.

    Raises:
        FileNotFoundError: If `log_path` does not exist.
        InvalidConfigError: If the JSONL header lacks valid board metadata.
    """
    meta_header, records = load_tournament_log(log_path)
    raw_board = meta_header.get("board_config")
    if not isinstance(raw_board, Mapping):
        raise InvalidConfigError(f"Tournament log '{log_path}' meta lacks valid board_config.")
    board_cfg = BoardConfig.from_dict(raw_board)
    board_dict = board_cfg.to_dict()
    budget_mode = str(meta_header.get("budget_mode", ""))
    tournament_name = str(meta_header.get("tournament_name", ""))

    raw_agents = meta_header.get("agents")
    meta_agents = list(raw_agents.keys()) if isinstance(raw_agents, Mapping) else []
    record_agents = sorted({rec.agent_x for rec in records} | {rec.agent_o for rec in records})
    participants = meta_agents if meta_agents else record_agents

    resolved_n_pairs = (
        int(expected_n_pairs)
        if expected_n_pairs is not None
        else int(meta_header.get("n_pairs", 0))
    )
    num_agents = len(participants)
    expected_matches = (
        num_agents * (num_agents - 1) * resolved_n_pairs
        if (num_agents >= 2 and resolved_n_pairs >= 1)
        else 0
    )

    seen_ids: set[str] = set()
    duplicate_match_ids: list[str] = []
    unbalanced_matchups: list[str] = []
    corrupt_or_illegal_matches: list[str] = []

    if expected_agents is not None and set(participants) != set(expected_agents):
        unbalanced_matchups.append(
            f"participant_mismatch: expected={sorted(expected_agents)}, "
            f"actual={sorted(participants)}"
        )

    directed_pairs: dict[tuple[str, str], dict[int, int]] = {}
    replay_game = ExtendedTicTacToe(
        rows=board_cfg.rows,
        cols=board_cfg.cols,
        win_k=board_cfg.win_k,
    )

    for rec in records:
        if rec.match_id in seen_ids:
            duplicate_match_ids.append(rec.match_id)
        seen_ids.add(rec.match_id)

        if rec.board_config != board_dict or rec.budget_mode != budget_mode:
            corrupt_or_illegal_matches.append(
                f"{rec.match_id}: metadata mismatch (board={rec.board_config}, "
                f"mode={rec.budget_mode!r})"
            )

        directed_pairs.setdefault((rec.agent_x, rec.agent_o), {})[rec.pair_id] = rec.seed

        if replay_moves:
            replay_err = _replay_match_record(replay_game, rec)
            if replay_err is not None:
                corrupt_or_illegal_matches.append(replay_err)

    for idx_a in range(num_agents):
        agent_a = participants[idx_a]
        for idx_b in range(idx_a + 1, num_agents):
            agent_b = participants[idx_b]
            ab_seeds = directed_pairs.get((agent_a, agent_b), {})
            ba_seeds = directed_pairs.get((agent_b, agent_a), {})
            if len(ab_seeds) != resolved_n_pairs or len(ba_seeds) != resolved_n_pairs:
                unbalanced_matchups.append(
                    f"{agent_a} vs {agent_b}: counts=({len(ab_seeds)}, {len(ba_seeds)}), "
                    f"expected=({resolved_n_pairs}, {resolved_n_pairs})"
                )
            elif ab_seeds != ba_seeds:
                unbalanced_matchups.append(
                    f"{agent_a} vs {agent_b}: paired X/O seeds or pair_ids do not align"
                )

    is_valid = (
        expected_matches > 0
        and len(records) == expected_matches
        and not duplicate_match_ids
        and not unbalanced_matchups
        and not corrupt_or_illegal_matches
    )
    return TournamentIntegrityReport(
        log_path=str(Path(log_path)),
        tournament_name=tournament_name,
        board_config=board_dict,
        budget_mode=budget_mode,
        participants=participants,
        n_pairs=resolved_n_pairs,
        total_matches=len(records),
        expected_matches=expected_matches,
        duplicate_match_ids=duplicate_match_ids,
        unbalanced_matchups=unbalanced_matchups,
        corrupt_or_illegal_matches=corrupt_or_illegal_matches,
        is_valid=is_valid,
    )


@dataclass(frozen=True, slots=True)
class OfficialTournamentSuiteResult:
    """Outputs and verification status produced by :func:`run_official_tournament_suite`."""

    log_paths: dict[str, Path]
    integrity_reports: dict[str, TournamentIntegrityReport]
    summaries: dict[str, Any]
    suite_csv_path: Path | None
    suite_markdown_path: Path | None


def _export_suite_consolidated_csv(
    ordered_runs: Sequence[tuple[str, str, Any]],
    output_csv_path: str | Path,
) -> Path:
    """Write a unified CSV combining ranked agent metrics across all suite tournaments."""
    path = Path(output_csv_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "board",
        "budget_mode",
        "total_tournament_matches",
        "rank",
        "agent_id",
        "total_games",
        "wins",
        "draws",
        "losses",
        "win_rate",
        "win_rate_ci_lower",
        "win_rate_ci_upper",
        "score",
        "score_ci_lower",
        "score_ci_upper",
        "score_x",
        "score_o",
        "delta_score",
        "delta_score_ci_lower",
        "delta_score_ci_upper",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "max_latency_ms",
        "avg_nodes_visited",
        "avg_simulations",
        "avg_depth",
        "total_moves",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for board_tag, mode_name, summary in ordered_runs:
            for rank, m in enumerate(summary.get_rankings(), start=1):
                writer.writerow(
                    {
                        "board": board_tag,
                        "budget_mode": mode_name,
                        "total_tournament_matches": summary.total_matches,
                        "rank": rank,
                        "agent_id": m.agent_id,
                        "total_games": m.total_games,
                        "wins": m.wins,
                        "draws": m.draws,
                        "losses": m.losses,
                        "win_rate": f"{m.win_rate:.4f}",
                        "win_rate_ci_lower": f"{m.win_rate_ci[0]:.4f}",
                        "win_rate_ci_upper": f"{m.win_rate_ci[1]:.4f}",
                        "score": f"{m.score:.4f}",
                        "score_ci_lower": f"{m.score_ci[0]:.4f}",
                        "score_ci_upper": f"{m.score_ci[1]:.4f}",
                        "score_x": f"{m.score_x:.4f}",
                        "score_o": f"{m.score_o:.4f}",
                        "delta_score": f"{m.delta_score:.4f}",
                        "delta_score_ci_lower": f"{m.delta_score_ci[0]:.4f}",
                        "delta_score_ci_upper": f"{m.delta_score_ci[1]:.4f}",
                        "mean_latency_ms": f"{m.mean_latency_ms:.4f}",
                        "median_latency_ms": f"{m.median_latency_ms:.4f}",
                        "p95_latency_ms": f"{m.p95_latency_ms:.4f}",
                        "max_latency_ms": f"{m.max_latency_ms:.4f}",
                        "avg_nodes_visited": f"{m.avg_nodes_visited:.2f}",
                        "avg_simulations": f"{m.avg_simulations:.2f}",
                        "avg_depth": f"{m.avg_depth:.2f}",
                        "total_moves": m.total_moves,
                    }
                )
    return path


def _export_suite_consolidated_markdown(
    ordered_runs: Sequence[tuple[str, str, Any]],
    integrity_reports: Mapping[str, TournamentIntegrityReport],
    format_summary_markdown_fn: Any,
    output_md_path: str | Path,
) -> Path:
    """Write a unified Markdown report with audit status and full tables for every suite run."""
    path = Path(output_md_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Official Paired Round-Robin Tournament Suite Summary (Task T4.3)",
        "",
        "## Suite Overview & Integrity Verification",
        "",
        "| Suite Key | Board | Budget Mode | Matches | Illegal Moves | 50/50 X/O Valid | "
        "Top Agent (Score) | Seat ΔScore [95% CI] |",
        "|:---|:---:|:---|---:|---:|:---:|:---|:---:|",
    ]
    for board_tag, mode_name, summary in ordered_runs:
        key = f"{board_tag}_{mode_name}"
        report = integrity_reports[key]
        rankings = summary.get_rankings()
        top_str = f"{rankings[0].agent_id} ({rankings[0].score:.3f})" if rankings else "N/A"
        sa = summary.seat_advantage
        ds_str = f"{sa.delta_score:+.3f} [{sa.delta_score_ci[0]:+.3f}, {sa.delta_score_ci[1]:+.3f}]"
        valid_str = "PASS" if report.is_valid else "FAIL"
        lines.append(
            f"| `{key}` | `{board_tag}` | `{mode_name}` | {report.total_matches} | "
            f"{len(report.corrupt_or_illegal_matches)} | {valid_str} | "
            f"{top_str} | {ds_str} |"
        )

    for board_tag, mode_name, summary in ordered_runs:
        lines.extend(["", "---", "", f"<!-- Suite Target: {board_tag}_{mode_name} -->", ""])
        lines.append(format_summary_markdown_fn(summary))

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_official_tournament_suite(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    reports_dir: str | Path | None = None,
    base_dir: str | Path | None = None,
    n_pairs_override: int | None = None,
    seed_override: int | None = None,
    targets: Sequence[tuple[str, str]] = DEFAULT_OFFICIAL_TOURNAMENT_TARGETS,
    resume: bool = True,
    n_bootstrap: int = 10000,
) -> OfficialTournamentSuiteResult:
    """Execute, audit, and summarize the official T4.3 Paired Round-Robin tournament suite.

    Default targets cover ``4x4_k4`` across ``fixed_10ms``, ``fixed_50ms``, and ``default``
    budgets plus the ``3x3_k3`` control tournament in ``default`` mode.

    Args:
        config_path: Path to ``configs/tournament.yaml``.
        output_dir: Directory where ``tournament_<board>_<mode>.jsonl`` logs are saved.
        reports_dir: Optional directory for per-tournament and suite CSV/Markdown summaries.
        base_dir: Project root used to resolve relative checkpoint paths.
        n_pairs_override: Optional override for ``n_pairs`` (defaults to YAML value, 100).
        seed_override: Optional override for base seed (defaults to YAML value, 42).
        targets: Sequence of ``(board_tag, budget_mode)`` pairs to execute.
        resume: Resume from existing matching JSONL logs when present.
        n_bootstrap: Bootstrap resamples for 95% confidence intervals (default 10,000).

    Returns:
        :class:`OfficialTournamentSuiteResult` holding log paths, audits, and summaries.

    Raises:
        InvalidConfigError: If configuration, mode, or log integrity verification fails.
    """
    from ttt_ai.benchmark.metrics import (
        export_summary_csv,
        format_summary_markdown,
        save_summary_markdown,
        summarize_tournament_log,
    )

    if not targets:
        raise InvalidConfigError("targets must contain at least one (board_tag, budget_mode) pair.")

    cfg = load_tournament_config(config_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rep_root = Path(reports_dir) if reports_dir is not None else None
    if rep_root is not None:
        rep_root.mkdir(parents=True, exist_ok=True)

    resolved_base = (
        Path(base_dir).resolve()
        if base_dir is not None
        else Path(config_path).resolve().parent.parent
    )
    n_pairs = int(n_pairs_override if n_pairs_override is not None else cfg.n_pairs)
    base_seed = int(seed_override if seed_override is not None else cfg.seed)

    log_paths: dict[str, Path] = {}
    integrity_reports: dict[str, TournamentIntegrityReport] = {}
    summaries: dict[str, Any] = {}
    ordered_runs: list[tuple[str, str, Any]] = []

    for board_tag, mode_name in targets:
        board_cfg = parse_board_preset(board_tag)
        if mode_name not in cfg.budget_modes:
            raise InvalidConfigError(
                f"Budget mode {mode_name!r} not defined in '{config_path}'. "
                f"Available: {sorted(cfg.budget_modes.keys())}."
            )
        mode_cfg = cfg.budget_modes[mode_name]
        roster = cfg.resolve_agents_for_board(board_cfg)
        key = f"{board_cfg.name}_{mode_cfg.name}"

        seed_everything(base_seed)
        runner = TournamentRunner(
            board_config=board_cfg,
            agent_specs=roster,
            n_pairs=n_pairs,
            base_seed=base_seed,
            budget_mode=mode_cfg.name,
            time_budget_override_ms=mode_cfg.time_budget_ms,
            n_simulations_override=mode_cfg.n_simulations,
            base_dir=resolved_base,
            tournament_name=f"{cfg.name}_{key}",
        )
        log_path = runner.run(out_root / f"tournament_{key}.jsonl", resume=resume)
        audit = verify_tournament_log_integrity(
            log_path,
            expected_agents=[spec.agent_id for spec in roster],
            expected_n_pairs=n_pairs,
            replay_moves=True,
        )
        if not audit.is_valid:
            raise InvalidConfigError(
                f"Tournament integrity check failed for '{log_path}': "
                f"duplicates={audit.duplicate_match_ids}, "
                f"unbalanced={audit.unbalanced_matchups}, "
                f"illegal={audit.corrupt_or_illegal_matches[:5]}."
            )

        summary = summarize_tournament_log(
            log_path,
            n_bootstrap=n_bootstrap,
            seed=base_seed,
        )
        if rep_root is not None:
            export_summary_csv(summary, rep_root / f"tournament_{key}_metrics.csv")
            save_summary_markdown(summary, rep_root / f"tournament_{key}_summary.md")

        log_paths[key] = log_path
        integrity_reports[key] = audit
        summaries[key] = summary
        ordered_runs.append((board_cfg.name, mode_cfg.name, summary))

    suite_csv: Path | None = None
    suite_md: Path | None = None
    if rep_root is not None:
        suite_csv = _export_suite_consolidated_csv(
            ordered_runs,
            rep_root / "tournament_suite_metrics.csv",
        )
        suite_md = _export_suite_consolidated_markdown(
            ordered_runs,
            integrity_reports,
            format_summary_markdown,
            rep_root / "tournament_suite_summary.md",
        )

    return OfficialTournamentSuiteResult(
        log_paths=log_paths,
        integrity_reports=integrity_reports,
        summaries=summaries,
        suite_csv_path=suite_csv,
        suite_markdown_path=suite_md,
    )
