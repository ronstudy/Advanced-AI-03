"""Fair round-robin tournament runner, statistical metrics, ablations, and plots."""

from __future__ import annotations

from ttt_ai.benchmark.tournament import (
    AgentSpec,
    BudgetModeConfig,
    MatchRecord,
    MoveRecord,
    TournamentConfig,
    TournamentRunner,
    collect_tournament_environment_meta,
    instantiate_agent,
    load_tournament_config,
    load_tournament_log,
    play_single_match,
)

__all__: list[str] = [
    "AgentSpec",
    "BudgetModeConfig",
    "MatchRecord",
    "MoveRecord",
    "TournamentConfig",
    "TournamentRunner",
    "collect_tournament_environment_meta",
    "instantiate_agent",
    "load_tournament_config",
    "load_tournament_log",
    "play_single_match",
]
