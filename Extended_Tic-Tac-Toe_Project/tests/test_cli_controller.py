"""Unit and smoke tests for Task T4.1 Interactive Game Controller and Rich CLI."""

from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest
from rich.console import Console

from ttt_ai import (
    AlphaBetaAgent,
    BoardConfig,
    CheckpointVerificationError,
    ExtendedTicTacToe,
    GameMode,
    HybridMCTSAgent,
    InteractiveMatchController,
    PolicyNetworkAgent,
    RandomAgent,
    RuleBasedAgent,
    create_agent_for_cli,
    parse_user_input,
    run_setup_wizard,
)
from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError
from ttt_ai.engine.rules import Player
from ttt_ai.ui import (
    build_dashboard_panel,
    build_heatmap_panel,
    format_extended_board_ascii,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"


def _make_memory_console(ascii_only: bool = True) -> tuple[Console, io.StringIO]:
    """Create a Rich Console backed by an in-memory UTF-8 StringIO buffer."""
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=False,
        color_system=None,
        width=120,
        safe_box=ascii_only,
        highlight=False,
    )
    return console, buf


class TestUserInputParsingAndAgentFactory:
    """Verify user input parsing, bounds checks, and difficulty/level factory."""

    def test_parse_valid_coordinates_indices_and_commands(self) -> None:
        game = ExtendedTicTacToe(4, 4, 4)
        game.step(0)

        assert parse_user_input("1,2", game).action == 6
        assert parse_user_input(" 2 , 3 ", game).action == 11
        assert parse_user_input("1 3", game).action == 7
        assert parse_user_input("(3,0)", game).action == 12
        assert parse_user_input("15", game).action == 15

        assert parse_user_input("u", game).kind == "undo"
        assert parse_user_input("UNDO", game).kind == "undo"
        assert parse_user_input("h", game).kind == "help"
        assert parse_user_input("help", game).kind == "help"
        assert parse_user_input("q", game).kind == "quit"
        assert parse_user_input("QUIT", game).kind == "quit"

    @pytest.mark.parametrize(
        "bad_input",
        [
            "",
            "   ",
            "abc",
            "1.5",
            "1,2,3",
            "1,",
            ",2",
            "0,0",  # Already occupied after step(0)
            "0",  # Already occupied flat index
            "-1,0",
            "0,-1",
            "4,0",  # Out-of-bounds row on 4x4
            "0,4",  # Out-of-bounds col on 4x4
            "-1",
            "16",
            "999",
        ],
    )
    def test_parse_invalid_inputs_raise_illegal_move_error(self, bad_input: str) -> None:
        game = ExtendedTicTacToe(4, 4, 4)
        game.step(0)
        with pytest.raises(IllegalMoveError):
            parse_user_input(bad_input, game)

    def test_create_agent_for_cli_presets_and_all_7_levels(self) -> None:
        cfg_4x4 = BoardConfig(4, 4, 4)
        assert isinstance(create_agent_for_cli("easy", cfg_4x4, seed=1), RandomAgent)
        assert isinstance(create_agent_for_cli("medium", cfg_4x4, seed=1), RuleBasedAgent)
        assert isinstance(create_agent_for_cli("hard", cfg_4x4, seed=1), AlphaBetaAgent)

        for lvl in range(1, 8):
            agent = create_agent_for_cli(
                lvl,
                cfg_4x4,
                seed=42,
                n_simulations=20,
                checkpoints_dir=CKPT_DIR,
            )
            assert agent.level == lvl

        cfg_3x3 = BoardConfig(3, 3, 3)
        lvl5 = create_agent_for_cli(5, cfg_3x3, checkpoints_dir=CKPT_DIR)
        lvl6 = create_agent_for_cli(6, cfg_3x3, checkpoints_dir=CKPT_DIR)
        lvl7 = create_agent_for_cli(7, cfg_3x3, n_simulations=20, checkpoints_dir=CKPT_DIR)
        assert isinstance(lvl5, PolicyNetworkAgent)
        assert lvl6.level == 6
        assert isinstance(lvl7, HybridMCTSAgent)

        with pytest.raises(InvalidConfigError):
            create_agent_for_cli("impossible", cfg_4x4)
        with pytest.raises(InvalidConfigError):
            create_agent_for_cli(8, cfg_4x4)
        with pytest.raises(CheckpointVerificationError):
            create_agent_for_cli(5, BoardConfig(5, 5, 4), checkpoints_dir=CKPT_DIR)


class TestInteractiveMatchControllerLoops:
    """Verify full game loops, outcome detection, undo/help/quit, and resilience."""

    def test_human_vs_human_win_x_win_o_and_draw(self) -> None:
        # 1. Player X wins on top row of 3x3 K=3.
        console, buf = _make_memory_console()
        x_win_inputs = iter(["0,0", "1,0", "0,1", "1,1", "0,2"])
        ctrl_x = InteractiveMatchController(
            game=ExtendedTicTacToe(3, 3, 3),
            mode=GameMode.HUMAN_VS_HUMAN,
            input_fn=lambda _: next(x_win_inputs),
            console=console,
            ascii_only=True,
        )
        res_x = ctrl_x.run_match()
        assert res_x.winner is Player.X
        assert not res_x.is_draw
        assert not res_x.quit_early
        assert res_x.move_count == 5
        assert res_x.winning_line == ((0, 0), (0, 1), (0, 2))
        assert "WINS" in buf.getvalue()

        # 2. Player O wins on middle row of 3x3 K=3.
        console_o, _ = _make_memory_console()
        o_win_inputs = iter(["0,0", "1,0", "0,1", "1,1", "2,2", "1,2"])
        ctrl_o = InteractiveMatchController(
            game=ExtendedTicTacToe(3, 3, 3),
            mode=GameMode.HUMAN_VS_HUMAN,
            input_fn=lambda _: next(o_win_inputs),
            console=console_o,
            ascii_only=True,
        )
        res_o = ctrl_o.run_match()
        assert res_o.winner is Player.O
        assert res_o.move_count == 6
        assert res_o.winning_line == ((1, 0), (1, 1), (1, 2))

        # 3. Full-board draw on 3x3 K=3: moves [0,1,2,4,3,6,7,5,8].
        console_d, buf_d = _make_memory_console()
        draw_inputs = iter(["0", "1", "2", "4", "3", "6", "7", "5", "8"])
        ctrl_d = InteractiveMatchController(
            game=ExtendedTicTacToe(3, 3, 3),
            mode=GameMode.HUMAN_VS_HUMAN,
            input_fn=lambda _: next(draw_inputs),
            console=console_d,
            ascii_only=True,
        )
        res_d = ctrl_d.run_match()
        assert res_d.winner is None
        assert res_d.is_draw
        assert res_d.move_count == 9
        assert "DRAW" in buf_d.getvalue()

    def test_human_vs_ai_invalid_input_recovery_undo_help_and_quit(self) -> None:
        cfg = BoardConfig(3, 3, 3)
        easy_ai = create_agent_for_cli("easy", cfg, seed=11)
        console, buf = _make_memory_console()

        scripted = iter([
            "bad_str",     # Garbage string
            "9,9",         # Out-of-bounds coordinates
            "99",          # Out-of-bounds flat index
            "h",           # Help command
            "0,0",         # Valid move 1 (Human X) -> AI plays reply
            "0,0",         # Duplicate occupied cell!
            "u",           # Undo 2 plies (AI + Human)
            "1,1",         # Valid move after undo
            "q",           # Quit cleanly
        ])
        ctrl = InteractiveMatchController(
            game=ExtendedTicTacToe(3, 3, 3),
            mode=GameMode.HUMAN_VS_AI,
            agent_x=None,
            agent_o=easy_ai,
            input_fn=lambda _: next(scripted),
            console=console,
            ascii_only=True,
        )
        summary = ctrl.run_match()
        assert summary.quit_early is True
        assert ctrl.invalid_input_count == 4
        assert summary.move_count == 2  # 1,1 + AI reply
        assert summary.move_history[0] == 4
        output = buf.getvalue()
        assert "Invalid input" in output
        assert "Commands:" in output

    def test_human_vs_ai_complete_match_with_presets(self) -> None:
        cfg = BoardConfig(4, 4, 4)
        for preset in ("easy", "medium", "hard"):
            ai_agent = create_agent_for_cli(
                preset,
                cfg,
                seed=7,
                time_budget_ms=25.0,
            )
            game = ExtendedTicTacToe(4, 4, 4)
            console, _ = _make_memory_console()

            def _pick_first_legal(_prompt: str, g: ExtendedTicTacToe = game) -> str:
                return str(g.legal_actions()[0])

            ctrl = InteractiveMatchController(
                game=game,
                mode=GameMode.HUMAN_VS_AI,
                agent_x=None,
                agent_o=ai_agent,
                input_fn=_pick_first_legal,
                console=console,
                ascii_only=True,
                budget_ms=25.0,
            )
            summary = ctrl.run_match()
            assert not summary.quit_early
            assert game.is_terminal()
            assert summary.move_count >= 7

    def test_ai_vs_ai_and_level7_heatmap_rendering(self) -> None:
        cfg = BoardConfig(4, 4, 4)
        agent_x = create_agent_for_cli(
            7,
            cfg,
            seed=42,
            n_simulations=20,
            checkpoints_dir=CKPT_DIR,
        )
        agent_o = create_agent_for_cli(2, cfg, seed=43)
        console, buf = _make_memory_console()

        game = ExtendedTicTacToe(4, 4, 4)
        ctrl = InteractiveMatchController(
            game=game,
            mode=GameMode.AI_VS_AI,
            agent_x=agent_x,
            agent_o=agent_o,
            console=console,
            step_delay_ms=0.0,
            ascii_only=True,
        )
        summary = ctrl.run_match()
        assert not summary.quit_early
        assert game.is_terminal()
        assert summary.move_count == len(summary.turn_records)
        rendered = buf.getvalue()
        assert "prior=" in rendered
        assert "Think Time:" in summary.final_board_text

        # Direct formatting & panel assertions on a live non-terminal board.
        preview_game = ExtendedTicTacToe(4, 4, 4)
        preview_game.step(5)
        dec = agent_x.select_action(preview_game)
        ascii_board = format_extended_board_ascii(
            preview_game,
            last_latency_ms=dec.latency_ms,
        )
        assert "Last Move: (1, 1) [#5]" in ascii_board
        heatmap_panel = build_heatmap_panel(preview_game, dec, ascii_only=True)
        dash_panel = build_dashboard_panel(
            preview_game,
            mode=GameMode.AI_VS_AI,
            actor_x_name="L7_HybridMCTS",
            actor_o_name="L2_RuleBased",
            turn_records=[],
            ascii_only=True,
        )
        assert heatmap_panel is not None
        assert dash_panel is not None

    def test_setup_wizard_with_mock_inputs(self) -> None:
        console, _ = _make_memory_console()
        wizard_inputs = iter([
            "3",       # Board preset 3: 3x3 K=3
            "2",       # Mode 2: AI vs AI
            "50",      # Budget ms
            "42",      # Seed
            "easy",    # Agent X
            "medium",  # Agent O
            "a",       # Auto pacing
        ])
        ctrl = run_setup_wizard(
            input_fn=lambda _: next(wizard_inputs),
            console=console,
            checkpoints_dir=CKPT_DIR,
            ascii_only=True,
        )
        assert ctrl.mode is GameMode.AI_VS_AI
        # Run the configured match with zero delay for fast test execution.
        ctrl._step_delay_ms = 0.0
        summary = ctrl.run_match()
        assert not summary.quit_early
        assert summary.move_count >= 5

    def test_ui_modules_do_not_access_private_engine_attributes(self) -> None:
        ui_dir = PROJECT_ROOT / "src" / "ttt_ai" / "ui"
        forbidden = {
            "_board",
            "_history",
            "_current_player",
            "_terminal",
            "_winner",
            "_winning_line",
        }
        for path in sorted(ui_dir.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    assert node.attr not in forbidden, (
                        f"Forbidden private attribute {node.attr} in {path}"
                    )
