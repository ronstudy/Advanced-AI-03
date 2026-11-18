"""Unit and smoke tests for Task T5.1 Rich CLI, Theme, Layout, and Interactive Controller."""

from __future__ import annotations

import ast
import importlib.util
import io
from pathlib import Path
from types import ModuleType

import pytest
from rich import box
from rich.console import Console
from rich.layout import Layout

from ttt_ai import (
    AlphaBetaAgent,
    BoardConfig,
    CheckpointVerificationError,
    ExtendedTicTacToe,
    GameMode,
    HybridMCTSAgent,
    InteractiveCLIApp,
    InteractiveMatchController,
    MatchTurnRecord,
    PolicyNetworkAgent,
    RandomAgent,
    RuleBasedAgent,
    create_agent_for_cli,
    get_available_agent_levels,
    parse_user_input,
    run_setup_wizard,
)
from ttt_ai.engine.exceptions import IllegalMoveError, InvalidConfigError
from ttt_ai.engine.rules import Player
from ttt_ai.ui import (
    ASCII_FALLBACK_CHARS,
    HEATMAP_HIGH_STYLE,
    HEATMAP_LOW_STYLE,
    HEATMAP_MED_STYLE,
    HEATMAP_OCCUPIED_STYLE,
    HEATMAP_TOP1_STYLE,
    UNICODE_CHARS,
    build_board_panel,
    build_dashboard_layout,
    build_dashboard_panel,
    build_heatmap_panel,
    build_telemetry_panel,
    classify_heatmap_tier,
    format_extended_board_ascii,
    format_latency_style,
    get_box_style,
    get_symbol_table,
    should_use_ascii,
    style_for_heatmap_tier,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"


def _load_play_script() -> ModuleType:
    """Load `scripts/play.py` dynamically for CLI argument parsing tests."""
    script_path = PROJECT_ROOT / "scripts" / "play.py"
    spec = importlib.util.spec_from_file_location("ttt_play_script", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_memory_console(
    ascii_only: bool = True,
    *,
    height: int | None = None,
) -> tuple[Console, io.StringIO]:
    """Create a Rich Console backed by an in-memory UTF-8 StringIO buffer."""
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=False,
        color_system=None,
        width=120,
        height=height,
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

    def test_ui_modules_do_not_use_bare_print(self) -> None:
        ui_dir = PROJECT_ROOT / "src" / "ttt_ai" / "ui"
        for path in sorted(ui_dir.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "print", f"Forbidden bare print() in {path}"


class TestRichThemeLayoutAndAllAgentHeatmaps:
    """Verify theme palette, ASCII fallbacks, 48/52 Layout, and Levels 1-7 heatmaps."""

    def test_theme_heatmap_tiers_latency_styles_and_ascii_fallback(self) -> None:
        assert InteractiveCLIApp is InteractiveMatchController
        assert "bright_yellow" in HEATMAP_TOP1_STYLE
        assert "green" in HEATMAP_HIGH_STYLE
        assert "yellow" in HEATMAP_MED_STYLE
        assert "dim white" in HEATMAP_LOW_STYLE

        assert style_for_heatmap_tier(0) == HEATMAP_OCCUPIED_STYLE
        assert style_for_heatmap_tier(1) == HEATMAP_LOW_STYLE
        assert style_for_heatmap_tier(2) == HEATMAP_MED_STYLE
        assert style_for_heatmap_tier(3) == HEATMAP_HIGH_STYLE
        assert style_for_heatmap_tier(4) == HEATMAP_TOP1_STYLE
        with pytest.raises(InvalidConfigError):
            style_for_heatmap_tier(5)

        assert classify_heatmap_tier(0.9, 0.0, 1.0, is_top1=True) == 4
        assert classify_heatmap_tier(0.75, 0.0, 1.0, is_top1=False) == 3
        assert classify_heatmap_tier(0.40, 0.0, 1.0, is_top1=False) == 2
        assert classify_heatmap_tier(0.10, 0.0, 1.0, is_top1=False) == 1
        assert classify_heatmap_tier(0.25, 0.25, 0.25, is_top1=False) == 2

        assert format_latency_style(10.0, budget_ms=50.0) == "bold green"
        assert format_latency_style(45.0, budget_ms=50.0) == "bold yellow"
        assert format_latency_style(55.0, budget_ms=50.0) == "bold red"

        assert get_box_style(ascii_only=False) is box.ROUNDED
        assert get_box_style(ascii_only=True) is box.ASCII
        assert should_use_ascii(force_ascii=True) is True
        assert get_symbol_table(ascii_only=True) == ASCII_FALLBACK_CHARS
        assert get_symbol_table(ascii_only=False) == UNICODE_CHARS

    def test_dashboard_layout_48_52_split_and_rendering(self) -> None:
        cfg = BoardConfig(4, 4, 4)
        agent_l7 = create_agent_for_cli(
            7,
            cfg,
            seed=42,
            n_simulations=20,
            checkpoints_dir=CKPT_DIR,
        )
        game = ExtendedTicTacToe(4, 4, 4)
        game.step(5)
        dec = agent_l7.select_action(game)
        row, col = divmod(dec.action, game.cols)
        turn_rec = MatchTurnRecord(
            turn_index=1,
            player=Player.O,
            action=dec.action,
            row=row,
            col=col,
            latency_ms=dec.latency_ms,
            agent_name=agent_l7.name,
            decision=dec,
            agent_level=7,
        )

        layout = build_dashboard_layout(
            game,
            mode=GameMode.HUMAN_VS_AI,
            actor_x_name="Human_X",
            actor_o_name=agent_l7.name,
            turn_records=[turn_rec],
            status_message="Ready for next move.",
            budget_ms=50.0,
            ascii_only=True,
        )
        assert isinstance(layout, Layout)
        assert layout["left"].ratio == 48
        assert layout["right"].ratio == 52

        console, buf = _make_memory_console(ascii_only=True, height=36)
        console.print(layout)
        rendered = buf.getvalue()
        assert "GAME BOARD" in rendered
        assert "AI TELEMETRY" in rendered
        assert "ACTION HEATMAP & TOP-5 MOVES" in rendered
        assert "visit=" in rendered
        assert "prior=" in rendered
        assert "HISTORY & CONTROLS" in rendered

    def test_heatmap_panel_all_seven_agent_levels(self) -> None:
        cfg = BoardConfig(4, 4, 4)
        game = ExtendedTicTacToe(4, 4, 4)
        game.step(5)
        game.step(10)

        for level in range(1, 8):
            agent = create_agent_for_cli(
                level,
                cfg,
                seed=42,
                time_budget_ms=50.0,
                n_simulations=20,
                checkpoints_dir=CKPT_DIR,
            )
            dec = agent.select_action(game)
            panel = build_heatmap_panel(
                game,
                dec,
                agent_level=level,
                ascii_only=True,
            )
            tel_panel = build_telemetry_panel(
                MatchTurnRecord(
                    turn_index=3,
                    player=Player.X,
                    action=dec.action,
                    row=dec.action // 4,
                    col=dec.action % 4,
                    latency_ms=dec.latency_ms,
                    agent_name=agent.name,
                    decision=dec,
                    agent_level=level,
                ),
                budget_ms=50.0,
                ascii_only=True,
            )
            console, buf = _make_memory_console(ascii_only=True)
            console.print(tel_panel)
            console.print(panel)
            out = buf.getvalue()
            assert "Top-5 Moves" in out
            assert f"[Lv.{level}]" in out
            if level in (1, 5):
                assert "prob=" in out
            elif level in (2, 3):
                assert "score=" in out
            elif level == 4:
                assert "visit=" in out
            elif level == 6:
                assert "Q=" in out
            elif level == 7:
                assert "visit=" in out
                assert "prior=" in out

    def test_board_panel_last_move_and_winning_line_markers(self) -> None:
        game = ExtendedTicTacToe(3, 3, 3)
        for move in (0, 3, 1, 4):
            game.step(move)
        console_mid, buf_mid = _make_memory_console(ascii_only=True)
        console_mid.print(build_board_panel(game, ascii_only=True))
        assert "[O]" in buf_mid.getvalue()

        game.step(2)  # Player X wins top row (0, 1, 2)
        console_win, buf_win = _make_memory_console(ascii_only=True)
        console_win.print(build_board_panel(game, ascii_only=True))
        win_out = buf_win.getvalue()
        assert "[X]" in win_out
        assert "*X*" in win_out

    def test_available_levels_and_wizard_filter_learned_checkpoints(self) -> None:
        avail_4x4_k4 = get_available_agent_levels(BoardConfig(4, 4, 4), CKPT_DIR)
        assert sorted(avail_4x4_k4.keys()) == [1, 2, 3, 4, 5, 6, 7]

        avail_4x4_k3 = get_available_agent_levels(BoardConfig(4, 4, 3), CKPT_DIR)
        assert sorted(avail_4x4_k3.keys()) == [1, 2, 3, 4, 6]

        avail_5x5_k4 = get_available_agent_levels(BoardConfig(5, 5, 4), CKPT_DIR)
        assert sorted(avail_5x5_k4.keys()) == [1, 2, 3, 4]

        console, buf = _make_memory_console(ascii_only=True)
        wizard_inputs = iter([
            "4",   # Preset 4: 5x5 K=4 (no checkpoints!)
            "1",   # Human vs AI
            "50",  # Budget
            "42",  # Seed
            "x",   # Play X
            "2",   # Level 2
        ])
        ctrl = run_setup_wizard(
            input_fn=lambda _: next(wizard_inputs),
            console=console,
            checkpoints_dir=CKPT_DIR,
            ascii_only=True,
        )
        assert isinstance(ctrl, InteractiveCLIApp)
        wizard_out = buf.getvalue()
        assert "[4] L4_MCTS" in wizard_out
        assert "[5] L5_PolicyNet" not in wizard_out
        assert "[7] L7_HybridMCTS" not in wizard_out

    def test_undo_two_steps_when_human_is_second_and_live_autoplay(self) -> None:
        cfg = BoardConfig(3, 3, 3)
        agent_x = create_agent_for_cli(1, cfg, seed=7)
        console, buf = _make_memory_console(ascii_only=True)
        inputs = iter([
            "u",   # Warning: no completed human move yet when Human O hasn't moved
            "4",   # Human O plays cell 4 -> AI X replies
            "u",   # Reverts AI X + Human O -> back to move_count == 1
            "q",   # Quit
        ])
        ctrl = InteractiveCLIApp(
            game=ExtendedTicTacToe(3, 3, 3),
            mode=GameMode.HUMAN_VS_AI,
            agent_x=agent_x,
            agent_o=None,
            input_fn=lambda _: next(inputs),
            console=console,
            ascii_only=True,
        )
        summary = ctrl.run_match()
        assert summary.quit_early is True
        assert summary.move_count == 1
        assert "No completed human move available to undo." in buf.getvalue()

        # Verify Live + Layout branch in AI vs AI mode completes cleanly.
        live_console, live_buf = _make_memory_console(ascii_only=True, height=36)
        live_ctrl = InteractiveCLIApp(
            game=ExtendedTicTacToe(3, 3, 3),
            mode=GameMode.AI_VS_AI,
            agent_x=create_agent_for_cli(2, cfg, seed=10),
            agent_o=create_agent_for_cli(2, cfg, seed=11),
            console=live_console,
            step_delay_ms=0.0,
            ascii_only=True,
            use_live=True,
            use_layout=True,
        )
        live_summary = live_ctrl.run_match()
        assert not live_summary.quit_early
        assert live_summary.is_draw is True
        assert "GAME BOARD" in live_buf.getvalue()

    def test_play_entrypoint_cli_flags_and_controller_builder(self) -> None:
        play_mod = _load_play_script()
        args = play_mod._parse_args(
            [
                "--board",
                "4x4_k4",
                "--mode",
                "ai_vs_ai",
                "--agent-x",
                "7",
                "--agent-o",
                "3",
                "--budget-ms",
                "25",
                "--step-delay",
                "150",
                "--n-simulations",
                "15",
                "--seed",
                "99",
                "--ascii",
            ]
        )
        assert args.board == "4x4_k4"
        assert args.mode == "ai_vs_ai"
        assert args.agent_x == "7"
        assert args.agent_o == "3"
        assert args.budget_ms == 25.0
        assert args.step_delay_ms == 150.0

        console, _ = _make_memory_console(ascii_only=True)
        ctrl = play_mod.build_controller_from_args(args, console=console, ascii_only=True)
        assert isinstance(ctrl, InteractiveCLIApp)
        assert ctrl.mode is GameMode.AI_VS_AI
        ctrl._step_delay_ms = 0.0
        summary = ctrl.run_match()
        assert summary.move_count >= 7

        # Verify --agent-x in human_vs_ai automatically puts Human on O.
        hv_args = play_mod._parse_args(
            ["--board", "3x3_k3", "--mode", "human_vs_ai", "--agent-x", "2"]
        )
        hv_ctrl = play_mod.build_controller_from_args(
            hv_args, console=console, ascii_only=True
        )
        assert hv_ctrl._agent_x is not None
        assert hv_ctrl._agent_o is None
