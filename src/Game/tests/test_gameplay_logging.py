import subprocess
import sys

import numpy as np

from Game.gym_env import BJJEnv


def _reset_and_play_one_move(env: BJJEnv) -> None:
    _, info = env.reset(seed=123)
    action = int(np.where(info["action_mask"])[0][0])
    env.step(action)


def test_default_env_is_silent(capsys) -> None:
    env = BJJEnv()
    capsys.readouterr()  # ignore construction-time output
    _reset_and_play_one_move(env)
    captured = capsys.readouterr()
    assert "Initializing game" not in captured.out
    assert " performed '" not in captured.out


def test_render_mode_human_emits_console_logs(capsys) -> None:
    env = BJJEnv(render_mode="human")
    capsys.readouterr()  # ignore construction-time output
    _reset_and_play_one_move(env)
    captured = capsys.readouterr()
    assert "Initializing game" in captured.out
    assert "performed" in captured.out


def test_file_logging_only_writes_file(tmp_path, capsys) -> None:
    log_path = tmp_path / "gameplay.log"
    env = BJJEnv(gameplay_log_path=log_path)
    capsys.readouterr()  # ignore construction-time output
    _reset_and_play_one_move(env)
    captured = capsys.readouterr()
    assert "Initializing game" not in captured.out
    assert " performed '" not in captured.out
    assert log_path.exists()
    contents = log_path.read_text()
    assert "Initializing game" in contents
    assert "performed" in contents


def test_render_and_file_logging_write_to_both(tmp_path, capsys) -> None:
    log_path = tmp_path / "gameplay.log"
    env = BJJEnv(render_mode="human", gameplay_log_path=log_path)
    capsys.readouterr()  # ignore construction-time output
    _reset_and_play_one_move(env)
    captured = capsys.readouterr()
    assert "Initializing game" in captured.out
    assert "performed" in captured.out
    assert log_path.exists()
    contents = log_path.read_text()
    assert "Initializing game" in contents
    assert "performed" in contents


def test_importing_play_game_has_no_demo_output() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import Game.play_game"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""


def test_importing_play_game_visualizer_has_no_demo_output() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import Game.play_game_visualizer"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""
