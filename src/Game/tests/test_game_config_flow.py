"""Tests for Game.game_config: try_apply_config and run_configured_game."""

from __future__ import annotations

from typing import Any, Callable, List, Optional
from unittest.mock import patch

import pytest

import Game.policies as policies_module
from Game.game_config import run_configured_game, try_apply_config
from Game.play_game import Game


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubServer:
    """Minimal stand-in for PositionServer used in game_config tests."""

    def __init__(self) -> None:
        self.on_game_config: Optional[Callable[[dict], None]] = None
        self.on_move_selected: Optional[Callable[[int], None]] = None
        self.config_options: Optional[dict] = None
        self.config_errors: List[str] = []

    def send_config_error(self, message: str) -> None:
        self.config_errors.append(message)

    def set_config_options(self, options: Optional[dict]) -> None:
        self.config_options = options

    def send_legal_moves(self, current_node: int, moves: List[dict]) -> None:
        pass  # no-op; prevents blocking in make_human_strategy


class StubVisualizer:
    """Minimal stand-in for Visualizer3D used in game_config tests."""

    def __init__(self) -> None:
        self.server: StubServer = StubServer()
        self.turn_delay: float = 0.0
        self.game_ref: Optional[Any] = None

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def visualizer() -> StubVisualizer:
    return StubVisualizer()


@pytest.fixture()
def defaults() -> dict:
    return {
        "max_turns_default": 30,
        "max_turns_min": 1,
        "max_turns_max": 200,
        "turn_delay_default": 0.0,
    }


@pytest.fixture()
def logger(caplog: pytest.LogCaptureFixture):
    import logging

    return logging.getLogger("test_game_config_flow")


# ---------------------------------------------------------------------------
# try_apply_config tests
# ---------------------------------------------------------------------------


def _random_vs_random_cfg(max_turns: int = 20) -> dict:
    return {
        "settings": {"max_turns": max_turns, "turn_delay": 0.0},
        "players": {
            "p1": {"type": "computer", "policy_id": "random"},
            "p2": {"type": "computer", "policy_id": "random"},
        },
    }


def test_valid_random_vs_random_returns_game(
    visualizer: StubVisualizer,
    defaults: dict,
    logger: Any,
) -> None:
    cfg = _random_vs_random_cfg(max_turns=20)

    game = try_apply_config(cfg, visualizer, defaults, logger)

    assert isinstance(game, Game)
    assert game.max_turns == 20
    assert game.player1.strategy == "random"
    assert game.player2.strategy == "random"
    assert visualizer.server.on_game_config is None
    assert visualizer.server.config_options is None


def test_unknown_policy_sends_config_error_and_returns_none(
    visualizer: StubVisualizer,
    defaults: dict,
    logger: Any,
) -> None:
    cfg = {
        "settings": {"max_turns": 10, "turn_delay": 0.0},
        "players": {
            "p1": {"type": "computer", "policy_id": "does-not-exist"},
            "p2": {"type": "computer", "policy_id": "random"},
        },
    }

    result = try_apply_config(cfg, visualizer, defaults, logger)

    assert result is None
    assert len(visualizer.server.config_errors) == 1
    # on_game_config and config_options should NOT have been cleared
    # (the error path leaves those untouched so the UI can retry)
    assert visualizer.server.on_game_config is None  # was never set by this call
    assert visualizer.server.config_options is None  # was never set by this call


def test_import_error_sends_friendly_message(
    visualizer: StubVisualizer,
    defaults: dict,
    logger: Any,
) -> None:
    cfg = _random_vs_random_cfg()

    with patch.object(
        policies_module,
        "resolve_strategies",
        side_effect=ImportError("No module named 'sb3_contrib'"),
    ):
        result = try_apply_config(cfg, visualizer, defaults, logger)

    assert result is None
    assert len(visualizer.server.config_errors) == 1
    assert "training" in visualizer.server.config_errors[0]


def test_human_vs_human_shares_same_strategy_object(
    visualizer: StubVisualizer,
    defaults: dict,
    logger: Any,
) -> None:
    """Both human players should share the same strategy callable (memoised by policy id)."""
    cfg = {
        "settings": {"max_turns": 5, "turn_delay": 0.0},
        "players": {
            "p1": {"type": "human"},
            "p2": {"type": "human"},
        },
    }

    # monkeypatch make_human_strategy so it returns a plain lambda (avoids blocking)
    dummy_strategy = lambda possible_moves: possible_moves[0]  # noqa: E731

    with patch("Game.human_player.make_human_strategy", return_value=dummy_strategy):
        game = try_apply_config(cfg, visualizer, defaults, logger)

    assert game is not None
    assert game.player1.strategy is game.player2.strategy


def test_config_source_provided_returns_game_without_queue(
    visualizer: StubVisualizer,
    defaults: dict,
    logger: Any,
) -> None:
    """run_configured_game with config_source should return directly, no queue installed."""
    cfg = _random_vs_random_cfg(max_turns=15)

    game = run_configured_game(
        visualizer,
        defaults=defaults,
        logger=logger,
        config_source=cfg,
    )

    assert isinstance(game, Game)
    assert game.max_turns == 15
    # server.on_game_config was never set by run_configured_game
    assert visualizer.server.on_game_config is None


def test_max_turns_clamped_to_defaults_range(
    visualizer: StubVisualizer,
    defaults: dict,
    logger: Any,
) -> None:
    cfg = {
        "settings": {"max_turns": 9999, "turn_delay": 0.0},
        "players": {
            "p1": {"type": "computer", "policy_id": "random"},
            "p2": {"type": "computer", "policy_id": "random"},
        },
    }

    game = try_apply_config(cfg, visualizer, defaults, logger)

    assert game is not None
    assert game.max_turns == defaults["max_turns_max"]  # clamped to 200
