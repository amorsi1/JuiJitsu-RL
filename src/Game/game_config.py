"""Game configuration flow: applies a browser config payload to create a ready-to-run Game."""

from __future__ import annotations

import logging
import queue
import random
from typing import Any, Optional


def try_apply_config(
    cfg: dict,
    visualizer: Any,
    defaults: dict,
    logger: logging.Logger,
) -> Optional[Any]:
    """Validate and apply a game_config payload, returning a configured Game or None on error.

    Args:
        cfg: Raw config payload from the browser (game_config WebSocket message).
        visualizer: A Visualizer3D (or compatible stub) with a `.server` and `.turn_delay`.
        defaults: Mapping with keys max_turns_default, max_turns_min, max_turns_max,
                  and turn_delay_default.
        logger: Logger for error/info messages.

    Returns:
        A fully configured, initialised Game ready to call .play_game() on, or None if
        config validation failed (in which case a config_error was sent to the browser).
    """
    settings = cfg.get("settings", {})

    # --- max_turns ---
    try:
        raw_turns = settings.get("max_turns", defaults["max_turns_default"])
        max_turns = int(raw_turns)
    except (TypeError, ValueError):
        max_turns = defaults["max_turns_default"]
    max_turns = max(defaults["max_turns_min"], min(defaults["max_turns_max"], max_turns))

    # --- turn_delay ---
    try:
        raw_delay = settings.get("turn_delay", defaults["turn_delay_default"])
        turn_delay = float(raw_delay)
    except (TypeError, ValueError):
        turn_delay = float(defaults["turn_delay_default"])

    # --- build Game ---
    from Game.play_game import Game

    game = Game("BJJ Match", max_turns=max_turns, visualizer=visualizer)

    # --- resolve strategies (may raise ValueError or ImportError) ---
    from Game.policies import StrategyContext, resolve_strategies

    ctx = StrategyContext(game=game, server=visualizer.server)
    try:
        strategies = resolve_strategies(cfg["players"], ctx)
    except ValueError as exc:
        visualizer.server.send_config_error(str(exc))
        logger.error("Config error (invalid policy): %s", exc)
        return None
    except ImportError as exc:
        visualizer.server.send_config_error(
            f"{exc} — try: uv sync --extra training"
        )
        logger.error("Config error (missing dependency): %s", exc)
        return None

    # --- apply settings ---
    visualizer.server.on_game_config = None
    visualizer.server.set_config_options(None)
    visualizer.turn_delay = turn_delay

    # Reseed from OS entropy so strategy loading (e.g. SB3's set_random_seed)
    # cannot pin the starting position to the same node on every Play Again.
    random.seed()
    game.initialize_game("Player 1", "Player 2")
    game.player1.strategy = strategies["p1"]
    game.player2.strategy = strategies["p2"]

    return game


def run_configured_game(
    visualizer: Any,
    *,
    defaults: dict,
    logger: logging.Logger,
    config_source: Optional[dict] = None,
) -> Any:
    """Block until a valid game config arrives, then return a ready-to-run Game.

    If `config_source` is provided the config is applied immediately without any
    queue or browser interaction.  Otherwise the function installs a listener on
    `visualizer.server.on_game_config`, broadcasts the config options UI to the
    browser, and loops until the browser sends a valid config.

    Args:
        visualizer: A Visualizer3D (or compatible stub).
        defaults: See try_apply_config.
        logger: Logger for status messages.
        config_source: If given, bypass the browser config loop entirely.

    Returns:
        A configured Game instance ready for .play_game().
    """
    if config_source is not None:
        game = try_apply_config(config_source, visualizer, defaults, logger)
        assert game is not None, "try_apply_config returned None for a direct config_source"
        return game

    from Game.policies import build_config_options

    options = build_config_options(
        max_turns_default=defaults["max_turns_default"],
        turn_delay_default=defaults["turn_delay_default"],
    )

    pending: queue.Queue[dict] = queue.Queue()
    visualizer.server.on_game_config = pending.put
    visualizer.server.set_config_options(options)

    while True:
        cfg = pending.get()
        game = try_apply_config(cfg, visualizer, defaults, logger)
        if game is None:
            # Error already sent to browser; re-arm the handler and wait for a retry.
            visualizer.server.on_game_config = pending.put
            continue
        return game
