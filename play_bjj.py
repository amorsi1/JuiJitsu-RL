#!/usr/bin/env python3
"""Entry point for a configurable BJJ match with browser UI."""

import argparse
import time
from pathlib import Path

from Game.game_config import run_configured_game
from Game.logging_utils import build_gameplay_logger
from Game.policies import discover_sb3_policies, register_policy
from Graph.graph_constructor import construct_graph
from render.visualizer3d import Visualizer3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a configurable BJJ match with browser UI.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--turn-delay", type=float, default=0.0)
    parser.add_argument(
        "--skip-gui",
        action="store_true",
        help="Skip the config overlay and use --p1/--p2 policy IDs directly.",
    )
    parser.add_argument("--p1", type=str, default="random", help="Policy ID for player 1.")
    parser.add_argument("--p2", type=str, default="random", help="Policy ID for player 2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = build_gameplay_logger("Game.gameplay.play_bjj", to_stdout=True)

    # Build the NetworkX graph in the background so the "Start Game" click is instant.
    # Runs in parallel with Visualizer3D startup (which loads the 3D viewer data).
    _graph_prefetch = threading.Thread(target=construct_graph, daemon=True, name="graph-prefetch")
    _graph_prefetch.start()

    # Register any SB3 model checkpoints found under models_dir.
    try:
        for spec in discover_sb3_policies(args.models_dir):
            try:
                register_policy(spec)
            except ValueError:
                pass  # already registered (idempotent on repeated calls)
    except ImportError:
        pass  # sb3 not installed; skip silently

    visualizer = Visualizer3D(
        port=args.port,
        open_browser=not args.no_browser,
    )

    defaults = {
        "max_turns_default": args.max_turns,
        "max_turns_min": 1,
        "max_turns_max": 500,
        "turn_delay_default": args.turn_delay,
    }

    if args.skip_gui:
        config_source: dict | None = {
            "settings": {
                "max_turns": args.max_turns,
                "turn_delay": args.turn_delay,
            },
            "players": {
                "p1": {"type": "computer", "policy_id": args.p1},
                "p2": {"type": "computer", "policy_id": args.p2},
            },
        }
    else:
        config_source = None

    game = run_configured_game(
        visualizer,
        defaults=defaults,
        logger=logger,
        config_source=config_source,
    )

    game.play_game()

    if game.winner is not None:
        winner_index = 0 if game.winner is game.player1 else 1
        visualizer.server.send_announcement(
            kind="win",
            winner_index=winner_index,
            win_type=game.win_reason,
        )

    time.sleep(2)
    visualizer.close()


if __name__ == "__main__":
    main()
