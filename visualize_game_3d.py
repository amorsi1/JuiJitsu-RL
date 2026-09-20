#!/usr/bin/env python3
"""
Run a dummy BJJ game and visualize it in 3D with the render feature.
This will open your browser showing the 3D positions and transitions.
"""

import time

from Game.logging_utils import build_gameplay_logger
from Game.play_game import Game


def play_game_with_3d_render(max_turns: int = 20):
    """Run a simple game and visualize it in 3D."""
    logger = build_gameplay_logger("Game.gameplay.visualize_game_3d", to_stdout=True)
    logger.info("Starting 3D visualizer...")

    game = Game(
        "3D Demo BJJ Match",
        max_turns=max_turns,
        visualize_3d=True,
        logger=logger,
        turn_delay=2.0,
    )
    game.initialize_game("Alice", "Bob")
    game.play_game()
    time.sleep(5.0)


if __name__ == "__main__":
    play_game_with_3d_render(max_turns=20)
