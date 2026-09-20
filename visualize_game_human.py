#!/usr/bin/env python3
"""Run a 3D game where one side is controlled via browser move selection."""

import argparse
from pathlib import Path

from Game.human_player import make_human_strategy
from Game.logging_utils import build_gameplay_logger
from Game.play_game import Game
from Game.sb3_strategy import make_sb3_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a human-vs-agent BJJ match in 3D.")
    parser.add_argument("--human-side", choices=["p1", "p2"], default="p1")
    parser.add_argument("--agent-type", choices=["random", "sb3"], default="random")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--max-turns", type=int, default=30)
    args = parser.parse_args()
    if args.agent_type == "sb3" and args.model_path is None:
        parser.error("--model-path is required when --agent-type sb3")
    return args


def main() -> None:
    args = parse_args()
    logger = build_gameplay_logger("Game.gameplay.visualize_game_human", to_stdout=True)

    game = Game("Human vs Agent", max_turns=args.max_turns, visualize_3d=True, logger=logger)
    game.initialize_game("Human", "Agent")

    human_player = game.player1 if args.human_side == "p1" else game.player2
    agent_player = game.player2 if human_player is game.player1 else game.player1

    human_player.strategy = make_human_strategy(game.visualizer.server, game.game_state)
    if args.agent_type == "sb3":
        agent_player.strategy = make_sb3_strategy(args.model_path, game)

    game.play_game()


if __name__ == "__main__":
    main()
