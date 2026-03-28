#!/usr/bin/env python3
"""
Simple script to run and visualize a single dummy BJJ game.
Run this from the project root: python visualize_game.py
"""

from Game.play_game_visualizer import Game

def main():
    # Create a game
    game = Game("Demo BJJ Match")

    # Initialize with two players
    game.initialize_game("Alice", "Bob")

    # Play the game with visualization (max 50 turns for a quick demo)
    print("Starting game visualization...")
    game.play_game(max_turns=50)

if __name__ == "__main__":
    main()
