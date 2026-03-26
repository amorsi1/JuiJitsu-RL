#!/usr/bin/env python3
"""
Run a dummy BJJ game and visualize it in 3D with the render feature.
This will open your browser showing the 3D positions and transitions.
"""

import sys
import time
from pathlib import Path

# Add Game directory to path for imports
game_dir = Path(__file__).parent / "Game"
sys.path.insert(0, str(game_dir))

from play_game import Board, GameState, Player
from Graph.graph_constructor import construct_graph
from render.visualizer3d import Visualizer3D


def turn_color(player, red_player):
    return 'red' if player is red_player else 'blue'


def play_game_with_3d_render(max_turns: int = 20):
    """Run a simple game and visualize it in 3D."""

    # Initialize the 3D visualizer (opens browser automatically)
    print("Starting 3D visualizer...")
    visualizer = Visualizer3D(host='localhost', port=8765, open_browser=True)

    # Create game components
    board = Board(construct_graph())
    game_state = GameState(board)
    game_state.initialize()

    # Create players
    player1 = Player("Alice")
    player2 = Player("Bob")

    # Randomly assign top/bottom
    import random
    player1.is_top = random.choice([True, False])
    player1.is_bottom = not player1.is_top
    player2.is_top = not player1.is_top
    player2.is_bottom = not player1.is_bottom

    print(f"Alice is on {'top' if player1.is_top else 'bottom'}")
    print(f"Bob is on {'top' if player2.is_top else 'bottom'}")

    # Play the game
    current_player = random.choice([player1, player2])

    # Show initial position
    print(f"\nStarting position: {board.get_node_data(game_state.current_node)['description']}")
    visualizer.update(game_state.current_node, active_turn=turn_color(current_player, player1))
    time.sleep(0.5)

    for turn in range(1, max_turns + 1):
        print(f"\n--- Turn {turn} ---")

        # Get possible moves
        possible_moves = game_state.get_possible_moves(current_player.is_top, current_player.is_bottom)

        if not possible_moves:
            print(f"No moves available. Reinitializing...")
            game_state.initialize()
            visualizer.update(game_state.current_node, active_turn=turn_color(current_player, player1))
            time.sleep(0.5)
            continue

        # Pick a random move
        move = random.choice(possible_moves)
        points, player_tapped, swap_players = game_state.process_move(move)
        current_player.points += points
        next_player = player2 if current_player is player1 else player1
        winner = game_state.check_winner()
        next_turn = turn_color(current_player, player1) if (player_tapped or winner) else turn_color(next_player, player1)

        # Get move description
        to_node = move[0]
        to_desc = board.get_node_data(to_node)['description']
        print(f"{current_player.name} moved to: {to_desc}")

        if points > 0:
            print(f"  → Earned {points} points")

        # Let the browser consume transitions continuously instead of pausing
        # between moves in the driver loop.
        visualizer.update(to_node, transition_id=move[1]['id'], active_turn=next_turn)

        # Check for tap
        if player_tapped:
            winner = player2 if current_player is player1 else player1
            print(f"\n{current_player.name} tapped! {winner.name} wins!")
            break

        # Check for position win
        if winner:
            winning_player = player1 if ((player1.is_top and winner == 'top') or
                                        (player1.is_bottom and winner == 'bottom')) else player2
            print(f"\n{winning_player.name} won by position!")
            break

        # Swap players
        current_player = next_player

    # Print final scores
    print(f"\n=== Game Over ===")
    print(f"Alice: {player1.points} points")
    print(f"Bob: {player2.points} points")

    if player1.points > player2.points:
        print("Alice wins!")
    elif player2.points > player1.points:
        print("Bob wins!")
    else:
        print("It's a tie!")

    print("\nClosing visualization in 5 seconds...")
    time.sleep(5)
    visualizer.close()


if __name__ == "__main__":
    play_game_with_3d_render(max_turns=20)
