from play_game import Board, GameState, Player
import random
import networkx as nx
import numpy as np
from typing import List, Optional
from Graph.graph_constructor import construct_graph
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class DynamicGraphVisualizer:
    def __init__(self, game, full_graph: nx.Graph):
        self.game = game
        self.full_graph = full_graph
        self.traversed_graph = nx.Graph()
        self.path: List[int] = []

        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.fig.suptitle('BJJ Game Graph Traversal', fontsize=16)

        # Initialize empty graph elements
        self.nodes = nx.draw_networkx_nodes(self.traversed_graph, pos={}, ax=self.ax, node_size=300,
                                            node_color='lightblue')
        self.edges = LineCollection([])
        self.ax.add_collection(self.edges)
        self.labels = {}  # Initialize as an empty dictionary
        self.current_node = nx.draw_networkx_nodes(self.traversed_graph, pos={}, ax=self.ax, nodelist=[],
                                                   node_color='r', node_size=500)

        self.text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes,
                                 verticalalignment='top', fontsize=10,
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def update(self):
        current_node = self.game.game_state.current_node
        if not self.path or current_node != self.path[-1]:
            self.path.append(current_node)
            if current_node not in self.traversed_graph:
                self.traversed_graph.add_node(current_node)
                if len(self.path) > 1:
                    self.traversed_graph.add_edge(self.path[-2], self.path[-1])

        # Update layout and graph elements
        pos = nx.spring_layout(self.traversed_graph, k=1, iterations=50)

        self.nodes.set_offsets([pos[n] for n in self.traversed_graph.nodes()])

        edge_pos = np.array([(pos[e[0]], pos[e[1]]) for e in self.traversed_graph.edges()])
        self.edges.set_segments(edge_pos)

        # Update labels
        for node, (x, y) in pos.items():
            if node not in self.labels:
                self.labels[node] = self.ax.text(x, y, str(node), fontsize=8, ha='center', va='center')
            else:
                self.labels[node].set_position((x, y))

        self.current_node.set_offsets([pos[current_node]])

        # Update text
        node_data = self.game.game_state.board.get_node_data(current_node)
        text = f"Turn: {self.game.turn_count}\n"
        text += f"Current Position: {node_data['description']}\n"
        text += f"{self.game.player1.name}: {self.game.player1.points} points\n"
        text += f"{self.game.player2.name}: {self.game.player2.points} points"
        self.text.set_text(text)

        # Adjust plot limits
        if pos:
            x_values, y_values = zip(*pos.values())
            x_margin = (max(x_values) - min(x_values)) * 0.1
            y_margin = (max(y_values) - min(y_values)) * 0.1
            self.ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
            self.ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class Game:
    def __init__(self, name: str):
        self.name = name
        self.board = Board(construct_graph())
        self.game_state = GameState(self.board)
        self.turn_count = 0
        self.player1: Optional[Player] = None
        self.player2: Optional[Player] = None
        self.current_player: Optional[Player] = None
        self.winner = None
        self.visualizer = DynamicGraphVisualizer(self, construct_graph())


    def choose_other_player(self, player: Player) -> Player:
        if player is self.player1:
            return self.player2
        elif player is self.player2:
            return self.player1

    def initialize_game(self, p1_name: str, p2_name: str):
        print(f'Initializing game: {self.name}')
        self.game_state.initialize()
        self.player1 = Player(p1_name)
        self.player2 = Player(p2_name)
        self._randomly_assign_positions()
        self.current_player = random.choice([self.player1, self.player2])

    def _randomly_assign_positions(self):
        """
        Randomly chooses whether a player is in the top or bottom position, and give the other player the opposite
        position.
        """
        self.player1.is_top = random.choice([True, False])
        self.player1.is_bottom = not self.player1.is_top

        self.player2.is_top = not self.player1.is_top
        self.player2.is_bottom = not self.player1.is_bottom

        print(f'{self.player1.name} is on {"top" if self.player1.is_top else "bottom"}')
        print(f'{self.player2.name} is on {"top" if self.player2.is_top else "bottom"}')

    def _swap_players_positions(self):
        """
        gives each player the other players' top and bottom position attributes
        """
        cache = (self.player1.is_top, self.player1.is_top)
        self.player1.is_top, self.player1.is_top = self.player2.is_top, self.player2.is_top
        self.player2.is_top, self.player2.is_top = cache

    def play_turn(self) -> bool:
        possible_moves = self.game_state.get_possible_moves(self.current_player.is_top, self.current_player.is_bottom)
        if not possible_moves:
            # if current state is a terminal node, but not associated with a win or loss
            if not self.game_state.board.get_outgoing_edges(self.game_state.current_node):
                # change to random node, then allow player to play their turn
                print('Terminal position encountered. switching to random position ')
                self.game_state.initialize()
                self.play_turn()
            else:
                print(f"No moves available for {self.current_player.name}. Switching players.")
                # note: maybe this shouldn't conclude the turn, and instead should switch players then call play_turn again
                return self._switch_players()
        else:
            move = self.current_player.choose_move(possible_moves)
            points, player_tapped, swap_players_positions = self.game_state.process_move(move)
            self.current_player.points += points
            print(f"{self.current_player.name} performed '{move[1]['description']}'")
            if points>0:
                print(f'Player earned {points} points for that move')

            # Update the visualization
            self.visualizer.update()

            if player_tapped:
                winning_player = self.choose_other_player(self.current_player)
                print(f"{self.current_player.name} tapped - {winning_player.name} has won! ")
                self.winner = winning_player
                return True

            winner = self.game_state.check_winner()
            if winner:
                # arriving at this node means that one of the players has won already
                winning_player = self.player1 if ((self.player1.is_top and winner == 'top') or
                                                  (self.player1.is_bottom and winner == 'bottom')) else self.player2
                self.winner = winning_player
                print(f"{winning_player.name} won by reaching a winning position!")
                return True

            if swap_players_positions:
                self._swap_players_positions()

            return self._switch_players()

    def _switch_players(self) -> bool:
        self.current_player = self.player2 if self.current_player is self.player1 else self.player1
        return False

    def check_for_points_win(self):
        if self.player1.points > self.player2.points:
            self.winner = self.player1
            print(f"{self.player1.name} wins!")
        elif self.player2.points > self.player1.points:
            self.winner = self.player2
            print(f"{self.player2.name} wins!")
        else:
            print("It's a tie!")

    def play_game(self, max_turns: int = 100):
        plt.ion()  # Turn on interactive mode
        self.visualizer.fig.show()
        for turn in range(1, max_turns + 1):
            self.turn_count += 1
            print(f"\nTurn {turn}:")
            if self.play_turn():
                break
        if not self.winner:
            self.check_for_points_win()
        self._print_game_result()

        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the final plot open

    def _print_game_result(self):
        print("\nGame over! Final scores:")
        print(f"{self.player1.name}: {self.player1.points}")
        print(f"{self.player2.name}: {self.player2.points}")

# Single game example
game = Game("BJJ Simulation")
game.initialize_game("Player 1", "Player 2")
game.play_game()