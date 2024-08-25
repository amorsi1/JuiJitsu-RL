import random
import networkx as nx
from typing import List, Tuple, Dict, Optional
from Graph.graph_constructor import construct_graph

class Board:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.rewards = {
            'sweep': 2, 'mount': 4, 'back': 4,
            'throw': 2, 'takedown': 4, 'pass': 3}

    def get_node_data(self, node: int) -> Dict:
        return self.graph.nodes[node]

    def get_edge_data(self, from_node: int, to_node: int) -> Dict:
        return self.graph.edges[from_node, to_node]

    def get_outgoing_edges(self, node: int) -> List[Dict]:
        return self.graph.nodes[node]['outgoing']

class GameState:
    def __init__(self, board: Board):
        self.board = board
        self.current_node = None

    def initialize(self):
        # position 94 is 'symmetric staggered standing'. A central node with many possible outgoing edges
        self.current_node = random.choice([94, random.choice(list(self.board.graph.nodes()))])

    def update(self, new_node: int):
        print(f'moving to position {self.board.get_node_data(new_node)['description']}')
        self.current_node = new_node

    def get_possible_moves(self, is_top: bool, is_bottom: bool) -> List[Tuple[int, Dict]]:
        #note: May not need to pass in both top and bottom position? need to explicitly test that they are always opposites
        #before any third state ([True,True] or [False,False]) will create issues if both attributes aren't passed in
        """
        Passes in Player's top/bottom position and calculates which moves are valid for the relative position of the player

        Note that this logic assumes the player can perform all moves, then filters out ones that are not possible in
        that position (i.e. top moves from bottom position and vice versa). This is an important distinction because it means
        that any player can do a move that doesn't have a top or bottom tag on it regardless of their position
        """
        possible_moves = []
        for outgoing_edge in self.board.get_node_data(self.current_node)['outgoing']:
            # player is on top position but the move is for bottom position
            if is_top and outgoing_edge['bottom']:
                pass
            # player is on bottom position but the move is for top position
            elif is_bottom and outgoing_edge['top']:
                pass
            # no conflicts found, the move is valid
            else:
                move = (outgoing_edge['to'], self.board.get_edge_data(outgoing_edge['from'], outgoing_edge['to']))
                possible_moves.append(move)

        return possible_moves

    def process_move(self, move: Tuple[int, Dict]) -> tuple[int, bool, bool]:
        new_node, edge_data = move
        points = self._calculate_points(edge_data)
        player_tapped = edge_data.get('tap', False)
        swap_players = edge_data.get('swaps_players', False)
        self.update(new_node)
        return points, player_tapped, swap_players

    def _calculate_points(self, edge_data: Dict) -> int:
        earned_points = 0  # iteratively add to this because a player may execute multiple maneuvers in the same move
        for maneuver, points in self.board.rewards.items():
            if edge_data.get(maneuver, False):
                print(f'{maneuver} executed, player wins {points} points')
                earned_points += points
        return earned_points
    def check_winner(self) -> Optional[str]:
        node_data = self.board.get_node_data(self.current_node)
        return node_data.get('winner')

class Player:
    # do I need the strategy property? revisit this
    def __init__(self, name: str, strategy: str = 'random'):
        self.name = name
        self.is_top = False
        self.is_bottom = False
        self.points = 0
        self.strategy = strategy

    def choose_move(self, possible_moves: List[Tuple[int, Dict]]) -> Tuple[int, Dict]:
        if self.strategy == 'random':
            return random.choice(possible_moves)
        # Implement other strategies here
        return random.choice(possible_moves)

# class MoveProcessor:
#     def __init__(self, game_state: GameState):
#         self.game_state = game_state
#
#     def get_possible_moves(self, player: Player) -> List[Tuple[int, Dict]]:
#         """Get all possible moves for the current player."""
#         node_data = self.game_state.get_node_data()
#         position = 'top' if player.is_top else 'bottom'
#         possible_moves = []
#         for edge in node_data['outgoing']:
#             edge_data = self.game_state.graph.edges[self.game_state.current_node, edge['transition']]
#             if position in edge_data.get('properties', []):
#                 possible_moves.append((edge['transition'], edge_data))
#         return possible_moves
#
#     def process_move(self, player: Player, move: Tuple[int, Dict]) -> Tuple[int, bool]:
#         """Process the chosen move and return the points earned and if the game ended."""
#         new_node, edge_data = move
#         points = 0
#         game_ended = False
#
#         # Check for point-earning moves
#         for move_type in ['sweep', 'mount', 'back', 'throw', 'takedown', 'pass']:
#             if edge_data.get(move_type, False):
#                 points += 2  # Assuming all moves are worth 2 points for simplicity
#
#         # Check for submission
#         if edge_data.get('tap', False):
#             game_ended = True
#
#         self.game_state.update(new_node)
#         return points, game_ended

class Game:
    def __init__(self, name: str, graph: nx.Graph):
        self.name = name
        self.board = Board(graph)
        self.game_state = GameState(self.board)
        self.player1: Optional[Player] = None
        self.player2: Optional[Player] = None
        self.current_player: Optional[Player] = None
        self.winner = None

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
                return self._switch_players()

        move = self.current_player.choose_move(possible_moves)
        points, player_tapped, swap_players_positions = self.game_state.process_move(move)
        self.current_player.points += points
        print(f"{self.current_player.name} made a move and earned {points} points.")

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
        for turn in range(1, max_turns + 1):
            print(f"\nTurn {turn}:")
            if self.play_turn():
                break
        if not self.winner:
            self.check_for_points_win()
        self._print_game_result()

    def _print_game_result(self):
        print("\nGame over! Final scores:")
        print(f"{self.player1.name}: {self.player1.points}")
        print(f"{self.player2.name}: {self.player2.points}")


# Usage
graph = construct_graph()
game = Game("BJJ Simulation", graph)
game.initialize_game("Player 1", "Player 2")
game.play_game()