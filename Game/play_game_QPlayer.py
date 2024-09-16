from play_game import Board, GameState, construct_graph
import random
import copy
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


class Player:
    def __init__(self, name: str, strategy: str = 'random'):
        self.name = name
        self.is_top = False
        self.is_bottom = False
        self.points = 0
        self.strategy = strategy
        self.agent = None  # Will be set to a QLearningAgent if strategy is 'q_learning'

    def choose_move(self, possible_moves: List[Tuple[int, Dict]]) -> Tuple[int, Dict]:
        if self.strategy == 'random':
            return random.choice(possible_moves)
        elif self.strategy == 'q_learning':
            return self.agent.choose_action(possible_moves)
        # Implement other strategies here
        return random.choice(possible_moves)


class QLearningAgent:
    def __init__(self, board: Board, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995):
        self.board = board
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay

        self.state_space = self._initialize_state_space()
        self.action_space = self._initialize_action_space()
        self.q_table = {}

    def _initialize_state_space(self) -> List[int]:
        return list(self.board.graph.nodes())

    def _initialize_action_space(self) -> Dict[int, List[int]]:
        action_space = {}
        for edge in self.board.graph.edges(data=True):
                processed_dict = copy.deepcopy(edge[2])
                # Change 'from' and 'to' to only contain the 'node' number
                processed_dict['from'] = move_dict['from']['node']
                processed_dict['to'] = move_dict['to']['node']
                # Remove the 'line_nr' item
                processed_dict.pop('line_nr', None)

                action_space[processed_dict['id']] = processed_dict
        return action_space

    def get_q_value(self, state: int, action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state: int, possible_moves: List[Tuple[int, Dict]]) -> Tuple[int, Dict]:
        if np.random.rand() <= self.exploration_rate:
            return random.choice(possible_moves)

        q_values = [self.get_q_value(state, move[0]) for move in possible_moves]
        max_q = max(q_values)
        best_moves = [move for move, q in zip(possible_moves, q_values) if q == max_q]
        return random.choice(best_moves)

    def update(self, state: int, action: int, reward: float, next_state: int,
               next_possible_moves: List[Tuple[int, Dict]]):
        current_q = self.get_q_value(state, action)
        if next_possible_moves:
            max_next_q = max(self.get_q_value(next_state, move[0]) for move in next_possible_moves)
        else:
            max_next_q = 0

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)


class QLearningSim:
    def __init__(self, num_games: int, q_agent: QLearningAgent):
        self.num_games = num_games
        self.q_agent = q_agent
        self.games = []
        self.results = []

    def initialize_games(self):
        print('Initializing games')
        self.games = [Game(f"Game_{i}") for i in range(self.num_games)]
        for game in tqdm(self.games):
            game.initialize_game("Q-Learning Player", "Random Player")
            game.player1.strategy = 'q_learning'
            game.player1.agent = self.q_agent
            game.player2.strategy = 'random'

    def run_games(self, max_turns: int = 100):
        print('Running games')
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.play_single_game, game, max_turns) for game in self.games]
            for future in tqdm(as_completed(futures)):
                self.results.append(future.result())

    def play_single_game(self, game: Game, max_turns: int) -> Dict:
        for _ in range(max_turns):
            if game.play_turn():
                break

        return {
            'game_name': game.name,
            'player1_name': game.player1.name,
            'player2_name': game.player2.name,
            'winner': game.winner.name if game.winner else 'Tie',
            'player1_points': game.player1.points,
            'player2_points': game.player2.points,
            'num_turns': game.turn_count
        }

    def agg_results(self) -> List[Dict]:
        print("SIMULATION RESULTS:")
        q_learning_wins = 0
        random_wins = 0
        num_ties = 0
        for result in self.results:
            if result['winner'] == "Q-Learning Player":
                q_learning_wins += 1
            elif result['winner'] == "Random Player":
                random_wins += 1
            else:
                num_ties += 1

        print(f'Q-Learning Player won {q_learning_wins} games out of {len(self.results)} ({round(q_learning_wins/len(self.results),2)})')
        print(f'Random Player won {random_wins} games out of {len(self.results)} ({round(random_wins/len(self.results),2)})')
        if num_ties > 0:
            print(f'There were {num_ties} ties')

        num_turns = [result['num_turns'] for result in self.results]
        print(f'On average, games lasted {np.mean(num_turns)} turns with a min of {np.min(num_turns)} and a max of {np.max(num_turns)}')

        return self.results

    def reset(self):
        self.games = []
        self.results = []

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
        current_player = self.player1 if self.current_player is self.player1 else self.player2
        state = self.game_state.current_node
        possible_moves = self.game_state.get_possible_moves(current_player.is_top, current_player.is_bottom)
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
            action = current_player.choose_move(state, possible_moves)
            points, player_tapped, swap_players_positions = self.game_state.process_move(action)
            current_player.points += points
            print(f"{current_player.name} performed '{action[1]['description']}'")
            if points > 0:
                print(f'Player earned {points} points for that move')

            next_state = self.game_state.current_node
            reward = points + (100 if player_tapped else 0)  # Simple reward function

            if current_player.strategy == 'q_learning':
                next_possible_moves = self.game_state.get_possible_moves(
                    not current_player.is_top if swap_players_positions else current_player.is_top,
                    not current_player.is_bottom if swap_players_positions else current_player.is_bottom
                )
                current_player.agent.update(state, action[0], reward, next_state, next_possible_moves)

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

            self.turn_count += 1
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
            self.turn_count += 1
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
board = Board(construct_graph())
state_size = len(board.graph)
action_size = sum(len(board.get_outgoing_edges(node)) for node in board.graph.nodes())
q_agent = QLearningAgent(state_size, action_size)
sim = QLearningSim(num_games=1000, q_agent=q_agent)
sim.run_games()
sim.print_results()

