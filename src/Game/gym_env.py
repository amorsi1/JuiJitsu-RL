import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .play_game import Game, Board, GameState, Player, tqdm
from typing import List, Tuple, Dict, Optional, Any
import random
def bool_to_int(value: bool) -> int:
    return 1 if value else 0
class BJJEnv(gym.Env):
    def __init__(self):
        self.game = Game("BJJ Match")
        self.game.initialize_game("Player1", "Player2")
        self.G = self.game.board.graph

        # Get edge IDs and create a mapping
        self.edge_ids = [data['id'] for _, _, data in self.G.edges(data=True)]
        self.id_to_index = {id: index for index, id in enumerate(self.edge_ids)}
        self.index_to_id = {index: id for index, id in enumerate(self.edge_ids)}
        self.edge_id_to_nodes = {data['id']: (start, end) for start, end, data in self.G.edges(data=True)}

        # get node IDs
        self.num_nodes = max(self.G.nodes()) + 1  # nodes are 0-indexed; +1 gives count and ensures Discrete covers all IDs
        # Define action space
        self.action_space = spaces.Discrete(len(self.edge_ids))
            # 'from_node': spaces.Discrete(num_nodes),
            # 'to_node': spaces.Discrete(num_nodes),
            # 'is_top': spaces.Discrete(2),
            # 'is_bottom': spaces.Discrete(2),
            # 'move_type': spaces.MultiBinary(7),
            # 'swaps_players': spaces.Discrete(2),
            # })

        # State space (not directly used by the agent)
        self.state_space = spaces.Dict({
            "position": spaces.Discrete(self.num_nodes),
            "current player": spaces.Discrete(2), # Player1 is 0 and Player2 is 1
            "current player on top": spaces.Discrete(2),  # 0 is False and 1 is True
            'player1_score': spaces.Box(low=0, high=200, shape=(1,), dtype=np.int32),
            'player2_score': spaces.Box(low=0, high=200, shape=(1,), dtype=np.int32),
            "turn_num": spaces.Box(low=0, high=self.game.max_turns, shape=(1,), dtype=np.int32)
        })

        # Define observation space
        # Note: SB3 will want a flat obervation space. In the future this may be consolidated into something like:

        self.observation_space = spaces.Box(                                                                                                
        low=np.array([0, -1e4, 0, 0, 0], dtype=np.float32),                                                                             
        high=np.array([self.num_nodes, # Position (Node ID)
                        1e4, # Point difference (can be negative if behind)
                        1, # on top (binary)
                        1, # on bottom (binary)
                        self.game.max_turns # number of turns left (can be 0 at end of game)
                    ], dtype=np.float32) 
        )


    def _get_state(self):
        # Return the full state (not directly used by the agent)
        return {
            "position": self.game.game_state.current_node,
            "current player": 0 if self.game.current_player is self.game.player1 else 1,
            "current player on top": 1 if self.game.current_player.is_top else 0,
            "player1_score": self.game.player1.points,
            "player2_score": self.game.player2.points,
            "turn_num": self.game.turn_count,
        }

    def _get_obs(self) -> np.ndarray:
        # Observation order: [current_position, point_difference, on_top, on_bottom, turns_left]
        current_player = self.game.current_player
        other_player = self.game.choose_other_player(current_player)

        return np.array([
            self.game.game_state.current_node,
            current_player.points - other_player.points,
            bool_to_int(current_player.is_top),
            bool_to_int(current_player.is_bottom),
            self.game.max_turns - self.game.turn_count
        ], dtype=np.float32)


    def _get_action_mask(self) -> np.ndarray:
        """
        Creates action mask for the current player based on the legal moves available to them

        Returns:
        np.ndarray: A boolean array of shape (n,) where n is the total number of moves in the game (~700).
                    Each element is either 0 (False) or 1 (True), where:
                    - 1 (True) indicates a legal move
                    - 0 (False) indicates an illegal move
        """
        possible_moves = self.game.game_state.get_possible_moves(
            self.game.current_player.is_top,
            self.game.current_player.is_bottom
        )
        mask = np.zeros(len(self.edge_ids), dtype=bool)  # mask is length of all possible actions in the entire game
        for move in possible_moves:
            #sets only the legal moves to 1
            edge_id = move[1]['id']
            mask[self.id_to_index[edge_id]] = 1
        return mask

    def action_masks(self) -> np.ndarray:
        """Public interface for sb3-contrib MaskablePPO action masking."""
        return self._get_action_mask()

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        super().reset(seed=seed)  # Seeds self.np_random
        if seed is not None:
            random.seed(seed)  # Also seed Python's random, used by Game internals

        # Fully reset the game
        self.game = Game("BJJ Match")  # Create a new game instance
        self.game.board = Board(self.G)  # Reset the board with the graph
        self.game.game_state = GameState(self.game.board)  # Reset the game state
        self.game.turn_count = 0

        self.game.initialize_game("Player1", "Player2")
        while not any(self._get_action_mask()):
            # Reinitialize if there are no valid moves for the starting position
            self.game.initialize_game("Player1", "Player2")

        return self._get_obs(), {"action_mask": self._get_action_mask()}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        edge_id = self.index_to_id[action]
        if edge_id >= len(self.edge_ids):
            # Invalid action, end turn without making a move
            self.game.switch_players()
            return self._get_obs(), -1, False, False, {}
        (start, end) = self.edge_id_to_nodes[edge_id]
        move = (start, self.game.board.get_edge_data(start, end))
        self.game.play_turn(move)
        self.game.turn_count += 1

        # terminated: game ended naturally (submission or position win)
        terminated = self.game.winner is not None
        # truncated: turn limit reached; check points to determine a winner if possible
        truncated = (not terminated) and (self.game.turn_count >= self.game.max_turns)

        if truncated:
            self.game.check_for_points_win()
            if self.game.winner is not None:
                terminated = True
                truncated = False

        obs = self._get_obs()
        reward = self._calculate_reward(obs)

        return obs, reward, terminated, truncated, {"action_mask": self._get_action_mask()}


    def _calculate_reward(self, obs) -> float:
        reward = 0
        other_player = self.game.choose_other_player(self.game.current_player)
        if self.game.winner == self.game.current_player:
            reward += 300
        elif self.game.winner == other_player:
            reward -= 300

        # TODO: currently rewards the cumulative score gap as a heuristic for being ahead of the opponent.
        # Consider switching to a marginal delta (points earned this turn only) to more precisely credit
        # the specific action that scored.
        reward += 1*obs[1]   # point_difference
        reward += 0.5*obs[2]  # on_top
        return reward
    def render(self, mode='human'):
        print(
            f"Current position: {self.game.game_state.board.get_node_data(self.game.game_state.current_node)['description']}")
        print(f"Player1 points: {self.game.player1.points}, Player2 points: {self.game.player2.points}")
        print(f"Player1 is on {'top' if self.game.player1.is_top else 'bottom'}")
        print(f"Turn count: {self.game.turn_count}")


def state_to_index(state=None, position=None, is_top=None) -> int:
    """
    Inputs:
        state (Dict[str, Any], optional): An instance of the agent's observation as returned by _get_obs.
            Contains position and is_top values
        position (int, optional): current node
        is_top (int, optional): relative top/bottom position of current player

    Identifies unique index value for Q-table that encodes absolute position of current player and top/bottom position
    Optional parameter lets you explicitly pass in an observation to identify the index, but will return the
    current state's observation by default
    """
    if state is not None and isinstance(state, np.ndarray):
        # Flat obs array: [current_position, point_difference, on_top, on_bottom, turns_left]
        return int(state[0]) * 2 + int(state[2])
    elif state is not None and isinstance(state, dict):
        return state['current_position'] * 2 + state['on_top']
    elif position is not None and is_top is not None:
        return position * 2 + is_top
    else:
        raise ValueError('no values passed in')


def get_masked_q_values(q_values: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """
    Apply an action mask to Q-values.

    Args:
    q_values (np.ndarray): 1D array of Q-values for all actions in a state.
                           Shape: (n,) where n is the number of moves.
    action_mask (np.ndarray): 1D binary boolean array indicating valid (1) and invalid (0) actions.
                              Shape: (n,) where n is the number of moves.

    Returns:
    np.ndarray: 1D array of masked Q-values with invalid actions set to -infinity.
                Shape: (n,) where n is the number of possible moves.
    """
    assert q_values.shape == action_mask.shape, \
        'Q-values and action masks lengths need to have the same shape for accurate element-wise operations'
    return np.where(action_mask, q_values, -np.inf)
def q_learning(env: BJJEnv, num_episodes, learning_rate=0.1, discount_factor=0.95,
               epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
    """
    initializes Q-table to have dimensions of num_states x num_actions, where each unique
    combination of current node and relative position constitutes a unique state (e.g. node 237
    and current player on top vs node 237 and current player on bottom)

    Note that this is a significantly lower dimensional state space that the observed BJJEnv one, and doesn't encode
    information that could potentially change how the agent acts, such as how many turns are left or the point
    difference between players.

    Args:
        epsilon: initial exploration rate (default 1.0 — fully exploratory at start)
        epsilon_min: floor for epsilon after decay (default 0.05)
        epsilon_decay: multiplicative decay applied after each episode (default 0.995)
    """

    # Initialize Q-table
    num_states = env.num_nodes * 2   # positions * (top/bottom)
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))

    for episode in tqdm(range(num_episodes)):
        state_obs, info = env.reset()
        done = False

        while not done:
            if not any(info['action_mask']):
                print('no valid moves available. Ending episode')
                break

            state_index = state_to_index(state_obs)

            if np.random.random() < epsilon:
                # exploratory move. Action space masked so that randomly chosen move is not invalid
                action = np.random.choice(np.where(info['action_mask'])[0])
            else:
                # sets all invalid moves to Q-value of -np.inf to avoid them being selected
                masked_q_values = get_masked_q_values(q_table[state_index,:], info['action_mask'])
                action = np.argmax(masked_q_values)

            next_state_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state_index = state_to_index(next_state_obs)

            # Q-learning update:
            # - On termination (tap/win), there is no next state to bootstrap from.
            # - On truncation (turn limit), the episode was cut short; bootstrap normally.
            if terminated:
                td_target = reward
            else:
                next_masked_q_values = get_masked_q_values(q_table[next_state_index], info['action_mask'])
                best_next_action = np.argmax(next_masked_q_values)
                td_target = reward + discount_factor * q_table[next_state_index][best_next_action]

            td_error = td_target - q_table[state_index][action]
            q_table[state_index][action] += learning_rate * td_error

            state_obs = next_state_obs

        # Decay exploration rate at the end of each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table

class QLearningAgent:
    def __init__(self, env: BJJEnv, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay

        self.state_space = self._initialize_state_space()
        self.action_space = self.env.action_space
        self.q_table = np.zeros((len(self.state_space), self.action_space.n))

    def _initialize_state_space(self) -> List[int]:
        return list(self.board.graph.nodes())

    def get_q_value(self, action: int, state=None, is_top=None) -> float:
        """
        inputs:
            action [int] : edge index associated with
            state [int, optional] : current node of the game. If not provided will be found by _get_obs
            is_top [int, optional] : Relative position of current player. If not provided will be found by _get_obs
        Outputs:
            Q-value from the Q-table indexed by the action and state
        """
        assert self.q_table is not None, \
            "Q-table is empty, ensure you train or provide a Q-table before using the strategy"
        if state and is_top:
            state_index = state_to_index(state, is_top)
            return self.q_table[state_index, action]
        else:
            state_obs = self.env._get_obs()
            state_index = state_to_index(state_obs)
            return self.q_table[state_index, action]

    def choose_action(self, state: int, possible_moves: List[Tuple[int, Dict]]) -> Tuple[int, Dict]:
        # Explore
        if np.random.rand() <= self.exploration_rate:
            return random.choice(possible_moves)

        # Exploit
        move_indices = []
        for move in possible_moves:
            move_index = self.env.id_to_index[move[1]['id']]
            move_indices.append(move_index)

        q_values = [self.get_q_value(move) for move in move_indices]
        max_q = max(q_values)

        best_moves = [move for move, q in zip(possible_moves, q_values) if q == max_q] #to break any potential ties
        #TODO: Consider removing best_moves tie break or making it faster
        return random.choice(best_moves)

    def update(self, state: int, action: int, reward: float, next_state: int,
               next_possible_moves: List[Tuple[int, Dict]]):
        current_q = self.get_q_value(action)
        if next_possible_moves:
            max_next_q = max(self.get_q_value(next_state, move[0]) for move in next_possible_moves)
        else:
            max_next_q = 0

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q

        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)


if __name__ == '__main__':
    env = BJJEnv()
    q_table = q_learning(env, num_episodes=100)

    # Test the learned policy
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_index = state_to_index(state)
        masked_q_values = get_masked_q_values(q_table[state_index], info['action_mask'])
        action = np.argmax(masked_q_values)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Total reward: {total_reward}")