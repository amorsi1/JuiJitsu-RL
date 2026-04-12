import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
from .logging_utils import build_gameplay_logger
from .play_game import Game, Board, GameState, Player, tqdm
from typing import Any, Dict, List, Tuple, Optional
import random
import time

from render.graph_renderer import MoveRecord
from render.hud_announcements import AnnouncementEvent, WinType


POINT_FLASH_DURATION: float = 2.0
def bool_to_int(value: bool) -> int:
    return 1 if value else 0


class BJJEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi", "graph"], "render_fps": 2, "render_transition_fps": 15}

    def __init__(self, render_mode: str | None = None, gameplay_log_path: Path | None = None) -> None:
        # Initialize renderers
        self.render_mode = render_mode
        self._renderer: object | None = None  # lazy FrameRenderer
        self._graph_renderer: object | None = None  # lazy GraphRenderer
        self._pending_transition_id: int | None = None
        self._pending_transition_target_node: int | None = None
        self._pending_announcement_event: AnnouncementEvent | None = None
        self._point_flash_p1: tuple[str, float] | None = None  # (message, timestamp)
        self._point_flash_p2: tuple[str, float] | None = None

        # Initialize gameplay logger
        self.gameplay_log_path = Path(gameplay_log_path) if gameplay_log_path is not None else None
        self.gameplay_logger = build_gameplay_logger(
            f"Game.gameplay.env.{id(self)}",
            to_stdout=(self.render_mode is not None),
            file_path=self.gameplay_log_path,
        )

        # Initialize game and graph
        self.game = Game("BJJ Match", logger=self.gameplay_logger)
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

        # Define vectorized observation space
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
        np.ndarray: An int8 array of shape (n,) where n is the total number of moves in the game (~700).
                    Each element is 0 or 1, where:
                    - 1 indicates a legal move
                    - 0 indicates an illegal move
        """
        possible_moves = self.game.game_state.get_possible_moves(
            self.game.current_player.is_top,
            self.game.current_player.is_bottom
        )
        mask = np.zeros(len(self.edge_ids), dtype=np.int8)  # mask is length of all possible actions in the entire game
        for move in possible_moves:
            #sets only the legal moves to 1
            edge_id = move[1]['id']
            mask[self.id_to_index[edge_id]] = 1
        return mask

    def action_masks(self) -> np.ndarray:
        """Public interface for sb3-contrib MaskablePPO action masking."""
        return self._get_action_mask()

    
    def _update_graph_history_after_step(
        self,
        pre_turn_node: int,
        selected_end: int,
        mover: int,
        p1_is_top_after_move: bool,
        announcement_event: AnnouncementEvent | None,
    ) -> None:
        self._ensure_graph_renderer()
        if announcement_event is not None and announcement_event.kind == "teleport":
            node = self.game.game_state.current_node
            node_data = self.game.game_state.board.get_node_data(node)
            self._graph_renderer.set_initial_state(
                node_id=node,
                description=node_data.get("description", str(node)),
                p1_is_top=self.game.player1.is_top,
            )
            return

        node_data = self.game.game_state.board.get_node_data(selected_end)
        self._graph_renderer.record_move(MoveRecord(
            from_node=pre_turn_node,
            to_node=selected_end,
            mover=mover,
            p1_is_top=p1_is_top_after_move,
            turn=self.game.turn_count,
            description=node_data.get("description", str(selected_end)),
        ))

    def _classify_win_type(
        self,
        edge_data: dict[str, object],
        points_win_triggered: bool,
    ) -> WinType:
        if edge_data.get("tap", False):
            return "submission"
        if points_win_triggered:
            return "points"
        return "position"

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        super().reset(seed=seed)  # Seeds self.np_random
        if seed is not None:
            random.seed(seed)  # Also seed Python's random, used by Game internals

        # Fully reset the game
        self.game = Game("BJJ Match", logger=self.gameplay_logger)  # Create a new game instance
        self.game.board = Board(self.G)  # Reset the board with the graph
        self.game.game_state = GameState(self.game.board, logger=self.gameplay_logger)  # Reset the game state
        self.game.turn_count = 0

        self.game.initialize_game("Player1", "Player2")
        while not any(self._get_action_mask()):
            # Reinitialize if there are no valid moves for the starting position
            self.game.initialize_game("Player1", "Player2")
        self._pending_transition_id = None
        self._pending_transition_target_node = None
        self._pending_announcement_event = None
        self._point_flash_p1 = None
        self._point_flash_p2 = None

        if self.render_mode == "graph":
            self._ensure_graph_renderer()
            node = self.game.game_state.current_node
            node_data = self.game.game_state.board.get_node_data(node)
            self._graph_renderer.set_initial_state(
                node_id=node,
                description=node_data.get("description", str(node)),
                p1_is_top=self.game.player1.is_top,
            )

        if self.render_mode in ("human", "graph"):
            self.render()

        return self._get_obs(), {"action_mask": self._get_action_mask()}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        edge_id = self.index_to_id[action]
        if edge_id >= len(self.edge_ids):
            # Invalid action, end turn without making a move
            self.game.switch_players()
            return self._get_obs(), -1, False, False, {}
        (start, selected_end) = self.edge_id_to_nodes[edge_id]
        move = (selected_end, self.game.board.get_edge_data(start, selected_end))
        acting_player: Player | None = self.game.current_player  # capture before play_turn() switches current_player
        self._pending_transition_id = move[1].get('id')
        self._pending_transition_target_node = selected_end

        pre_turn_node = self.game.game_state.current_node
        mover = 0 if self.game.current_player is self.game.player1 else 1
        p1_is_top_after_move = self.game.player1.is_top
        if move[1].get("swaps_players", False):
            p1_is_top_after_move = not p1_is_top_after_move
        announcement_event: AnnouncementEvent | None = None

        p1_pts_before = self.game.player1.points
        p2_pts_before = self.game.player2.points
        edge_data = move[1]

        # play turn and update game state
        self.game.play_turn(move)
        self.game.turn_count += 1

        p1_delta = self.game.player1.points - p1_pts_before
        p2_delta = self.game.player2.points - p2_pts_before

        if p1_delta or p2_delta:
            # one of the player scored points this move
            scoring_moves = [name.capitalize() for name in self.game.board.rewards
                             if edge_data.get(name, False)
            ]
            label = " + ".join(scoring_moves) if scoring_moves else "Points"

            
            if p1_delta > 0:
                self._point_flash_p1 = (f"{label}! +{p1_delta} pts", time.time())
            if p2_delta > 0:
                self._point_flash_p2 = (f"{label}! +{p2_delta} pts", time.time())

        # terminated: game ended naturally (submission or position win)
        terminated = self.game.winner is not None
        # truncated: turn limit reached; check points to determine a winner if possible
        truncated = (not terminated) and (self.game.turn_count >= self.game.max_turns)
        points_win_triggered = False

        if truncated:
            self.game.check_for_points_win()
            points_win_triggered = self.game.winner is not None
            if self.game.winner is not None:
                terminated = True
                truncated = False

        if self.game.winner is not None:
            winner_index = 0 if self.game.winner is self.game.player1 else 1
            announcement_event = AnnouncementEvent(
                kind="win",
                winner_index=winner_index,
                win_type=self._classify_win_type(edge_data, points_win_triggered),
            )
        elif self.game.game_state.current_node != selected_end:
            announcement_event = AnnouncementEvent(kind="teleport")

        if self.render_mode == "graph":
            self._update_graph_history_after_step(
                pre_turn_node=pre_turn_node,
                selected_end=selected_end,
                mover=mover,
                p1_is_top_after_move=p1_is_top_after_move,
                announcement_event=announcement_event,
            )

        obs = self._get_obs()
        reward = self._calculate_reward(obs)

        info: dict[str, Any] = {"action_mask": self._get_action_mask()}
        self._pending_announcement_event = announcement_event
        if self.render_mode in ("human", "graph"):
            self.render(announcement_event=announcement_event)

        if terminated or truncated:
            other_player = self.game.choose_other_player(acting_player)
            info["is_win"] = self.game.winner == acting_player
            info["is_loss"] = self.game.winner == other_player
            info["point_diff"] = float(acting_player.points - other_player.points)
        return obs, reward, terminated, truncated, info


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
    def _ensure_renderer(self) -> None:
        """Lazily create the FrameRenderer on first graphical render call."""
        if self._renderer is None:
            import render.frame_renderer
            self._renderer = render.frame_renderer.FrameRenderer()

    def _ensure_graph_renderer(self) -> None:
        if self._graph_renderer is None:
            from render.graph_renderer import GraphRenderer
            self._graph_renderer = GraphRenderer(source_graph=self.G)

    def _get_player_info(self, announcement_event: AnnouncementEvent | None = None) -> dict[str, object]:
        """Build display-info dict for the renderer overlay text."""
        node = self.game.game_state.current_node
        node_data = self.game.game_state.board.get_node_data(node)
        info: dict[str, object] = {
            "description": node_data.get("description", str(node)),
            "p1_points": self.game.player1.points,
            "p2_points": self.game.player2.points,
            "turn": self.game.turn_count,
            "announcement_event": announcement_event,
        }
        now = time.time()
        if self._point_flash_p1:
            if now - self._point_flash_p1[1] < POINT_FLASH_DURATION:
                info["p1_flash"] = self._point_flash_p1[0]
            else:
                self._point_flash_p1 = None
        if self._point_flash_p2:
            if now - self._point_flash_p2[1] < POINT_FLASH_DURATION:
                info["p2_flash"] = self._point_flash_p2[0]
            else:
                self._point_flash_p2 = None
        return info

    def render(self, announcement_event: AnnouncementEvent | None = None) -> np.ndarray | list[np.ndarray] | str | None:
        """Render the environment according to render_mode."""
        if announcement_event is None:
            announcement_event = self._pending_announcement_event
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "graph":
            self._ensure_graph_renderer()
            result = self._graph_renderer.render_graph(
                render_mode="human",
                fps=self.metadata["render_fps"],
                player_info=self._get_player_info(announcement_event),
                announcement_event=announcement_event,
            )
            self._pending_announcement_event = None
            return result

        if self._pending_transition_id is not None and self.render_mode in ("human", "rgb_array"):
            self._ensure_renderer()
            target_node = self._pending_transition_target_node
            if target_node is None:
                target_node = self.game.game_state.current_node
            transition_frames = self._renderer.render_transition(  # type: ignore[union-attr]
                transition_id=self._pending_transition_id,
                node_id=target_node,
                render_mode=self.render_mode,
                fps=self.metadata["render_transition_fps"],
                player_info=self._get_player_info(None),
            )
            self._pending_transition_id = None
            self._pending_transition_target_node = None

            if announcement_event is None:
                self._pending_announcement_event = None
                return transition_frames

            final_frame = self._renderer.render_frame(  # type: ignore[union-attr]
                node_id=self.game.game_state.current_node,
                render_mode=self.render_mode,
                fps=self.metadata["render_fps"],
                player_info=self._get_player_info(announcement_event),
                announcement_event=announcement_event,
            )
            self._pending_announcement_event = None
            if self.render_mode == "rgb_array":
                if transition_frames is None:
                    return [final_frame] if final_frame is not None else []
                assert isinstance(transition_frames, list)
                return transition_frames + ([final_frame] if final_frame is not None else [])
            return final_frame

        # "human" or "rgb_array"
        self._ensure_renderer()
        result = self._renderer.render_frame(  # type: ignore[union-attr]
            node_id=self.game.game_state.current_node,
            render_mode=self.render_mode,
            fps=self.metadata["render_fps"],
            player_info=self._get_player_info(announcement_event),
            announcement_event=announcement_event,
        )
        self._pending_announcement_event = None
        return result

    def _render_ansi(self) -> str:
        """Return a text representation of the current game state."""
        node = self.game.game_state.current_node
        node_data = self.game.game_state.board.get_node_data(node)
        desc = node_data.get("description", str(node))
        p1_pos = "top" if self.game.player1.is_top else "bottom"
        return (
            f"Position: {desc}\n"
            f"P1: {self.game.player1.points} pts ({p1_pos})  "
            f"P2: {self.game.player2.points} pts\n"
            f"Turn: {self.game.turn_count}"
        )

    def close(self) -> None:
        """Clean up rendering resources."""
        if self._renderer is not None:
            self._renderer.close()  # type: ignore[union-attr]
            self._renderer = None
        if self._graph_renderer is not None:
            self._graph_renderer.close()  # type: ignore[union-attr]
            self._graph_renderer = None
        super().close()


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