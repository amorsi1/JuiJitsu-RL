from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from Game.gym_env import build_action_edge_maps, build_obs
from Game.play_game import Game


def _load_maskable_ppo(model_path: str | Path):
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(str(model_path))


def make_sb3_strategy(
    model_path: str | Path,
    game: Game,
) -> Callable[[List[Tuple[int, Dict]]], Tuple[int, Dict]]:
    """Create a Player strategy callable backed by a MaskablePPO checkpoint."""
    model = _load_maskable_ppo(model_path)
    edge_ids, id_to_index, index_to_id, _ = build_action_edge_maps(game.board.graph)
    action_count = len(edge_ids)

    def strategy(possible_moves: List[Tuple[int, Dict]]) -> Tuple[int, Dict]:
        assert possible_moves, "empty list of possible_moves passed to choose_move"
        current_player = game.current_player
        other_player = game.choose_other_player(current_player)
        obs = build_obs(
            game.game_state,
            current_player,
            turns_left=game.max_turns - game.turn_count,
            other_player=other_player,
        )

        mask = np.zeros(action_count, dtype=bool)
        for _, edge_data in possible_moves:
            action_index = id_to_index[edge_data["id"]]
            mask[action_index] = True

        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        action_index = int(np.asarray(action).item())
        if action_index < 0 or action_index >= action_count:
            raise ValueError(f"Predicted invalid action index {action_index}")
        if not mask[action_index]:
            raise ValueError(f"Predicted masked action index {action_index}")

        selected_edge_id = index_to_id[action_index]
        for move in possible_moves:
            if move[1]["id"] == selected_edge_id:
                return move
        raise ValueError(f"Predicted edge id {selected_edge_id} not among legal moves")

    return strategy
