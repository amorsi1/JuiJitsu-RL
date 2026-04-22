import queue
from typing import Callable, Dict, List, Tuple

from Game.play_game import GameState
from render.position_server import PositionServer


def make_human_strategy(
    server: PositionServer, game_state: GameState
) -> Callable[[List[Tuple[int, Dict]]], Tuple[int, Dict]]:
    """Create a blocking strategy callable that waits for browser move selection."""
    selected_nodes: queue.Queue[int] = queue.Queue()

    def on_move_selected(to_node: int) -> None:
        selected_nodes.put(int(to_node))

    server.on_move_selected = on_move_selected

    def strategy(possible_moves: List[Tuple[int, Dict]]) -> Tuple[int, Dict]:
        assert possible_moves, "empty list of possible_moves passed to choose_move"
        if len(possible_moves) == 1:
            return possible_moves[0]
        server.send_legal_moves(
            current_node=game_state.current_node,
            moves=[
                {
                    "to_node": to_node,
                    "transition_id": edge_data["id"],
                    "description": edge_data.get("description", ""),
                }
                for to_node, edge_data in possible_moves
            ],
        )
        to_node = selected_nodes.get()
        for move in possible_moves:
            if move[0] == to_node:
                return move
        raise ValueError(f"Selected to_node {to_node} not among legal moves")

    return strategy
