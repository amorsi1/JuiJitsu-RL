import logging
import queue
from typing import Callable, Dict, List, Tuple

from Game.play_game import GameState
from render.position_server import PositionServer

logger = logging.getLogger(__name__)


def _drain_stale_selections(selections: "queue.Queue[int]") -> int:
    """Discard selections queued before this turn's prompt. Returns the count dropped.

    Why this is needed
    ------------------
    The viewer marks every node in a ``legal_moves`` message clickable, including the
    lone node of a forced move. But ``strategy`` below returns a forced move
    immediately *without* consuming the queue, so a click on that node leaves an entry
    behind. The next human turn would then consume it as that turn's choice, silently
    desyncing every turn after — a failure that surfaces far from its cause.

    Why the call site matters
    -------------------------
    This must run *before* ``send_legal_moves``, never after. Before the prompt, the
    browser has not been asked for anything this turn, so every queued entry is
    provably stale. Draining after the prompt would race with the player's real click
    and could swallow it.

    Reviewer's note
    ---------------
    This is a defensive net, not a fix at the source, and is the part of the
    ``move_selected`` desync work most likely to be judged suboptimal later. Two
    narrower alternatives, if this proves wrong: stop marking forced moves clickable in
    the viewer, or have the forced-move path consume the queue like any other turn.
    Both were passed over because they live in code paths (viewer overlay state; the
    unconditional ``send_legal_moves`` added in ``d021014``) that are harder to reason
    about than an unconditional drain here. The cost of this approach is that it hides
    protocol bugs rather than surfacing them — hence the warning log on every drop.
    """
    dropped = 0
    while True:
        try:
            stale = selections.get_nowait()
        except queue.Empty:
            return dropped
        dropped += 1
        logger.warning(
            "Discarding stale move_selected for node %s queued before this turn", stale
        )


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
        _drain_stale_selections(selected_nodes)
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
        if len(possible_moves) == 1:
            return possible_moves[0]
        to_node = selected_nodes.get()
        for move in possible_moves:
            if move[0] == to_node:
                return move
        raise ValueError(f"Selected to_node {to_node} not among legal moves")

    return strategy
