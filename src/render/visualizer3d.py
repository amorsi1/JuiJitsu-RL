"""
Visualizer3D class — starts the WebSocket server and opens the browser viewer.
Provides update() method to push positions/transitions during gameplay.
"""
import os
import time
import webbrowser
from typing import Optional

from .position_server import PositionServer


class Visualizer3D:
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8765,
        open_browser: bool = True,
        turn_delay: float = 0.0,
    ):
        self.server = PositionServer(host=host, port=port)
        self.server.load_data()
        self._last_node_id = None
        self.game_ref = None
        self.turn_delay = turn_delay
        self.server.start()
        # Give the server a moment to bind
        time.sleep(0.3)
        if open_browser:
            viewer_path = os.path.join(os.path.dirname(__file__), 'viewer', 'index.html')
            webbrowser.open('file://' + os.path.abspath(viewer_path))
            connected = self.server.wait_for_connection(timeout=15.0)
            if not connected:
                # Continue gameplay even when no viewer connects.
                pass

    def set_turn(self, active_turn: str):
        self.server.set_turn(active_turn)

    def _broadcast_hud(self) -> None:
        if self.game_ref is None:
            return
        game = self.game_ref
        try:
            node_id = game.game_state.current_node
            position_name = game.game_state.board.get_node_data(node_id).get('description', '')
        except (AttributeError, KeyError):
            position_name = ''
        self.server.send_hud_state(
            turn_number=getattr(game, 'turn_count', 0),
            position_name=position_name,
            p1_points=getattr(game.player1, 'points', 0) if game.player1 else 0,
            p2_points=getattr(game.player2, 'points', 0) if game.player2 else 0,
        )

    def update(
        self,
        node_id: int,
        transition_id: Optional[int] = None,
        active_turn: Optional[str] = None,
    ):
        """
        Send current position to the viewer. If transition_id is provided,
        send the transition frames for animation (the frames already include
        the start and end poses). Otherwise send a static position.

        active_turn is embedded in the message so the viewer can sync the
        turn indicator with the animation queue rather than flipping immediately.
        """
        if transition_id is not None:
            reverse = False
            transition = self.server.transition_frames.get(transition_id)
            if transition is not None and self._last_node_id is not None:
                if (
                    self._last_node_id == transition['to_node']
                    and node_id == transition['from_node']
                ):
                    reverse = True
            self.server.send_transition(transition_id, reverse=reverse, turn=active_turn)
        else:
            self.server.send_position(node_id, turn=active_turn)
        self._last_node_id = node_id
        self._broadcast_hud()
        if self.turn_delay > 0:
            time.sleep(self.turn_delay)

    def close(self):
        self.server.stop()
