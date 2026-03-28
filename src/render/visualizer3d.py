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
    def __init__(self, host: str = 'localhost', port: int = 8765, open_browser: bool = True):
        self.server = PositionServer(host=host, port=port)
        self.server.load_data()
        self._last_node_id = None
        self.server.start()
        # Give the server a moment to bind
        time.sleep(0.3)
        if open_browser:
            viewer_path = os.path.join(os.path.dirname(__file__), 'viewer', 'index.html')
            webbrowser.open('file://' + os.path.abspath(viewer_path))
            # Wait for the browser to connect
            time.sleep(1.0)

    def set_turn(self, active_turn: str):
        self.server.set_turn(active_turn)

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

    def close(self):
        self.server.stop()
