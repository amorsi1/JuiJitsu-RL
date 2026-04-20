"""
WebSocket server that pushes 3D position and transition frame data to the browser viewer.
"""
import asyncio
import contextlib
import errno
import json
import os
import signal
import subprocess
import threading
import websockets
from typing import Callable, Dict, List, Optional

from render.position_loader import load_all


def _free_port(port: int) -> None:
    """Terminate any process listening on the given TCP port."""
    result = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
    for pid_str in result.stdout.strip().splitlines():
        with contextlib.suppress(Exception):
            os.kill(int(pid_str), signal.SIGTERM)


class PositionServer:
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.positions: Dict = {}
        self.transition_frames: Dict = {}
        self.current_turn: Optional[str] = None
        self._clients: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._stop_event: Optional[asyncio.Event] = None
        self.on_move_selected: Optional[Callable[[int], None]] = None

    def load_data(self, nodes_path: str = None, transitions_path: str = None):
        self.positions, self.transition_frames = load_all(nodes_path, transitions_path)

    def _turn_message(self) -> str:
        return json.dumps({
            'type': 'turn_state',
            'turn': self.current_turn,
        })

    async def _handler(self, websocket):
        self._clients.add(websocket)
        try:
            async for message in websocket:
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if payload.get('type') == 'get_turn':
                    await websocket.send(self._turn_message())
                elif payload.get('type') == 'move_selected':
                    to_node = payload.get('to_node')
                    if self.on_move_selected is not None:
                        try:
                            self.on_move_selected(int(to_node))
                        except (TypeError, ValueError):
                            continue
        finally:
            self._clients.discard(websocket)

    async def _broadcast(self, message: str):
        if self._clients:
            await asyncio.gather(
                *[client.send(message) for client in self._clients],
                return_exceptions=True
            )

    def _run_server(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def serve():
            self._stop_event = asyncio.Event()
            try:
                self._server = await websockets.serve(self._handler, self.host, self.port)
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                _free_port(self.port)
                await asyncio.sleep(0.5)
                self._server = await websockets.serve(self._handler, self.host, self.port)
            await self._stop_event.wait()
            self._server.close()
            await self._server.wait_closed()

        self._loop.run_until_complete(serve())
        self._loop.close()

    def start(self):
        if not self.positions:
            self.load_data()
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def send_position(self, node_id: int, turn: Optional[str] = None):
        """Send a static position (pose) for the given node."""
        if node_id not in self.positions:
            return
        payload: dict = {
            'type': 'position',
            'node_id': node_id,
            'data': self.positions[node_id]
        }
        if turn is not None:
            payload['turn'] = turn
        msg = json.dumps(payload)
        if self._loop and self._clients:
            asyncio.run_coroutine_threadsafe(self._broadcast(msg), self._loop)

    def send_turn_state(self):
        """Broadcast the active turn to all connected clients."""
        if self._loop and self._clients:
            asyncio.run_coroutine_threadsafe(self._broadcast(self._turn_message()), self._loop)

    def send_legal_moves(self, current_node: int, moves: List[dict]):
        """Broadcast legal moves for the current human turn."""
        payload = {
            'type': 'legal_moves',
            'current_node': current_node,
            'moves': moves,
        }
        msg = json.dumps(payload)
        if self._loop and self._clients:
            asyncio.run_coroutine_threadsafe(self._broadcast(msg), self._loop)

    def set_turn(self, turn: str):
        """Store and broadcast whose turn it is."""
        if turn not in {'blue', 'red'}:
            raise ValueError("turn must be 'blue' or 'red'")
        self.current_turn = turn
        self.send_turn_state()

    def send_transition(self, transition_id: int, reverse: bool = False, turn: Optional[str] = None):
        """Send transition frame sequence for animation."""
        if transition_id not in self.transition_frames:
            return
        data = self.transition_frames[transition_id]
        payload: dict = {
            'type': 'transition',
            'transition_id': transition_id,
            'reverse': reverse,
            'frames': data['frames'],
            'detailed': data['detailed'],
            'from_node': data['from_node'],
            'to_node': data['to_node'],
            'from_reo': data['from_reo'],
            'to_reo': data['to_reo'],
        }
        if turn is not None:
            payload['turn'] = turn
        msg = json.dumps(payload)
        if self._loop and self._clients:
            asyncio.run_coroutine_threadsafe(self._broadcast(msg), self._loop)

    def stop(self):
        if self._loop and not self._loop.is_closed() and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=3.0)
