import asyncio
import json
import time
import pytest
import websockets
from render.position_server import PositionServer

NUM_JOINTS = 23
NUM_PLAYERS = 2

# Use a unique port per test module to avoid conflicts
BASE_PORT = 9100


@pytest.fixture
def server(tmp_path):
    """Create a PositionServer with minimal synthetic data, start it, yield, then stop."""
    srv = PositionServer(host='localhost', port=BASE_PORT)

    # Inject synthetic data directly instead of loading from files
    srv.positions = {
        1: [[[0.1 * j, 0.0, 0.0] for j in range(NUM_JOINTS)] for _ in range(NUM_PLAYERS)],
        2: [[[0.0, 0.1 * j, 0.0] for j in range(NUM_JOINTS)] for _ in range(NUM_PLAYERS)],
    }
    srv.transition_frames = {
        10: {
            'frames': [
                [[[0.0, 0.0, 0.0] for _ in range(NUM_JOINTS)] for _ in range(NUM_PLAYERS)],
                [[[1.0, 1.0, 1.0] for _ in range(NUM_JOINTS)] for _ in range(NUM_PLAYERS)],
            ],
            'detailed': False,
            'from_node': 1,
            'to_node': 2,
            'from_reo': {'mirror': False, 'swap_players': False, 'angle': 0.0, 'offset': [0.0, 0.0, 0.0]},
            'to_reo': {'mirror': False, 'swap_players': False, 'angle': 0.25, 'offset': [0.1, 0.0, -0.2]},
        },
    }
    srv.start()
    time.sleep(0.3)
    yield srv
    srv.stop()


@pytest.fixture
def server_real_data():
    """Create a PositionServer loaded with the real GrappleMap data."""
    srv = PositionServer(host='localhost', port=BASE_PORT + 1)
    srv.load_data()
    srv.start()
    time.sleep(0.3)
    yield srv
    srv.stop()


async def _connect_and_receive(port, timeout=2.0):
    """Connect as a WebSocket client and return the first message received."""
    async with websockets.connect(f'ws://localhost:{port}') as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
        return json.loads(msg)


# --- Server lifecycle ---

def test_server_starts_and_stops(server):
    """Server should start without error and have a running event loop."""
    assert server._loop is not None
    assert server._thread is not None
    assert server._thread.is_alive()


def test_server_stops_cleanly(tmp_path):
    srv = PositionServer(host='localhost', port=BASE_PORT + 2)
    srv.positions = {1: [[[0.0, 0.0, 0.0]] * NUM_JOINTS] * NUM_PLAYERS}
    srv.transition_frames = {}
    srv.start()
    time.sleep(0.3)
    srv.stop()
    time.sleep(0.3)
    assert not srv._thread.is_alive()


# --- Client connection ---

def test_client_can_connect(server):
    """A WebSocket client should be able to connect to the server."""
    async def _test():
        async with websockets.connect(f'ws://localhost:{BASE_PORT}') as ws:
            if hasattr(ws, "open"):
                assert ws.open
            else:
                assert ws.state.name == "OPEN"

    asyncio.run(_test())


def test_client_tracked_on_connect(server):
    """The server should track connected clients."""
    async def _test():
        async with websockets.connect(f'ws://localhost:{BASE_PORT}') as ws:
            await asyncio.sleep(0.1)
            assert len(server._clients) == 1

    asyncio.run(_test())


def test_client_removed_on_disconnect(server):
    """After client disconnects, it should be removed from the set."""
    async def _test():
        ws = await websockets.connect(f'ws://localhost:{BASE_PORT}')
        await asyncio.sleep(0.1)
        assert len(server._clients) == 1
        await ws.close()
        await asyncio.sleep(0.2)
        assert len(server._clients) == 0

    asyncio.run(_test())


# --- send_position ---

def test_send_position_delivers_message(server):
    """send_position should deliver a JSON message with the correct structure."""
    async def _test():
        async with websockets.connect(f'ws://localhost:{BASE_PORT}') as ws:
            await asyncio.sleep(0.1)
            server.send_position(1)
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(msg)
            assert data['type'] == 'position'
            assert data['node_id'] == 1
            assert len(data['data']) == NUM_PLAYERS
            assert len(data['data'][0]) == NUM_JOINTS

    asyncio.run(_test())


def test_send_position_unknown_node_is_silent(server):
    """send_position with a non-existent node_id should not crash or send anything."""
    async def _test():
        async with websockets.connect(f'ws://localhost:{BASE_PORT}') as ws:
            await asyncio.sleep(0.1)
            server.send_position(9999)  # doesn't exist
            # Should not receive anything — use a short timeout
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ws.recv(), timeout=0.5)

    asyncio.run(_test())


def test_send_position_no_clients_no_crash(server):
    """send_position with no connected clients should not raise."""
    server.send_position(1)  # no exception


# --- turn state ---

def test_get_turn_returns_current_turn(server):
    """Clients can request the current turn via the WebSocket API."""
    async def _test():
        async with websockets.connect(f'ws://localhost:{BASE_PORT}') as ws:
            await asyncio.sleep(0.1)
            server.set_turn('blue')
            await ws.send(json.dumps({'type': 'get_turn'}))
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
            assert data['type'] == 'turn_state'
            assert data['turn'] == 'blue'

    asyncio.run(_test())


def test_set_turn_broadcasts_to_connected_clients(server):
    """set_turn should push the updated turn state to connected clients."""
    async def _test():
        async with websockets.connect(f'ws://localhost:{BASE_PORT}') as ws:
            await asyncio.sleep(0.1)
            server.set_turn('red')
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
            assert data['type'] == 'turn_state'
            assert data['turn'] == 'red'

    asyncio.run(_test())


def test_set_turn_rejects_invalid_value(server):
    with pytest.raises(ValueError):
        server.set_turn('green')


# --- send_transition ---

def test_send_transition_delivers_frames(server):
    """send_transition should deliver a message with frames array."""
    async def _test():
        async with websockets.connect(f'ws://localhost:{BASE_PORT}') as ws:
            await asyncio.sleep(0.1)
            server.send_transition(10, reverse=True)
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(msg)
            assert data['type'] == 'transition'
            assert data['transition_id'] == 10
            assert data['reverse'] is True
            assert len(data['frames']) == 2  # 2 keyframes in our test data
            assert data['from_node'] == 1
            assert data['to_node'] == 2
            assert data['to_reo']['angle'] == pytest.approx(0.25)

    asyncio.run(_test())


def test_send_transition_unknown_id_is_silent(server):
    """send_transition with a non-existent ID should not crash or send."""
    async def _test():
        async with websockets.connect(f'ws://localhost:{BASE_PORT}') as ws:
            await asyncio.sleep(0.1)
            server.send_transition(9999)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ws.recv(), timeout=0.5)

    asyncio.run(_test())


# --- Multiple clients ---

def test_broadcast_reaches_all_clients(server):
    """A position message should be sent to all connected clients."""
    async def _test():
        ws1 = await websockets.connect(f'ws://localhost:{BASE_PORT}')
        ws2 = await websockets.connect(f'ws://localhost:{BASE_PORT}')
        await asyncio.sleep(0.1)
        assert len(server._clients) == 2

        server.send_position(2)

        msg1 = json.loads(await asyncio.wait_for(ws1.recv(), timeout=2.0))
        msg2 = json.loads(await asyncio.wait_for(ws2.recv(), timeout=2.0))

        assert msg1['node_id'] == 2
        assert msg2['node_id'] == 2

        await ws1.close()
        await ws2.close()

    asyncio.run(_test())


# --- Real data integration ---

def test_send_position_node94_real_data(server_real_data):
    """Verify that the real starting position (node 94) sends valid data."""
    async def _test():
        async with websockets.connect(f'ws://localhost:{BASE_PORT + 1}') as ws:
            await asyncio.sleep(0.1)
            server_real_data.send_position(94)
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(msg)
            assert data['type'] == 'position'
            assert data['node_id'] == 94
            assert len(data['data']) == NUM_PLAYERS
            assert len(data['data'][0]) == NUM_JOINTS
            # Each joint should be [x, y, z]
            for joint in data['data'][0]:
                assert len(joint) == 3

    asyncio.run(_test())


def test_send_transition_0_real_data(server_real_data):
    """Verify that real transition 0 sends valid frame data."""
    async def _test():
        async with websockets.connect(f'ws://localhost:{BASE_PORT + 1}') as ws:
            await asyncio.sleep(0.1)
            server_real_data.send_transition(0)
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(msg)
            assert data['type'] == 'transition'
            assert data['transition_id'] == 0
            assert data['reverse'] is False
            assert len(data['frames']) >= 1
            assert 'from_reo' in data
            assert 'to_reo' in data
            # Check first frame structure
            frame = data['frames'][0]
            assert len(frame) == NUM_PLAYERS
            assert len(frame[0]) == NUM_JOINTS

    asyncio.run(_test())
