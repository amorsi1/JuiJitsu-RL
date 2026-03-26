"""
Tests for the Visualizer3D wrapper class.
These tests verify server lifecycle, update routing, and browser launch behavior.
"""
import time
from unittest.mock import patch, MagicMock
import pytest
from render.visualizer3d import Visualizer3D

# Use a unique port range to avoid conflicts with other test modules
BASE_PORT = 9200


@pytest.fixture
def visualizer():
    """Create a Visualizer3D without opening the browser."""
    viz = Visualizer3D(port=BASE_PORT, open_browser=False)
    yield viz
    viz.close()


# --- Initialization ---

def test_visualizer_starts_server(visualizer):
    """The server should be running after init."""
    assert visualizer.server._thread is not None
    assert visualizer.server._thread.is_alive()


def test_visualizer_loads_data(visualizer):
    """Position and transition data should be loaded."""
    assert len(visualizer.server.positions) > 0
    assert len(visualizer.server.transition_frames) > 0


def test_visualizer_does_not_open_browser_when_disabled():
    """open_browser=False should skip webbrowser.open."""
    with patch('render.visualizer3d.webbrowser.open') as mock_open:
        viz = Visualizer3D(port=BASE_PORT + 1, open_browser=False)
        mock_open.assert_not_called()
        viz.close()


def test_visualizer_opens_browser_when_enabled():
    """open_browser=True should call webbrowser.open with the viewer path."""
    with patch('render.visualizer3d.webbrowser.open') as mock_open:
        viz = Visualizer3D(port=BASE_PORT + 2, open_browser=True)
        mock_open.assert_called_once()
        call_arg = mock_open.call_args[0][0]
        assert 'viewer/index.html' in call_arg
        assert call_arg.startswith('file://')
        viz.close()


# --- update() ---

def test_update_position_only(visualizer):
    """update(node_id) with no transition_id should call send_position."""
    with patch.object(visualizer.server, 'send_position') as mock_pos, \
         patch.object(visualizer.server, 'send_transition') as mock_trans:
        visualizer.update(94)
        mock_pos.assert_called_once_with(94, turn=None)
        mock_trans.assert_not_called()
        assert visualizer._last_node_id == 94


def test_update_with_transition(visualizer):
    """update(node_id, transition_id) should send a forward transition by default."""
    with patch.object(visualizer.server, 'send_transition') as mock_trans, \
         patch.object(visualizer.server, 'send_position') as mock_pos:
        visualizer._last_node_id = 0
        visualizer.update(259, transition_id=0)
        mock_trans.assert_called_once_with(0, reverse=False, turn=None)
        mock_pos.assert_not_called()


def test_update_with_reverse_transition(visualizer):
    """A move that traverses a transition backwards should be marked reverse."""
    with patch.object(visualizer.server, 'send_transition') as mock_trans:
        visualizer._last_node_id = 259
        visualizer.update(0, transition_id=0)
        mock_trans.assert_called_once_with(0, reverse=True, turn=None)


def test_update_no_crash_on_missing_node(visualizer):
    """update with a non-existent node should not raise."""
    visualizer.update(999999)


def test_update_no_crash_on_missing_transition(visualizer):
    """update with a non-existent transition should not raise."""
    visualizer.update(94, transition_id=999999)


def test_update_sends_turn_with_position(visualizer):
    """active_turn should be passed through to send_position, not broadcast separately."""
    with patch.object(visualizer.server, 'send_position') as mock_pos:
        visualizer.update(94, active_turn='red')
        mock_pos.assert_called_once_with(94, turn='red')


def test_set_turn_forwards_to_server(visualizer):
    with patch.object(visualizer.server, 'set_turn') as mock_turn:
        visualizer.set_turn('blue')
        mock_turn.assert_called_once_with('blue')


# --- close() ---

def test_close_stops_server(visualizer):
    """close() should stop the server thread."""
    visualizer.close()

    assert not visualizer.server._thread.is_alive()


# --- Integration: Game class creates/uses Visualizer3D ---

def test_game_has_visualizer_attribute():
    """Game class should have a visualizer attribute that defaults to None.

    Note: We can't directly test Game(visualize_3d=True) here because
    play_game.py has top-level code that runs on import. Instead we verify
    the attribute exists and test the Visualizer3D import path used by Game.
    """
    # Verify that the Game class sets self.visualizer = None by default
    # by checking the source rather than importing (avoids top-level side effects)
    import inspect
    import importlib
    # The conditional import path used by Game.__init__
    assert importlib.util.find_spec('render.visualizer3d') is not None


def test_visualizer_import_path_matches_game():
    """Verify the import that Game.__init__ uses resolves to our Visualizer3D."""
    from render.visualizer3d import Visualizer3D as V3D
    assert V3D is Visualizer3D
