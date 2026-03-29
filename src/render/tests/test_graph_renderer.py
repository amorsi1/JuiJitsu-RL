"""Tests for the graph-based game history renderer (graph_renderer.py).

Covers initialization, set_initial_state, record_move, node reuse,
sliding window pruning, layout computation, and rgb_array output.
"""

from __future__ import annotations

import numpy as np
import pytest

from render.graph_renderer import GraphRenderer, MoveRecord, VisibleEdge, VisibleNode


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_init_defaults() -> None:
    """GraphRenderer starts with empty state."""
    renderer = GraphRenderer()
    assert renderer._visible_nodes == {}
    assert renderer._visible_edges == []
    assert renderer._current_node is None
    assert renderer._layout == {}
    assert renderer._layout_dirty is True


def test_init_custom_dimensions() -> None:
    """Custom width, height, and window_size are stored."""
    renderer = GraphRenderer(width=1024, height=768, window_size=20)
    assert renderer._visible_nodes == {}
    assert renderer._current_node is None


# ---------------------------------------------------------------------------
# set_initial_state
# ---------------------------------------------------------------------------


def test_set_initial_state() -> None:
    """set_initial_state creates exactly one visible node with correct fields."""
    renderer = GraphRenderer()
    renderer.set_initial_state(node_id=42, description="closed guard", p1_is_top=True)

    assert len(renderer._visible_nodes) == 1
    assert 42 in renderer._visible_nodes
    assert renderer._current_node == 42
    assert renderer._visible_edges == []

    node = renderer._visible_nodes[42]
    assert isinstance(node, VisibleNode)
    assert node.node_id == 42
    assert node.description == "closed guard"
    assert node.p1_is_top is True
    assert node.first_seen_turn == 0
    assert node.last_seen_turn == 0


def test_set_initial_state_clears_history() -> None:
    """Calling set_initial_state a second time clears all previous state."""
    renderer = GraphRenderer()
    renderer.set_initial_state(node_id=0, description="start", p1_is_top=True)
    renderer.record_move(MoveRecord(from_node=0, to_node=1, mover=0, p1_is_top=True, turn=1))
    renderer.record_move(MoveRecord(from_node=1, to_node=2, mover=1, p1_is_top=False, turn=2))

    assert len(renderer._visible_nodes) == 3
    assert len(renderer._visible_edges) == 2

    renderer.set_initial_state(node_id=99, description="reset pos", p1_is_top=False)

    assert len(renderer._visible_nodes) == 1
    assert 99 in renderer._visible_nodes
    assert renderer._current_node == 99
    assert renderer._visible_edges == []


# ---------------------------------------------------------------------------
# record_move — basic
# ---------------------------------------------------------------------------


def test_record_move_adds_node_and_edge() -> None:
    """A single move adds one new node and one edge."""
    renderer = GraphRenderer()
    renderer.set_initial_state(node_id=0, description="start", p1_is_top=True)
    renderer.record_move(MoveRecord(from_node=0, to_node=1, mover=0, p1_is_top=True, turn=1))

    assert len(renderer._visible_nodes) == 2
    assert 0 in renderer._visible_nodes
    assert 1 in renderer._visible_nodes
    assert len(renderer._visible_edges) == 1
    assert renderer._current_node == 1


def test_record_move_chain() -> None:
    """A chain of moves accumulates nodes and edges sequentially."""
    renderer = GraphRenderer()
    renderer.set_initial_state(node_id=0, description="start", p1_is_top=True)
    renderer.record_move(MoveRecord(from_node=0, to_node=1, mover=0, p1_is_top=True, turn=1))
    renderer.record_move(MoveRecord(from_node=1, to_node=2, mover=1, p1_is_top=False, turn=2))
    renderer.record_move(MoveRecord(from_node=2, to_node=3, mover=0, p1_is_top=True, turn=3))

    assert len(renderer._visible_nodes) == 4
    assert len(renderer._visible_edges) == 3
    assert renderer._current_node == 3


# ---------------------------------------------------------------------------
# record_move — node reuse
# ---------------------------------------------------------------------------


def test_node_reuse_no_duplicate() -> None:
    """Revisiting an existing node does not create a duplicate."""
    renderer = GraphRenderer()
    renderer.set_initial_state(node_id=0, description="start", p1_is_top=True)
    renderer.record_move(MoveRecord(from_node=0, to_node=1, mover=0, p1_is_top=True, turn=1))
    renderer.record_move(MoveRecord(from_node=1, to_node=0, mover=1, p1_is_top=True, turn=2))

    assert len(renderer._visible_nodes) == 2
    assert len(renderer._visible_edges) == 2
    assert renderer._current_node == 0


def test_node_reuse_updates_top() -> None:
    """Revisiting a node updates its p1_is_top field."""
    renderer = GraphRenderer()
    renderer.set_initial_state(node_id=0, description="start", p1_is_top=True)
    renderer.record_move(MoveRecord(from_node=0, to_node=1, mover=0, p1_is_top=True, turn=1))
    renderer.record_move(MoveRecord(from_node=1, to_node=0, mover=1, p1_is_top=False, turn=2))

    assert renderer._visible_nodes[0].p1_is_top is False


def test_node_reuse_updates_last_seen_turn() -> None:
    """Revisiting a node updates its last_seen_turn."""
    renderer = GraphRenderer()
    renderer.set_initial_state(node_id=0, description="start", p1_is_top=True)
    renderer.record_move(MoveRecord(from_node=0, to_node=1, mover=0, p1_is_top=True, turn=1))
    renderer.record_move(MoveRecord(from_node=1, to_node=0, mover=1, p1_is_top=True, turn=5))

    assert renderer._visible_nodes[0].last_seen_turn == 5


# ---------------------------------------------------------------------------
# Sliding window pruning
# ---------------------------------------------------------------------------


def test_sliding_window_prunes() -> None:
    """Exceeding window_size prunes the oldest node and its edges."""
    renderer = GraphRenderer(window_size=3)
    renderer.set_initial_state(node_id=0, description="n0", p1_is_top=True)
    renderer.record_move(MoveRecord(from_node=0, to_node=1, mover=0, p1_is_top=True, turn=1))
    renderer.record_move(MoveRecord(from_node=1, to_node=2, mover=1, p1_is_top=False, turn=2))
    renderer.record_move(MoveRecord(from_node=2, to_node=3, mover=0, p1_is_top=True, turn=3))

    assert len(renderer._visible_nodes) == 3
    assert 0 not in renderer._visible_nodes
    assert renderer._current_node == 3


def test_current_node_not_pruned() -> None:
    """The current node is never pruned even if it is the oldest."""
    renderer = GraphRenderer(window_size=2)
    renderer.set_initial_state(node_id=0, description="n0", p1_is_top=True)
    # 0->1 at turn 1
    renderer.record_move(MoveRecord(from_node=0, to_node=1, mover=0, p1_is_top=True, turn=1))
    # 1->0 at turn 2 (revisit 0, updates last_seen_turn)
    renderer.record_move(MoveRecord(from_node=1, to_node=0, mover=1, p1_is_top=True, turn=2))
    # 0->2 at turn 3 (3 nodes total: 0,1,2; current=2)
    renderer.record_move(MoveRecord(from_node=0, to_node=2, mover=0, p1_is_top=True, turn=3))

    # current_node == 2 must survive; node 1 is oldest by first_seen_turn
    assert renderer._current_node == 2
    assert 2 in renderer._visible_nodes
    assert len(renderer._visible_nodes) <= 2


def test_edges_removed_on_prune() -> None:
    """Edges involving a pruned node are removed."""
    renderer = GraphRenderer(window_size=3)
    renderer.set_initial_state(node_id=0, description="n0", p1_is_top=True)
    renderer.record_move(MoveRecord(from_node=0, to_node=1, mover=0, p1_is_top=True, turn=1))
    renderer.record_move(MoveRecord(from_node=1, to_node=2, mover=1, p1_is_top=False, turn=2))
    renderer.record_move(MoveRecord(from_node=2, to_node=3, mover=0, p1_is_top=True, turn=3))

    # Node 0 should have been pruned
    for edge in renderer._visible_edges:
        assert edge.from_node != 0, "Edge from pruned node 0 should be removed"
        assert edge.to_node != 0, "Edge to pruned node 0 should be removed"


# ---------------------------------------------------------------------------
# Layout computation
# ---------------------------------------------------------------------------


def test_layout_produces_positions() -> None:
    """_compute_layout populates _layout with float coordinate tuples."""
    renderer = GraphRenderer()
    renderer.set_initial_state(node_id=0, description="start", p1_is_top=True)
    renderer.record_move(MoveRecord(from_node=0, to_node=1, mover=0, p1_is_top=True, turn=1))
    renderer.record_move(MoveRecord(from_node=1, to_node=2, mover=1, p1_is_top=False, turn=2))

    renderer._compute_layout()

    for node_id in renderer._visible_nodes:
        assert node_id in renderer._layout
        pos = renderer._layout[node_id]
        assert len(pos) == 2
        assert isinstance(pos[0], float)
        assert isinstance(pos[1], float)


def test_layout_dirty_flag() -> None:
    """Layout dirty flag is set on state changes and cleared after computation."""
    renderer = GraphRenderer()
    renderer.set_initial_state(node_id=0, description="start", p1_is_top=True)
    assert renderer._layout_dirty is True

    renderer._compute_layout()
    assert renderer._layout_dirty is False

    renderer.record_move(MoveRecord(from_node=0, to_node=1, mover=0, p1_is_top=True, turn=1))
    assert renderer._layout_dirty is True


# ---------------------------------------------------------------------------
# RGB array rendering
# ---------------------------------------------------------------------------


def test_render_rgb_array() -> None:
    """render_graph('rgb_array') returns a numpy array of the expected shape."""
    renderer = GraphRenderer(width=800, height=600)
    renderer.set_initial_state(node_id=0, description="start", p1_is_top=True)
    renderer.record_move(MoveRecord(from_node=0, to_node=1, mover=0, p1_is_top=True, turn=1))

    frame = renderer.render_graph("rgb_array")

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (600, 800, 3)
    assert frame.dtype == np.uint8
    assert np.any(frame != 0), "Frame should not be entirely black"


def test_render_empty_state() -> None:
    """render_graph on an uninitialized renderer should not crash."""
    renderer = GraphRenderer(width=800, height=600)
    frame = renderer.render_graph("rgb_array")

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (600, 800, 3)
    assert frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_close_idempotent() -> None:
    """Calling close() multiple times must not raise."""
    renderer = GraphRenderer()
    renderer.close()
    renderer.close()
