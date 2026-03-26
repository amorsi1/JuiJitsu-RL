import json
import os
import tempfile
import pytest
from render.position_loader import (
    _parse_joint, _parse_frame, _parse_reo, load_positions, load_transition_frames, load_all
)

NUM_JOINTS = 23
NUM_PLAYERS = 2


# --- _parse_joint ---

def test_parse_joint_extracts_xyz():
    j = {"_isDirty": True, "_x": 1.5, "_y": -0.3, "_z": 0.0}
    assert _parse_joint(j) == [1.5, -0.3, 0.0]


def test_parse_joint_ignores_extra_keys():
    j = {"_isDirty": True, "_x": 0, "_y": 0, "_z": 0, "extra": 99}
    assert _parse_joint(j) == [0, 0, 0]


# --- _parse_frame ---

def _make_joint(x=0.0, y=0.0, z=0.0):
    return {"_isDirty": True, "_x": x, "_y": y, "_z": z}


def _make_frame(num_joints=NUM_JOINTS):
    """Create a valid frame with 2 players, each with num_joints joints."""
    return [
        [_make_joint(x=j * 0.1) for j in range(num_joints)],
        [_make_joint(x=j * -0.1) for j in range(num_joints)],
    ]


def test_parse_frame_shape():
    frame = _make_frame()
    parsed = _parse_frame(frame)
    assert len(parsed) == NUM_PLAYERS
    assert len(parsed[0]) == NUM_JOINTS
    assert len(parsed[1]) == NUM_JOINTS


def test_parse_frame_values():
    frame = _make_frame()
    parsed = _parse_frame(frame)
    # Player 0, joint 5 should be [0.5, 0.0, 0.0]
    assert parsed[0][5] == pytest.approx([0.5, 0.0, 0.0])
    # Player 1, joint 3 should be [-0.3, 0.0, 0.0]
    assert parsed[1][3] == pytest.approx([-0.3, 0.0, 0.0])


def test_parse_frame_each_joint_is_3_floats():
    parsed = _parse_frame(_make_frame())
    for player in parsed:
        for joint in player:
            assert len(joint) == 3
            assert all(isinstance(v, float) for v in joint)


def test_parse_reo_extracts_primitives():
    reo = {
        "mirror": True,
        "swap_players": False,
        "angle": 0.25,
        "offset": _make_joint(1.0, 0.0, -0.5),
    }
    assert _parse_reo(reo) == {
        "mirror": True,
        "swap_players": False,
        "angle": 0.25,
        "offset": [1.0, 0.0, -0.5],
    }


# --- load_positions (with real data) ---

def test_load_positions_returns_dict():
    positions = load_positions()
    assert isinstance(positions, dict)
    assert len(positions) > 0


def test_load_positions_contains_starting_node():
    """Node 94 ('symmetric staggered standing') is the default start position."""
    positions = load_positions()
    assert 94 in positions


def test_load_positions_shape():
    positions = load_positions()
    for node_id, pos in list(positions.items())[:10]:
        assert len(pos) == NUM_PLAYERS, f"Node {node_id}: expected {NUM_PLAYERS} players"
        for pl in range(NUM_PLAYERS):
            assert len(pos[pl]) == NUM_JOINTS, f"Node {node_id} player {pl}: expected {NUM_JOINTS} joints"
            for j, joint in enumerate(pos[pl]):
                assert len(joint) == 3, f"Node {node_id} p{pl} j{j}: expected 3 coords"


def test_load_positions_values_are_finite():
    """Joint coordinates should be reasonable floating point values."""
    positions = load_positions()
    for node_id, pos in list(positions.items())[:20]:
        for pl in range(NUM_PLAYERS):
            for j, joint in enumerate(pos[pl]):
                for coord in joint:
                    assert isinstance(coord, (int, float))
                    assert abs(coord) < 100, f"Node {node_id} p{pl} j{j}: unreasonable coord {coord}"


# --- load_transition_frames (with real data) ---

def test_load_transition_frames_returns_dict():
    frames = load_transition_frames()
    assert isinstance(frames, dict)
    assert len(frames) > 0


def test_load_transition_frames_have_multiple_frames():
    """Each transition should have at least 1 frame."""
    data = load_transition_frames()
    for tid, entry in list(data.items())[:10]:
        assert len(entry['frames']) >= 1, f"Transition {tid}: expected at least 1 frame"


def test_load_transition_frames_shape():
    data = load_transition_frames()
    for tid, entry in list(data.items())[:10]:
        for fi, frame in enumerate(entry['frames']):
            assert len(frame) == NUM_PLAYERS, f"Transition {tid} frame {fi}: expected {NUM_PLAYERS} players"
            for pl in range(NUM_PLAYERS):
                assert len(frame[pl]) == NUM_JOINTS, f"Transition {tid} frame {fi} player {pl}: expected {NUM_JOINTS} joints"


def test_load_transition_frames_has_detailed_flag():
    data = load_transition_frames()
    for tid, entry in list(data.items())[:10]:
        assert 'detailed' in entry
        assert isinstance(entry['detailed'], bool)


def test_load_transition_frames_include_reorientation_metadata():
    data = load_transition_frames()
    for tid, entry in list(data.items())[:10]:
        assert isinstance(entry['from_node'], int)
        assert isinstance(entry['to_node'], int)
        for key in ('from_reo', 'to_reo'):
            reo = entry[key]
            assert set(reo.keys()) == {'mirror', 'swap_players', 'angle', 'offset'}
            assert isinstance(reo['mirror'], bool)
            assert isinstance(reo['swap_players'], bool)
            assert isinstance(reo['angle'], (int, float))
            assert len(reo['offset']) == 3


# --- load_all ---

def test_load_all_returns_both():
    positions, transitions = load_all()
    assert len(positions) > 0
    assert len(transitions) > 0


# --- Custom file paths ---

def _write_test_json(data, suffix=".json"):
    f = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
    json.dump(data, f)
    f.close()
    return f.name


def test_load_positions_from_custom_path():
    nodes = [
        {"id": 1, "position": _make_frame(NUM_JOINTS)},
        {"id": 2, "position": _make_frame(NUM_JOINTS)},
    ]
    path = _write_test_json(nodes)
    try:
        positions = load_positions(path)
        assert set(positions.keys()) == {1, 2}
        assert len(positions[1]) == NUM_PLAYERS
        assert len(positions[1][0]) == NUM_JOINTS
    finally:
        os.unlink(path)


def test_load_positions_skips_nodes_without_position():
    nodes = [
        {"id": 1, "position": _make_frame(NUM_JOINTS)},
        {"id": 2, "description": "no position data"},
    ]
    path = _write_test_json(nodes)
    try:
        positions = load_positions(path)
        assert 1 in positions
        assert 2 not in positions
    finally:
        os.unlink(path)


def test_load_transition_frames_from_custom_path():
    transitions = [
        {
            "id": 10,
            "frames": [_make_frame(NUM_JOINTS), _make_frame(NUM_JOINTS)],
            "from": {"node": 1, "reo": {"mirror": False, "swap_players": False, "angle": 0.0, "offset": _make_joint()}},
            "to": {"node": 2, "reo": {"mirror": True, "swap_players": True, "angle": 0.1, "offset": _make_joint(0.1, 0.0, -0.1)}},
        },
    ]
    path = _write_test_json(transitions)
    try:
        frames = load_transition_frames(path)
        assert 10 in frames
        assert len(frames[10]['frames']) == 2
        assert frames[10]['from_node'] == 1
        assert frames[10]['to_node'] == 2
        assert frames[10]['to_reo']['mirror'] is True
    finally:
        os.unlink(path)


def test_load_transition_frames_skips_entries_without_frames():
    transitions = [
        {
            "id": 10,
            "frames": [_make_frame(NUM_JOINTS)],
            "from": {"node": 1, "reo": {"mirror": False, "swap_players": False, "angle": 0.0, "offset": _make_joint()}},
            "to": {"node": 2, "reo": {"mirror": False, "swap_players": False, "angle": 0.0, "offset": _make_joint()}},
        },
        {"id": 11, "description": "no frames"},
    ]
    path = _write_test_json(transitions)
    try:
        frames = load_transition_frames(path)
        assert 10 in frames
        assert 11 not in frames
    finally:
        os.unlink(path)
