import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from Game.gym_env import BJJEnv
from render.frame_renderer import FrameRenderer
from render.position_loader import load_raw_transition_index, parse_transition


def _make_joint(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> dict:
    return {"_isDirty": True, "_x": x, "_y": y, "_z": z}


def _make_frame(value: float = 0.0) -> list:
    return [
        [_make_joint(x=value + j, y=0.0, z=float(j % 3)) for j in range(23)],
        [_make_joint(x=-(value + j), y=0.0, z=float((j + 1) % 3)) for j in range(23)],
    ]


def _make_raw_transition(transition_id: int = 100, from_node: int = 1, to_node: int = 2) -> dict:
    return {
        "id": transition_id,
        "frames": [_make_frame(0.0), _make_frame(10.0)],
        "from": {
            "node": from_node,
            "reo": {
                "mirror": False,
                "swap_players": False,
                "angle": 0.0,
                "offset": _make_joint(0.0, 0.0, 0.0),
            },
        },
        "to": {
            "node": to_node,
            "reo": {
                "mirror": True,
                "swap_players": True,
                "angle": 0.2,
                "offset": _make_joint(1.0, 0.0, -1.0),
            },
        },
    }


def test_load_raw_transition_index(tmp_path: Path) -> None:
    transitions = [
        _make_raw_transition(transition_id=11),
        {"id": 12, "description": "no frames"},
    ]
    path = tmp_path / "transitions.json"
    path.write_text(json.dumps(transitions))

    index = load_raw_transition_index(str(path))

    assert set(index.keys()) == {11}
    assert isinstance(next(iter(index.keys())), int)
    assert "frames" in index[11]


def test_parse_transition() -> None:
    parsed = parse_transition(_make_raw_transition(transition_id=20, from_node=7, to_node=9))

    assert set(parsed.keys()) == {
        "frames",
        "detailed",
        "from_node",
        "to_node",
        "from_reo",
        "to_reo",
    }
    assert parsed["from_node"] == 7
    assert parsed["to_node"] == 9
    assert isinstance(parsed["detailed"], bool)
    assert len(parsed["frames"]) == 2
    assert len(parsed["frames"][0]) == 2
    assert len(parsed["frames"][0][0]) == 23
    assert len(parsed["frames"][0][0][0]) == 3


def test_interpolate_midpoints() -> None:
    frame0 = np.zeros((2, 23, 3), dtype=float).tolist()
    frame1 = np.full((2, 23, 3), 2.0, dtype=float).tolist()

    result = FrameRenderer._interpolate_midpoints([frame0, frame1])

    assert len(result) == 3
    np.testing.assert_allclose(np.array(result[0]), 0.0)
    np.testing.assert_allclose(np.array(result[1]), 1.0)
    np.testing.assert_allclose(np.array(result[2]), 2.0)


def test_get_transition_caches() -> None:
    renderer = FrameRenderer()
    renderer._raw_transitions = {42: _make_raw_transition(transition_id=42)}

    with patch("render.position_loader.parse_transition") as mock_parse:
        mock_parse.return_value = {"frames": [], "detailed": False, "from_node": 1, "to_node": 2}

        first = renderer._get_transition(42)
        second = renderer._get_transition(42)

    assert first is second
    assert mock_parse.call_count == 1


def test_get_transition_frees_raw() -> None:
    renderer = FrameRenderer()
    renderer._raw_transitions = {77: _make_raw_transition(transition_id=77)}

    renderer._get_transition(77)

    assert 77 not in renderer._raw_transitions


def test_draw_players_accepts_raw_joints() -> None:
    import pygame

    renderer = FrameRenderer(width=300, height=200)
    surface = pygame.Surface((300, 200))
    players = parse_transition(_make_raw_transition())["frames"][0]

    renderer._draw_players(surface, players, player_info=None)


def test_render_transition_fallback() -> None:
    renderer = FrameRenderer(width=200, height=150)
    expected = np.zeros((150, 200, 3), dtype=np.uint8)

    renderer._ensure_positions = lambda: None  # type: ignore[method-assign]
    renderer._get_transition = lambda _: None  # type: ignore[method-assign]
    renderer.render_frame = MagicMock(return_value=expected)  # type: ignore[method-assign]

    result = renderer.render_transition(
        transition_id=-1,
        node_id=94,
        render_mode="rgb_array",
    )

    assert isinstance(result, list)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], expected)
    renderer.render_frame.assert_called_once()


def test_reverse_detection() -> None:
    renderer = FrameRenderer(width=120, height=80)
    raw = _make_raw_transition(transition_id=88, from_node=10, to_node=11)
    parsed = parse_transition(raw)

    seen_first_joint_x: list[float] = []

    def _capture(_surface: object, players: list, _player_info: dict | None) -> None:
        seen_first_joint_x.append(players[0][0][0])

    renderer._ensure_positions = lambda: None  # type: ignore[method-assign]
    renderer._get_transition = lambda _: parsed  # type: ignore[method-assign]
    renderer._draw_players = _capture  # type: ignore[method-assign]
    renderer._last_node_id = parsed["to_node"]

    renderer.render_transition(
        transition_id=88,
        node_id=parsed["from_node"],
        render_mode="rgb_array",
    )

    assert len(seen_first_joint_x) >= 2
    # Should start with what was originally the second frame (value ~10.0)
    assert seen_first_joint_x[0] == pytest.approx(10.0)


def test_step_sets_pending_transition() -> None:
    env = BJJEnv()
    try:
        _, info = env.reset()
        action = int(np.where(info["action_mask"])[0][0])
        expected_transition_id = env.index_to_id[action]

        env.step(action)

        assert env._pending_transition_id == expected_transition_id
    finally:
        env.close()


def test_render_clears_pending_transition() -> None:
    env = BJJEnv(render_mode="rgb_array")
    try:
        _, info = env.reset()
        action = int(np.where(info["action_mask"])[0][0])
        env.step(action)

        mock_renderer = MagicMock()
        fake_frames = [np.zeros((400, 600, 3), dtype=np.uint8)]
        mock_renderer.render_transition.return_value = fake_frames
        env._renderer = mock_renderer

        result = env.render()

        assert result == fake_frames
        assert env._pending_transition_id is None
        mock_renderer.render_transition.assert_called_once()
    finally:
        env.close()
