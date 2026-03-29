"""Tests for BJJEnv render_mode support.

Verifies that BJJEnv correctly implements Gymnasium's render_mode protocol:
- metadata declares supported render modes
- render() dispatches based on render_mode (None, "ansi", "rgb_array", "human", "graph")
- FrameRenderer is lazily initialized and properly cleaned up
- Auto-rendering in step() and reset() for human mode
- gymnasium.make() passes render_mode through kwargs
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import gymnasium
import numpy as np
import pytest

import Game  # noqa: F401 -- side-effect import: triggers __init__.py registration
from Game.gym_env import BJJEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def env_no_render() -> BJJEnv:
    """BJJEnv with default render_mode=None."""
    env = BJJEnv()
    yield env
    env.close()


@pytest.fixture()
def env_ansi() -> BJJEnv:
    """BJJEnv with render_mode='ansi'."""
    env = BJJEnv(render_mode="ansi")
    yield env
    env.close()


@pytest.fixture()
def env_rgb() -> BJJEnv:
    """BJJEnv with render_mode='rgb_array'."""
    env = BJJEnv(render_mode="rgb_array")
    yield env
    env.close()


# ---------------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------------


def test_metadata_declares_render_modes(env_no_render: BJJEnv) -> None:
    """BJJEnv.metadata must list 'human', 'rgb_array', and 'ansi' as supported render modes."""
    assert "render_modes" in env_no_render.metadata
    expected = ["human", "rgb_array", "ansi"]
    assert env_no_render.metadata["render_modes"] == expected


# ---------------------------------------------------------------------------
# 2. Default render_mode
# ---------------------------------------------------------------------------


def test_default_render_mode_is_none(env_no_render: BJJEnv) -> None:
    """Default construction must set render_mode to None."""
    assert env_no_render.render_mode is None


# ---------------------------------------------------------------------------
# 3. render() with no mode
# ---------------------------------------------------------------------------


def test_render_returns_none_when_no_mode(env_no_render: BJJEnv) -> None:
    """render() must return None when render_mode is None."""
    env_no_render.reset()
    result = env_no_render.render()
    assert result is None


# ---------------------------------------------------------------------------
# 4. ANSI mode
# ---------------------------------------------------------------------------


def test_render_ansi_returns_string(env_ansi: BJJEnv) -> None:
    """render() must return a string containing position info when mode='ansi'."""
    env_ansi.reset()
    result = env_ansi.render()
    assert isinstance(result, str)
    assert len(result) > 0
    # The string should contain some position-related information
    # (e.g., current position description, points, turn count)


# ---------------------------------------------------------------------------
# 5. rgb_array mode (mocked renderer)
# ---------------------------------------------------------------------------


@patch("render.frame_renderer.FrameRenderer")
def test_render_rgb_array_returns_numpy(mock_renderer_cls: MagicMock) -> None:
    """render() with mode='rgb_array' must return a uint8 numpy array of shape (H, W, 3)."""
    width, height = 600, 400
    fake_frame = np.zeros((height, width, 3), dtype=np.uint8)
    mock_instance = MagicMock()
    mock_instance.render_frame.return_value = fake_frame
    mock_renderer_cls.return_value = mock_instance

    env = BJJEnv(render_mode="rgb_array")
    try:
        env.reset()
        result = env.render()

        assert isinstance(result, np.ndarray)
        assert result.shape == (height, width, 3)
        assert result.dtype == np.uint8
        mock_instance.render_frame.assert_called_once()
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 6. Lazy initialization
# ---------------------------------------------------------------------------


def test_renderer_lazy_init(env_rgb: BJJEnv) -> None:
    """After construction with render_mode='rgb_array', _renderer must be None until render() is called."""
    assert env_rgb._renderer is None

    with patch("render.frame_renderer.FrameRenderer") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.render_frame.return_value = np.zeros((400, 600, 3), dtype=np.uint8)
        mock_cls.return_value = mock_instance

        env_rgb.reset()
        env_rgb.render()

        # After render(), the renderer should have been created
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# 7. No renderer created when mode is None
# ---------------------------------------------------------------------------


@patch("render.frame_renderer.FrameRenderer")
def test_renderer_not_created_when_mode_none(mock_renderer_cls: MagicMock) -> None:
    """With render_mode=None, calling render() must never instantiate a renderer."""
    env = BJJEnv()
    try:
        env.reset()
        env.render()
        mock_renderer_cls.assert_not_called()
        assert env._renderer is None
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 8. close() cleans up renderer
# ---------------------------------------------------------------------------


@patch("render.frame_renderer.FrameRenderer")
def test_close_cleans_up_renderer(mock_renderer_cls: MagicMock) -> None:
    """After rendering, close() must call renderer.close() and set _renderer to None."""
    mock_instance = MagicMock()
    mock_instance.render_frame.return_value = np.zeros((400, 600, 3), dtype=np.uint8)
    mock_renderer_cls.return_value = mock_instance

    env = BJJEnv(render_mode="rgb_array")
    env.reset()
    env.render()  # triggers lazy init

    env.close()

    mock_instance.close.assert_called_once()
    assert env._renderer is None


# ---------------------------------------------------------------------------
# 9. close() is idempotent
# ---------------------------------------------------------------------------


@patch("render.frame_renderer.FrameRenderer")
def test_close_idempotent(mock_renderer_cls: MagicMock) -> None:
    """Calling close() twice must not raise."""
    mock_instance = MagicMock()
    mock_instance.render_frame.return_value = np.zeros((400, 600, 3), dtype=np.uint8)
    mock_renderer_cls.return_value = mock_instance

    env = BJJEnv(render_mode="rgb_array")
    env.reset()
    env.render()

    env.close()
    env.close()  # second call must not raise


# ---------------------------------------------------------------------------
# 10. Auto-render in step() for human mode
# ---------------------------------------------------------------------------


@patch("render.frame_renderer.FrameRenderer")
def test_step_auto_renders_in_human_mode(mock_renderer_cls: MagicMock) -> None:
    """In human mode, step() must call render() automatically."""
    mock_instance = MagicMock()
    mock_instance.render_frame.return_value = None  # human mode returns None
    mock_renderer_cls.return_value = mock_instance

    env = BJJEnv(render_mode="human")
    try:
        obs, info = env.reset()
        # Clear any calls from reset's auto-render
        mock_instance.render_frame.reset_mock()

        # Pick a valid action from the mask
        valid_actions = np.where(info["action_mask"])[0]
        assert len(valid_actions) > 0, "Expected at least one valid action"
        action = valid_actions[0]

        env.step(action)
        mock_instance.render_frame.assert_called()
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 11. Auto-render in reset() for human mode
# ---------------------------------------------------------------------------


@patch("render.frame_renderer.FrameRenderer")
def test_reset_auto_renders_in_human_mode(mock_renderer_cls: MagicMock) -> None:
    """In human mode, reset() must call render() automatically."""
    mock_instance = MagicMock()
    mock_instance.render_frame.return_value = None
    mock_renderer_cls.return_value = mock_instance

    env = BJJEnv(render_mode="human")
    try:
        env.reset()
        mock_instance.render_frame.assert_called()
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 12. gymnasium.make() passes render_mode through
# ---------------------------------------------------------------------------


def test_make_with_render_mode() -> None:
    """gymnasium.make('BJJEnv-v0', render_mode='rgb_array') must pass render_mode to BJJEnv."""
    env = gymnasium.make("BJJEnv-v0", render_mode="rgb_array")
    try:
        assert env.unwrapped.render_mode == "rgb_array"
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 13. Graph render mode — metadata
# ---------------------------------------------------------------------------


def test_metadata_includes_graph() -> None:
    """BJJEnv.metadata must include 'graph' as a supported render mode."""
    assert "graph" in BJJEnv.metadata["render_modes"]


# ---------------------------------------------------------------------------
# 14. Graph render mode — lazy init
# ---------------------------------------------------------------------------


def test_graph_renderer_lazy_init() -> None:
    """After construction with render_mode='graph', _graph_renderer must be None."""
    env = BJJEnv(render_mode="graph")
    try:
        assert env._graph_renderer is None
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 15. Graph render mode — close cleans up
# ---------------------------------------------------------------------------


@patch("render.graph_renderer.GraphRenderer")
def test_close_cleans_graph_renderer(mock_graph_cls: MagicMock) -> None:
    """close() must call graph_renderer.close() and set _graph_renderer to None."""
    mock_instance = MagicMock()
    mock_graph_cls.return_value = mock_instance

    env = BJJEnv(render_mode="graph")
    # Simulate lazy init by assigning the mock directly
    env._graph_renderer = mock_instance

    env.close()

    mock_instance.close.assert_called_once()
    assert env._graph_renderer is None


# ---------------------------------------------------------------------------
# 16. Graph render mode — gymnasium.make
# ---------------------------------------------------------------------------


def test_make_with_graph_mode() -> None:
    """gymnasium.make('BJJEnv-v0', render_mode='graph') must pass render_mode through."""
    env = gymnasium.make("BJJEnv-v0", render_mode="graph")
    try:
        assert env.unwrapped.render_mode == "graph"
    finally:
        env.close()
