"""Tests for the enhanced 2D stick figure renderer (frame_renderer.py).

Covers segment definitions, joint radii, depth shading, projection with z-values,
and rgb_array rendering output.
"""

from __future__ import annotations

import numpy as np
import pytest

from render.frame_renderer import (
    BG_COLOR,
    DEPTH_MAX_BRIGHTNESS,
    DEPTH_MIN_BRIGHTNESS,
    JOINT_RADII,
    NUM_JOINTS,
    SEGMENT_DEFS,
    FrameRenderer,
    ProjectionResult,
    SegmentDef,
    _depth_shaded_color,
)

# The 26 original segment pairs (before wrist-finger additions)
ORIGINAL_SEGMENTS: list[tuple[int, int]] = [
    (0, 2), (0, 4), (2, 4), (4, 6), (6, 8), (8, 20),
    (20, 10), (10, 12), (12, 14), (14, 16), (16, 18),
    (1, 3), (1, 5), (3, 5), (5, 7), (7, 9), (9, 20),
    (20, 11), (11, 13), (13, 15), (15, 17), (17, 19),
    (8, 9), (10, 21), (11, 21), (21, 22),
]


# ---------------------------------------------------------------------------
# Segment definitions
# ---------------------------------------------------------------------------


def test_segment_defs_count() -> None:
    """SEGMENT_DEFS should have 28 entries (26 original + 2 wrist-finger)."""
    assert len(SEGMENT_DEFS) == 28


def test_segment_defs_are_named_tuples() -> None:
    """Each entry in SEGMENT_DEFS should be a SegmentDef with expected fields."""
    for seg in SEGMENT_DEFS:
        assert isinstance(seg, SegmentDef)
        assert hasattr(seg, "joint_from")
        assert hasattr(seg, "joint_to")
        assert hasattr(seg, "radius_center")


def test_segment_defs_cover_original_connectivity() -> None:
    """All 26 original segment pairs must be present in SEGMENT_DEFS."""
    actual_pairs = {(s.joint_from, s.joint_to) for s in SEGMENT_DEFS}
    for pair in ORIGINAL_SEGMENTS:
        assert pair in actual_pairs, f"Original segment {pair} missing from SEGMENT_DEFS"


# ---------------------------------------------------------------------------
# Joint radii
# ---------------------------------------------------------------------------


def test_joint_radii_complete() -> None:
    """JOINT_RADII should have an entry for every joint index 0..NUM_JOINTS-1."""
    assert len(JOINT_RADII) == NUM_JOINTS
    for idx in range(NUM_JOINTS):
        assert idx in JOINT_RADII, f"Missing joint radius for index {idx}"


def test_joint_radii_proportional() -> None:
    """Joint radii should follow anatomical proportions."""
    head = JOINT_RADII[22]      # HEAD
    hip = JOINT_RADII[8]        # LEFT_HIP
    shoulder = JOINT_RADII[10]  # LEFT_SHOULDER
    knee = JOINT_RADII[6]       # LEFT_KNEE
    ankle = JOINT_RADII[4]      # LEFT_ANKLE
    finger = JOINT_RADII[18]    # LEFT_FINGERS

    assert head > hip > shoulder > knee > ankle > finger


def test_joint_radii_all_positive() -> None:
    """Every joint radius must be a positive float."""
    for idx, radius in JOINT_RADII.items():
        assert isinstance(radius, float)
        assert radius > 0, f"Joint {idx} radius should be positive, got {radius}"


# ---------------------------------------------------------------------------
# Depth shading
# ---------------------------------------------------------------------------


def test_depth_shaded_color_closer_is_brighter() -> None:
    """Closer objects (lower z) should produce brighter colors."""
    base: tuple[int, int, int] = (200, 100, 50)
    closer = _depth_shaded_color(base, z=0.0, z_min=0.0, z_range=1.0)
    farther = _depth_shaded_color(base, z=1.0, z_min=0.0, z_range=1.0)

    for ch in range(3):
        assert closer[ch] >= farther[ch], (
            f"Channel {ch}: closer ({closer[ch]}) should be >= farther ({farther[ch]})"
        )


def test_depth_shaded_color_at_extremes() -> None:
    """At z=z_min brightness=max (1.0); at z=z_min+z_range brightness=min (0.4)."""
    base: tuple[int, int, int] = (200, 100, 50)

    closest = _depth_shaded_color(base, z=0.0, z_min=0.0, z_range=1.0)
    assert closest == base

    farthest = _depth_shaded_color(base, z=1.0, z_min=0.0, z_range=1.0)
    expected = (
        int(base[0] * DEPTH_MIN_BRIGHTNESS),
        int(base[1] * DEPTH_MIN_BRIGHTNESS),
        int(base[2] * DEPTH_MIN_BRIGHTNESS),
    )
    assert farthest == expected


def test_depth_shaded_color_clamps_rgb() -> None:
    """RGB channels should never exceed 255 or go below 0."""
    white: tuple[int, int, int] = (255, 255, 255)

    for z in [-1.0, 0.0, 0.5, 1.0, 2.0]:
        color = _depth_shaded_color(white, z=z, z_min=0.0, z_range=1.0)
        for ch in range(3):
            assert 0 <= color[ch] <= 255, (
                f"Channel {ch} out of range at z={z}: got {color[ch]}"
            )


def test_depth_shaded_color_zero_range() -> None:
    """When z_range=0, should not crash and should return a valid color."""
    base: tuple[int, int, int] = (200, 100, 50)
    color = _depth_shaded_color(base, z=0.5, z_min=0.0, z_range=0.0)

    assert len(color) == 3
    for ch in range(3):
        assert 0 <= color[ch] <= 255


# ---------------------------------------------------------------------------
# Projection with z-values
# ---------------------------------------------------------------------------


def test_projection_returns_projection_result() -> None:
    """_project_joints should return a ProjectionResult NamedTuple."""
    renderer = FrameRenderer()
    player0: list[list[float]] = [[0.0, 1.0, 0.5]] * NUM_JOINTS
    player1: list[list[float]] = [[1.0, 0.0, 1.5]] * NUM_JOINTS

    result = renderer._project_joints([player0, player1])
    assert isinstance(result, ProjectionResult)


def test_projection_returns_z_values() -> None:
    """Projected z_values should match the input z coordinates."""
    renderer = FrameRenderer()
    player0: list[list[float]] = [[0.0, 1.0, 0.5]] * NUM_JOINTS
    player1: list[list[float]] = [[1.0, 0.0, 1.5]] * NUM_JOINTS

    result = renderer._project_joints([player0, player1])

    assert len(result.z_values) == 2
    assert len(result.z_values[0]) == NUM_JOINTS
    assert len(result.z_values[1]) == NUM_JOINTS

    for z in result.z_values[0]:
        assert z == pytest.approx(0.5)
    for z in result.z_values[1]:
        assert z == pytest.approx(1.5)


def test_projection_returns_positive_scale() -> None:
    """The projection scale factor should be positive."""
    renderer = FrameRenderer()
    player0: list[list[float]] = [[0.0, 1.0, 0.5]] * NUM_JOINTS
    player1: list[list[float]] = [[1.0, 0.0, 1.5]] * NUM_JOINTS

    result = renderer._project_joints([player0, player1])
    assert result.scale > 0


def test_projection_screen_coords_shape() -> None:
    """screen_coords: 2 players, each with NUM_JOINTS (x, y) tuples."""
    renderer = FrameRenderer()
    player0: list[list[float]] = [
        [float(i), float(i) * 0.5, 0.0] for i in range(NUM_JOINTS)
    ]
    player1: list[list[float]] = [
        [float(i) * -1.0, float(i) * 0.3, 1.0] for i in range(NUM_JOINTS)
    ]

    result = renderer._project_joints([player0, player1])

    assert len(result.screen_coords) == 2
    assert len(result.screen_coords[0]) == NUM_JOINTS
    assert len(result.screen_coords[1]) == NUM_JOINTS

    for px, py in result.screen_coords[0]:
        assert isinstance(px, int)
        assert isinstance(py, int)


# ---------------------------------------------------------------------------
# RGB array rendering (integration)
# ---------------------------------------------------------------------------


@pytest.fixture()
def rgb_renderer() -> FrameRenderer:
    """FrameRenderer with positions loaded for rgb_array tests."""
    renderer = FrameRenderer(width=600, height=400)
    renderer._ensure_positions()
    return renderer


def _first_available_node(renderer: FrameRenderer) -> int:
    """Return the first available node_id from loaded positions."""
    assert renderer._positions is not None and len(renderer._positions) > 0
    return next(iter(renderer._positions))


def test_render_rgb_array_shape(rgb_renderer: FrameRenderer) -> None:
    """render_frame with 'rgb_array' should return (400, 600, 3) uint8 array."""
    node_id = _first_available_node(rgb_renderer)
    frame = rgb_renderer.render_frame(node_id, render_mode="rgb_array")

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (400, 600, 3)
    assert frame.dtype == np.uint8


def test_render_rgb_array_not_all_background(rgb_renderer: FrameRenderer) -> None:
    """Rendered frame should contain non-background pixels (actual body parts)."""
    node_id = _first_available_node(rgb_renderer)
    frame = rgb_renderer.render_frame(node_id, render_mode="rgb_array")

    assert frame is not None
    bg = np.array(BG_COLOR, dtype=np.uint8)
    is_bg = np.all(frame == bg, axis=-1)
    non_bg_count = np.sum(~is_bg)
    assert non_bg_count > 0, "Frame is entirely background color; no body parts drawn"


def test_render_rgb_array_invalid_node(rgb_renderer: FrameRenderer) -> None:
    """Rendering a non-existent node should return a black frame, not crash."""
    frame = rgb_renderer.render_frame(-999, render_mode="rgb_array")

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (400, 600, 3)
    assert np.all(frame == 0)
