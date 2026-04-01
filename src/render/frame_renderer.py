"""Pygame-based 2D stick figure renderer for BJJ positions.

Renders an orthographic front-view (XY plane) of both players using joint
coordinates from nodes.json. Features z-depth shading, anatomical segment
widths, and painter's algorithm draw order. Supports both 'human' (window)
and 'rgb_array' (numpy array) Gymnasium render modes.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Joint name constants (23 joints per player, matching GrappleMap convention)
# ---------------------------------------------------------------------------

LEFT_TOE = 0
RIGHT_TOE = 1
LEFT_HEEL = 2
RIGHT_HEEL = 3
LEFT_ANKLE = 4
RIGHT_ANKLE = 5
LEFT_KNEE = 6
RIGHT_KNEE = 7
LEFT_HIP = 8
RIGHT_HIP = 9
LEFT_SHOULDER = 10
RIGHT_SHOULDER = 11
LEFT_ELBOW = 12
RIGHT_ELBOW = 13
LEFT_WRIST = 14
RIGHT_WRIST = 15
LEFT_HAND = 16
RIGHT_HAND = 17
LEFT_FINGERS = 18
RIGHT_FINGERS = 19
CORE = 20
NECK = 21
HEAD = 22

NUM_JOINTS = 23

# ---------------------------------------------------------------------------
# Segment connectivity with anatomical radii — ported from viewer/index.html
# radius_center controls the pixel width of each segment.
# ---------------------------------------------------------------------------


class SegmentDef(NamedTuple):
    joint_from: int
    joint_to: int
    radius_center: float


SEGMENT_DEFS: list[SegmentDef] = [
    # Left side
    SegmentDef(LEFT_TOE, LEFT_HEEL, 0.025),
    SegmentDef(LEFT_TOE, LEFT_ANKLE, 0.025),
    SegmentDef(LEFT_HEEL, LEFT_ANKLE, 0.025),
    SegmentDef(LEFT_ANKLE, LEFT_KNEE, 0.055),
    SegmentDef(LEFT_KNEE, LEFT_HIP, 0.085),
    SegmentDef(LEFT_HIP, CORE, 0.1),
    SegmentDef(CORE, LEFT_SHOULDER, 0.075),
    SegmentDef(LEFT_SHOULDER, LEFT_ELBOW, 0.06),
    SegmentDef(LEFT_ELBOW, LEFT_WRIST, 0.03),
    SegmentDef(LEFT_WRIST, LEFT_HAND, 0.02),
    SegmentDef(LEFT_HAND, LEFT_FINGERS, 0.02),
    SegmentDef(LEFT_WRIST, LEFT_FINGERS, 0.02),
    # Right side
    SegmentDef(RIGHT_TOE, RIGHT_HEEL, 0.025),
    SegmentDef(RIGHT_TOE, RIGHT_ANKLE, 0.025),
    SegmentDef(RIGHT_HEEL, RIGHT_ANKLE, 0.025),
    SegmentDef(RIGHT_ANKLE, RIGHT_KNEE, 0.055),
    SegmentDef(RIGHT_KNEE, RIGHT_HIP, 0.085),
    SegmentDef(RIGHT_HIP, CORE, 0.1),
    SegmentDef(CORE, RIGHT_SHOULDER, 0.075),
    SegmentDef(RIGHT_SHOULDER, RIGHT_ELBOW, 0.06),
    SegmentDef(RIGHT_ELBOW, RIGHT_WRIST, 0.03),
    SegmentDef(RIGHT_WRIST, RIGHT_HAND, 0.02),
    SegmentDef(RIGHT_HAND, RIGHT_FINGERS, 0.02),
    SegmentDef(RIGHT_WRIST, RIGHT_FINGERS, 0.02),
    # Cross connections
    SegmentDef(LEFT_HIP, RIGHT_HIP, 0.1),
    SegmentDef(LEFT_SHOULDER, NECK, 0.065),
    SegmentDef(RIGHT_SHOULDER, NECK, 0.065),
    SegmentDef(NECK, HEAD, 0.05),
]

# Per-joint radii — from viewer/index.html lines 120-127
JOINT_RADII: dict[int, float] = {
    LEFT_TOE: 0.025, RIGHT_TOE: 0.025,
    LEFT_HEEL: 0.03, RIGHT_HEEL: 0.03,
    LEFT_ANKLE: 0.03, RIGHT_ANKLE: 0.03,
    LEFT_KNEE: 0.05, RIGHT_KNEE: 0.05,
    LEFT_HIP: 0.09, RIGHT_HIP: 0.09,
    LEFT_SHOULDER: 0.08, RIGHT_SHOULDER: 0.08,
    LEFT_ELBOW: 0.045, RIGHT_ELBOW: 0.045,
    LEFT_WRIST: 0.02, RIGHT_WRIST: 0.02,
    LEFT_HAND: 0.02, RIGHT_HAND: 0.02,
    LEFT_FINGERS: 0.02, RIGHT_FINGERS: 0.02,
    CORE: 0.1,
    NECK: 0.05,
    HEAD: 0.11,
}

# Joints to render as circles (matching the JS viewer's render_as_sphere flag)
SPHERE_JOINTS: list[int] = [
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_HIP, RIGHT_HIP,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_HAND, RIGHT_HAND,
    HEAD,
]

# Player colors (RGB)
PLAYER_COLORS: list[tuple[int, int, int]] = [
    (220, 60, 60),   # Player 0 — red
    (60, 100, 220),  # Player 1 — blue
]

BG_COLOR: tuple[int, int, int] = (25, 25, 45)
TEXT_COLOR: tuple[int, int, int] = (200, 200, 200)

# ---------------------------------------------------------------------------
# Z-depth shading
# ---------------------------------------------------------------------------

DEPTH_MIN_BRIGHTNESS: float = 0.4
DEPTH_MAX_BRIGHTNESS: float = 1.0


def _depth_shaded_color(
    base_color: tuple[int, int, int],
    z: float,
    z_min: float,
    z_range: float,
) -> tuple[int, int, int]:
    """Scale RGB brightness based on z-depth. Closer (lower z) = brighter."""
    if z_range <= 0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (z - z_min) / z_range))
    brightness = DEPTH_MAX_BRIGHTNESS - t * (DEPTH_MAX_BRIGHTNESS - DEPTH_MIN_BRIGHTNESS)
    return (
        min(255, int(base_color[0] * brightness)),
        min(255, int(base_color[1] * brightness)),
        min(255, int(base_color[2] * brightness)),
    )


# ---------------------------------------------------------------------------
# Projection result
# ---------------------------------------------------------------------------


class ProjectionResult(NamedTuple):
    screen_coords: list[list[tuple[int, int]]]
    z_values: list[list[float]]
    scale: float


class FrameRenderer:
    """Renders BJJ positions as 2D stick figures using pygame."""

    def __init__(self, width: int = 600, height: int = 400) -> None:
        self._width = width
        self._height = height
        self._screen: object | None = None  # pygame.Surface (human mode only)
        self._clock: object | None = None   # pygame.time.Clock
        self._positions: dict[int, list[list[list[float]]]] | None = None
        self._font: object | None = None

    def _ensure_positions(self) -> None:
        """Load position data on first use."""
        if self._positions is None:
            from render.position_loader import load_positions
            self._positions = load_positions()

    def _ensure_display(self) -> None:
        """Initialize pygame display for human mode."""
        if self._screen is None:
            import pygame
            pygame.init()
            self._screen = pygame.display.set_mode((self._width, self._height))
            pygame.display.set_caption("BJJEnv")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("monospace", 14)

    def _project_joints(
        self, players: list[list[list[float]]]
    ) -> ProjectionResult:
        """Project 3D joint positions to 2D screen coordinates.

        Uses orthographic XY projection (front view, looking along Z axis).
        Returns screen coordinates, z-values, and the projection scale factor.
        """
        all_x: list[float] = []
        all_y: list[float] = []
        for player_joints in players:
            for joint in player_joints:
                all_x.append(joint[0])
                all_y.append(joint[1])

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        margin = 40
        range_x = max_x - min_x or 1.0
        range_y = max_y - min_y or 1.0

        draw_w = self._width - 2 * margin
        draw_h = self._height - 2 * margin - 30  # reserve top for text

        scale = min(draw_w / range_x, draw_h / range_y)

        cx = margin + (draw_w - range_x * scale) / 2
        cy = margin + 30 + (draw_h - range_y * scale) / 2

        screen_coords: list[list[tuple[int, int]]] = []
        z_values: list[list[float]] = []
        for player_joints in players:
            pts: list[tuple[int, int]] = []
            zs: list[float] = []
            for joint in player_joints:
                px = int(cx + (joint[0] - min_x) * scale)
                py = int(cy + (max_y - joint[1]) * scale)
                pts.append((px, py))
                zs.append(joint[2])
            screen_coords.append(pts)
            z_values.append(zs)

        return ProjectionResult(screen_coords=screen_coords, z_values=z_values, scale=scale)

    def _draw_players(
        self,
        surface: object,
        players: list[list[list[float]]],
        player_info: dict[str, object] | None,
    ) -> None:
        """Draw stick figures from raw joint data (2 players × 23 joints × [x,y,z])."""
        import pygame

        surface.fill(BG_COLOR)  # type: ignore[union-attr]

        proj = self._project_joints(players)

        # Compute z range across all joints of both players
        all_z = [z for player_zs in proj.z_values for z in player_zs]
        z_min = min(all_z)
        z_max = max(all_z)
        z_range = z_max - z_min

        # Build draw commands: (z_depth, draw_callable)
        draw_cmds: list[tuple[float, object]] = []

        for p_idx in range(len(proj.screen_coords)):
            pts = proj.screen_coords[p_idx]
            zs = proj.z_values[p_idx]
            base_color = PLAYER_COLORS[p_idx]

            # Segment draw commands
            for seg in SEGMENT_DEFS:
                if seg.joint_from >= len(pts) or seg.joint_to >= len(pts):
                    continue
                avg_z = (zs[seg.joint_from] + zs[seg.joint_to]) / 2.0
                color = _depth_shaded_color(base_color, avg_z, z_min, z_range)
                pixel_width = max(2, int(seg.radius_center * 2 * proj.scale))
                pt_a = pts[seg.joint_from]
                pt_b = pts[seg.joint_to]
                draw_cmds.append((
                    avg_z,
                    lambda s=surface, c=color, a=pt_a, b=pt_b, w=pixel_width: (
                        pygame.draw.line(s, c, a, b, w)
                    ),
                ))

            # Joint sphere draw commands
            for j_idx in SPHERE_JOINTS:
                if j_idx >= len(pts):
                    continue
                z_val = zs[j_idx]
                color = _depth_shaded_color(base_color, z_val, z_min, z_range)
                pixel_radius = max(2, int(JOINT_RADII[j_idx] * proj.scale))
                center = pts[j_idx]
                draw_cmds.append((
                    z_val,
                    lambda s=surface, c=color, ct=center, r=pixel_radius: (
                        pygame.draw.circle(s, c, ct, r)
                    ),
                ))

        # Painter's algorithm: draw furthest (highest z) first
        draw_cmds.sort(key=lambda cmd: cmd[0], reverse=True)
        for _, draw_fn in draw_cmds:
            draw_fn()

        # Overlay text (always on top)
        if self._font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.SysFont("monospace", 14)

        if player_info:
            lines = [
                f"Position: {player_info.get('description', '')}",
                f"P1: {player_info.get('p1_points', 0)} pts  "
                f"P2: {player_info.get('p2_points', 0)} pts  "
                f"Turn: {player_info.get('turn', '?')}",
            ]
            for i, line in enumerate(lines):
                text_surf = self._font.render(line, True, TEXT_COLOR)
                surface.blit(text_surf, (8, 4 + i * 16))  # type: ignore[union-attr]

    def _draw_frame(
        self,
        surface: object,
        node_id: int,
        player_info: dict[str, object] | None,
    ) -> None:
        """Draw stick figures with depth shading and anatomical widths."""
        players = self._positions[node_id]  # type: ignore[index]
        self._draw_players(surface, players, player_info)

    def render_frame(
        self,
        node_id: int,
        render_mode: str,
        fps: int = 2,
        player_info: dict[str, object] | None = None,
    ) -> np.ndarray | None:
        """Render the current position as stick figures.

        Returns a numpy array (H, W, 3) uint8 for 'rgb_array', None for 'human'.
        """
        import pygame

        self._ensure_positions()

        if node_id not in self._positions:  # type: ignore[operator]
            return np.zeros((self._height, self._width, 3), dtype=np.uint8) if render_mode == "rgb_array" else None

        if render_mode == "human":
            self._ensure_display()
            # Pump event queue so the window stays responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            self._draw_frame(self._screen, node_id, player_info)
            pygame.display.flip()
            self._clock.tick(fps)  # type: ignore[union-attr]
            return None

        # rgb_array — render to offscreen surface
        offscreen = pygame.Surface((self._width, self._height))
        self._draw_frame(offscreen, node_id, player_info)
        # surfarray gives (W, H, 3); transpose to (H, W, 3)
        arr = pygame.surfarray.array3d(offscreen)
        return arr.transpose(1, 0, 2).astype(np.uint8)

    def close(self) -> None:
        """Clean up pygame resources."""
        if self._screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._clock = None
            self._font = None
