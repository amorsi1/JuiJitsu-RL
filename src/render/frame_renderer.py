"""Pygame-based 2D stick figure renderer for BJJ positions.

Renders an orthographic front-view (XY plane) of both players as stick figures
using joint coordinates from nodes.json. Supports both 'human' (window) and
'rgb_array' (numpy array) Gymnasium render modes.
"""

from __future__ import annotations

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
# Segment connectivity — ported from viewer/index.html
# Each tuple is (from_joint, to_joint).
# ---------------------------------------------------------------------------

SEGMENTS: list[tuple[int, int]] = [
    # Left side
    (LEFT_TOE, LEFT_HEEL),
    (LEFT_TOE, LEFT_ANKLE),
    (LEFT_HEEL, LEFT_ANKLE),
    (LEFT_ANKLE, LEFT_KNEE),
    (LEFT_KNEE, LEFT_HIP),
    (LEFT_HIP, CORE),
    (CORE, LEFT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (LEFT_WRIST, LEFT_HAND),
    (LEFT_HAND, LEFT_FINGERS),
    # Right side
    (RIGHT_TOE, RIGHT_HEEL),
    (RIGHT_TOE, RIGHT_ANKLE),
    (RIGHT_HEEL, RIGHT_ANKLE),
    (RIGHT_ANKLE, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_HIP),
    (RIGHT_HIP, CORE),
    (CORE, RIGHT_SHOULDER),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (RIGHT_WRIST, RIGHT_HAND),
    (RIGHT_HAND, RIGHT_FINGERS),
    # Cross connections
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, NECK),
    (RIGHT_SHOULDER, NECK),
    (NECK, HEAD),
]

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
    ) -> list[list[tuple[int, int]]]:
        """Project 3D joint positions to 2D screen coordinates.

        Uses orthographic XY projection (front view, looking along Z axis).
        Returns pixel coordinates for each player's joints.
        """
        # Collect all x, y values across both players for normalization
        all_x: list[float] = []
        all_y: list[float] = []
        for player_joints in players:
            for joint in player_joints:
                all_x.append(joint[0])
                all_y.append(joint[1])

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Add padding
        margin = 40
        range_x = max_x - min_x or 1.0
        range_y = max_y - min_y or 1.0

        draw_w = self._width - 2 * margin
        draw_h = self._height - 2 * margin - 30  # reserve top for text

        # Uniform scale to preserve aspect ratio
        scale = min(draw_w / range_x, draw_h / range_y)

        # Center offset
        cx = margin + (draw_w - range_x * scale) / 2
        cy = margin + 30 + (draw_h - range_y * scale) / 2

        projected: list[list[tuple[int, int]]] = []
        for player_joints in players:
            pts: list[tuple[int, int]] = []
            for joint in player_joints:
                px = int(cx + (joint[0] - min_x) * scale)
                # Flip Y so up is up on screen
                py = int(cy + (max_y - joint[1]) * scale)
                pts.append((px, py))
            projected.append(pts)
        return projected

    def _draw_frame(
        self,
        surface: object,
        node_id: int,
        player_info: dict[str, object] | None,
    ) -> None:
        """Draw stick figures and overlay text onto a pygame surface."""
        import pygame

        surface.fill(BG_COLOR)

        players = self._positions[node_id]  # type: ignore[index]
        projected = self._project_joints(players)

        for p_idx, pts in enumerate(projected):
            color = PLAYER_COLORS[p_idx]

            # Draw segments
            for j_from, j_to in SEGMENTS:
                if j_from < len(pts) and j_to < len(pts):
                    pygame.draw.line(surface, color, pts[j_from], pts[j_to], 3)

            # Draw joint spheres
            for j_idx in SPHERE_JOINTS:
                if j_idx < len(pts):
                    radius = 6 if j_idx == HEAD else 4
                    pygame.draw.circle(surface, color, pts[j_idx], radius)

        # Overlay text
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", 14)

        if player_info:
            lines = [
                f"Position: {player_info.get('description', node_id)}",
                f"P1: {player_info.get('p1_points', 0)} pts  "
                f"P2: {player_info.get('p2_points', 0)} pts  "
                f"Turn: {player_info.get('turn', '?')}",
            ]
        else:
            lines = [f"Node: {node_id}"]

        for i, line in enumerate(lines):
            text_surf = self._font.render(line, True, TEXT_COLOR)
            surface.blit(text_surf, (8, 4 + i * 16))

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
