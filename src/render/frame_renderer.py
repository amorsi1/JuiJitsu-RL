"""Pygame-based 2D stick figure renderer for BJJ positions.

Renders an orthographic front-view (XY plane) of both players using joint
coordinates from nodes.json. Features z-depth shading, anatomical segment
widths, and painter's algorithm draw order. Supports both 'human' (window)
and 'rgb_array' (numpy array) Gymnasium render modes.
"""

from __future__ import annotations

import time
from typing import Callable, NamedTuple

import numpy as np

from render.hud_announcements import (
    AnnouncementEvent,
    AnnouncementSpec,
    WIN_PANEL_BORDER,
    WIN_PANEL_FILL,
    WIN_PANEL_SHADOW,
    build_announcement,
)

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
# HUD overlay constants
# ---------------------------------------------------------------------------

HUD_LARGE_FONT_SIZE: int = 36
HUD_MEDIUM_FONT_SIZE: int = 15
HUD_TURN_FONT_SIZE: int = 20
HUD_MARGIN: int = 8
# Pixels reserved at the top of the frame for the HUD; keep in sync with
# layout offsets in _project_joints and graph_renderer.MARGIN_TOP.
HUD_TOP_RESERVE: int = 100

# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------


_OUTLINE_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
_OUTLINE_COLOR: tuple[int, int, int] = (0, 0, 0)


def draw_hud_overlay(
    surface: object,
    player_info: dict[str, object] | None,
    width: int,
    height: int,
    font_large: object,
    font_medium: object,
    font_turn: object,
    announcement_font_getter: Callable[[int], object],
) -> None:
    """Draw the scoreboard HUD onto *surface*.

    Layout:
    - Very top center: turn number (20 px, outlined).
    - Below center:     current position name in medium text (outlined).
    - Below, top-left:  P1 points in large bold red (outlined).
    - Below, top-right: P2 points in large bold blue (outlined).

    Text outline is achieved by blitting 8 black copies at ±1-px offsets
    before the coloured text — no background rectangle.

    Args:
        surface: pygame Surface to draw onto.
        player_info: Dict with keys 'p1_points', 'p2_points', 'description', 'turn'.
        width: Surface pixel width (used to centre turn and right-align P2).
        font_large: Initialised pygame Font for large player scores.
        font_medium: Initialised pygame Font for position info.
        font_turn: Initialised pygame Font for the turn counter.
    """
    if not player_info:
        return

    p1_pts = player_info.get("p1_points", 0)
    p2_pts = player_info.get("p2_points", 0)
    description = str(player_info.get("description", ""))
    turn = player_info.get("turn", "?")

    # Row 1 — turn number, centred at very top
    turn_text = f"Turn {turn}"
    turn_surf = font_turn.render(turn_text, True, TEXT_COLOR)  # type: ignore[union-attr]
    tw = turn_surf.get_width()
    turn_x = (width - tw) // 2
    th_turn = _blit_outlined(surface, turn_text, TEXT_COLOR, font_turn, turn_x, HUD_MARGIN)

    # Row 2 — position name, centred under turn
    desc_surf = font_medium.render(description, True, TEXT_COLOR)  # type: ignore[union-attr]
    desc_x = (width - desc_surf.get_width()) // 2
    desc_y = HUD_MARGIN + th_turn + 4
    th_desc = _blit_outlined(surface, description, TEXT_COLOR, font_medium, desc_x, desc_y)

    # Row 3 — large player scores
    score_y = desc_y + th_desc + 4
    th_large = _blit_outlined(
        surface,
        f"P1: {p1_pts}", PLAYER_COLORS[0], font_large, HUD_MARGIN, score_y,
    )
    _blit_outlined(
        surface,
        f"P2: {p2_pts}", PLAYER_COLORS[1], font_large,
        width - HUD_MARGIN, score_y, right_align=True,
    )

    # Row 4 — point flash messages (temporary, keyed off p1_flash / p2_flash)
    flash_y = score_y + th_large + 3
    p1_flash = player_info.get("p1_flash")
    if p1_flash:
        _blit_outlined(surface, str(p1_flash), PLAYER_COLORS[0], font_medium, HUD_MARGIN, flash_y)
    p2_flash = player_info.get("p2_flash")
    if p2_flash:
        _blit_outlined(surface, str(p2_flash), PLAYER_COLORS[1], font_medium,
                       width - HUD_MARGIN, flash_y, right_align=True)

    announcement_event = player_info.get("announcement_event")
    if isinstance(announcement_event, AnnouncementEvent):
        spec = build_announcement(announcement_event)
        _draw_announcement(surface, width, height, spec, announcement_font_getter)


def _blit_outlined(
    surface: object,
    text: str,
    color: tuple[int, int, int],
    font: object,
    x: int,
    y: int,
    right_align: bool = False,
) -> int:
    """Blit text with a 1-px black outline. Returns rendered height."""
    colored_surf = font.render(text, True, color)  # type: ignore[union-attr]
    outline_surf = font.render(text, True, _OUTLINE_COLOR)  # type: ignore[union-attr]
    tw, th = colored_surf.get_size()
    rx = x - tw if right_align else x
    for dx, dy in _OUTLINE_OFFSETS:
        surface.blit(outline_surf, (rx + dx, y + dy))  # type: ignore[union-attr]
    surface.blit(colored_surf, (rx, y))  # type: ignore[union-attr]
    return th


def _draw_announcement(
    surface: object,
    width: int,
    height: int,
    spec: AnnouncementSpec,
    announcement_font_getter: Callable[[int], object],
) -> None:
    import pygame

    font = announcement_font_getter(spec.font_size)
    y_center = height // 2 if spec.placement == "center" else height // 3

    if spec.panel_style == "text_only":
        text_surf = font.render(spec.text, True, spec.text_color)  # type: ignore[union-attr]
        text_x = (width - text_surf.get_width()) // 2
        text_y = y_center - (text_surf.get_height() // 2)
        _blit_outlined(surface, spec.text, spec.text_color, font, text_x, text_y)
        return

    panel_width = int(width * 0.7)
    panel_padding_x = 24
    panel_padding_y = 16
    line_spacing = 6
    lines = [spec.text]
    text_width = font.size(spec.text)[0]  # type: ignore[union-attr]
    max_text_width = panel_width - (panel_padding_x * 2)
    if text_width > max_text_width and " won by " in spec.text:
        prefix, suffix = spec.text.split(" won by ", 1)
        lines = [prefix, f"won by {suffix}"]

    line_heights = [font.size(line)[1] for line in lines]  # type: ignore[union-attr]
    text_block_height = sum(line_heights) + line_spacing * max(0, len(lines) - 1)
    panel_height = text_block_height + panel_padding_y * 2

    panel_x = (width - panel_width) // 2
    panel_y = y_center - (panel_height // 2)
    panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
    shadow_rect = panel_rect.move(6, 6)
    corner_radius = 18

    pygame.draw.rect(surface, WIN_PANEL_SHADOW, shadow_rect, border_radius=corner_radius)
    pygame.draw.rect(surface, WIN_PANEL_FILL, panel_rect, border_radius=corner_radius)
    pygame.draw.rect(surface, WIN_PANEL_BORDER, panel_rect, width=4, border_radius=corner_radius)

    y = panel_rect.centery - (text_block_height // 2)
    for i, line in enumerate(lines):
        line_width = font.size(line)[0]  # type: ignore[union-attr]
        line_x = panel_rect.centerx - (line_width // 2)
        _blit_outlined(surface, line, spec.text_color, font, line_x, y)
        y += line_heights[i] + line_spacing


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
        self._font_large: object | None = None
        self._font_medium: object | None = None
        self._font_turn: object | None = None
        self._announcement_fonts: dict[int, object] = {}
        self._raw_transitions: dict[int, dict] | None = None
        self._frame_cache: dict[int, dict] = {}
        self._last_node_id: int | None = None

    def _ensure_positions(self) -> None:
        """Load position data on first use."""
        if self._positions is None:
            from render.position_loader import load_positions
            self._positions = load_positions()

    def _ensure_fonts(self) -> None:
        """Initialize all fonts (safe to call before display is up, e.g. rgb_array mode)."""
        import pygame
        if not pygame.font.get_init():
            pygame.font.init()
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", 14)
        if self._font_large is None:
            self._font_large = pygame.font.SysFont("monospace", HUD_LARGE_FONT_SIZE, bold=True)
        if self._font_medium is None:
            self._font_medium = pygame.font.SysFont("monospace", HUD_MEDIUM_FONT_SIZE)
        if self._font_turn is None:
            self._font_turn = pygame.font.SysFont("monospace", HUD_TURN_FONT_SIZE)

    def _get_announcement_font(self, size: int) -> object:
        import pygame

        if size not in self._announcement_fonts:
            if not pygame.font.get_init():
                pygame.font.init()
            self._announcement_fonts[size] = pygame.font.SysFont("monospace", size, bold=True)
        return self._announcement_fonts[size]

    def _ensure_display(self) -> None:
        """Initialize pygame display for human mode."""
        if self._screen is None:
            import pygame
            pygame.init()
            self._screen = pygame.display.set_mode((self._width, self._height))
            pygame.display.set_caption("BJJEnv")
            self._clock = pygame.time.Clock()
            self._ensure_fonts()

    def _ensure_raw_transitions(self) -> None:
        """Load raw transition index on first use."""
        if self._raw_transitions is None:
            from render.position_loader import load_raw_transition_index
            self._raw_transitions = load_raw_transition_index()

    def _get_transition(self, transition_id: int) -> dict | None:
        """Get parsed transition data, parsing and caching on first access."""
        if transition_id in self._frame_cache:
            return self._frame_cache[transition_id]
        self._ensure_raw_transitions()
        raw = self._raw_transitions.get(transition_id)  # type: ignore[union-attr]
        if raw is None:
            return None
        from render.position_loader import parse_transition
        parsed = parse_transition(raw)
        self._frame_cache[transition_id] = parsed
        del self._raw_transitions[transition_id]  # type: ignore[union-attr]
        return parsed

    def _hold_display(self, duration_s: float, fps: int) -> None:
        """Keep the current frame visible in human mode without redrawing."""
        import pygame

        if self._clock is None or duration_s <= 0:
            return
        stop_at = time.perf_counter() + duration_s
        while time.perf_counter() < stop_at:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            self._clock.tick(max(1, fps))  # type: ignore[union-attr]

    @staticmethod
    def _interpolate_midpoints(
        frames: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Insert linearly interpolated midpoints between consecutive frames.

        Used when detailed=False to double frame count for smoother playback.
        Each frame is 2 players × 23 joints × [x,y,z].
        """
        if len(frames) < 2:
            return frames
        result: list[list[list[list[float]]]] = []
        frames_arr = np.array(frames)  # shape: (N, 2, 23, 3)
        for i in range(len(frames) - 1):
            result.append(frames[i])
            mid = ((frames_arr[i] + frames_arr[i + 1]) / 2.0).tolist()
            result.append(mid)
        result.append(frames[-1])
        return result

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
        draw_h = self._height - 2 * margin - HUD_TOP_RESERVE

        scale = min(draw_w / range_x, draw_h / range_y)

        cx = margin + (draw_w - range_x * scale) / 2
        cy = margin + HUD_TOP_RESERVE + (draw_h - range_y * scale) / 2

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

        # HUD overlay (always on top)
        self._ensure_fonts()
        draw_hud_overlay(
            surface, player_info, self._width, self._height,
            self._font_large, self._font_medium, self._font_turn,
            self._get_announcement_font,
        )

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
        announcement_event: AnnouncementEvent | None = None,
    ) -> np.ndarray | None:
        """Render the current position as stick figures.

        Returns a numpy array (H, W, 3) uint8 for 'rgb_array', None for 'human'.
        """
        import pygame

        self._ensure_positions()
        if announcement_event is not None:
            player_info = dict(player_info or {})
            player_info["announcement_event"] = announcement_event

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
            announcement = player_info.get("announcement_event") if player_info else None
            if isinstance(announcement, AnnouncementEvent):
                spec = build_announcement(announcement)
                self._hold_display(spec.hold_seconds, fps)
            return None

        # rgb_array — render to offscreen surface
        offscreen = pygame.Surface((self._width, self._height))
        self._draw_frame(offscreen, node_id, player_info)
        # surfarray gives (W, H, 3); transpose to (H, W, 3)
        arr = pygame.surfarray.array3d(offscreen)
        return arr.transpose(1, 0, 2).astype(np.uint8)

    def render_transition(
        self,
        transition_id: int,
        node_id: int,
        render_mode: str,
        fps: int = 15,
        player_info: dict[str, object] | None = None,
        announcement_event: AnnouncementEvent | None = None,
    ) -> list[np.ndarray] | None:
        """Render a smooth transition animation.

        Plays back keyframes from the transition at the given fps.
        Returns list of np.ndarray frames for 'rgb_array', None for 'human'.
        Falls back to a static render_frame() if transition data is not found.
        """
        import pygame

        self._ensure_positions()
        if announcement_event is not None:
            player_info = dict(player_info or {})
            player_info["announcement_event"] = announcement_event
        transition = self._get_transition(transition_id)

        if transition is None:
            result = self.render_frame(
                node_id,
                render_mode,
                fps,
                player_info,
                announcement_event=announcement_event,
            )
            return [result] if render_mode == "rgb_array" and result is not None else result

        frames = list(transition['frames'])

        # Reverse detection: if we arrived at from_node from to_node, play backwards
        if (
            self._last_node_id == transition['to_node']
            and node_id == transition['from_node']
        ):
            frames = frames[::-1]

        if not transition['detailed']:
            frames = self._interpolate_midpoints(frames)

        if render_mode == "human":
            self._ensure_display()
            output_frames = None
            for frame_data in frames:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        self._last_node_id = node_id
                        return None
                self._draw_players(self._screen, frame_data, player_info)
                pygame.display.flip()
                self._clock.tick(fps)  # type: ignore[union-attr]
            announcement = player_info.get("announcement_event") if player_info else None
            if isinstance(announcement, AnnouncementEvent):
                spec = build_announcement(announcement)
                self._hold_display(spec.hold_seconds, fps)
        else:  # rgb_array
            output_frames = []
            for frame_data in frames:
                offscreen = pygame.Surface((self._width, self._height))
                self._draw_players(offscreen, frame_data, player_info)
                arr = pygame.surfarray.array3d(offscreen)
                output_frames.append(arr.transpose(1, 0, 2).astype(np.uint8))

        self._last_node_id = node_id
        return output_frames if render_mode == "rgb_array" else None

    def close(self) -> None:
        """Clean up pygame resources."""
        if self._screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._clock = None
            self._font = None
            self._font_large = None
            self._font_medium = None
            self._font_turn = None
            self._announcement_fonts = {}
