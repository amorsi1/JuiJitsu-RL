"""Graph-based renderer for BJJ game traversal visualization.

Draws a directed graph of positions visited during gameplay, with nodes
representing BJJ positions and directed edges showing transitions colored
by which player made the move. Structural edges from the source graph are
shown as dim background lines for context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np

from render.frame_renderer import (
    BG_COLOR,
    HUD_LARGE_FONT_SIZE,
    HUD_MEDIUM_FONT_SIZE,
    HUD_TOP_RESERVE,
    HUD_TURN_FONT_SIZE,
    PLAYER_COLORS,
    TEXT_COLOR,
    draw_hud_overlay,
)

# ---------------------------------------------------------------------------
# Graph-specific colors
# ---------------------------------------------------------------------------

GREY: tuple[int, int, int] = (150, 150, 150)
NODE_FILL: tuple[int, int, int] = (40, 40, 60)
STRUCTURAL_EDGE_COLOR: tuple[int, int, int] = (55, 60, 80)

# Node sizing
NORMAL_RADIUS: int = 28
CURRENT_RADIUS: int = 35
NORMAL_OUTLINE: int = 2
CURRENT_OUTLINE: int = 4

# Arrow sizing
ARROWHEAD_SIZE: int = 10
EDGE_WIDTH: int = 2

# Layout margins
MARGIN_X: int = 60
MARGIN_TOP: int = HUD_TOP_RESERVE  # keep in sync with frame_renderer.HUD_TOP_RESERVE

# Animation
ANIM_FRAMES: int = 12
ANIM_FPS: int = 24

# Hierarchical layout
Y_SCALE: float = 0.4       # layout units per terminal-distance level
MIN_NODE_SEP: float = 0.35  # min 2D distance between any two nodes


# ---------------------------------------------------------------------------
# Terminal-distance computation
# ---------------------------------------------------------------------------


def _compute_terminal_distances(
    graph: nx.DiGraph,
) -> tuple[dict[int, int], int]:
    """Compute shortest directed distance from every node to the nearest terminal.

    Uses multi-source BFS on the reversed graph so a single pass covers all sources.
    Terminal nodes are:
      - Nodes with a truthy 'winner' attribute
      - Nodes with zero out-edges (dead ends)
      - Destinations of edges with tap=True

    Args:
        graph: The source directed graph.

    Returns:
        Tuple of (distances, max_dist) where distances maps node_id -> int distance.
        Unreachable nodes are assigned max_dist + 1.
    """
    terminal_nodes: set[int] = set()

    for nid, data in graph.nodes(data=True):
        if data.get("winner") or graph.out_degree(nid) == 0:
            terminal_nodes.add(nid)

    for _u, v, data in graph.edges(data=True):
        if data.get("tap"):
            terminal_nodes.add(v)

    if not terminal_nodes:
        return {}, 0

    rev = graph.reverse(copy=False)
    distances: dict[int, int] = {nid: 0 for nid in terminal_nodes}
    queue: list[int] = list(terminal_nodes)
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for neighbor in rev.successors(node):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    max_dist = max(distances.values()) if distances else 0
    unreachable_dist = max_dist + 1
    for nid in graph.nodes():
        if nid not in distances:
            distances[nid] = unreachable_dist

    return distances, max_dist


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MoveRecord:
    from_node: int
    to_node: int
    mover: int  # 0 = player1, 1 = player2
    p1_is_top: bool
    turn: int
    description: str = ""


@dataclass
class VisibleNode:
    node_id: int
    description: str
    p1_is_top: bool
    first_seen_turn: int
    last_seen_turn: int


@dataclass
class VisibleEdge:
    from_node: int
    to_node: int
    mover: int
    turn: int


# ---------------------------------------------------------------------------
# GraphRenderer
# ---------------------------------------------------------------------------


class GraphRenderer:

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        window_size: int = 10,
        source_graph: nx.DiGraph | None = None,
    ) -> None:
        self._width = width
        self._height = height
        self._window_size = window_size
        self._source_graph = source_graph

        if source_graph is not None:
            self._terminal_distances, self._max_terminal_dist = _compute_terminal_distances(source_graph)
        else:
            self._terminal_distances: dict[int, int] = {}
            self._max_terminal_dist: int = 0

        self._visible_nodes: dict[int, VisibleNode] = {}
        self._visible_edges: list[VisibleEdge] = []
        self._current_node: int | None = None

        # Layout: positions in abstract space, pinned once placed
        self._layout: dict[int, tuple[float, float]] = {}
        self._layout_dirty: bool = True

        # Viewport bounds (only grows; reset on prune or episode reset)
        self._viewport: tuple[float, float, float, float] | None = None

        # Animation: node_id -> progress [0.0, 1.0]
        self._anim_progress: dict[int, float] = {}
        self._anim_parent: dict[int, int] = {}  # animating node -> parent node

        # Pygame resources (lazy init)
        self._screen: object | None = None
        self._clock: object | None = None
        self._font: object | None = None
        self._font_large: object | None = None
        self._font_medium: object | None = None
        self._font_turn: object | None = None
        self._small_font: object | None = None

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def set_initial_state(self, node_id: int, description: str, p1_is_top: bool) -> None:
        self._visible_nodes.clear()
        self._visible_edges.clear()
        self._layout.clear()
        self._current_node = node_id
        self._layout_dirty = True
        self._viewport = None
        self._anim_progress.clear()
        self._anim_parent.clear()

        self._visible_nodes[node_id] = VisibleNode(
            node_id=node_id,
            description=description,
            p1_is_top=p1_is_top,
            first_seen_turn=0,
            last_seen_turn=0,
        )

    def record_move(self, move: MoveRecord) -> None:
        is_new_node = move.to_node not in self._visible_nodes

        if is_new_node:
            self._visible_nodes[move.to_node] = VisibleNode(
                node_id=move.to_node,
                description=move.description or str(move.to_node),
                p1_is_top=move.p1_is_top,
                first_seen_turn=move.turn,
                last_seen_turn=move.turn,
            )
            # Start entrance animation
            self._anim_progress[move.to_node] = 0.0
            self._anim_parent[move.to_node] = move.from_node
            self._layout_dirty = True
        else:
            vn = self._visible_nodes[move.to_node]
            vn.last_seen_turn = move.turn
            vn.p1_is_top = move.p1_is_top

        self._visible_edges.append(VisibleEdge(
            from_node=move.from_node,
            to_node=move.to_node,
            mover=move.mover,
            turn=move.turn,
        ))
        self._current_node = move.to_node
        self._prune_window()

    def render_graph(
        self,
        render_mode: str,
        fps: int = 2,
        player_info: dict[str, object] | None = None,
    ) -> np.ndarray | None:
        import pygame

        self._compute_layout()

        if render_mode == "human":
            self._ensure_display()

            if self._anim_progress:
                # Animate new node entrance over ANIM_FRAMES at ANIM_FPS
                for _ in range(ANIM_FRAMES):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.close()
                            return None
                    # Advance animation
                    finished: list[int] = []
                    for nid in self._anim_progress:
                        self._anim_progress[nid] += 1.0 / ANIM_FRAMES
                        if self._anim_progress[nid] >= 1.0:
                            finished.append(nid)
                    for nid in finished:
                        del self._anim_progress[nid]
                        self._anim_parent.pop(nid, None)
                    self._draw_graph(self._screen, player_info)
                    pygame.display.flip()
                    self._clock.tick(ANIM_FPS)  # type: ignore[union-attr]
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return None
                self._draw_graph(self._screen, player_info)
                pygame.display.flip()
                self._clock.tick(fps)  # type: ignore[union-attr]

            return None

        # rgb_array — skip animation, draw final state
        self._anim_progress.clear()
        self._anim_parent.clear()
        offscreen = pygame.Surface((self._width, self._height))
        self._draw_graph(offscreen, player_info)
        arr = pygame.surfarray.array3d(offscreen)
        return arr.transpose(1, 0, 2).astype(np.uint8)

    def close(self) -> None:
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
            self._small_font = None

    # -------------------------------------------------------------------
    # Internal — pruning
    # -------------------------------------------------------------------

    def _prune_window(self) -> None:
        pruned = False
        while len(self._visible_nodes) > self._window_size:
            oldest_id: int | None = None
            oldest_turn: int = float("inf")  # type: ignore[assignment]
            for nid, vn in self._visible_nodes.items():
                if nid == self._current_node:
                    continue
                if vn.first_seen_turn < oldest_turn:
                    oldest_turn = vn.first_seen_turn
                    oldest_id = nid
            if oldest_id is None:
                break
            del self._visible_nodes[oldest_id]
            self._layout.pop(oldest_id, None)
            self._anim_progress.pop(oldest_id, None)
            self._anim_parent.pop(oldest_id, None)
            self._visible_edges = [
                e for e in self._visible_edges
                if e.from_node != oldest_id and e.to_node != oldest_id
            ]
            pruned = True

        if pruned:
            # Recalculate viewport from remaining nodes so it can shrink
            self._viewport = None
            self._update_viewport()
            self._layout_dirty = True

    # -------------------------------------------------------------------
    # Internal — layout (pinned: existing nodes never move)
    # -------------------------------------------------------------------

    def _compute_layout(self) -> dict[int, tuple[float, float]]:
        if not self._layout_dirty and self._layout:
            return self._layout

        if not self._visible_nodes:
            self._layout = {}
            self._layout_dirty = False
            return self._layout

        new_nodes = [nid for nid in self._visible_nodes if nid not in self._layout]

        if len(self._visible_nodes) == 1:
            nid = next(iter(self._visible_nodes))
            if nid not in self._layout:
                dist = self._terminal_distances.get(nid, self._max_terminal_dist + 1)
                self._layout[nid] = (0.0, -dist * Y_SCALE)
            self._layout_dirty = False
            self._update_viewport()
            return self._layout

        if not new_nodes:
            self._layout_dirty = False
            return self._layout

        for nid in new_nodes:
            dist = self._terminal_distances.get(nid, self._max_terminal_dist + 1)
            target_y = -dist * Y_SCALE
            parent = next(
                (e.from_node for e in self._visible_edges
                 if e.to_node == nid and e.from_node in self._layout),
                None,
            )
            parent_x = self._layout[parent][0] if parent is not None else 0.0
            self._layout[nid] = (self._find_x_position(parent_x, target_y), target_y)

        self._layout_dirty = False
        self._update_viewport()
        return self._layout

    def _find_x_position(self, parent_x: float, target_y: float) -> float:
        """Find an X coordinate at target_y that maintains MIN_NODE_SEP from all existing nodes.

        Tries parent_x first, then alternates ±MIN_NODE_SEP, ±2×MIN_NODE_SEP, ...
        Falls back to a small random offset after 20 attempts.

        Args:
            parent_x: X coordinate of the parent node.
            target_y: Target Y coordinate for the new node.

        Returns:
            X coordinate satisfying the minimum separation constraint.
        """
        existing = list(self._layout.values())

        def _clear(cx: float) -> bool:
            return all(
                math.sqrt((cx - ex) ** 2 + (target_y - ey) ** 2) >= MIN_NODE_SEP
                for ex, ey in existing
            )

        if _clear(parent_x):
            return parent_x

        for step in range(1, 21):
            for sign in (1, -1):
                candidate = parent_x + sign * step * MIN_NODE_SEP
                if _clear(candidate):
                    return candidate

        return parent_x + np.random.uniform(-MIN_NODE_SEP * 0.5, MIN_NODE_SEP * 0.5)

    # -------------------------------------------------------------------
    # Internal — viewport
    # -------------------------------------------------------------------

    def _update_viewport(self) -> None:
        """Expand viewport to encompass all laid-out positions with padding."""
        if not self._layout:
            self._viewport = None
            return

        xs = [p[0] for p in self._layout.values()]
        ys = [p[1] for p in self._layout.values()]
        padding = 0.4
        new_bounds = (min(xs) - padding, min(ys) - padding,
                      max(xs) + padding, max(ys) + padding)

        if self._viewport is None:
            self._viewport = new_bounds
        else:
            # Only expand, never shrink (shrinking happens on prune via reset)
            self._viewport = (
                min(self._viewport[0], new_bounds[0]),
                min(self._viewport[1], new_bounds[1]),
                max(self._viewport[2], new_bounds[2]),
                max(self._viewport[3], new_bounds[3]),
            )

    # -------------------------------------------------------------------
    # Internal — drawing
    # -------------------------------------------------------------------

    def _ensure_display(self) -> None:
        if self._screen is None:
            import pygame
            pygame.init()
            self._screen = pygame.display.set_mode((self._width, self._height))
            pygame.display.set_caption("BJJEnv \u2014 Graph View")
            self._clock = pygame.time.Clock()
            self._ensure_fonts()

    def _ensure_fonts(self) -> None:
        import pygame
        if not pygame.font.get_init():
            pygame.font.init()
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", 14)
        if self._small_font is None:
            self._small_font = pygame.font.SysFont("monospace", 9)
        if self._font_large is None:
            self._font_large = pygame.font.SysFont("monospace", HUD_LARGE_FONT_SIZE, bold=True)
        if self._font_medium is None:
            self._font_medium = pygame.font.SysFont("monospace", HUD_MEDIUM_FONT_SIZE)
        if self._font_turn is None:
            self._font_turn = pygame.font.SysFont("monospace", HUD_TURN_FONT_SIZE)

    def _layout_to_screen(
        self, layout: dict[int, tuple[float, float]]
    ) -> dict[int, tuple[int, int]]:
        """Map layout coordinates to screen pixels using a stable viewport."""
        if not layout or self._viewport is None:
            return {}

        vp_min_x, vp_min_y, vp_max_x, vp_max_y = self._viewport
        vp_w = vp_max_x - vp_min_x or 1.0
        vp_h = vp_max_y - vp_min_y or 1.0

        draw_w = self._width - 2 * MARGIN_X
        draw_h = self._height - MARGIN_TOP - MARGIN_X

        # Uniform scale preserves aspect ratio
        scale = min(draw_w / vp_w, draw_h / vp_h)
        offset_x = MARGIN_X + (draw_w - vp_w * scale) / 2
        offset_y = MARGIN_TOP + (draw_h - vp_h * scale) / 2

        screen_pos: dict[int, tuple[int, int]] = {}
        for nid, (lx, ly) in layout.items():
            sx = int(offset_x + (lx - vp_min_x) * scale)
            sy = int(offset_y + (ly - vp_min_y) * scale)
            screen_pos[nid] = (sx, sy)

        return screen_pos

    def _draw_graph(
        self,
        surface: object,
        player_info: dict[str, object] | None,
    ) -> None:
        import pygame

        surface.fill(BG_COLOR)  # type: ignore[union-attr]

        layout = self._compute_layout()
        if not layout:
            self._draw_overlay(surface, player_info)
            return

        self._ensure_fonts()

        screen_pos = self._layout_to_screen(layout)
        if not screen_pos:
            self._draw_overlay(surface, player_info)
            return

        # Compute effective positions (with animation interpolation)
        effective_pos = self._effective_positions(screen_pos)

        # 1. Structural edges (dim, underneath everything)
        self._draw_structural_edges(surface, effective_pos)

        # 2. Traversal edges (arrows)
        for edge in self._visible_edges:
            if edge.from_node in effective_pos and edge.to_node in effective_pos:
                color = PLAYER_COLORS[edge.mover]
                start = effective_pos[edge.from_node]
                end = effective_pos[edge.to_node]
                from_r = CURRENT_RADIUS if edge.from_node == self._current_node else NORMAL_RADIUS
                to_r = CURRENT_RADIUS if edge.to_node == self._current_node else NORMAL_RADIUS
                # Scale target radius for animating nodes
                if edge.to_node in self._anim_progress:
                    t = _ease_out(self._anim_progress[edge.to_node])
                    to_r = max(4, int(to_r * t))
                self._draw_arrow(surface, start, end, color, from_r, to_r)

        # 3. Nodes (on top)
        for nid, vn in self._visible_nodes.items():
            if nid not in effective_pos:
                continue
            is_current = nid == self._current_node
            full_radius = CURRENT_RADIUS if is_current else NORMAL_RADIUS
            outline_w = CURRENT_OUTLINE if is_current else NORMAL_OUTLINE
            center = effective_pos[nid]

            # Animation scaling
            if nid in self._anim_progress:
                t = _ease_out(self._anim_progress[nid])
                radius = max(4, int(full_radius * t))
                outline_w = max(1, int(outline_w * t))
            else:
                radius = full_radius

            outline_color = PLAYER_COLORS[0] if vn.p1_is_top else PLAYER_COLORS[1]

            pygame.draw.circle(surface, NODE_FILL, center, radius)
            pygame.draw.circle(surface, outline_color, center, radius, outline_w)

            # Only draw text when animation is mostly complete
            if nid not in self._anim_progress or self._anim_progress[nid] > 0.6:
                lines = self._wrap_text(vn.description, max_chars_per_line=12)[:3]
                total_h = len(lines) * 11
                y_start = center[1] - total_h // 2
                for i, line in enumerate(lines):
                    text_surf = self._small_font.render(line, True, TEXT_COLOR)
                    text_rect = text_surf.get_rect(center=(center[0], y_start + i * 11))
                    surface.blit(text_surf, text_rect)  # type: ignore[union-attr]

        self._draw_overlay(surface, player_info)

    def _effective_positions(
        self, screen_pos: dict[int, tuple[int, int]]
    ) -> dict[int, tuple[int, int]]:
        """Apply animation interpolation: animating nodes lerp from parent to target."""
        if not self._anim_progress:
            return screen_pos

        result = dict(screen_pos)
        for nid, progress in self._anim_progress.items():
            if nid not in screen_pos:
                continue
            parent_id = self._anim_parent.get(nid)
            if parent_id is None or parent_id not in screen_pos:
                continue
            t = _ease_out(progress)
            ox, oy = screen_pos[parent_id]
            tx, ty = screen_pos[nid]
            result[nid] = (int(ox + t * (tx - ox)), int(oy + t * (ty - oy)))
        return result

    def _draw_structural_edges(
        self, surface: object, screen_pos: dict[int, tuple[int, int]]
    ) -> None:
        """Draw dim lines for source-graph edges between visible nodes (not traversed)."""
        import pygame

        if self._source_graph is None:
            return

        traversed = {(e.from_node, e.to_node) for e in self._visible_edges}
        visible_ids = set(self._visible_nodes.keys())
        drawn: set[tuple[int, int]] = set()

        for nid in visible_ids:
            if nid not in self._source_graph or nid not in screen_pos:
                continue
            for neighbor in self._source_graph.successors(nid):
                if (neighbor in visible_ids
                        and neighbor in screen_pos
                        and (nid, neighbor) not in traversed
                        and (nid, neighbor) not in drawn):
                    pygame.draw.line(
                        surface, STRUCTURAL_EDGE_COLOR,
                        screen_pos[nid], screen_pos[neighbor], 1,
                    )
                    drawn.add((nid, neighbor))

    def _draw_arrow(
        self,
        surface: object,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        from_radius: int,
        to_radius: int,
    ) -> None:
        import pygame

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1:
            return

        ux = dx / length
        uy = dy / length

        sx = start[0] + ux * from_radius
        sy = start[1] + uy * from_radius
        ex = end[0] - ux * to_radius
        ey = end[1] - uy * to_radius

        shortened_len = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
        if shortened_len < 1:
            return

        pygame.draw.line(surface, color, (int(sx), int(sy)), (int(ex), int(ey)), EDGE_WIDTH)

        # Arrowhead triangle
        px, py = -uy, ux
        base_x = ex - ux * ARROWHEAD_SIZE
        base_y = ey - uy * ARROWHEAD_SIZE
        left = (int(base_x + px * ARROWHEAD_SIZE * 0.5), int(base_y + py * ARROWHEAD_SIZE * 0.5))
        right = (int(base_x - px * ARROWHEAD_SIZE * 0.5), int(base_y - py * ARROWHEAD_SIZE * 0.5))
        pygame.draw.polygon(surface, color, [(int(ex), int(ey)), left, right])

    def _draw_overlay(
        self,
        surface: object,
        player_info: dict[str, object] | None,
    ) -> None:
        self._ensure_fonts()

        if player_info:
            draw_hud_overlay(
                surface, player_info, self._width,
                self._font_large, self._font_medium, self._font_turn,
            )
        else:
            import pygame
            text_surf = self._font.render(  # type: ignore[union-attr]
                f"Node: {self._current_node}", True, TEXT_COLOR,
            )
            surface.blit(text_surf, (8, 4))  # type: ignore[union-attr]

    @staticmethod
    def _wrap_text(text: str, max_chars_per_line: int = 12) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current_line = ""
        for word in words:
            if not current_line:
                current_line = word
            elif len(current_line) + 1 + len(word) <= max_chars_per_line:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ease_out(t: float) -> float:
    """Quadratic ease-out: starts fast, decelerates to stop."""
    return 1.0 - (1.0 - min(t, 1.0)) ** 2
