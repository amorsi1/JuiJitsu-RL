"""Graph-based renderer for BJJ game traversal visualization.

Draws a directed graph of positions visited during gameplay, with nodes
representing BJJ positions and directed edges showing transitions colored
by which player made the move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from render.frame_renderer import BG_COLOR, PLAYER_COLORS, TEXT_COLOR

# ---------------------------------------------------------------------------
# Graph-specific colors
# ---------------------------------------------------------------------------

GREY: tuple[int, int, int] = (150, 150, 150)
NODE_FILL: tuple[int, int, int] = (40, 40, 60)

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
MARGIN_TOP: int = 40


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

    def __init__(self, width: int = 800, height: int = 600, window_size: int = 10) -> None:
        self._width = width
        self._height = height
        self._window_size = window_size

        self._visible_nodes: dict[int, VisibleNode] = {}
        self._visible_edges: list[VisibleEdge] = []
        self._current_node: int | None = None

        # Layout caching
        self._cached_layout: dict[int, tuple[float, float]] = {}
        self._layout_dirty: bool = True

        # Pygame resources (lazy init)
        self._screen: object | None = None
        self._clock: object | None = None
        self._font: object | None = None
        self._small_font: object | None = None

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def set_initial_state(self, node_id: int, description: str, p1_is_top: bool) -> None:
        self._visible_nodes.clear()
        self._visible_edges.clear()
        self._cached_layout.clear()
        self._current_node = node_id
        self._layout_dirty = True

        self._visible_nodes[node_id] = VisibleNode(
            node_id=node_id,
            description=description,
            p1_is_top=p1_is_top,
            first_seen_turn=0,
            last_seen_turn=0,
        )

    def record_move(self, move: MoveRecord) -> None:
        # Update or create to_node
        if move.to_node in self._visible_nodes:
            vn = self._visible_nodes[move.to_node]
            vn.last_seen_turn = move.turn
            vn.p1_is_top = move.p1_is_top
        else:
            # Need description from the graph — use node_id as fallback
            self._visible_nodes[move.to_node] = VisibleNode(
                node_id=move.to_node,
                description=str(move.to_node),
                p1_is_top=move.p1_is_top,
                first_seen_turn=move.turn,
                last_seen_turn=move.turn,
            )
            # Seed new node position near parent in cached layout
            if move.from_node in self._cached_layout:
                px, py = self._cached_layout[move.from_node]
                jitter_x = (np.random.random() - 0.5) * 0.2
                jitter_y = (np.random.random() - 0.5) * 0.2
                self._cached_layout[move.to_node] = (px + jitter_x, py + jitter_y)
            self._layout_dirty = True

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

        if render_mode == "human":
            self._ensure_display()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            self._draw_graph(self._screen, player_info)
            pygame.display.flip()
            self._clock.tick(fps)  # type: ignore[union-attr]
            return None

        # rgb_array
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
            self._small_font = None

    # -------------------------------------------------------------------
    # Internal — pruning
    # -------------------------------------------------------------------

    def _prune_window(self) -> None:
        while len(self._visible_nodes) > self._window_size:
            # Find the node with the oldest first_seen_turn, never prune current
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
            self._cached_layout.pop(oldest_id, None)
            # Remove edges referencing the pruned node
            self._visible_edges = [
                e for e in self._visible_edges
                if e.from_node != oldest_id and e.to_node != oldest_id
            ]
            self._layout_dirty = True

    # -------------------------------------------------------------------
    # Internal — layout
    # -------------------------------------------------------------------

    def _compute_layout(self) -> dict[int, tuple[float, float]]:
        if not self._layout_dirty and self._cached_layout:
            return self._cached_layout

        g = nx.DiGraph()
        for nid in self._visible_nodes:
            g.add_node(nid)
        for edge in self._visible_edges:
            if edge.from_node in self._visible_nodes and edge.to_node in self._visible_nodes:
                g.add_edge(edge.from_node, edge.to_node)

        # Seed positions from cache for stable layout
        pos_seed: dict[int, tuple[float, float]] | None = None
        if self._cached_layout:
            pos_seed = {
                nid: self._cached_layout[nid]
                for nid in g.nodes
                if nid in self._cached_layout
            }
            if not pos_seed:
                pos_seed = None

        if len(g.nodes) == 0:
            self._cached_layout = {}
        elif len(g.nodes) == 1:
            nid = next(iter(g.nodes))
            self._cached_layout = {nid: (0.0, 0.0)}
        else:
            self._cached_layout = nx.spring_layout(
                g, pos=pos_seed, seed=42, iterations=50, k=1.5 / math.sqrt(len(g.nodes))
            )

        self._layout_dirty = False
        return self._cached_layout

    # -------------------------------------------------------------------
    # Internal — drawing
    # -------------------------------------------------------------------

    def _ensure_display(self) -> None:
        if self._screen is None:
            import pygame
            pygame.init()
            self._screen = pygame.display.set_mode((self._width, self._height))
            pygame.display.set_caption("BJJEnv — Graph View")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("monospace", 14)
            self._small_font = pygame.font.SysFont("monospace", 9)

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

        # Ensure fonts are initialized
        if self._font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.SysFont("monospace", 14)
        if self._small_font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._small_font = pygame.font.SysFont("monospace", 9)

        # Map layout coordinates [-1, 1] to screen pixels with margins
        screen_pos = self._layout_to_screen(layout)

        # Draw edges first (below nodes)
        for edge in self._visible_edges:
            if edge.from_node in screen_pos and edge.to_node in screen_pos:
                color = PLAYER_COLORS[edge.mover]
                start = screen_pos[edge.from_node]
                end = screen_pos[edge.to_node]
                from_radius = CURRENT_RADIUS if edge.from_node == self._current_node else NORMAL_RADIUS
                to_radius = CURRENT_RADIUS if edge.to_node == self._current_node else NORMAL_RADIUS
                self._draw_arrow(surface, start, end, color, from_radius, to_radius)

        # Draw nodes on top
        for nid, vn in self._visible_nodes.items():
            if nid not in screen_pos:
                continue
            is_current = nid == self._current_node
            radius = CURRENT_RADIUS if is_current else NORMAL_RADIUS
            outline_w = CURRENT_OUTLINE if is_current else NORMAL_OUTLINE
            center = screen_pos[nid]

            # Outline color based on top/bottom
            outline_color = PLAYER_COLORS[0] if vn.p1_is_top else PLAYER_COLORS[1]

            # Filled circle
            pygame.draw.circle(surface, NODE_FILL, center, radius)
            pygame.draw.circle(surface, outline_color, center, radius, outline_w)

            # Text inside node
            lines = self._wrap_text(vn.description, max_chars_per_line=12)
            lines = lines[:3]  # max 3 lines
            total_h = len(lines) * 11
            y_start = center[1] - total_h // 2
            for i, line in enumerate(lines):
                text_surf = self._small_font.render(line, True, TEXT_COLOR)
                text_rect = text_surf.get_rect(center=(center[0], y_start + i * 11))
                surface.blit(text_surf, text_rect)  # type: ignore[union-attr]

        self._draw_overlay(surface, player_info)

    def _layout_to_screen(
        self, layout: dict[int, tuple[float, float]]
    ) -> dict[int, tuple[int, int]]:
        if not layout:
            return {}

        xs = [p[0] for p in layout.values()]
        ys = [p[1] for p in layout.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        range_x = max_x - min_x or 1.0
        range_y = max_y - min_y or 1.0

        draw_w = self._width - 2 * MARGIN_X
        draw_h = self._height - MARGIN_TOP - MARGIN_X  # bottom margin same as side

        screen_pos: dict[int, tuple[int, int]] = {}
        for nid, (lx, ly) in layout.items():
            sx = int(MARGIN_X + (lx - min_x) / range_x * draw_w)
            sy = int(MARGIN_TOP + (ly - min_y) / range_y * draw_h)
            screen_pos[nid] = (sx, sy)

        return screen_pos

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

        # Unit direction
        ux = dx / length
        uy = dy / length

        # Shorten both ends by respective node radii
        sx = start[0] + ux * from_radius
        sy = start[1] + uy * from_radius
        ex = end[0] - ux * to_radius
        ey = end[1] - uy * to_radius

        # Check we still have a positive-length line after shortening
        shortened_dx = ex - sx
        shortened_dy = ey - sy
        shortened_len = math.sqrt(shortened_dx * shortened_dx + shortened_dy * shortened_dy)
        if shortened_len < 1:
            return

        pygame.draw.line(surface, color, (int(sx), int(sy)), (int(ex), int(ey)), EDGE_WIDTH)

        # Arrowhead triangle at target end
        # Perpendicular direction
        px = -uy
        py = ux
        tip_x, tip_y = ex, ey
        base_x = ex - ux * ARROWHEAD_SIZE
        base_y = ey - uy * ARROWHEAD_SIZE
        left = (int(base_x + px * ARROWHEAD_SIZE * 0.5), int(base_y + py * ARROWHEAD_SIZE * 0.5))
        right = (int(base_x - px * ARROWHEAD_SIZE * 0.5), int(base_y - py * ARROWHEAD_SIZE * 0.5))
        pygame.draw.polygon(surface, color, [(int(tip_x), int(tip_y)), left, right])

    def _draw_overlay(
        self,
        surface: object,
        player_info: dict[str, object] | None,
    ) -> None:
        import pygame

        if self._font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.SysFont("monospace", 14)

        if player_info:
            lines = [
                f"Position: {player_info.get('description', '?')}",
                f"P1: {player_info.get('p1_points', 0)} pts  "
                f"P2: {player_info.get('p2_points', 0)} pts  "
                f"Turn: {player_info.get('turn', '?')}",
            ]
        else:
            lines = [f"Node: {self._current_node}"]

        for i, line in enumerate(lines):
            text_surf = self._font.render(line, True, TEXT_COLOR)
            surface.blit(text_surf, (8, 4 + i * 16))  # type: ignore[union-attr]

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
