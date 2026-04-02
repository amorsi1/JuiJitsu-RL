# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Working Files

All AI-generated notes, plans, reports, and scratch `.md` files must go in `.claude/` — never in the project root. The root is for project files only.

## Commands

This project uses **uv** as the package manager (Python 3.12). The project is configured with a hatchling build backend and installs in editable mode via `uv sync`, making `Game`, `Graph`, and `render` importable directly (no `src.` prefix needed).

```bash
uv sync --extra dev        # Install project + dev dependencies (pytest)
```

Run tests:

```bash
uv run pytest tests/                         # Core game/graph tests
uv run pytest src/Game/tests/               # Gym env + registration tests
uv run pytest src/render/tests/              # Visualization tests
uv run pytest tests/test_position.py::test_swap_players_positions  # Single test
```

Environment variables are loaded via `python-dotenv`. Copy `.env.template` to `.env` and set `GRAPH_FILES_DIR` to the path of the `Graph/files/` directory. The graph constructor uses this variable to locate GrappleMap JSON data.

## Architecture

The project has three layers: **graph**, **game engine**, and **RL environment**.

## Import Convention

All imports use bare package names — never `src.` prefix, never `sys.path` manipulation:

```python
from Game.play_game import Board, GameState, Player
from Graph.graph_constructor import construct_graph
from render.visualizer3d import Visualizer3D
```

This works because `uv sync` installs the project in editable mode, registering `src/Game`, `src/Graph`, and `src/render` as top-level packages.

### 1. Graph Layer (`Graph/`)

`graph_constructor.py` builds a NetworkX `DiGraph` from GrappleMap JSON files:
- **Nodes**: 800 BJJ positions with 3D coordinates, descriptions, and tag metadata
- **Edges**: 1400+ transitions annotated with `swaps_players`, `top`, `bottom`, `reversible` flags

`reward.py` enriches edge attributes in-place after graph construction, flagging point-earning moves (sweep, mount, back, throw, takedown, pass) and tap (submission) edges by matching node tags from `tags.json`. Four tap transitions have known mismatched position properties; three mount transitions have top/bottom tags that need fixing.

### 2. Game Engine (`Game/play_game.py`)

Classes in dependency order:
- **`Board`** — thin wrapper around the NetworkX graph
- **`GameState`** — tracks the current node
- **`Player`** — tracks top/bottom position, points, and strategy
- **`Game`** — orchestrates turns; players alternate selecting from legal outgoing edges filtered by their `top`/`bottom` position; game ends on tap, terminal win state, or max turns
- **`Simulation`** — runs N games in parallel via `ThreadPoolExecutor`

Move legality is based on the `top`/`bottom` edge attributes relative to the acting player's position. When an edge has `swaps_players=True`, the top/bottom assignments flip after the move.

`play_turn()` always leaves the game in a playable state via `ensure_playable_state()`, called after `switch_players()`. This handles two post-turn edge cases: dead-end nodes (reinitialize) and positions where all outgoing edges require the opposite player's role (switch again). Since players have opposite positions, at most one extra switch is needed for non-dead-end nodes.

### 3. RL Environment (`Game/gym_env.py`)

`BJJEnv` wraps the game engine as a Gymnasium environment:
- **Registered as**: `"BJJEnv-v0"` — use `gymnasium.make("BJJEnv-v0")` as the standard entry point. Registration fires on `import Game` via `Game/__init__.py`. An idempotency guard prevents errors on module reload.
- **`disable_env_checker=True`** in the registry — no `PassiveEnvChecker` wrapper overhead during training. Use `gymnasium.make("BJJEnv-v0", disable_env_checker=False)` for development/debugging.
- **`max_episode_steps=None`** — no `TimeLimit` wrapper; `BJJEnv` manages truncation internally via `game.max_turns`.
- **Action space**: `Discrete(n)` where n = total edges in graph (~700+); illegal actions are masked via `info['action_mask']`
- **Observation space**: flat `Box(shape=(5,), dtype=float32)` — `[current_position, point_difference, on_top, on_bottom, turns_left]` — SB3-compatible
- **State index for Q-table**: `position * 2 + is_top` (encoded by `state_to_index()`)
- **Rewards**: +300 win / -300 loss, +1×cumulative point gap (TODO: switch to marginal delta), +0.5×on_top
- **Termination**: `terminated=True` on tap/position win; `truncated=True` on turn limit
- **Render modes**: `"human"` (pygame window with stick figures), `"rgb_array"` (numpy array), `"ansi"` (text), `"graph"` (pygame window with directed graph of traversed positions). Pass via `gymnasium.make("BJJEnv-v0", render_mode="human")`. Renderers are lazily initialized — no pygame overhead during training. Auto-renders on `step()`/`reset()` in human and graph modes. Graph mode records moves before `play_turn()` to capture pre-mutation state.

`q_learning()` is the active standalone training function. It uses epsilon-greedy exploration with configurable decay (`epsilon`, `epsilon_min`, `epsilon_decay` params). `QLearningAgent` is an incomplete class-based wrapper — `_initialize_state_space` still references `self.board` (should be `self.env.G`) and is not used for training.

`check_env(BJJEnv())` passes cleanly as of 2026-03-28.

### 4. Visualization (`render/`)

**Native renderer** (`render/frame_renderer.py`): `FrameRenderer` draws 2D figures using pygame with orthographic XY projection. Features: z-depth shading (closer parts brighter, 0.4–1.0 brightness range), anatomical segment widths from JS viewer proportions (`SEGMENT_DEFS` with `radius_center`), proportional joint radii (`JOINT_RADII`), and painter's algorithm draw order (back-to-front across both players for correct occlusion). Uses `position_loader.load_positions()` (nodes.json only, 4.4 MB) and 28-segment connectivity with `SegmentDef` NamedTuples ported from the JS viewer. Supports `"human"` (pygame window) and `"rgb_array"` (numpy array) modes. Integrated into BJJEnv via `render_mode`.

HUD layout (shared between human and graph render modes via `draw_hud_overlay`):
- Row 1: turn number, centred, 20 px (`HUD_TURN_FONT_SIZE`)
- Row 2: current position name, centred, 15 px (`HUD_MEDIUM_FONT_SIZE`)
- Row 3: P1 score top-left in bold red, P2 score top-right in bold blue, 36 px (`HUD_LARGE_FONT_SIZE`)
- Text outline: 8 black copies at ±1 px offsets before the coloured text — no background rectangles
- `HUD_TOP_RESERVE = 100` px is the vertical space reserved above the figure/graph area

**Graph renderer** (`render/graph_renderer.py`): `GraphRenderer` draws a directed graph of positions visited during gameplay using pygame. Nodes show truncated position names inside circles; node outline color indicates which player is on top (red=P1, blue=P2). Edges are directed arrows colored by who made the move. Uses `networkx.spring_layout` with position seeding for stable incremental layout. A sliding window (default 10 nodes) prunes old nodes to keep the view readable. Integrated into BJJEnv via `render_mode="graph"`. Key dataclasses: `MoveRecord` (captured before `play_turn()` mutates state), `VisibleNode`, `VisibleEdge`.

Layout stability features: existing nodes are pinned via `spring_layout(fixed=...)` so they never move after placement. New nodes are seeded at the angle that maximizes separation from the parent's existing neighbors (`_best_angle`), creating natural branching. Viewport only grows (never shrinks except on prune/reset) with uniform-scale screen mapping. Dim structural edges from the source graph (`source_graph` param, passed as `self.G` from BJJEnv) show connections between visible nodes that weren't traversed, giving topological context. New nodes animate in via ease-out interpolation over ~500ms (12 frames at 24fps).

**Browser renderer** (`render/visualizer3d.py`): `Visualizer3D` uses `position_server.py` (WebSocket server) to stream game state to a browser viewer in real time. Entry points: `visualize_game.py` and `visualize_game_3d.py` at the repo root. Independent of `render_mode`.

## Data Files

All GrappleMap data lives in `Graph/files/`:
- `nodes.json` (4.4 MB) — positions
- `transitions.json` (47.6 MB) — transitions
- `tags.json` — position metadata used for reward tagging
- `terminal_node_winstate.json` — positions flagged as wins

The `GRAPH_FILES_DIR` env var must point to this directory. `graph_constructor.py` currently has a hardcoded fallback path (`/Users/afmorsi/dev/JJ_RL/Graph/files/`) which should be replaced with the env var.
