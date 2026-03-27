# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Working Files

All AI-generated notes, plans, reports, and scratch `.md` files must go in `.claude/` — never in the project root. The root is for project files only.

## Commands

This project uses **uv** as the package manager (Python 3.12).

```bash
uv sync              # Install dependencies
uv sync --extra dev        # Include dev dependencies (pytest)
```

Run tests:
make sure `uv synv --extra dev` was run beforehand

```bash
uv run pytest tests/                         # Core game/graph tests
uv run pytest render/tests/                  # Visualization tests
uv run pytest tests/test_position.py::test_swap_players_positions  # Single test
```

Environment variables are loaded via `python-dotenv`. Copy `.env.template` to `.env` and set `GRAPH_FILES_DIR` to the path of the `Graph/files/` directory. The graph constructor uses this variable to locate GrappleMap JSON data.

## Architecture

The project has three layers: **graph**, **game engine**, and **RL environment**.

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

### 3. RL Environment (`Game/gym_env.py`)

`BJJEnv` wraps the game engine as a Gymnasium environment:
- **Action space**: `Discrete(n)` where n = total edges in graph (~700+); illegal actions are masked via `info['action_mask']`
- **Observation space**: flat `Box(shape=(5,), dtype=float32)` — `[current_position, point_difference, on_top, on_bottom, turns_left]` — SB3-compatible
- **State index for Q-table**: `position * 2 + is_top` (encoded by `state_to_index()`)
- **Rewards**: +300 win / -300 loss, +1×cumulative point gap (TODO: switch to marginal delta), +0.5×on_top
- **Termination**: `terminated=True` on tap/position win; `truncated=True` on turn limit

`q_learning()` is the active standalone training function. It uses epsilon-greedy exploration with configurable decay (`epsilon`, `epsilon_min`, `epsilon_decay` params). `QLearningAgent` is an incomplete class-based wrapper — `_initialize_state_space` still references `self.board` (should be `self.env.G`) and is not used for training.

`check_env(BJJEnv())` passes cleanly as of 2026-03-27.

### 4. Visualization (`render/`)

`Visualizer3D` uses `position_server.py` (WebSocket server) to stream game state to a browser viewer in real time. Entry points: `visualize_game.py` and `visualize_game_3d.py` at the repo root.

## Data Files

All GrappleMap data lives in `Graph/files/`:
- `nodes.json` (4.4 MB) — positions
- `transitions.json` (47.6 MB) — transitions
- `tags.json` — position metadata used for reward tagging
- `terminal_node_winstate.json` — positions flagged as wins

The `GRAPH_FILES_DIR` env var must point to this directory. `graph_constructor.py` currently has a hardcoded fallback path (`/Users/afmorsi/dev/JJ_RL/Graph/files/`) which should be replaced with the env var.
