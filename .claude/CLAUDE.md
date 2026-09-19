# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Working Files

All AI-generated notes, plans, reports, and scratch `.md` files must go in `.claude/` — never in the project root. The root is for project files only.

## Commands

This project uses **uv** as the package manager (Python 3.12). The project is configured with a hatchling build backend and installs in editable mode via `uv sync`, making `Game`, `Graph`, and `render` importable directly (no `src.` prefix needed).

```bash
uv sync --extra dev                  # Install project + dev dependencies (pytest)
uv sync --extra training --extra dev # Also install SB3 training dependencies (sb3-contrib, torch)
```

Run tests:

```bash
uv run pytest tests/                         # Core game/graph tests
uv run pytest src/Game/tests/               # Gym env + registration + SB3 tests
uv run pytest src/render/tests/              # Visualization tests
uv run pytest tests/test_position.py::test_swap_players_positions  # Single test
uv run pytest src/Game/tests/test_human_player.py src/Game/tests/test_sb3_strategy.py src/render/tests/test_position_server.py  # Human UI + protocol
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

`Player.strategy` can be either:
- a string strategy key (currently `random`)
- a callable that accepts `possible_moves: list[tuple[int, dict]]` and returns a selected move tuple

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

`action_masks()` is the public interface for sb3-contrib's MaskablePPO (delegates to `_get_action_mask()`).

Reusable helper functions are module-level for cross-module reuse:
- `build_action_edge_maps(graph)` — edge-ID/action-index mapping used by `BJJEnv` and SB3 strategy adapter
- `build_obs(game_state, player, turns_left, other_player)` — canonical flat observation builder

`check_env(BJJEnv())` passes cleanly as of 2026-03-28.

### 3b. SB3 Training (`Game/train_sb3.py`)

`train()` trains an agent on BJJEnv with action masking. Supports two algorithms via the `ALGORITHMS` registry:

| Key | Class | Default n_steps | Default batch_size |
|-----|-------|----------------|-------------------|
| `"maskable_ppo"` | `MaskablePPO` | 2048 | 64 |
| `"recurrent_ppo"` | `MaskableRecurrentPPO` | 128 | 128 |

Run via CLI:
```bash
uv run python -m Game.train_sb3                                    # MaskablePPO (default)
uv run python -m Game.train_sb3 --algorithm recurrent_ppo          # MaskableRecurrentPPO
uv run python -m Game.train_sb3 --algorithm recurrent_ppo --total-timesteps 100000
```

`load_and_evaluate(model_path, algorithm)` reloads and evaluates a saved model. Both `train()` and `load_and_evaluate()` accept `algorithm` to dispatch to the correct class. Requires `uv sync --extra training`.

### 3c. MaskableRecurrentPPO (`Game/maskable_recurrent/`)

Custom algorithm combining `RecurrentPPO`'s LSTM memory with `MaskablePPO`'s action masking. No diamond inheritance — `MaskableRecurrentActorCriticPolicy` single-inherits from `RecurrentActorCriticPolicy` and grafts in masking by replacing `self.action_dist` with `make_masked_proba_distribution()` after `super().__init__()`, then rebuilding `action_net` and the optimizer.

Key design details:
- **Buffer**: `MaskableRecurrentRolloutBuffer` stores `action_masks` shape `(buffer_size, n_envs, n_actions)`, initialised all-ones. Padded timesteps use `padding_value=1.0` (not 0.0) to avoid `log(0)` in masked distributions.
- **Policy alias**: `MlpLstmPolicy = MaskableRecurrentActorCriticPolicy` — use `"MlpLstmPolicy"` as the policy string.
- **Training**: `model.learn(total_timesteps=N, use_masking=True)`
- **Prediction**: `model.predict(obs, action_masks=masks)` — masks forwarded through policy to distribution.

### 3d. Human + SB3 Strategy Adapters (`Game/human_player.py`, `Game/sb3_strategy.py`)

- `make_human_strategy(server, game_state)`:
  - sends current legal moves via `PositionServer.send_legal_moves(...)`
  - drains stale queue entries **before** sending the prompt (`_drain_stale_selections`), never after — forced moves return without consuming the queue, so a click on a forced move would otherwise be eaten by the next turn
  - blocks on a queue until the browser sends `move_selected`
  - validates selected node against legal `possible_moves`

  Invariant: **one `move_selected` per `legal_moves` message.** Python does exactly one
  queue read per human turn, so a duplicate send desyncs every turn after. The browser
  side enforces this with `overlayState.moveSent`, reset only in
  `renderLegalMovesOverlay`; the drain above is the server-side net.

  `onOverlayNodeClick` sends `move_selected` **at click time**, not from the pan's
  `onDone` — `animateOverlayPanToNode`'s staleness guard returns before invoking
  `onDone`, so a superseded pan (e.g. a window resize) used to cancel the send and
  deadlock the game. Keep protocol sends out of animation callbacks.

  Test stubs must answer the `legal_moves` prompt (`StubPositionServer.reply_with`)
  rather than pre-queueing a selection — pre-queued entries are now drained.
- `make_sb3_strategy(model_path, game)`:
  - loads a `MaskablePPO` checkpoint
  - builds obs from live `Game` state using `build_obs(...)`
  - builds action mask from current `possible_moves`
  - chooses a legal move from `model.predict(..., action_masks=mask, deterministic=True)`

### 3e. Policy Registry (`Game/policies.py`)

Extensible registry for `Player.strategy` factories. All built-ins are registered at import time.

Key types:
- `PolicySpec(id, label, kind, factory, description)` — frozen dataclass; `factory: Callable[[StrategyContext], Strategy]`
- `StrategyContext(game, server=None)` — context passed to each factory

Key functions:
- `register_policy(spec)` — raises `ValueError` on duplicate id
- `get_policy(id)` — raises `ValueError` listing known ids on unknown
- `available_policies()` — sorted list of all registered specs
- `build_config_options(max_turns_default, turn_delay_default)` — JSON-ready dict with `settings` and `policies` keys (no factory field); sent to browser as `config_options` message
- `discover_sb3_policies(models_dir: Path)` — globs `**/*.zip`, returns fresh `PolicySpec` list (id format: `sb3:<relpath-sans-ext>`); does NOT import torch/sb3 at discovery time; does NOT register (caller does)
- `resolve_strategies(config_players, ctx)` — maps `{player_key: {"type": ..., "policy_id": ...}}` to `{player_key: Strategy}`; memoizes per policy_id so human-vs-human shares one instance

Built-in policy ids: `"random"` (kind `computer`), `"human"` (kind `human`; raises if `ctx.server is None`).

To add a new policy:
```python
from Game.policies import PolicySpec, StrategyContext, register_policy
register_policy(PolicySpec(
    id="my-policy", label="My Policy", kind="computer",
    factory=lambda ctx: my_strategy_function,
))
```

### 3f. Game Config Flow (`Game/game_config.py`)

Bridges the browser config UI with game construction.

- `try_apply_config(cfg, visualizer, defaults, logger) -> Game | None`:
  - Validates/clamps `max_turns` and `turn_delay` from `cfg['settings']`
  - Calls `resolve_strategies` before `initialize_game`
  - On `ValueError`/`ImportError`: calls `visualizer.server.send_config_error(...)` and returns None
  - On success: clears `on_game_config`/`config_options`, sets `visualizer.turn_delay`, calls `game.initialize_game`, assigns strategies, returns Game

- `run_configured_game(visualizer, *, defaults, logger, config_source=None) -> Game`:
  - If `config_source` given: applies directly (no browser interaction)
  - Otherwise: broadcasts `build_config_options()` via `set_config_options`, loops on `on_game_config` queue until a valid config arrives

`defaults` dict keys: `max_turns_default`, `max_turns_min`, `max_turns_max`, `turn_delay_default`.

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

**Browser renderer** (`render/visualizer3d.py`): `Visualizer3D` uses `position_server.py` (WebSocket server) to stream game state to a browser viewer in real time. Entry points: `visualize_game.py`, `visualize_game_3d.py`, `visualize_game_human.py`, and `play_bjj.py` at the repo root. Independent of `render_mode`.

`Visualizer3D(port=8765, open_browser=True, ...)` appends `?port={port}` to the `file://` URL so the viewer connects to the correct server port automatically.

**Config protocol** (new in `position_server.py`):
- `set_config_options(options: dict | None)` — stores and broadcasts `{type: 'config_options', ...options}` to connected clients; `None` clears (no broadcast)
- `send_config_error(message: str)` — broadcasts `{type: 'config_error', message}`
- `on_game_config: Callable[[dict], None] | None` — set this to receive inbound `game_config` messages from the browser
- `config_options: dict | None` — replayed to newly connecting clients when set before they connect

`play_bjj.py` is the primary entry point for configurable matches:
- Opens the browser config overlay automatically
- Discovers `models/*.zip` SB3 checkpoints and makes them selectable
- Supports `--skip-gui --p1 <policy_id> --p2 <policy_id>` to bypass the overlay

`visualize_game_human.py` starts a legacy fixed-role human-vs-agent match (no config overlay):
- `--human-side {p1,p2}`: choose which player the human controls
- `--agent-type {random,sb3}`: random opponent (default) or MaskablePPO checkpoint
- `--model-path PATH`: required when `--agent-type sb3`

`position_server.py` WebSocket protocol summary:
- outbound `config_options`: config UI payload from `build_config_options()`
- outbound `config_error`: error from `send_config_error()`
- outbound `legal_moves`: `{type, current_node, moves:[{to_node, transition_id, description}]}`
- outbound `transition`: `{type, transition_id, reverse, frames, detailed, from_node, to_node, from_reo, to_reo}` — **asymmetric contract**: `from_node`/`to_node` are direction-corrected (actual origin → actual destination); `reverse`, `from_reo`, `to_reo` stay canonical so the 3D frame player can pair reos with frame iteration order.
- inbound `game_config`: full config payload from browser start button
- inbound `move_selected`: `{type: "move_selected", to_node}`

`viewer/index.html` overlay behaviors:
- `#configOverlay` shown on `config_options`; Start button sends `game_config`; error shown on `config_error`
- `#graphOverlay` (D3 legal-moves) shown on `legal_moves`; click sends `move_selected`
- Both overlays cleared on `position`/`transition` messages

## Data Files

All GrappleMap data lives in `Graph/files/`:
- `nodes.json` (4.4 MB) — positions
- `transitions.json` (47.6 MB) — transitions
- `tags.json` — position metadata used for reward tagging
- `terminal_node_winstate.json` — positions flagged as wins

The `GRAPH_FILES_DIR` env var must point to this directory. `graph_constructor.py` currently has a hardcoded fallback path (`/Users/afmorsi/dev/JJ_RL/Graph/files/`) which should be replaced with the env var.
