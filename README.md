## WIP

High level Jiu Jitsu is about strategy as much as technique — [Gordan Ryan used to write down how he would win a match, then execute it against some of the best competitors in the game](https://www.flograppling.com/video/6943607-gordon-ryan-writes-down-his-prediction-finishes-with-triangle-submission). This is a personal project that aims to train reinforcement learning agents on the game of jui jistu and pin them against each other. To my knowledge this hasn't been done before in Jui Jitsu and is an opportunity to test different models and representations for their effectiveness.

The statespace is based on the [GrappleMap](https://github.com/Eelis/GrappleMap) database, which is a directed graph of 800 positions and 1400 transitions.
![ezgif-faster-grapplemap](https://github.com/user-attachments/assets/d73b2b10-61f6-44e6-93b8-2a1db634f61f)


Rules:
* The game is turn-based, with possible moves being defined by the current node of the graph and the relative position of the player. Currently, players are always in either a top or bottom position, which restricts what move a player can make
* Wins are defined by either a submission (one player's only move is to tap) or based on points at the end of a game
* Points are awarded for executing certain transitions, according to International Brazilian Jiu-Jitsu Federation (IBJJF) rules

Currently, only Q-learning is implemented, but I want to extend this to algorithms like monte-carlo tree search which place more emphasis on planning. Additionally, epsilon-greedy policies can change how often a player goes for submission wins compared to point wins. Statistically, most profesiional BJJ games are won by submissions, so in a perfect simulation agents that go for more submissions should win more 



## Setup

This repo uses `uv` and editable installs for `Game`, `Graph`, and `render`.

```bash
uv sync --extra dev
```

If you want SB3 model-backed play/training:

```bash
uv sync --extra training --extra dev
```

Environment setup:

```bash
cp .env.template .env
```

Set `GRAPH_FILES_DIR` in `.env` to the local path containing GrappleMap files (`nodes.json`, `transitions.json`, `tags.json`, `terminal_node_winstate.json`).

## 3D Visualization Entrypoints

Baseline 3D autoplay:

```bash
uv run python visualize_game_3d.py
```

Human-in-the-loop 3D play:

```bash
uv run python visualize_game_human.py
```

Human UI options:

```bash
# Human as player 2 (blue), random opponent
uv run python visualize_game_human.py --human-side p2

# Human vs MaskablePPO checkpoint
uv run python visualize_game_human.py --agent-type sb3 --model-path models/best.zip
```

CLI help:

```bash
uv run python visualize_game_human.py --help
```

## Human Player UI Design

`visualize_game_human.py` wires `Game` with:

- `Game.human_player.make_human_strategy(...)`: blocks `Player.choose_move(...)` until browser sends a selected destination node.
- `Game.sb3_strategy.make_sb3_strategy(...)`: loads a MaskablePPO checkpoint and chooses legal actions using an action mask built from current `possible_moves`.
- `render.position_server.PositionServer`: broadcasts `legal_moves` and consumes `move_selected`.
- `render/viewer/index.html`: renders a D3 force-layout legal-moves overlay and sends click selections back over WebSocket.

Behavior:

- Graph overlay is shown only on human turns (on `legal_moves` message).
- Overlay is cleared during animation updates (`position`/`transition` messages).
- Human selection is validated against legal moves before a move is applied.

## How To Test Locally

### 1) Fast automated checks for this feature

```bash
uv run pytest \
  src/Game/tests/test_gym_env_helpers.py \
  src/Game/tests/test_player_strategy.py \
  src/Game/tests/test_human_player.py \
  src/Game/tests/test_sb3_strategy.py \
  src/render/tests/test_position_server.py -v
```

### 2) Manual browser verification

Run:

```bash
uv run python visualize_game_human.py
```

Expected flow:

1. Browser opens and renders the Babylon 3D scene.
2. On non-human turns: no graph overlay is visible.
3. On human turns: legal-moves overlay appears with spring-layout motion.
4. Hover increases node opacity; click sends move selection.
5. After click: non-selected nodes dim, selected node recenters, then 3D transition plays.
6. Overlay disappears until the next human turn.

### 3) Manual SB3-opponent verification

Run:

```bash
uv run python visualize_game_human.py --agent-type sb3 --model-path models/best.zip
```

Expected:

- Same human overlay behavior.
- Opponent turns are chosen by the checkpoint policy (no overlay shown).

