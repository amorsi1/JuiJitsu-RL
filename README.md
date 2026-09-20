High level Jiu Jitsu is about strategy as much as technique — [Gordan Ryan used to write down how he would win a match, then execute it against some of the best competitors in the game](https://www.flograppling.com/video/6943607-gordon-ryan-writes-down-his-prediction-finishes-with-triangle-submission). This is a personal project that aims to train reinforcement learning agents on the game of jui jistu and pin them against each other. To my knowledge this hasn't been done before in Jui Jitsu and is an opportunity to test different models and representations for their effectiveness.

The statespace is based on the [GrappleMap](https://github.com/Eelis/GrappleMap) database, which is a directed graph of 800 positions and 1400 transitions.
![ezgif-faster-grapplemap](https://github.com/user-attachments/assets/d73b2b10-61f6-44e6-93b8-2a1db634f61f)


**Rules**:
* The game is turn-based, with possible moves being defined by the current node of the graph and the relative position of the player. Currently, players are always in either a top or bottom position, which restricts what move a player can make
* Wins are defined by either a submission (one player's only move is to tap) or based on points at the end of a game
* Points are awarded for executing certain transitions, according to International Brazilian Jiu-Jitsu Federation (IBJJF) rules

Currently, only Q-learning and PPO policies are implemented, but I want to extend this to algorithms like monte-carlo tree search which place more emphasis on planning. Additionally, epsilon-greedy policies can change how often a player goes for submission wins compared to point wins. Statistically, most profesiional BJJ games are won by submissions, so in a perfect simulation agents that go for more submissions should win more 

## Quickstart
This repo uses `uv` and editable installs for `Game`, `Graph`, and `render`.

```
git clone <repo-url> && cd JJ_RL
uv sync                        # installs Python 3.12 + dependencies
cp .env.template .env          # GrappleMap data is bundled in GrappleMap_files/
uv run python play_bjj.py      # opens the browser 3D viewer
```
`GRAPH_FILES_DIR` in `.env` will work by default, referring to the local path containing GrappleMap files (`nodes.json`, `transitions.json`, `tags.json`, `terminal_node_winstate.json`).

If you want SB3 model-backed play/training or to run tests:

```bash
uv sync --extra training --extra dev
```


## Playing a Game

### Configurable match with browser UI (recommended)

`play_bjj.py` is the primary entry point. It opens the browser and shows a config overlay where you choose policies for each player and set max turns / turn delay:
```bash
uv run python play_bjj.py
```

<img width="757" height="443" alt="play_bjj" src="https://github.com/user-attachments/assets/dd5dcc1e-aeed-4483-b219-8735ee5c8125" />

There are several options to run the script headless and set game parameters in the CLI. Discover using:
```bash
uv run python play_bjj.py --help
```

SB3 checkpoints are auto-discovered from `--models-dir` (default: `models/`). Any `.zip` file under that directory is available as policy id `sb3:<relpath-without-extension>` (e.g. `sb3:best`, `sb3:agents/v2`).

### Adding a new agent policy

```python
from Game.policies import PolicySpec, register_policy

register_policy(PolicySpec(
    id="my-agent",
    label="My Agent",
    kind="computer",   # or "human"
    factory=lambda ctx: my_strategy_callable,
    description="Shown in the browser config overlay",
))
```

`factory` receives a `StrategyContext(game, server)` and must return either the string `"random"` or a callable `(possible_moves: list[tuple[int, dict]]) -> tuple[int, dict]`.

### Other Visualization Entrypoints

Baseline 3D autoplay (no browser UI config):

```bash
uv run python visualize_game_3d.py
```

The Gymnasium environment has built-in render modes that can scalably be used to monitor training.
```bash
uv run python -m Game.train_sb3 --eval-render-mode human
```
<img width="400" height="303" alt="render_mode_human" src="https://github.com/user-attachments/assets/cfebf86d-d0c1-4267-9479-f06c4cfbd070" />

```bash
uv run python -m Game.train_sb3 --eval-render-mode graph
```
<img width="400" height="321" alt="render_graph" src="https://github.com/user-attachments/assets/e1935c49-7156-4240-a958-33f67357ca22" /> (e.g. of some undesirable behavior)


## How To Test Locally

### Automated checks

```bash
uv run pytest \
  src/Game/tests/test_gym_env_helpers.py \
  src/Game/tests/test_player_strategy.py \
  src/Game/tests/test_human_player.py \
  src/Game/tests/test_policies.py \
  src/Game/tests/test_game_config_flow.py \
  src/Game/tests/test_game_visualizer_injection.py \
  src/render/tests/test_position_server.py -v
```

### Manual browser verification

Run:

```bash
uv run python play_bjj.py
```

Expected flow:

1. Browser opens and renders the Babylon 3D scene with a config overlay.
2. Choose player policies, max turns, and turn delay; click Start.
3. Overlay disappears and the match begins.
4. On human turns: D3 legal-moves overlay appears; click a node to move.
5. After click: overlay clears, 3D transition plays, game continues.
6. Win announcement shown at end of match.

### Manual SB3-opponent verification

```bash
uv run python play_bjj.py --skip-gui --p1 human --p2 sb3:best
```

Expected:

- Human overlay shown on your turns.
- SB3 checkpoint makes moves on opponent turns (no overlay).

