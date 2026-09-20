# BJJ-RL Training Audit

**Date:** 2026-03-26 | **Last updated:** 2026-03-27
**Scope:** `Game/gym_env.py`, `Game/play_game_QPlayer.py`, `Game/play_game.py`, `Graph/reward.py`

---

## Executive Summary

The game engine and graph construction are largely sound. The Gymnasium environment wrapper and Q-learning implementation have several **critical bugs that prevent correct training**, plus deeper design issues that would limit learning even if the bugs were fixed. The issues are ordered from most to least impactful.

**Status as of 2026-03-27:** All critical bugs (Part 1) resolved. `check_env(BJJEnv())` passes. `play_game_QPlayer.py` deleted. See updated checklist at end of document.

---

## Part 1 — Critical Bugs ✅ All Resolved

### 1.1 Loss penalty is unreachable — **FIXED**

Was: duplicate `elif` condition made `-300` dead code.
Fix: `elif self.game.winner == other_player:` where `other_player = self.game.choose_other_player(self.game.current_player)`.

---

### 1.2 Q-table assigned a type, not an array — **FIXED**

Was: `self.q_table = np.ndarray`.
Fix: `self.q_table = np.zeros((len(self.state_space), self.action_space.n))`.

---

### 1.3 `QLearningAgent._initialize_state_space` references `self.board` — **OUTSTANDING**

`QLearningAgent` in `gym_env.py` is not used for training (the standalone `q_learning()` function is). This class is deferred for a later refactor. The method still references `self.board` instead of `self.env.G`.

---

### 1.4 Wrong argument order in `QLearningAgent.get_q_value` — **OUTSTANDING**

Same class as 1.3 — deferred.

---

### 1.5 `play_game_QPlayer._initialize_action_space` crashes immediately — **DELETED**

`play_game_QPlayer.py` was deleted entirely. Its `Game`, `QLearningAgent`, and `QLearningSim` classes were a divergent fork of `play_game.py` with no Gymnasium integration. The canonical path is `BJJEnv` + `q_learning()`.

---

### 1.6 `_swap_players_positions` does not swap `is_bottom` — **FIXED**

Fixed in both `play_game.py` and `play_game_QPlayer.py` (now deleted). Both tuple positions now correctly cache and assign both `is_top` and `is_bottom`.

---

### 1.7 Module-level execution code — **FIXED**

Both files' top-level training code wrapped in `if __name__ == '__main__':`. `play_game_QPlayer.py` also fixed its wrong API calls before deletion.

---

## Part 2 — Q-Learning Design Issues

### 2.1 Q-table does not encode all observation variables (partial Markov property violation)

The Q-table state index is:
```python
state_index = position * 2 + is_top
```

But the observation returned to the agent includes `point_difference` and `turns_left`, which the Q-table completely ignores. The agent observes these values but cannot act differently based on them because they are not reflected in the state key.

**Consequence:** The agent cannot learn strategies like "I'm ahead by 6 points with 3 turns left — play defensively" vs "I'm behind by 6 points — go for the submission." This is the single most limiting design issue for learning nuanced strategy.

**Options (in order of complexity):**
1. **Remove unused observations** from `_get_obs` — simplest; make the declared observation match what the Q-table actually uses.
2. **Encode score bucket + turns bucket into the state index** — `state_index = position * 2 * num_score_buckets * num_turn_buckets + ...` — grows the table.
3. **Switch to function approximation (DQN)** — the right long-term direction; see Part 4.

---

### 2.2 State space vs. action space size is unfavourable for tabular Q-learning

| Dimension | Size |
|---|---|
| States (`num_nodes * 2`) | ~1,600 |
| Actions (`len(edge_ids)`) | ~700 |
| Q-table entries | ~1,120,000 |

Most state-action pairs are **never valid** (a node has on average ~4 outgoing edges, not 700). The Q-table is 99.4% unused. This wastes memory and slows convergence because zeros in masked positions look identical to "legitimately valued" entries.

**Better representations:**
- Sparse dict-based Q-table keyed on `(state, action)` — only store entries encountered during training.
- Per-node Q-vector of length = number of outgoing edges from that node — eliminates invalid entries entirely.
- DQN with action masking (see Part 4).

---

### 2.3 Action masking applied inconsistently in bootstrapping — `gym_env.py:227`

During the Q-update, the best next action is:
```python
best_next_action = np.argmax(get_masked_q_values(q_table[next_state_index], next_action_mask))
```

This is correct. However, `get_masked_q_values` sets invalid actions to `-inf`. If the Q-table is all zeros initially and no valid action has been visited yet, `np.argmax` on a vector of `-inf` and `0` will pick an unmasked zero entry rather than reflecting true uncertainty. The update should guard against this:

```python
valid_actions = np.where(next_action_mask)[0]
if len(valid_actions) == 0:
    td_target = reward   # terminal; no bootstrapping
else:
    best_next_q = np.max(q_table[next_state_index][valid_actions])
    td_target = reward + discount_factor * best_next_q
```

---

### 2.4 No epsilon decay schedule visible in `q_learning()` — **FIXED**

Added `epsilon_min=0.05` and `epsilon_decay=0.995` as parameters (default `epsilon=1.0`). Decay applied at end of each episode: `epsilon = max(epsilon_min, epsilon * epsilon_decay)`.

---

### 2.5 Reward is cumulative point difference, not marginal points earned — **DEFERRED (TODO)**

`reward += 1 * obs[1]` still uses the cumulative score gap. A TODO comment marks this for future revisit. The current behaviour provides a heuristic reward for being ahead of the opponent, which may be acceptable for the baseline.

---

### 2.6 `terminated` and `truncated` not distinguished — **FIXED**

`step()` now returns:
```python
terminated = self.game.winner is not None   # tap or position win
truncated = (not terminated) and (self.game.turn_count >= self.game.max_turns)
```
The Q-learning update in `q_learning()` skips bootstrapping on `terminated` and bootstraps normally on `truncated`. Also fixed a double `turn_count` increment (play_turn already increments; `step()` was incrementing again).

---

## Part 3 — Gymnasium Environment Compliance

### 3.1 `reset()` may not call `super().reset(seed=seed)`

Gymnasium requires this call to properly seed `self.np_random`, which is the correct way to get reproducible random numbers inside the environment. Without it, `check_env` will warn and seeded reproducibility is not guaranteed.

```python
def reset(self, seed=None, **kwargs):
    super().reset(seed=seed)
    ...
```

### 3.2 Observation bounds not tight

The `current_position` is declared as `Discrete(num_nodes)` where `num_nodes = max(G.nodes())`. If node IDs are not zero-indexed (they are not — GrappleMap uses arbitrary IDs), valid node IDs will fall outside `[0, num_nodes)` and `check_env` will flag an out-of-bounds observation.

**Fix:** Map node IDs to a contiguous `[0, N)` range, or use a `Box(low=0, high=max_node_id, dtype=int)` that exactly covers the real range.

### 3.3 `point_difference` has no declared bounds

If declared as an unbounded `Box` or `Discrete`, this is fine — but it should be explicitly bounded based on the maximum achievable score differential per game. In the worst case: 100 turns × max points per turn. Declare this explicitly so `check_env` can validate it.

### 3.4 Run `check_env` before any training

```python
from gymnasium.utils.env_checker import check_env
env = BJJEnv()
check_env(env)  # will surface the issues above and more
```

This is a prerequisite step. Do not attempt to train until `check_env` passes cleanly.

---

## Part 4 — Plan for Training Effective Models

### Phase 0 — Fix and Validate (prerequisite)

| # | Task | File |
|---|---|---|
| 0.1 | Fix unreachable loss penalty | `gym_env.py:137` |
| 0.2 | Fix Q-table initialisation in `QLearningAgent` | `gym_env.py:256` |
| 0.3 | Fix terminated/truncated split | `gym_env.py:step` |
| 0.4 | Fix reward delta (marginal, not cumulative) | `gym_env.py:142` |
| 0.5 | Add epsilon decay to `q_learning()` | `gym_env.py` |
| 0.6 | Move module-level code to `__main__` | `gym_env.py:313` |
| 0.7 | Fix `_swap_players_positions` in both game files | `play_game.py:143`, `play_game_QPlayer.py:192` |
| 0.8 | Call `super().reset(seed=seed)` in `reset()` | `gym_env.py` |
| 0.9 | Run `check_env(BJJEnv())` with zero failures | — |

### Phase 1 — Baseline Q-Learning (Tabular)

**Goal:** Establish a working baseline that beats a random opponent reliably.

Configuration:
- State: `(position, is_top)` — keep it simple for the baseline
- Q-table: sparse `dict` keyed on `(state, action_edge_id)`
- Epsilon: starts at 1.0, decays by 0.995 per episode, floor 0.05
- Learning rate: 0.1 | Discount: 0.9
- Opponent: random player (already implemented in `play_game.py`)
- Episodes: 50,000 minimum
- Evaluation: every 1,000 episodes, run 200 games against a random opponent and record win rate

**Expected outcome:** Win rate should converge above 60% against a random opponent. If it does not, there is still a bug or the reward signal is not informative enough.

**Logging to add (minimum):**
```python
metrics = {
    'episode': ep,
    'win_rate': wins / eval_games,
    'avg_ep_length': sum(lengths) / len(lengths),
    'epsilon': epsilon,
    'submission_rate': submissions / eval_games,
}
```

### Phase 2 — Richer State Encoding

Once the baseline is working, extend the state to distinguish strategically different situations:

```python
score_bucket = np.clip(point_difference // 3, -3, 3)  # -3 to +3
turn_bucket = min(turns_left // 20, 4)                 # 0 to 4
state_index = (position * 2 + is_top) * 7 * 5 + (score_bucket + 3) * 5 + turn_bucket
```

This grows the table to ~56,000 states (still manageable) and lets the agent learn time- and score-aware strategies.

**Hypothesis to test:** Submission rate should increase when the agent is behind on points and time is running out.

### Phase 3 — Self-Play

Replace the random opponent with a copy of the trained Q-agent to produce a stronger curriculum:

1. Train agent A against random for N episodes.
2. Freeze A as the opponent.
3. Train agent B against frozen A.
4. Evaluate B against random and against A.
5. Repeat, keeping the stronger agent as the new opponent baseline.

Self-play is critical for finding the true optimal strategy because a random opponent doesn't punish exploitable policies.

### Phase 4 — Function Approximation (DQN)

Tabular Q-learning will plateau because:
- It cannot generalise across similar positions (each node is treated independently)
- The 3D positional coordinates on nodes are completely ignored

A DQN uses the node's feature vector as input, enabling generalisation. Recommended architecture:

```
Input: [node_embedding or positional_coords (3D), is_top, score_bucket, turn_bucket]
→ FC(128) → ReLU
→ FC(128) → ReLU
→ FC(num_actions)  with action masking applied to output
```

Libraries to use:
- **Stable-Baselines3** with `MaskablePPO` (from `sb3-contrib`) — easiest integration path, handles the action mask natively
- Alternatively, implement a DQN manually using PyTorch

---

## Summary Checklist

**Bugs to fix before training:**
- [ ] Unreachable loss penalty (`gym_env.py:137`)
- [ ] Q-table initialisation (`gym_env.py:256`)
- [ ] `_swap_players_positions` both files
- [ ] `terminated` vs `truncated` split
- [ ] Reward uses marginal delta, not cumulative difference
- [ ] Epsilon decay added to training loop
- [ ] Module-level execution code moved to `__main__`

**Environment compliance:**
- [ ] `super().reset(seed=seed)` called
- [ ] Observation bounds tight and correct
- [ ] `check_env(BJJEnv())` passes with no warnings

**Training milestones:**
- [ ] Random baseline win rate established (should be ~50%)
- [ ] Tabular Q-agent beats random at >60% after 50k episodes
- [ ] Richer state encoding tested with score/time buckets
- [ ] Self-play loop implemented
- [ ] Submission rate tracked as a separate metric (core research hypothesis)
