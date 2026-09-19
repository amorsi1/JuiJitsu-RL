# TODOs — graph overlay auto-move work

Derived from the work on `f0f1dbc` (auto-move graph animation) and `c033d89` (human
strategy test fix), plus the verification pass over that change. Scoped to what that
work surfaced; not a full-repo audit.

Ordered roughly by severity.

> Items 1 and 2 have a full write-up in [`overlay-bugs.md`](./overlay-bugs.md) —
> root-cause traces, fix options with tradeoffs, and verification plans.

---

## P1 — Bugs

### 1. Interrupted pan drops `move_selected`, hanging the human turn
**`src/render/viewer/index.html:1081`, `:1120`, `:1659-1660`; `src/Game/human_player.py:34`**

`animateOverlayPanToNode` guards stale RAF loops with:

```js
if (overlayState.panAnimId !== myAnimId) return;
```

That returns *before* reaching `onDone`. A superseded pan's callback never fires. The
human click path sends `move_selected` from inside that callback (`:1120-1126`), so
anything that bumps `panAnimId` mid-pan silently cancels the move.

The reachable trigger is the window `resize` handler (`:1659-1660`), which calls
`centerOverlayOnNode` → `panAnimId += 1` (`:800`). Resize the window during the 900 ms
click pan and Python blocks forever at `selected_nodes.get()` (`human_player.py:34`).
The game is dead with no error.

Pre-existing, not introduced by `f0f1dbc`.

**Fix direction:** don't couple protocol side effects to animation completion. Send
`move_selected` on click and let the animation run independently, or give
`animateOverlayPanToNode` an `onCancel`/always-run completion path so superseded pans
still settle their state.

### ~~2. Reverse transitions highlight the wrong node~~ ✅ FIXED (`c451f92`)
**`src/render/position_server.py` — `send_transition` now direction-corrects node ids**

`send_transition` now swaps `from_node`/`to_node` when `reverse=True` so the wire
format means "actual origin → actual destination". `reverse`, `from_reo`, and `to_reo`
stay canonical so `queueTransitionFrames` can pair reos with its frame iteration order
independently.

> **Note:** The original fix direction in this item said *"then drop the `msg.reverse`
> branches in `queueTransitionFrames`"*. **That second half was wrong.** The reo
> composition (`:1568-1570`, `:1589`, `:1599`) is matched to frame iteration direction;
> dropping those branches breaks 3D figure orientation silently. Do not follow that
> instruction. The correct fix is node-id swap only — reos and `reverse` stay canonical.

---

## P2 — Unintended behavior

### 3. Auto-preview orange lands on a node that is already highlighted
**`src/render/viewer/index.html:1344`, `:1367-1368`, CSS `:213` / `:219`**

`mergeTransitionOverlay` sets `toNode.current = true` (`:1344`) before the preview sets
`pendingSelectedId = toId` (`:1367`). The rendered class list is
`graph-node current selected`, so the preview reads as amber `#ffd88f` → orange
`#ffac64` on an *already* highlighted node.

The click path is different: there, orange marks a not-yet-current, dimmer selectable
node, so the highlight is the whole point. The auto path's version is a subtle hue
shift on a node that's already the brightest thing on screen.

Needs a human to look at it. If it doesn't read, options: hold off setting
`toNode.current` until the preview settles, or give `.selected` a distinct treatment
(stroke, radius bump) rather than relying on fill alone.

### 4. Auto-preview state is only cleaned up in the pan callback
**`src/render/viewer/index.html:1377-1386`**

Same root cause as #1. The preview clears `pendingSelectedId`, `moveCommitted`, and
`transitionLink.selectedPath` inside the 200 ms pan's `onDone`. If that pan is
superseded (resize mid-preview), none of it runs and the graph stays dimmed with a
stale orange node.

Not a permanent stuck state — every superseding path (`renderLegalMovesOverlay:1221`,
`mergeTransitionOverlay:1349`, `createOverlayState:725`) clears the flags, and
`clickCommitted` was added specifically so a rapid follow-up transition can't be
misread. But there is a visible interim window where the overlay is wrong.

Falls out of fixing #1 properly.

### 5. `turn_delay` defaults to `0.0`
**`src/render/visualizer3d.py:19`, `:86-87`**

The overlay animations assume some spacing between turns. At the default there is none,
so in computer-vs-computer play transitions arrive back to back and each new pan
supersedes the last mid-flight. `clickCommitted` keeps the *classification* correct, but
pans still stack and previews get cut short.

Consider a minimum inter-transition spacing on the browser side (queue transitions
rather than preempting), or a non-zero default when a visualizer is attached.

---

## P3 — Tech debt

### 6. No test coverage for the overlay state machine
Nothing asserts any of it. Grepping `moveCommitted|mergeTransition|renderLegalMoves|pendingSelected`
across `src/render/tests/` and `src/Game/tests/` returns exactly one hit —
`test_visualizer3d.py:53`, which only checks `'viewer/index.html' in call_arg` for the
browser-open path.

Every behavior in `f0f1dbc` — the `clickCommitted` discriminator, the dimming
exemption, the preview lifecycle — is verified only by reading the code. Both bugs above
are exactly the kind a test would have caught.

The overlay logic is pure state manipulation over `overlayState`; it does not need a
real browser. Extracting it from `index.html` into a module testable under node (or
jsdom) is the unlock. That is a real refactor — worth scoping separately.

### 7. Link dimming now has two overlapping mechanisms
**`src/render/viewer/index.html:1157-1160`**

```js
if (!moveCommitted) return false;
if (d.selectedPath) return false;
return !(d.source.id === currentId && d.target.id === pendingId);
```

The `selectedPath` exemption was added in `f0f1dbc` because during a preview `currentId`
and `pendingId` are both the destination, so the source/target test can never match. But
the two clauses now express the same intent twice.

`selectedPath` was dead state before this — set in three places, threaded through
`buildOverlayRenderLinks:979`, read by nothing. Now that it is live, the source/target
clause is redundant: `onOverlayNodeClick:1105-1109` already sets `selectedPath` on
exactly the link that clause matches. Collapse to `return !d.selectedPath;` and verify
against the click path.

### 8. `.selected` beats `.current` only by CSS source order
**`src/render/viewer/index.html:213-222`**

`.graph-node.current` and `.graph-node.selected` have equal specificity. The preview
renders orange purely because `.selected` is declared second. Reordering the stylesheet
silently breaks the animation with no error anywhere.

Make the win explicit, or stop relying on both classes being present at once (see #3).

### 9. `900` is a literal in two places; `AUTO_MOVE_PAN_MS` is a constant
**`src/render/viewer/index.html:704`, `:1120`, `:1389`**

`f0f1dbc` introduced `AUTO_MOVE_PAN_MS = 200` but left the human-click duration as a
bare `900` at both `:1120` and `:1389`. Extract `HUMAN_MOVE_PAN_MS = 900` alongside it.

### 10. Seed-dependent test branches hide stale assertions
**`src/Game/tests/test_human_player.py:80-101`**

`c033d89` fixed two assertions that `d021014` had invalidated. One of them
(`test_game_play_turn_works_with_human_strategy`) sat behind
`if len(possible_moves) > 1: ... else: ...` against a real unseeded `Game`, so it only
failed when the starting position happened to have exactly one legal move — roughly one
run in three. It read as flake, not staleness, which is why it survived a commit.

The conditional is gone, but the test still constructs an unseeded `Game`. Seed it, or
build a fixture graph with known structure the way the other tests in the file do.

### 11. `uv sync --extra dev` silently removes training extras
**`.claude/CLAUDE.md` (Commands section)**

Running the documented dev-install command uninstalls `torch`, `sb3-contrib`,
`stable-baselines3`, `tensorboard`, and `optuna`. Recovering needs
`uv sync --extra training --extra dev`. Hit during this session's test run.

Relatedly: when the venv lacks `pytest`, `uv run pytest` falls through to the ambient
anaconda Python 3.8 and produces five confusing collection errors rather than a clear
"not installed".

Document the footgun in CLAUDE.md, or make the dev command additive.

---

## Not yet verified

`f0f1dbc` was verified by code reading, a `node --check` on the extracted script, and a
green suite (293 passed). **Nobody has watched the animation.** Plan verification steps
1–4 are browser checks that remain outstanding:

1. Computer turn shows orange preview → 200 ms glide → settles gold
2. Forced human move animates the same way rather than jumping
3. Normal human click still runs the 900 ms path with no duplicate orange
4. Back-to-back transitions leave no ghost `pendingSelectedId`

Open question from #3: whether 200 ms reads as a preview or as a flicker.

---

## Pre-existing markers (surfaced incidentally, out of scope for this work)

- `src/Game/gym_env.py:346` — reward uses cumulative score gap; TODO says switch to
  marginal delta
- `src/Game/gym_env.py:632` — TODO: reconsider or speed up the `best_moves` tie-break
