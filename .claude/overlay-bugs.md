# Two graph-overlay bugs: dropped `move_selected` and unswapped reverse transitions

Written for an agent picking these up cold. Both are **pre-existing** — neither was
introduced by `f0f1dbc` (auto-move graph animation) — but that commit makes #2 more
visible and inherits #1's fragility.

Line numbers are against `ef9ed74` on `amorsi1/human-player-ui`.

---

## Shared context

### The message flow

One WebSocket, Python → browser, defined in `src/render/position_server.py` and driven
by `src/render/visualizer3d.py`.

`Visualizer3D.update(node_id, transition_id=None, active_turn=None)`
(`visualizer3d.py:57-87`) sends **either** a `transition` **or** a `position`, never
both. Call sites:

| Call site | Message | When |
|---|---|---|
| `play_game.py:166` (`initialize_game`) | `position` | once, at game start |
| `play_game.py:255` (`play_turn`) | `transition` | every move |

So in a running game, **every** move is a `transition`; `position` only fires at init
(including after "Play Again"). This matters for both bugs.

`send_legal_moves` is additionally sent by the human strategy each turn
(`human_player.py:21-31`), unconditionally — including when the move is forced
(commit `d021014`).

### Browser message handler

`index.html:1940-2010`. Relevant dispatch:

- `legal_moves` → `renderLegalMovesOverlay(msg)` (`:1968`)
- `position` → reset heuristic + figure setup (`:1974-1994`)
- `transition` → `mergeTransitionOverlay(msg)` then `queueTransitionFrames(msg)`
  (`:1996-2000`)

Two independent consumers of a `transition`: the **3D figure player**
(`queueTransitionFrames`) and the **2D graph overlay** (`mergeTransitionOverlay`). They
do not share state. Bug #2 is entirely about these two disagreeing.

### Running it

```bash
uv sync --extra training --extra dev   # NB: `--extra dev` alone removes torch/sb3
uv run python play_bjj.py              # browser config overlay
uv run python visualize_game_human.py --human-side p1 --agent-type random
```

Tests: `uv run pytest tests/ src/Game/tests/ src/render/tests/` (293 passing).
**No test covers the overlay state machine** — see "Testing" under each bug.

---

# Bug 1 — Interrupted pan silently drops `move_selected`, hanging the game

## Impact

The game deadlocks. The Python game thread blocks forever, the browser stays
responsive and connected, and nothing indicates an error. The user sees a board that
has stopped accepting input.

Only affects human players (the only path that sends `move_selected`).

## Reproduction

1. Start a human game: `uv run python visualize_game_human.py --human-side p1`
2. On the human's turn, click a legal move in the graph overlay.
3. **Within the next ~1.1 s, resize the browser window.**
4. The orange selection stays on screen. No transition ever plays. The game is dead.

The window is `900 ms` (pan) + `220 ms` (setTimeout) = **1120 ms** from click to send.

## Root cause

`onOverlayNodeClick` (`index.html:1097-1130`) sends the move from inside the pan's
completion callback:

```js
animateOverlayPanToNode(node, 900, () => {
    pruneOffscreenHistory();
    renderOverlayData();
    drawContinuityArrow(sourceNode, node);
    window.setTimeout(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'move_selected', to_node: Number(node.id) }));
        }
    }, 220);
});
```

`animateOverlayPanToNode` (`:1050-1095`) guards against stale RAF loops by comparing a
generation counter:

```js
overlayState.panAnimId += 1;              // :1070
const myAnimId = overlayState.panAnimId;

function animate(now) {
    if (overlayState.panAnimId !== myAnimId) return;   // :1081  ← returns BEFORE onDone
    ...
    if (t < 1) {
        requestAnimationFrame(animate);
    } else if (onDone) {
        onDone();                                      // :1089  ← never reached
    }
}
```

**The guard returns before `onDone`.** A superseded pan's callback never runs, so the
`move_selected` send is cancelled along with the animation.

Note the two early-return paths *do* call `onDone`: null node (`:1051-1053`) and
zero/negative duration (`:1062-1068`). Only the RAF path can drop it.

### What bumps `panAnimId`

| Site | Function | Reachable during the click pan? |
|---|---|---|
| `:1070` | `animateOverlayPanToNode` | No — Python is blocked, so no new transition |
| `:800` | `centerOverlayOnNode` | **Yes, via the resize handler** |

`centerOverlayOnNode` has two callers:

- `renderLegalMovesOverlay:1292` — start of a turn, *before* any click. Not a trigger.
- **the window `resize` handler, `:1659-1660`** — the live trigger:

```js
if (overlayState.currentNodeId && overlayState.nodesById.has(overlayState.currentNodeId)) {
    centerOverlayOnNode(overlayState.nodesById.get(overlayState.currentNodeId));
}
```

Also `resetOverlayScene:812` replaces the whole `overlayState` object; `animate` closes
over the outer `let`, so it reads the fresh object's `panAnimId === 0`, mismatches, and
bails. Reachable via `ws.onclose` (`:2014`).

### Why nothing recovers

Python is parked on an unguarded, untimed blocking read —
`human_player.py:32-38`:

```python
if len(possible_moves) == 1:
    return possible_moves[0]
to_node = selected_nodes.get()        # :34 — no timeout
```

`queue.Queue.get()` with no timeout. Nothing ever puts. There is no retry, no
heartbeat, and no user-visible error. The WebSocket stays open because the server runs
on its own thread, which makes the hang look like a UI freeze rather than a protocol
failure.

Note also that because Python is blocked, **no server message can arrive during the
pan** — which is exactly why resize is the only realistic trigger, and why this is rare
enough to have survived this long.

## Fix options

**The real defect is coupling a protocol side effect to animation completion.** Pick
based on how much choreography you want to preserve.

### Option A — send on click, decouple entirely (recommended)

Move the `ws.send` to the top of `onOverlayNodeClick`, alongside the state mutations.
Let the animation run independently.

- *Pro:* structurally eliminates the bug class. No callback can drop the send.
- *Con:* changes pacing. Python currently can't respond for ~1.1 s; after this it
  responds immediately, so the `transition` — and therefore `mergeTransitionOverlay` —
  can land while the click animation is still running. The existing 900 ms pan +
  220 ms delay appears to be deliberate choreography to avoid exactly that.
- *Mitigation:* `mergeTransitionOverlay` would need to tolerate arriving mid-click-pan.
  Note `clickCommitted` (added in `f0f1dbc`) already exists to classify this case
  correctly, so the 900 ms branch would still be selected. Verify the two pans don't
  fight — the second supersedes the first, which is visually fine but skips the first's
  `onDone`.

### Option B — give `animateOverlayPanToNode` cancel semantics

Have the guard at `:1081` invoke a cancel path rather than returning bare — e.g.
`onDone(/* cancelled */ true)`, or a separate `onCancel`.

- *Pro:* fixes every current and future `onDone`-dependent side effect at once,
  including Bug 1's sibling (auto-preview cleanup, item #4 in `.claude/todos.md`).
- *Con:* **double-send hazard.** If a pan is superseded by another pan to the same
  node, a naive cancel path fires `onDone` twice and sends `move_selected` twice.
  Python would consume the first and leave the second in the queue, desyncing every
  subsequent turn. Any implementation *must* make the send idempotent — e.g. a
  `overlayState.pendingMoveSend` node id that is cleared on send.

### Option C — don't bump `panAnimId` on resize while a pan is in flight

Narrowest possible change.

- *Pro:* one-line, near-zero blast radius.
- *Con:* treats the symptom. The next side effect wired to an `onDone` re-opens the
  same hole. Does not fix `resetOverlayScene`-via-`ws.onclose`.

**Recommendation:** A, with B's `pendingMoveSend` idempotency guard as the safety net.
C only if you want a minimal hotfix before a proper fix.

### Also worth considering

Give `selected_nodes.get()` a timeout and a visible error. Even with the browser fixed,
a dropped WebSocket frame or a closed tab currently hangs the game with no diagnostic.

## Verification

Manual (no automated coverage exists):

1. Click a move, resize during the pan → transition must still play.
2. Click a move, resize *twice* rapidly → exactly one `move_selected` on the wire.
   Watch with the browser devtools WS frame inspector, or log in `on_move_selected`
   (`human_player.py:14-15`).
3. Click a move, let it finish untouched → unchanged 900 ms choreography, one send.
4. Forced move (single legal move) → no click involved; must be unaffected.
5. Close the tab mid-pan → server should not wedge (only if you add the timeout).

Check the frame inspector for `move_selected` count in every case. Double-sends are
the main regression risk and will not be obvious — they desync a turn *later*.

---

# Bug 2 — Reverse transitions highlight the wrong node ✅ FIXED (`c451f92`)

**Fix:** `send_transition` now swaps `from_node`/`to_node` when `reverse=True` so the
wire format is direction-corrected. `reverse`, `from_reo`, `to_reo` stay canonical.
Both JS consumers (`mergeTransitionOverlay:1306-1307` and `:1999`) are correct
automatically — no JS changes needed.

The second reader (`:1999`, `lastTransitionToNodeId`) was not in the original report;
it feeds the reset heuristic and is also corrected by the server-side fix.

---

## Impact (archived)

On a reverse traversal, the graph overlay marks the **origin** as the destination:
wrong node goes gold/current, the traversed edge is recorded backwards, the viewport
pans to the node the players just left, and `overlayState.currentNodeId` is set wrong.

Since `f0f1dbc`, the new orange auto-move preview now *spotlights* the wrong node
instead of silently panning to it — which is how this surfaced.

The 3D figures animate **correctly**; only the graph overlay is wrong. That mismatch
(figures show one thing, graph another) is the user-visible symptom.

## Why `reverse` exists

GrappleMap transitions are stored with a canonical direction. `position_loader.py:73-76`
and `:104-107` load `from_node`/`to_node` straight off the raw record:

```python
'from_node': t['from']['node'],
'to_node':   t['to']['node'],
'from_reo':  _parse_reo(t['from']['reo']),
'to_reo':    _parse_reo(t['to']['reo']),
```

The game can traverse an edge in either direction (`reversible` edges). `Visualizer3D`
detects which way by comparing against the previous node — `visualizer3d.py:73-82`:

```python
reverse = False
transition = self.server.transition_frames.get(transition_id)
if transition is not None and self._last_node_id is not None:
    if (
        self._last_node_id == transition['to_node']
        and node_id == transition['from_node']
    ):
        reverse = True
self.server.send_transition(transition_id, reverse=reverse, turn=active_turn, hud=hud)
```

So `reverse=True` means: **the game actually went `to_node` → `from_node`**, the
opposite of the canonical record.

## Root cause

`send_transition` (`position_server.py:228-249`) forwards the canonical nodes verbatim
and never swaps them:

```python
payload: dict = {
    'type': 'transition',
    'transition_id': transition_id,
    'reverse': reverse,
    'frames': data['frames'],
    'detailed': data['detailed'],
    'from_node': data['from_node'],     # :245 — canonical, NOT swapped
    'to_node': data['to_node'],         # :246 — canonical, NOT swapped
    'from_reo': data['from_reo'],
    'to_reo': data['to_reo'],
}
```

The contract is therefore: *`from_node`/`to_node` are canonical; the client must
consult `reverse` to learn the real direction.*

**`queueTransitionFrames` honours that contract** (`:1558-1601`) — it branches on
`msg.reverse` in three places: start reo (`:1568-1570`), start frame (`:1573`), and
frame iteration order + end reo (`:1580-1599`).

**Two consumers do not:**

1. `mergeTransitionOverlay:1306-1307` — the graph overlay:
   ```js
   const fromId = String(msg.from_node);
   const toId = String(msg.to_node);
   ```
   Everything downstream inherits the error: `ensureOverlayLink(fromId, toId)` (`:1334`)
   records a backwards edge, `toNode.current = true` (`:1344`) lights the wrong node,
   `overlayState.currentNodeId = toId` (`:1345`) corrupts overlay state, and the pan
   (`:1377`/`:1389`) glides to the wrong place.

2. The message handler, `:1999`:
   ```js
   lastTransitionToNodeId = String(msg.to_node);
   ```
   Feeds the reset heuristic at `:1978-1982`, which decides whether an incoming
   `position` should wipe the scene. Lower impact in practice — `position` only fires at
   game init (see Shared context) — but it is the same defect and should be fixed
   together.

### Partial self-healing (why this is intermittent)

`renderLegalMovesOverlay:1246` sets `overlayState.currentNodeId` from the server's
authoritative `msg.current_node`, so a **human** game re-syncs on the next turn. The
corrupted node highlight and the backwards edge persist in the history, but the
"current" pointer recovers.

**Computer-vs-computer games never send `legal_moves`**, so there is no correction and
the error compounds across the match. Test with `play_bjj.py --skip-gui --p1 random
--p2 random`.

## Fix

**Swap `from_node`/`to_node` in `send_transition` when `reverse=True`**, so the wire
format means "actual origin → actual destination". Then both JS consumers can read the
fields directly with no `reverse` awareness.

```python
from_node, to_node = data['from_node'], data['to_node']
if reverse:
    from_node, to_node = to_node, from_node
```

### Critical constraint: do NOT swap the reos

`queueTransitionFrames` still needs `reverse`, `from_reo`, and `to_reo` in their
**canonical** pairing — the reo composition at `:1568-1570` and `:1589`/`:1599` is
matched to the frame iteration direction. Swapping the reos, or removing the `reverse`
flag, will break the 3D figure orientation in a way that is subtle and easy to miss
(figures drift or flip between moves rather than failing loudly).

Confirmed safe: `queueTransitionFrames` reads only `frames`, `detailed`, `reverse`,
`from_reo`, `to_reo` — **never** `from_node`/`to_node`. Verified by grep; the only
`from_node`/`to_node` readers in `index.html` are `:1306-1307` and `:1999`.

So: swap the node ids, keep `reverse` and the reos exactly as they are.

### Then

- `mergeTransitionOverlay:1306-1307` — no change needed once the wire is fixed.
- `:1999` — no change needed either.
- Consider a comment on `send_transition` documenting that node ids are
  direction-corrected while reos are canonical, since that asymmetry is surprising.

### Alternative considered

Fixing it client-side (swap in `mergeTransitionOverlay` when `msg.reverse`) works but
leaves a third place that has to remember the rule, and `:1999` would need the same
treatment. Server-side is the single choke point.

## Verification

1. **Find a reverse traversal.** Add a temporary log in `visualizer3d.py:81` where
   `reverse = True` is set, printing `transition_id`, `_last_node_id`, `node_id`. Run
   `play_bjj.py --skip-gui --p1 random --p2 random` until it fires.
2. With that seed/position, confirm pre-fix that the overlay's gold node disagrees with
   the 3D figures, and post-fix that they agree.
3. Confirm the edge arrow points the way the game actually moved.
4. **Regression-check the 3D figures.** Play several non-reverse moves and at least one
   reverse move; figure orientation must be unchanged from before. This is the main
   risk of the fix — verify visually, orientation bugs won't throw.
5. Run a long computer-vs-computer game and confirm the accumulated graph history has no
   backwards edges.

---

## Testing note (applies to both)

Nothing asserts overlay behavior. Grepping
`moveCommitted|mergeTransition|renderLegalMoves|pendingSelected` across
`src/render/tests/` and `src/Game/tests/` returns exactly one hit —
`test_visualizer3d.py:53`, which only checks `'viewer/index.html' in call_arg`.

Both bugs are the kind a test would have caught. Bug 2 in particular is cheaply
testable **on the Python side today**, with no browser: assert that
`send_transition(tid, reverse=True)` emits a payload whose `from_node`/`to_node` match
the actual traversal direction. That test is worth writing as part of the fix.

Bug 1 needs the JS state machine extracted from `index.html` to be testable — a larger
refactor tracked as item #6 in `.claude/todos.md`.
