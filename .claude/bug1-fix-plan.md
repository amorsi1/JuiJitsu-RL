# Plan — Bug 1: interrupted pan drops `move_selected`

Scope: **Bug 1 only.** Bug 2 (reverse transitions) is a separate, independent fix.

Branch: stay on `amorsi1/human-player-ui` — it already owns the graph-overlay code
and is not `master`. No new branch.

Reference: `.claude/overlay-bugs.md` (Bug 1). Line numbers below are against `3b16a80`.

---

## Verified against the code

| Claim | Status |
|---|---|
| Window is 1120 ms (900 pan + 220 timeout), `:1120-1129` | confirmed |
| `:1081` guard returns before `onDone` at `:1089` | confirmed |
| Only reachable trigger is the resize handler `:1660` | confirmed — `renderLegalMovesOverlay:1292` fires pre-click; `animateOverlayPanToNode:1070` needs a server message, and Python is parked on `selected_nodes.get()` (`human_player.py:34`) |
| Double-send hazard on a naive cancel path | confirmed as the reason to build the guard first |

One new finding, not in the report: **`turn_delay` does not buffer the response.**
`Visualizer3D.update` sends the transition at `visualizer3d.py:82` and *then* sleeps
(`:86-87`). So under Option A the `transition` lands within milliseconds of the click,
not after the delay. That makes Step 3 below a near-certainty rather than a maybe.

---

## Design decisions

**Option A (send on click), with a per-turn idempotency guard.** Matches the report's
recommendation. The guard is `overlayState.moveSent`, reset **only** in
`renderLegalMovesOverlay` (`:1221-1223`).

Why that reset site and not the node id the report suggested: `selectable` is set only
in `renderLegalMovesOverlay:1272` and cleared for every node in
`mergeTransitionOverlay:1336-1340`, so one `renderLegalMovesOverlay` call brackets
exactly one click opportunity. That makes the flag enforce **one `move_selected` per
`legal_moves` message** — precisely the invariant Python's one-`get()`-per-human-turn
depends on. A node-id key would also wrongly block re-selecting the same node on a
later turn.

Not doing Option B (cancel semantics on `animateOverlayPanToNode`) — it is the fix for
todos #4 (auto-preview cleanup), a different bug, and bundling them widens the blast
radius. Not doing Option C; it is a symptom patch.

---

## Steps

### 1. Idempotency guard — commit alone, before the behavior change

- `createOverlayState()` (`:706-734`): add `moveSent: false`.
- New `sendMoveSelected(nodeId)`: no-op if `overlayState.moveSent` or the socket is not
  OPEN; otherwise set the flag and send.
- `renderLegalMovesOverlay` (`:1221-1223`): reset `moveSent = false` alongside the
  existing three resets.
- `onOverlayNodeClick` (`:1124-1128`): route the existing send through the helper,
  leaving it inside the timeout for now.

**Verify:** behavior is byte-for-byte unchanged in normal play — one `move_selected`
per turn in the devtools WS frame inspector, transitions play as before. This commit is
a pure net; it must not change anything observable.

### 2. Move the send to click time — the actual fix

In `onOverlayNodeClick`, call `sendMoveSelected(node.id)` immediately after the three
flag mutations at `:1101-1103` (flags first, so a re-entrant click can't slip past the
`:1098` guard). Delete the `window.setTimeout` wrapper at `:1124-1128`.

Leave the pan and its callback in place — they are now purely visual.

**Verify** (manual; no automated coverage exists for the overlay state machine):
1. Click a move, resize during the pan → transition still plays. **This is the bug.**
2. Click, then resize twice rapidly → exactly one `move_selected` on the wire.
3. Click, leave it alone → one send, transition plays.
4. Forced move (single legal move) → unaffected.
5. Reconnect / "Play Again" → next turn still accepts a click (guard reset fires).

Use the devtools WS frame inspector for the counts. Double-sends are the main
regression risk and they desync a turn *later*, so count frames explicitly — do not
infer from the game looking fine.

### 3. Restore the click's selection feedback — contingent on Step 2's observed result

After Step 2 the transition arrives ~immediately, so `mergeTransitionOverlay:1349-1350`
clears `moveCommitted`/`pendingSelectedId` within milliseconds of the click. The orange
selection highlight and the dimming of unselected nodes effectively stop being visible.
The click pan is also superseded by merge's own pan to the same node, so the click
pan's `onDone` (prune / render / `drawContinuityArrow`) stops running — harmless, since
merge's `onDone` prunes and renders and `:1304` removes the continuity arrow anyway,
but it means `drawContinuityArrow` becomes dead on the click path.

If the highlight loss is visible in Step 2 (expected), fix it by making the click branch
of `mergeTransitionOverlay` (`:1387-1394`) hold the selection through its pan and clear
in `onDone`, exactly as the auto-move branch already does at `:1369-1386`. The two
branches then differ only in pan duration (`900` vs `AUTO_MOVE_PAN_MS`) and collapse
into one path with a duration variable.

**Verify:** click feedback lasts ~900 ms as before; auto-moves and forced moves keep
their 200 ms compressed preview; no double pan-fight.

**Decision point:** if Step 2 looks fine on its own, skip this. Don't build it blind.

---

## Flagged, not in scope — needs your call

These are real and adjacent, but each is a separate change and I have not assumed
approval for any of them.

1. **`selected_nodes.get()` has no timeout** (`human_player.py:34`). Even with the
   browser fixed, a dropped frame or a closed tab hangs the game with no diagnostic.
   The report raises this too.

2. **Forced moves can poison the queue — pre-existing, same desync family.** Since
   `d021014`, `send_legal_moves` fires unconditionally, and `renderLegalMovesOverlay`
   marks the single move `selectable`. But `human_player.py:32-33` returns early on
   `len(possible_moves) == 1` *without* consuming the queue. A user who clicks that node
   before the transition clears it leaves a stale entry that the **next** human turn
   consumes — desyncing exactly the way the report warns about, and it would look like a
   regression from this fix. Cheapest fix is draining stale entries before the blocking
   `get()`. This one is worth doing soon; it will otherwise muddy Step 2's verification.

3. **No automated coverage.** Node 22 is available but the repo has no JS test infra
   (no `package.json`), and the overlay state machine is inline in a 2024-line
   `index.html` using d3 and live DOM. Building a harness is todos #6, a refactor an
   order of magnitude larger than this fix. If you want the parallel test-writer /
   implementer agent workflow from `CLAUDE.md`, item 2 above is the only part of this
   work with a writable pytest ("a duplicate `move_selected` does not desync the next
   turn"); Steps 1–3 have no test target without that refactor.

---

## Commits (atomic, in order)

1. `Add per-turn idempotency guard for move_selected sends`
2. `Send move_selected on click instead of after the pan animation`
3. *(if Step 3 runs)* `Hold click selection highlight through the transition pan`
