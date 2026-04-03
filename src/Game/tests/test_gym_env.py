import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from Game.gym_env import BJJEnv, get_masked_q_values, state_to_index
from render.hud_announcements import AnnouncementEvent


@pytest.fixture(scope="module")
def env() -> BJJEnv:
    """Single BJJEnv instance shared across the module.

    Construction is expensive (~1 s, loads 47 MB of graph data). Each test calls
    env.reset() itself to get a clean game state; the graph and spaces are reused.
    """
    return BJJEnv()


# ---------------------------------------------------------------------------
# 1. Smoke test
# ---------------------------------------------------------------------------


def test_check_env_passes():
    """Gymnasium's check_env must pass with no errors or warnings.

    If this test fails, the fine-grained tests below identify the root cause:
    - reset/step contract tests → wrong return shape or types
    - observation bounds tests → an obs element is outside its declared range
    - game-logic tests → action mask or termination logic is broken
    """
    check_env(BJJEnv(), warn=False)


# ---------------------------------------------------------------------------
# 2. reset() contract
# ---------------------------------------------------------------------------


def test_reset_returns_two_tuple(env):
    result = env.reset()
    assert len(result) == 2


def test_reset_obs_in_observation_space(env):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), (
        f"obs {obs} not in observation_space "
        f"(low={env.observation_space.low}, high={env.observation_space.high})"
    )


def test_reset_obs_dtype(env):
    obs, _ = env.reset()
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"


def test_reset_obs_shape(env):
    obs, _ = env.reset()
    assert obs.shape == (5,), f"Expected shape (5,), got {obs.shape}"


def test_reset_info_has_action_mask(env):
    _, info = env.reset()
    assert "action_mask" in info, f"info keys: {list(info.keys())}"


def test_reset_action_mask_length(env):
    _, info = env.reset()
    assert len(info["action_mask"]) == env.action_space.n, (
        f"Mask length {len(info['action_mask'])} != action_space.n {env.action_space.n}"
    )


def test_reset_has_at_least_one_valid_action(env):
    _, info = env.reset()
    assert any(info["action_mask"]), "No valid actions available after reset"


def test_reset_seeded_reproducibility(env):
    """Same seed must produce the same initial observation (check_env seed contract)."""
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# 3. step() contract
# ---------------------------------------------------------------------------


def _first_valid_action(info: dict) -> int:
    return int(np.where(info["action_mask"])[0][0])


def test_step_returns_five_tuple(env):
    _, info = env.reset()
    result = env.step(_first_valid_action(info))
    assert len(result) == 5


def test_step_obs_in_observation_space(env):
    _, info = env.reset()
    obs, _, _, _, _ = env.step(_first_valid_action(info))
    assert env.observation_space.contains(obs), (
        f"obs {obs} not in observation_space after step "
        f"(low={env.observation_space.low}, high={env.observation_space.high})"
    )


def test_step_reward_is_scalar(env):
    _, info = env.reset()
    _, reward, _, _, _ = env.step(_first_valid_action(info))
    assert np.isscalar(reward) or (isinstance(reward, np.ndarray) and reward.ndim == 0), (
        f"reward {reward!r} is not a scalar"
    )


def test_step_terminated_is_bool(env):
    _, info = env.reset()
    _, _, terminated, _, _ = env.step(_first_valid_action(info))
    assert isinstance(terminated, bool), f"terminated has type {type(terminated)}"


def test_step_truncated_is_bool(env):
    _, info = env.reset()
    _, _, _, truncated, _ = env.step(_first_valid_action(info))
    assert isinstance(truncated, bool), f"truncated has type {type(truncated)}"


def test_step_not_both_terminated_and_truncated(env):
    _, info = env.reset()
    _, _, terminated, truncated, _ = env.step(_first_valid_action(info))
    assert not (terminated and truncated), "terminated and truncated are both True"


def test_step_info_has_action_mask(env):
    _, info = env.reset()
    _, _, _, _, info = env.step(_first_valid_action(info))
    assert "action_mask" in info, f"info keys after step: {list(info.keys())}"


def test_step_action_mask_length(env):
    _, info = env.reset()
    _, _, _, _, info = env.step(_first_valid_action(info))
    assert len(info["action_mask"]) == env.action_space.n


# ---------------------------------------------------------------------------
# 4. Observation bounds — per-element diagnostics
#    These pinpoint which element of obs caused observation_space.contains() to fail.
# ---------------------------------------------------------------------------


def test_obs_position_in_bounds(env):
    obs, _ = env.reset()
    lo, hi = env.observation_space.low[0], env.observation_space.high[0]
    assert lo <= obs[0] <= hi, (
        f"obs[0] (position) = {obs[0]} outside [{lo}, {hi}]"
    )


def test_obs_point_difference_in_bounds(env):
    obs, _ = env.reset()
    lo, hi = env.observation_space.low[1], env.observation_space.high[1]
    assert lo <= obs[1] <= hi, (
        f"obs[1] (point_difference) = {obs[1]} outside [{lo}, {hi}]"
    )


def test_obs_on_top_is_binary(env):
    obs, _ = env.reset()
    assert obs[2] in (0.0, 1.0), f"obs[2] (on_top) = {obs[2]} is not 0 or 1"


def test_obs_on_bottom_is_binary(env):
    obs, _ = env.reset()
    assert obs[3] in (0.0, 1.0), f"obs[3] (on_bottom) = {obs[3]} is not 0 or 1"


def test_obs_turns_left_in_bounds(env):
    obs, _ = env.reset()
    lo, hi = env.observation_space.low[4], env.observation_space.high[4]
    assert lo <= obs[4] <= hi, (
        f"obs[4] (turns_left) = {obs[4]} outside [{lo}, {hi}]"
    )


# ---------------------------------------------------------------------------
# 5. Game-logic correctness
# ---------------------------------------------------------------------------


def test_turn_count_increments_after_step(env):
    env.reset()
    before = env.game.turn_count
    _, info = env.reset()
    env.step(_first_valid_action(info))
    assert env.game.turn_count == before + 1, (
        f"turn_count did not increment: was {before}, now {env.game.turn_count}"
    )


def test_truncated_when_turn_count_reaches_max_turns():
    """truncated=True (and terminated=False) fires when turn_count reaches max_turns."""
    env = BJJEnv()
    _, info = env.reset()
    # Override max_turns after reset so the observation_space bounds remain valid
    # (obs high[4] stays at the original 100; turns_left of 1 is well within range)
    env.game.max_turns = 1
    _, _, terminated, truncated, _ = env.step(_first_valid_action(info))
    assert not (terminated and truncated), "terminated and truncated are both True"
    if not terminated:
        assert truncated, (
            "Expected truncated=True when turn_count reached max_turns without a winner"
        )


def test_masked_actions_are_legal_moves(env):
    """Every action flagged True in the mask must be a legal move for the current player."""
    obs, info = env.reset()
    mask = info["action_mask"]
    is_top = bool(obs[2])
    is_bottom = bool(obs[3])

    possible_moves = env.game.game_state.get_possible_moves(is_top, is_bottom)
    legal_edge_ids = {move[1]["id"] for move in possible_moves}

    for idx in np.where(mask)[0]:
        edge_id = env.index_to_id[idx]
        assert edge_id in legal_edge_ids, (
            f"Action index {idx} (edge_id={edge_id}) is unmasked "
            f"but not in legal moves for is_top={is_top}, is_bottom={is_bottom}"
        )


def test_step_moves_to_destination_node(env):
    """After a step, current_node must be the edge's destination, not its source."""
    _, info = env.reset()
    action = _first_valid_action(info)
    edge_id = env.index_to_id[action]
    _, expected_dest = env.edge_id_to_nodes[edge_id]
    env.step(action)
    assert env.game.game_state.current_node == expected_dest, (
        f"current_node is {env.game.game_state.current_node} "
        f"but expected destination {expected_dest}"
    )


def test_position_changes_during_episode(env):
    """The graph position must actually change as the agent takes actions."""
    _, info = env.reset(seed=42)
    initial_node = env.game.game_state.current_node
    visited_nodes = {initial_node}
    for _ in range(20):
        valid = np.where(info["action_mask"])[0]
        if len(valid) == 0:
            break
        action = int(valid[0])
        _, _, terminated, truncated, info = env.step(action)
        visited_nodes.add(env.game.game_state.current_node)
        if terminated or truncated:
            break
    assert len(visited_nodes) > 1, (
        f"Agent stayed at node {initial_node} for the entire episode — "
        f"position never updated"
    )
    
# ---------------------------------------------------------------------------
# 5b. Announcement event payloads
# ---------------------------------------------------------------------------


def test_submission_win_emits_win_announcement() -> None:
    env = BJJEnv()
    try:
        _, info = env.reset()
        action = _first_valid_action(info)
        edge_id = env.index_to_id[action]
        start, end = env.edge_id_to_nodes[edge_id]
        edge_data = env.game.board.get_edge_data(start, end)

        original_tap = edge_data.get("tap", False)
        edge_data["tap"] = True

        def _fake_play_turn(move: tuple[int, dict[str, object]]) -> None:
            env.game.game_state.current_node = move[0]
            env.game.winner = env.game.player1

        env.game.play_turn = _fake_play_turn  # type: ignore[method-assign]
        env.step(action)

        event = env._pending_announcement_event
        assert event == AnnouncementEvent(kind="win", winner_index=0, win_type="submission")
        edge_data["tap"] = original_tap
    finally:
        env.close()


def test_turn_limit_points_win_emits_points_announcement() -> None:
    env = BJJEnv()
    try:
        _, info = env.reset()
        action = _first_valid_action(info)
        edge_id = env.index_to_id[action]
        start, end = env.edge_id_to_nodes[edge_id]
        edge_data = env.game.board.get_edge_data(start, end)
        edge_data["tap"] = False
        env.game.max_turns = 1

        def _fake_play_turn(move: tuple[int, dict[str, object]]) -> None:
            env.game.game_state.current_node = move[0]
            env.game.winner = None

        def _fake_check_for_points_win() -> None:
            env.game.winner = env.game.player2

        env.game.play_turn = _fake_play_turn  # type: ignore[method-assign]
        env.game.check_for_points_win = _fake_check_for_points_win  # type: ignore[method-assign]
        env.step(action)

        event = env._pending_announcement_event
        assert event == AnnouncementEvent(kind="win", winner_index=1, win_type="points")
    finally:
        env.close()


def test_position_win_emits_position_announcement() -> None:
    env = BJJEnv()
    try:
        _, info = env.reset()
        action = _first_valid_action(info)
        edge_id = env.index_to_id[action]
        start, end = env.edge_id_to_nodes[edge_id]
        edge_data = env.game.board.get_edge_data(start, end)
        edge_data["tap"] = False

        def _fake_play_turn(move: tuple[int, dict[str, object]]) -> None:
            env.game.game_state.current_node = move[0]
            env.game.winner = env.game.player2

        env.game.play_turn = _fake_play_turn  # type: ignore[method-assign]
        env.step(action)

        event = env._pending_announcement_event
        assert event == AnnouncementEvent(kind="win", winner_index=1, win_type="position")
    finally:
        env.close()


def test_non_winning_reset_emits_teleport_announcement() -> None:
    env = BJJEnv()
    try:
        _, info = env.reset()
        action = _first_valid_action(info)
        edge_id = env.index_to_id[action]
        start, end = env.edge_id_to_nodes[edge_id]
        edge_data = env.game.board.get_edge_data(start, end)
        edge_data["tap"] = False

        def _fake_play_turn(move: tuple[int, dict[str, object]]) -> None:
            env.game.game_state.current_node = move[0] + 1
            env.game.winner = None

        env.game.play_turn = _fake_play_turn  # type: ignore[method-assign]
        env.step(action)

        event = env._pending_announcement_event
        assert event == AnnouncementEvent(kind="teleport")
    finally:
        env.close()


def test_event_payload_excludes_render_style_fields() -> None:
    env = BJJEnv()
    try:
        payload = AnnouncementEvent(kind="teleport")
        info = env._get_player_info(payload)
        event = info["announcement_event"]
        assert isinstance(event, AnnouncementEvent)
        for field in ("hold_seconds", "font_size", "text_color", "placement"):
            assert not hasattr(event, field)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 6. Utility functions
# ---------------------------------------------------------------------------


def test_state_to_index_numpy_array():
    obs = np.array([94.0, 0.0, 1.0, 0.0, 50.0], dtype=np.float32)
    assert state_to_index(obs) == 94 * 2 + 1


def test_state_to_index_numpy_array_bottom():
    obs = np.array([10.0, -3.0, 0.0, 1.0, 20.0], dtype=np.float32)
    assert state_to_index(obs) == 10 * 2 + 0


def test_state_to_index_dict():
    assert state_to_index({"current_position": 5, "on_top": 0}) == 10


def test_state_to_index_explicit_args():
    assert state_to_index(position=3, is_top=1) == 7


def test_state_to_index_raises_with_no_args():
    with pytest.raises(ValueError):
        state_to_index()


def test_get_masked_q_values_masks_invalid_to_neg_inf():
    q = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, False, True])
    result = get_masked_q_values(q, mask)
    assert result[1] == -np.inf
    assert result[0] == pytest.approx(1.0)
    assert result[2] == pytest.approx(3.0)


def test_get_masked_q_values_all_valid_unchanged():
    q = np.array([0.5, -1.0, 2.5])
    mask = np.ones(3, dtype=bool)
    result = get_masked_q_values(q, mask)
    np.testing.assert_array_almost_equal(result, q)


def test_get_masked_q_values_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        get_masked_q_values(np.array([1.0, 2.0]), np.array([True]))
