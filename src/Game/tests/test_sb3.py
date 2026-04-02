"""Tests for SB3 (sb3-contrib MaskablePPO) integration with BJJEnv-v0.

These are smoke tests that verify the env is compatible with MaskablePPO's
action-masking API. Training timesteps are kept very low (256-512) since
correctness — not convergence — is the goal.
"""

import numpy as np
import pytest

import Game  # noqa: F401 — side-effect import: triggers gymnasium registration
from Game.gym_env import BJJEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env() -> BJJEnv:
    """Single BJJEnv instance shared across the module.

    Construction is expensive (~1 s, loads 47 MB of graph data). Each test
    calls env.reset() itself to get a clean game state; the graph and spaces
    are reused.
    """
    return BJJEnv()


@pytest.fixture(scope="module")
def trained_model(env: BJJEnv) -> MaskablePPO:
    """MaskablePPO model trained for a minimal number of timesteps."""
    model = MaskablePPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=512)
    return model


# ---------------------------------------------------------------------------
# 1. action_masks() public interface
# ---------------------------------------------------------------------------


def test_action_masks_method_exists(env: BJJEnv) -> None:
    """BJJEnv must expose a public action_masks() method for MaskablePPO."""
    env.reset()
    assert hasattr(env, "action_masks"), (
        "BJJEnv is missing the public action_masks() method required by MaskablePPO"
    )
    assert callable(env.action_masks)


def test_action_masks_returns_correct_shape_and_dtype(env: BJJEnv) -> None:
    """action_masks() must return a bool ndarray of shape (action_space.n,)."""
    env.reset()
    masks = env.action_masks()
    assert isinstance(masks, np.ndarray), (
        f"Expected np.ndarray, got {type(masks).__name__}"
    )
    assert masks.shape == (env.action_space.n,), (
        f"Expected shape ({env.action_space.n},), got {masks.shape}"
    )
    assert masks.dtype == bool, f"Expected bool dtype, got {masks.dtype}"


def test_action_masks_matches_private_method(env: BJJEnv) -> None:
    """action_masks() must return the same result as _get_action_mask()."""
    env.reset()
    public = env.action_masks()
    private = env._get_action_mask()
    np.testing.assert_array_equal(public, private)


# ---------------------------------------------------------------------------
# 2. MaskablePPO instantiation
# ---------------------------------------------------------------------------


def test_maskable_ppo_instantiation(env: BJJEnv) -> None:
    """MaskablePPO('MlpPolicy', env) must create without error."""
    model = MaskablePPO("MlpPolicy", env, verbose=0)
    assert model is not None
    assert model.policy is not None


# ---------------------------------------------------------------------------
# 3. MaskablePPO training (short smoke test)
# ---------------------------------------------------------------------------


def test_maskable_ppo_learn_short(env: BJJEnv) -> None:
    """MaskablePPO.learn(total_timesteps=512) must complete without error."""
    model = MaskablePPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=512)
    # If we reach here without exception, the test passes


# ---------------------------------------------------------------------------
# 4. MaskablePPO prediction with masks
# ---------------------------------------------------------------------------


def test_maskable_ppo_predict_with_masks(
    env: BJJEnv, trained_model: MaskablePPO
) -> None:
    """predict() with action_masks must return a valid (unmasked) action."""
    obs, info = env.reset()
    masks = env.action_masks()

    action, _states = trained_model.predict(obs, action_masks=masks, deterministic=True)

    action_int = int(action)
    assert 0 <= action_int < env.action_space.n, (
        f"Action {action_int} outside valid range [0, {env.action_space.n})"
    )
    assert masks[action_int], (
        f"Predicted action {action_int} is masked (illegal)"
    )


# ---------------------------------------------------------------------------
# 5. Model save / load round-trip
# ---------------------------------------------------------------------------


def test_model_save_load_roundtrip(
    env: BJJEnv, trained_model: MaskablePPO, tmp_path: pytest.TempPathFactory
) -> None:
    """A saved MaskablePPO model must load and predict identically."""
    save_path = tmp_path / "test_maskable_ppo"
    trained_model.save(str(save_path))

    loaded_model = MaskablePPO.load(str(save_path), env=env)

    obs, info = env.reset()
    masks = env.action_masks()

    original_action, _ = trained_model.predict(
        obs, action_masks=masks, deterministic=True
    )
    loaded_action, _ = loaded_model.predict(
        obs, action_masks=masks, deterministic=True
    )

    assert int(original_action) == int(loaded_action), (
        f"Original model predicted {int(original_action)}, "
        f"loaded model predicted {int(loaded_action)}"
    )
    # Verify the loaded model's action is legal
    assert masks[int(loaded_action)], (
        f"Loaded model predicted masked action {int(loaded_action)}"
    )


# ---------------------------------------------------------------------------
# 6. Terminal info contains BJJ metrics
# ---------------------------------------------------------------------------


def test_terminal_info_contains_bjj_metrics(env: BJJEnv) -> None:
    """On episode termination/truncation, info must contain is_win, is_loss, point_diff."""
    obs, _ = env.reset()
    terminated = truncated = False
    terminal_info: dict | None = None

    while not (terminated or truncated):
        masks = env.action_masks()
        legal_actions = np.where(masks)[0]
        if len(legal_actions) == 0:
            break
        action = int(np.random.choice(legal_actions))
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            terminal_info = info

    assert terminal_info is not None, "Episode never reached a terminal state"
    assert "is_win" in terminal_info, "info missing 'is_win' key at termination"
    assert "is_loss" in terminal_info, "info missing 'is_loss' key at termination"
    assert "point_diff" in terminal_info, "info missing 'point_diff' key at termination"
    assert isinstance(terminal_info["is_win"], bool), (
        f"Expected bool for is_win, got {type(terminal_info['is_win']).__name__}"
    )
    assert isinstance(terminal_info["is_loss"], bool), (
        f"Expected bool for is_loss, got {type(terminal_info['is_loss']).__name__}"
    )
    assert isinstance(terminal_info["point_diff"], float), (
        f"Expected float for point_diff, got {type(terminal_info['point_diff']).__name__}"
    )


# ---------------------------------------------------------------------------
# 7. MaskableEvalCallback instantiation
# ---------------------------------------------------------------------------


def test_maskable_eval_callback_instantiation() -> None:
    """MaskableEvalCallback must instantiate without error."""
    eval_env = BJJEnv()
    callback = MaskableEvalCallback(
        eval_env=eval_env,
        eval_freq=1000,
        n_eval_episodes=3,
        use_masking=True,
        verbose=0,
    )
    assert callback is not None


# ---------------------------------------------------------------------------
# 8. BJJMetricsCallback extracts and clears metrics on rollout end
# ---------------------------------------------------------------------------


def test_bjj_metrics_callback_extracts_metrics() -> None:
    """BJJMetricsCallback must accumulate terminal infos and clear buffers on rollout end."""
    from Game.train_sb3 import BJJMetricsCallback

    callback = BJJMetricsCallback()

    # Provide a minimal model with logger configured so init_callback succeeds
    from stable_baselines3.common.logger import configure as sb3_configure

    model = MaskablePPO("MlpPolicy", BJJEnv(), verbose=0)
    model.set_logger(sb3_configure(format_strings=[]))  # silent logger for unit test
    callback.init_callback(model)

    # Simulate three SB3 _on_step() calls — two terminal, one non-terminal
    terminal_infos = [
        {"is_win": True, "is_loss": False, "point_diff": 5.0},
        {"is_win": False, "is_loss": True, "point_diff": -3.0},
        {},  # non-terminal step — no BJJ keys
    ]
    for info in terminal_infos:
        callback.locals = {"infos": [info]}
        callback._on_step()

    # Before rollout end: 2 terminal episodes accumulated
    assert len(callback._win_buffer) == 2, (
        f"Expected 2 entries in win buffer before rollout end, got {len(callback._win_buffer)}"
    )
    assert list(callback._win_buffer) == [True, False], (
        f"Unexpected win buffer contents: {list(callback._win_buffer)}"
    )
    assert len(callback._point_diff_buffer) == 2, (
        f"Expected 2 entries in point_diff buffer before rollout end, "
        f"got {len(callback._point_diff_buffer)}"
    )
    assert list(callback._point_diff_buffer) == [5.0, -3.0], (
        f"Unexpected point_diff buffer contents: {list(callback._point_diff_buffer)}"
    )

    # After rollout end: buffers must be cleared
    callback._on_rollout_end()
    assert len(callback._win_buffer) == 0, (
        f"Expected empty win buffer after rollout end, got {len(callback._win_buffer)}"
    )
    assert len(callback._point_diff_buffer) == 0, (
        f"Expected empty point_diff buffer after rollout end, "
        f"got {len(callback._point_diff_buffer)}"
    )


# ---------------------------------------------------------------------------
# 9. train() smoke test with callbacks
# ---------------------------------------------------------------------------


def test_train_with_callbacks(tmp_path: pytest.TempPathFactory) -> None:
    """train() with callback parameters must complete and save a model file."""
    from Game.train_sb3 import train

    model = train(
        total_timesteps=512,
        save_path=tmp_path / "model",
        tensorboard_log=tmp_path / "logs",
        eval_freq=256,
        n_eval_episodes=2,
        checkpoint_freq=256,
        checkpoint_dir=tmp_path / "checkpoints",
        best_model_dir=tmp_path / "best_model",
        verbose=0,
    )
    assert model is not None
    assert (tmp_path / "model.zip").exists()
