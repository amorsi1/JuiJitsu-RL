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
