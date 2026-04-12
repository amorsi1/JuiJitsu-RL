"""Smoke tests for MaskableRecurrentPPO integration with BJJEnv-v0.

Verifies that the MaskableRecurrentPPO wrapper is compatible with BJJEnv's
action-masking API. Training timesteps are kept very low (256) since
correctness — not convergence — is the goal.
"""

import numpy as np
import pytest

import Game  # noqa: F401 — triggers BJJEnv-v0 registration
from Game.gym_env import BJJEnv
from Game.maskable_recurrent import MaskableRecurrentPPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env() -> BJJEnv:
    """Single BJJEnv shared across the module (graph loading is expensive)."""
    return BJJEnv()


@pytest.fixture(scope="module")
def trained_model(env: BJJEnv) -> MaskableRecurrentPPO:
    """MaskableRecurrentPPO trained for minimal timesteps."""
    model = MaskableRecurrentPPO("MlpLstmPolicy", env, verbose=0, n_steps=128, batch_size=128)
    model.learn(total_timesteps=256, use_masking=True)
    return model


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------


def test_maskable_recurrent_ppo_instantiation(env: BJJEnv) -> None:
    """MaskableRecurrentPPO('MlpLstmPolicy', env) must create without error."""
    model = MaskableRecurrentPPO("MlpLstmPolicy", env, verbose=0)
    assert model is not None
    assert model.policy is not None


# ---------------------------------------------------------------------------
# 2. Learning (short smoke test)
# ---------------------------------------------------------------------------


def test_maskable_recurrent_ppo_learn(env: BJJEnv) -> None:
    """MaskableRecurrentPPO.learn(total_timesteps=256, use_masking=True) must complete without error."""
    model = MaskableRecurrentPPO("MlpLstmPolicy", env, verbose=0, n_steps=128, batch_size=128)
    model.learn(total_timesteps=256, use_masking=True)


# ---------------------------------------------------------------------------
# 3. Prediction with action masks
# ---------------------------------------------------------------------------


def test_maskable_recurrent_ppo_predict_with_masks(
    env: BJJEnv, trained_model: MaskableRecurrentPPO
) -> None:
    """predict() with action_masks must return a valid (unmasked) action."""
    obs, _info = env.reset()
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
# 4. Save / load round-trip
# ---------------------------------------------------------------------------


def test_maskable_recurrent_ppo_save_load(
    env: BJJEnv, trained_model: MaskableRecurrentPPO, tmp_path: pytest.TempPathFactory
) -> None:
    """A saved MaskableRecurrentPPO model must load and predict identically."""
    save_path = tmp_path / "test_maskable_recurrent_ppo"
    trained_model.save(str(save_path))

    loaded_model = MaskableRecurrentPPO.load(str(save_path), env=env)

    obs, _info = env.reset()
    masks = env.action_masks()

    original_action, _ = trained_model.predict(obs, action_masks=masks, deterministic=True)
    loaded_action, _ = loaded_model.predict(obs, action_masks=masks, deterministic=True)

    assert int(original_action) == int(loaded_action), (
        f"Original model predicted {int(original_action)}, "
        f"loaded model predicted {int(loaded_action)}"
    )


# ---------------------------------------------------------------------------
# 5. Rollout buffer includes action masks
# ---------------------------------------------------------------------------


def test_buffer_includes_action_masks(env: BJJEnv) -> None:
    """Rollout buffer must store action_masks after learn() completes."""
    model = MaskableRecurrentPPO("MlpLstmPolicy", env, verbose=0, n_steps=32, batch_size=32)
    model.learn(total_timesteps=64, use_masking=True)

    assert hasattr(model.rollout_buffer, "action_masks"), (
        "rollout_buffer is missing the action_masks attribute"
    )
    assert isinstance(model.rollout_buffer.action_masks, np.ndarray), (
        f"Expected np.ndarray for rollout_buffer.action_masks, "
        f"got {type(model.rollout_buffer.action_masks).__name__}"
    )

    for sample in model.rollout_buffer.get(batch_size=32):
        assert hasattr(sample, "action_masks"), (
            "Rollout buffer sample is missing the action_masks attribute"
        )
        assert sample.action_masks.shape[-1] == env.action_space.n, (
            f"Expected last dim {env.action_space.n}, got {sample.action_masks.shape[-1]}"
        )
        break  # Only need to check one batch


# ---------------------------------------------------------------------------
# 6. evaluate_policy compatibility
# ---------------------------------------------------------------------------


def test_evaluate_policy_compat(env: BJJEnv, trained_model: MaskableRecurrentPPO) -> None:
    """evaluate_policy from sb3_contrib must run without error on a trained model."""
    evaluate_policy(trained_model, env, n_eval_episodes=2)
