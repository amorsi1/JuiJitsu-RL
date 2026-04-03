"""Tests for the refactored train_sb3.py algorithm registry and CLI interface.

Verifies that ALGORITHMS contains the expected entries with the correct
classes and configuration keys, and that train() dispatches to the right
algorithm class based on the `algorithm` parameter.
"""

import pytest

import Game  # noqa: F401 — triggers BJJEnv-v0 registration
from Game.gym_env import BJJEnv
from sb3_contrib import MaskablePPO
from Game.maskable_recurrent import MaskableRecurrentPPO
from Game.train_sb3 import ALGORITHMS, train


# ---------------------------------------------------------------------------
# 1. ALGORITHMS registry structure
# ---------------------------------------------------------------------------


def test_algorithms_registry_has_expected_keys() -> None:
    """ALGORITHMS must contain 'maskable_ppo' and 'recurrent_ppo' with required keys."""
    assert "maskable_ppo" in ALGORITHMS, "ALGORITHMS missing 'maskable_ppo' key"
    assert "recurrent_ppo" in ALGORITHMS, "ALGORITHMS missing 'recurrent_ppo' key"

    required_keys = {"class", "policy", "default_n_steps", "default_batch_size"}
    for algo_name in ("maskable_ppo", "recurrent_ppo"):
        missing = required_keys - set(ALGORITHMS[algo_name].keys())
        assert not missing, (
            f"ALGORITHMS['{algo_name}'] missing required keys: {missing}"
        )


def test_maskable_ppo_algorithm_class() -> None:
    """ALGORITHMS['maskable_ppo']['class'] must be MaskablePPO."""
    assert ALGORITHMS["maskable_ppo"]["class"] is MaskablePPO


def test_recurrent_ppo_algorithm_class() -> None:
    """ALGORITHMS['recurrent_ppo']['class'] must be MaskableRecurrentPPO."""
    assert ALGORITHMS["recurrent_ppo"]["class"] is MaskableRecurrentPPO


# ---------------------------------------------------------------------------
# 2. train() dispatch
# ---------------------------------------------------------------------------


def test_train_default_algorithm_returns_maskable_ppo(tmp_path: pytest.TempPathFactory) -> None:
    """train() without algorithm argument must return a MaskablePPO model."""
    model, _mean, _std = train(
        total_timesteps=256,
        save_path=tmp_path / "m",
        tensorboard_log=tmp_path / "logs",
        eval_freq=128,
        n_eval_episodes=2,
        checkpoint_freq=256,
        checkpoint_dir=tmp_path / "cp",
        best_model_dir=tmp_path / "best",
        verbose=0,
    )
    assert isinstance(model, MaskablePPO)


def test_train_recurrent_ppo_returns_maskable_recurrent_ppo(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """train(algorithm='recurrent_ppo') must return a MaskableRecurrentPPO model."""
    model, _mean, _std = train(
        algorithm="recurrent_ppo",
        total_timesteps=256,
        save_path=tmp_path / "m",
        tensorboard_log=tmp_path / "logs",
        eval_freq=128,
        n_eval_episodes=2,
        checkpoint_freq=256,
        checkpoint_dir=tmp_path / "cp",
        best_model_dir=tmp_path / "best",
        verbose=0,
    )
    assert isinstance(model, MaskableRecurrentPPO)


# ---------------------------------------------------------------------------
# 3. Default n_steps values
# ---------------------------------------------------------------------------


def test_n_steps_defaults_differ_by_algorithm() -> None:
    """maskable_ppo and recurrent_ppo must have different default_n_steps (2048 vs 128)."""
    assert ALGORITHMS["maskable_ppo"]["default_n_steps"] != ALGORITHMS["recurrent_ppo"]["default_n_steps"], (
        "Expected different default_n_steps for maskable_ppo and recurrent_ppo"
    )


# ---------------------------------------------------------------------------
# 4. Explicit n_steps override
# ---------------------------------------------------------------------------


def test_train_with_explicit_n_steps_overrides_default(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """train() with explicit n_steps must use the provided value, not the default."""
    model, _mean, _std = train(
        algorithm="recurrent_ppo",
        total_timesteps=256,
        n_steps=64,
        batch_size=64,
        save_path=tmp_path / "m",
        tensorboard_log=tmp_path / "logs",
        eval_freq=128,
        n_eval_episodes=2,
        checkpoint_freq=256,
        checkpoint_dir=tmp_path / "cp",
        best_model_dir=tmp_path / "best",
        verbose=0,
    )
    assert isinstance(model, MaskableRecurrentPPO)
    assert model.n_steps == 64, (
        f"Expected model.n_steps == 64, got {model.n_steps}"
    )


# ---------------------------------------------------------------------------
# 5. Model file saved to disk
# ---------------------------------------------------------------------------


def test_train_saves_model_file(tmp_path: pytest.TempPathFactory) -> None:
    """train() must save a .zip model file at the specified save_path."""
    train(
        total_timesteps=256,
        save_path=tmp_path / "model",
        tensorboard_log=tmp_path / "logs",
        eval_freq=128,
        n_eval_episodes=2,
        checkpoint_freq=256,
        checkpoint_dir=tmp_path / "cp",
        best_model_dir=tmp_path / "best",
        verbose=0,
    )
    assert (tmp_path / "model.zip").exists(), (
        f"Expected model file at {tmp_path / 'model.zip'} but it was not found"
    )
