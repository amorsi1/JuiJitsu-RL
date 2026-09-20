"""Tests for BJJEnv-v0 gymnasium registry registration.

Verifies that importing Game triggers registration and that
gymnasium.make("BJJEnv-v0") returns a correctly configured environment.
"""
import importlib

import gymnasium
import numpy as np
import pytest

import Game  # noqa: F401 — side-effect import: triggers __init__.py registration
from Game.gym_env import BJJEnv


# ---------------------------------------------------------------------------
# 1. Registry presence — no environment construction needed
# ---------------------------------------------------------------------------


def test_bjjenv_is_in_gymnasium_registry() -> None:
    """BJJEnv-v0 must appear in gymnasium.registry after importing Game."""
    assert "BJJEnv-v0" in gymnasium.registry


def test_gymnasium_spec_id() -> None:
    """gymnasium.spec() must resolve to an EnvSpec with id='BJJEnv-v0'."""
    spec = gymnasium.spec("BJJEnv-v0")
    assert spec.id == "BJJEnv-v0"


def test_gymnasium_spec_entry_point() -> None:
    """Registry entry_point must point to Game.gym_env:BJJEnv."""
    spec = gymnasium.spec("BJJEnv-v0")
    assert spec.entry_point == "Game.gym_env:BJJEnv"


def test_gymnasium_spec_no_max_episode_steps() -> None:
    """max_episode_steps must be None so BJJEnv manages its own truncation."""
    spec = gymnasium.spec("BJJEnv-v0")
    assert spec.max_episode_steps is None


# ---------------------------------------------------------------------------
# 2. gymnasium.make() — construction and wrapper chain
# ---------------------------------------------------------------------------


def test_make_does_not_raise() -> None:
    """gymnasium.make('BJJEnv-v0') must not raise."""
    env = gymnasium.make("BJJEnv-v0")
    env.close()


def test_make_unwrapped_is_bjjenv_instance() -> None:
    """env.unwrapped must be a BJJEnv instance, not a generic wrapper."""
    env = gymnasium.make("BJJEnv-v0")
    try:
        assert isinstance(env.unwrapped, BJJEnv), (
            f"Expected BJJEnv, got {type(env.unwrapped).__name__}"
        )
    finally:
        env.close()


def test_make_no_time_limit_wrapper() -> None:
    """No TimeLimit wrapper should be present — BJJEnv manages truncation internally."""
    env = gymnasium.make("BJJEnv-v0")
    try:
        e = env
        while hasattr(e, "env"):
            assert not isinstance(e, gymnasium.wrappers.TimeLimit), (
                "Unexpected TimeLimit wrapper; set max_episode_steps=None in gymnasium.register()"
            )
            e = e.env
    finally:
        env.close()


def test_make_no_passive_env_checker_wrapper() -> None:
    """PassiveEnvChecker must be absent because disable_env_checker=True."""
    env = gymnasium.make("BJJEnv-v0")
    try:
        e = env
        while hasattr(e, "env"):
            assert not isinstance(e, gymnasium.wrappers.PassiveEnvChecker), (
                "Unexpected PassiveEnvChecker; set disable_env_checker=True in gymnasium.register()"
            )
            e = e.env
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 3. Spaces are preserved through the wrapper chain
# ---------------------------------------------------------------------------


def test_make_observation_space_shape() -> None:
    """Observation space shape must be (5,) through gymnasium.make()."""
    env = gymnasium.make("BJJEnv-v0")
    try:
        assert env.observation_space.shape == (5,), (
            f"Expected (5,), got {env.observation_space.shape}"
        )
    finally:
        env.close()


def test_make_observation_space_dtype() -> None:
    """Observation space dtype must be float32 through gymnasium.make()."""
    env = gymnasium.make("BJJEnv-v0")
    try:
        assert env.observation_space.dtype == np.float32, (
            f"Expected float32, got {env.observation_space.dtype}"
        )
    finally:
        env.close()


def test_make_action_space_is_discrete() -> None:
    """Action space must be Discrete through gymnasium.make()."""
    env = gymnasium.make("BJJEnv-v0")
    try:
        assert isinstance(env.action_space, gymnasium.spaces.Discrete), (
            f"Expected Discrete, got {type(env.action_space).__name__}"
        )
    finally:
        env.close()


def test_make_action_space_n_is_positive() -> None:
    """action_space.n must be positive (>=700 edges in GrappleMap)."""
    env = gymnasium.make("BJJEnv-v0")
    try:
        assert env.action_space.n > 0, f"action_space.n={env.action_space.n}"
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 4. Registration is idempotent
# ---------------------------------------------------------------------------


def test_double_registration_does_not_raise() -> None:
    """Re-running __init__.py via module reload must not raise.

    gymnasium.register() raises gymnasium.error.Error on duplicate IDs.
    The idempotency guard prevents this on repeated imports.
    """
    game_module = importlib.import_module("Game")
    importlib.reload(game_module)  # re-runs __init__.py; must not raise
    assert "BJJEnv-v0" in gymnasium.registry


def test_registration_count_is_one() -> None:
    """BJJEnv-v0 must appear exactly once in the registry."""
    count = sum(1 for k in gymnasium.registry if k == "BJJEnv-v0")
    assert count == 1, f"Expected 1 registration, found {count}"
