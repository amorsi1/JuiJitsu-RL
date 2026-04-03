"""Tests for the HPO module (src/Game/hpo.py).

Covers objective(), run_hpo(), and train_with_hpo() — verifying that Optuna
studies are configured correctly, hyperparameter ranges are honoured, JSON
artifacts are written, and the full orchestration pipeline returns a trained
MaskablePPO model.

Training timesteps are intentionally tiny (256–512) so these run in CI.
"""

import json
import math

import optuna
import pytest
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from sb3_contrib import MaskablePPO

from Game.hpo import objective, run_hpo, train_with_hpo

# Suppress noisy Optuna progress logs for the entire module.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Module-scoped fixture — 2 trials in-memory, shared across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hpo_study() -> optuna.Study:
    """Run 2 HPO trials once and reuse the study object across all tests."""
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=0))
    study.optimize(objective, n_trials=2)
    return study


# ---------------------------------------------------------------------------
# 1. Study creation
# ---------------------------------------------------------------------------


def test_study_creation_in_memory() -> None:
    """optuna.create_study(direction='maximize') must return an optuna.Study."""
    study = optuna.create_study(direction="maximize")
    assert isinstance(study, optuna.Study)


# ---------------------------------------------------------------------------
# 2. Study direction
# ---------------------------------------------------------------------------


def test_study_direction_is_maximize() -> None:
    """A study created with direction='maximize' must report StudyDirection.MAXIMIZE."""
    study = optuna.create_study(direction="maximize")
    assert study.direction == StudyDirection.MAXIMIZE


# ---------------------------------------------------------------------------
# 3. TPESampler acceptance
# ---------------------------------------------------------------------------


def test_tpe_sampler_accepted() -> None:
    """TPESampler must be accepted by create_study without raising."""
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    assert isinstance(study, optuna.Study)


# ---------------------------------------------------------------------------
# 4. objective is callable
# ---------------------------------------------------------------------------


def test_objective_is_callable() -> None:
    """objective must be a callable."""
    assert callable(objective)


# ---------------------------------------------------------------------------
# 5. objective returns float values
# ---------------------------------------------------------------------------


def test_objective_returns_float(hpo_study: optuna.Study) -> None:
    """Every completed trial must have a non-None float value."""
    for trial in hpo_study.trials:
        assert trial.value is not None, f"Trial {trial.number} has None value"
        assert isinstance(trial.value, float), (
            f"Trial {trial.number} value is {type(trial.value).__name__}, expected float"
        )


# ---------------------------------------------------------------------------
# 6. objective reward is finite
# ---------------------------------------------------------------------------


def test_objective_reward_is_finite(hpo_study: optuna.Study) -> None:
    """Every completed trial's reward must be a finite float (no NaN/inf)."""
    for trial in hpo_study.trials:
        assert math.isfinite(trial.value), (
            f"Trial {trial.number} value {trial.value} is not finite"
        )


# ---------------------------------------------------------------------------
# 7. best_params has exactly the expected 4 keys
# ---------------------------------------------------------------------------


def test_hpo_study_has_best_params(hpo_study: optuna.Study) -> None:
    """study.best_params must be a dict with exactly the 4 expected hyperparameter keys."""
    expected_keys = {"learning_rate", "n_steps", "batch_size", "gamma"}
    best_params = hpo_study.best_params
    assert isinstance(best_params, dict), (
        f"best_params is {type(best_params).__name__}, expected dict"
    )
    assert set(best_params.keys()) == expected_keys, (
        f"best_params keys {set(best_params.keys())} != expected {expected_keys}"
    )


# ---------------------------------------------------------------------------
# 8. best_params value types and ranges
# ---------------------------------------------------------------------------


def test_best_params_value_types(hpo_study: optuna.Study) -> None:
    """Each hyperparameter in best_params must have the correct type and lie in its valid range."""
    params = hpo_study.best_params

    lr = params["learning_rate"]
    assert isinstance(lr, float), f"learning_rate is {type(lr).__name__}, expected float"
    assert 1e-5 <= lr <= 1e-2, f"learning_rate {lr} outside [1e-5, 1e-2]"

    n_steps = params["n_steps"]
    assert isinstance(n_steps, int), f"n_steps is {type(n_steps).__name__}, expected int"
    assert n_steps in {512, 1024, 2048}, f"n_steps {n_steps} not in {{512, 1024, 2048}}"

    batch_size = params["batch_size"]
    assert isinstance(batch_size, int), (
        f"batch_size is {type(batch_size).__name__}, expected int"
    )
    assert batch_size in {32, 64, 128, 256}, (
        f"batch_size {batch_size} not in {{32, 64, 128, 256}}"
    )

    gamma = params["gamma"]
    assert isinstance(gamma, float), f"gamma is {type(gamma).__name__}, expected float"
    assert 0.95 <= gamma <= 0.999, f"gamma {gamma} outside [0.95, 0.999]"


# ---------------------------------------------------------------------------
# 9. run_hpo saves best_params.json
# ---------------------------------------------------------------------------


def test_run_hpo_saves_best_params_json(tmp_path: pytest.TempPathFactory) -> None:
    """run_hpo() must write best_params.json containing 'best_value' and 'best_params' keys."""
    run_hpo(n_trials=2, study_name="test", storage=None, output_dir=tmp_path, seed=0)

    json_path = tmp_path / "best_params.json"
    assert json_path.exists(), f"best_params.json not found in {tmp_path}"

    with json_path.open() as f:
        data = json.load(f)

    assert "best_value" in data, f"'best_value' key missing from best_params.json: {data}"
    assert "best_params" in data, f"'best_params' key missing from best_params.json: {data}"


# ---------------------------------------------------------------------------
# 10. run_hpo returns optuna.Study
# ---------------------------------------------------------------------------


def test_run_hpo_returns_study_object(tmp_path: pytest.TempPathFactory) -> None:
    """run_hpo() must return an optuna.Study instance."""
    result = run_hpo(n_trials=2, study_name="test", storage=None, output_dir=tmp_path, seed=0)
    assert isinstance(result, optuna.Study), (
        f"run_hpo() returned {type(result).__name__}, expected optuna.Study"
    )


# ---------------------------------------------------------------------------
# 11. train_with_hpo returns (MaskablePPO, float, float)
# ---------------------------------------------------------------------------


def test_train_with_hpo_returns_model(tmp_path: pytest.TempPathFactory) -> None:
    """train_with_hpo() must return a 3-tuple whose first element is MaskablePPO."""
    result = train_with_hpo(
        n_trials=1,
        total_timesteps=512,
        seed=0,
        save_path=tmp_path / "model",
        tensorboard_log=None,
        output_dir=tmp_path / "hpo_out",
        study_name="test_hpo",
        storage=None,
    )
    assert isinstance(result, tuple), (
        f"train_with_hpo() returned {type(result).__name__}, expected tuple"
    )
    assert len(result) == 3, f"Expected tuple of length 3, got length {len(result)}"
    model, mean_reward, std_reward = result
    assert isinstance(model, MaskablePPO), (
        f"First element is {type(model).__name__}, expected MaskablePPO"
    )


# ---------------------------------------------------------------------------
# 12. train_with_hpo saves model.zip
# ---------------------------------------------------------------------------


def test_train_with_hpo_saves_model(tmp_path: pytest.TempPathFactory) -> None:
    """train_with_hpo() must save a .zip model file at save_path."""
    train_with_hpo(
        n_trials=1,
        total_timesteps=512,
        seed=0,
        save_path=tmp_path / "model",
        tensorboard_log=None,
        output_dir=tmp_path / "hpo_out",
        study_name="test_hpo",
        storage=None,
    )
    assert (tmp_path / "model.zip").exists(), (
        f"model.zip not found in {tmp_path} after train_with_hpo()"
    )


# ---------------------------------------------------------------------------
# 13. train_with_hpo saves best_params.json in output_dir
# ---------------------------------------------------------------------------


def test_train_with_hpo_saves_best_params_json(tmp_path: pytest.TempPathFactory) -> None:
    """train_with_hpo() must write best_params.json with 'best_value' and 'best_params' keys."""
    output_dir = tmp_path / "hpo_out"
    train_with_hpo(
        n_trials=1,
        total_timesteps=512,
        seed=0,
        save_path=tmp_path / "model",
        tensorboard_log=None,
        output_dir=output_dir,
        study_name="test_hpo",
        storage=None,
    )

    json_path = output_dir / "best_params.json"
    assert json_path.exists(), f"best_params.json not found in {output_dir}"

    with json_path.open() as f:
        data = json.load(f)

    assert "best_value" in data, f"'best_value' key missing from best_params.json: {data}"
    assert "best_params" in data, f"'best_params' key missing from best_params.json: {data}"
