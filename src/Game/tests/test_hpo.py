"""Tests for the HPO module (src/Game/hpo.py).

Covers objective(), run_hpo(), and train_with_hpo() — verifying that Optuna
studies are configured correctly, hyperparameter ranges are honoured, JSON
artifacts are written, and the full orchestration pipeline returns a trained
MaskablePPO model.

Performance strategy
--------------------
The ``objective()`` function normally uses 10 000 training steps per trial.
Tests use ``make_objective(trial_timesteps=256)`` and pass ``trial_timesteps=256``
to ``run_hpo()`` / ``train_with_hpo()`` so each trial completes in seconds.

Module-scoped fixtures share expensive computation across all tests that need it:
  - ``hpo_study``: 2 HPO trials, used by tests 5-8
  - ``run_hpo_result``: run_hpo() output, used by tests 9-10
  - ``hpo_run``: train_with_hpo() output, used by tests 11-13
"""

import json
import math

import optuna
import pytest
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from sb3_contrib import MaskablePPO

from Game.hpo import make_objective, objective, run_hpo, train_with_hpo

# Suppress noisy Optuna progress logs for the entire module.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Module-scoped fixtures — run expensive training once per test session
# ---------------------------------------------------------------------------

_TRIAL_TIMESTEPS = 256  # tiny budget; fast but end-to-end real training


@pytest.fixture(scope="module")
def hpo_study() -> optuna.Study:
    """Run 2 HPO trials once and reuse the study object across tests 5-8."""
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=0))
    study.optimize(make_objective(trial_timesteps=_TRIAL_TIMESTEPS), n_trials=2)
    return study


@pytest.fixture(scope="module")
def run_hpo_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run run_hpo() once and share output across tests 9-10."""
    base = tmp_path_factory.mktemp("run_hpo")
    study = run_hpo(
        n_trials=2,
        study_name="test_run_hpo",
        storage=None,
        output_dir=base,
        seed=0,
        trial_timesteps=_TRIAL_TIMESTEPS,
    )
    return {"study": study, "output_dir": base}


@pytest.fixture(scope="module")
def hpo_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run train_with_hpo() once and share result across tests 11-13."""
    base = tmp_path_factory.mktemp("hpo_run")
    save_path = base / "model"
    output_dir = base / "hpo_out"
    result = train_with_hpo(
        n_trials=1,
        total_timesteps=512,
        trial_timesteps=_TRIAL_TIMESTEPS,
        seed=0,
        save_path=save_path,
        tensorboard_log=None,
        output_dir=output_dir,
        study_name="test_hpo",
        storage=None,
    )
    return {"result": result, "save_path": save_path, "output_dir": output_dir}


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


def test_run_hpo_saves_best_params_json(run_hpo_result: dict) -> None:
    """run_hpo() must write best_params.json containing 'best_value' and 'best_params' keys."""
    json_path = run_hpo_result["output_dir"] / "best_params.json"
    assert json_path.exists(), f"best_params.json not found in {run_hpo_result['output_dir']}"

    with json_path.open() as f:
        data = json.load(f)

    assert "best_value" in data, f"'best_value' key missing from best_params.json: {data}"
    assert "best_params" in data, f"'best_params' key missing from best_params.json: {data}"


# ---------------------------------------------------------------------------
# 10. run_hpo returns optuna.Study
# ---------------------------------------------------------------------------


def test_run_hpo_returns_study_object(run_hpo_result: dict) -> None:
    """run_hpo() must return an optuna.Study instance."""
    assert isinstance(run_hpo_result["study"], optuna.Study), (
        f"run_hpo() returned {type(run_hpo_result['study']).__name__}, expected optuna.Study"
    )


# ---------------------------------------------------------------------------
# 11. train_with_hpo returns (MaskablePPO, float, float)
# ---------------------------------------------------------------------------


def test_train_with_hpo_returns_model(hpo_run: dict) -> None:
    """train_with_hpo() must return a 3-tuple whose first element is MaskablePPO."""
    result = hpo_run["result"]
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


def test_train_with_hpo_saves_model(hpo_run: dict) -> None:
    """train_with_hpo() must save a .zip model file at save_path."""
    model_zip = hpo_run["save_path"].with_suffix(".zip")
    assert model_zip.exists(), f"model.zip not found at {model_zip} after train_with_hpo()"


# ---------------------------------------------------------------------------
# 13. train_with_hpo saves best_params.json in output_dir
# ---------------------------------------------------------------------------


def test_train_with_hpo_saves_best_params_json(hpo_run: dict) -> None:
    """train_with_hpo() must write best_params.json with 'best_value' and 'best_params' keys."""
    json_path = hpo_run["output_dir"] / "best_params.json"
    assert json_path.exists(), f"best_params.json not found in {hpo_run['output_dir']}"

    with json_path.open() as f:
        data = json.load(f)

    assert "best_value" in data, f"'best_value' key missing from best_params.json: {data}"
    assert "best_params" in data, f"'best_params' key missing from best_params.json: {data}"
