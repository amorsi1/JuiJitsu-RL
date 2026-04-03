"""Optuna HPO orchestration for MaskablePPO on BJJEnv-v0."""

import json
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sb3_contrib import MaskablePPO

from Game.train_sb3 import train


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function that samples hyperparameters and runs a short trial.

    Args:
        trial: Optuna trial object used to sample hyperparameter values.

    Returns:
        Mean reward from the short training run.
    """
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)

    _, mean_reward, _ = train(
        total_timesteps=10_000,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        verbose=0,
        tensorboard_log=None,
        eval_freq=999_999,
        n_eval_episodes=5,
        save_path=Path(f"/tmp/hpo_trial_{trial.number}"),
        checkpoint_dir=Path("/tmp/hpo_checkpoints"),
        best_model_dir=Path("/tmp/hpo_best"),
    )
    return mean_reward


def run_hpo(
    n_trials: int,
    study_name: str,
    storage: str | None,
    output_dir: Path,
    seed: int,
) -> optuna.Study:
    """Create an Optuna study and run hyperparameter optimisation.

    Uses TPESampler for acquisition and MedianPruner for early stopping.
    Saves best params to ``output_dir/best_params.json`` after optimisation.

    Args:
        n_trials: Number of trials to run.
        study_name: Name of the Optuna study (used for storage lookup).
        storage: Optuna storage URL (e.g. ``"sqlite:///hpo.db"``). None uses
            in-memory storage.
        output_dir: Directory where ``best_params.json`` is written.
        seed: Random seed passed to TPESampler for reproducibility.

    Returns:
        The completed Optuna study object.
    """
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = output_dir / "best_params.json"
    best_params_path.write_text(
        json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2)
    )

    return study


def train_with_hpo(
    n_trials: int = 50,
    total_timesteps: int = 500_000,
    seed: int = 42,
    save_path: Path = Path("models/maskable_ppo_bjj_hpo"),
    tensorboard_log: Path | None = Path("logs/maskable_ppo_bjj_hpo"),
    output_dir: Path = Path("hpo_results"),
    study_name: str = "bjj_hpo",
    storage: str | None = None,
) -> tuple[MaskablePPO, float, float]:
    """Run HPO then train a full model using the best discovered hyperparameters.

    Orchestrates the full pipeline:
    1. Run ``n_trials`` Optuna trials to find optimal hyperparameters.
    2. Train a production model for ``total_timesteps`` using those params.

    Args:
        n_trials: Number of HPO trials to run.
        total_timesteps: Timesteps for the final full-budget training run.
        seed: Random seed for both HPO sampler and final training.
        save_path: Path to save the final trained model.
        tensorboard_log: TensorBoard log directory for the final training run.
            None disables logging.
        output_dir: Directory for HPO outputs (``best_params.json``).
        study_name: Optuna study name.
        storage: Optuna storage URL. None uses in-memory storage.

    Returns:
        Tuple of (model, mean_reward, std_reward) from the final training run.
    """
    study = run_hpo(
        n_trials=n_trials,
        output_dir=output_dir,
        study_name=study_name,
        storage=storage,
        seed=seed,
    )
    best_params = study.best_params
    print(
        f"HPO complete. Best params: {best_params}. "
        f"Running full training with {total_timesteps} steps."
    )
    return train(
        **best_params,
        total_timesteps=total_timesteps,
        seed=seed,
        save_path=save_path,
        tensorboard_log=tensorboard_log,
    )


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train_with_hpo()
