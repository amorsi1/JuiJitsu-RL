"""MaskablePPO training script for BJJEnv-v0."""

from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

import Game  # noqa: F401 – triggers BJJEnv-v0 registration
from Game.gym_env import BJJEnv


class BJJMetricsCallback(BaseCallback):
    """Logs BJJ-specific win rate and point differential to TensorBoard.

    Reads ``is_win`` and ``point_diff`` keys from terminal-step info dicts
    (populated by BJJEnv.step()) and records aggregated stats once per PPO
    rollout via SB3's logger.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._win_buffer: list[bool] = []
        self._point_diff_buffer: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "is_win" in info:
                self._win_buffer.append(info["is_win"])
                self._point_diff_buffer.append(info["point_diff"])
        return True

    def _on_rollout_end(self) -> None:
        if self._win_buffer:
            win_rate = sum(self._win_buffer) / len(self._win_buffer)
            mean_point_diff = sum(self._point_diff_buffer) / len(self._point_diff_buffer)
            self.logger.record("bjj/win_rate", win_rate)
            self.logger.record("bjj/mean_point_diff", mean_point_diff)
        self._win_buffer = []
        self._point_diff_buffer = []


def train(
    total_timesteps: int = 50_000,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    verbose: int = 1,
    save_path: Path = Path("models/maskable_ppo_bjj"),
    tensorboard_log: Path | None = Path("logs/maskable_ppo_bjj"),
    eval_freq: int = 5_000,
    n_eval_episodes: int = 10,
    checkpoint_freq: int = 10_000,
    checkpoint_dir: Path = Path("models/checkpoints"),
    best_model_dir: Path = Path("models/best_model"),
) -> tuple[MaskablePPO, float, float]:
    """Train a MaskablePPO agent on BJJEnv-v0.

    Args:
        total_timesteps: Number of environment steps to train for.
        seed: Random seed for reproducibility.
        learning_rate: PPO learning rate.
        n_steps: Rollout length per update.
        batch_size: Minibatch size for PPO updates.
        gamma: Discount factor.
        verbose: Verbosity level (0 = silent, 1 = info).
        save_path: Where to save the trained model (directory created if needed).
        tensorboard_log: Directory for TensorBoard logs. None disables logging.
        eval_freq: Evaluate every this many timesteps.
        n_eval_episodes: Episodes per evaluation.
        checkpoint_freq: Save a checkpoint every this many timesteps.
        checkpoint_dir: Directory for periodic checkpoint saves.
        best_model_dir: Directory to save the best model found during eval.

    Returns:
        Tuple of (model, mean_reward, std_reward) — the trained model and its
        final evaluation statistics.
    """
    env = BJJEnv()
    env.reset(seed=seed)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        seed=seed,
        verbose=verbose,
        tensorboard_log=str(tensorboard_log) if tensorboard_log is not None else None,
    )

    eval_env = BJJEnv()
    callbacks: list[Any] = [
        MaskableEvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(best_model_dir),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            use_masking=True,
            verbose=verbose,
        ),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="maskable_ppo_bjj",
            verbose=verbose,
        ),
        BJJMetricsCallback(verbose=verbose),
    ]

    model.learn(
        total_timesteps=total_timesteps,
        use_masking=True,
        callback=CallbackList(callbacks),
    )

    # Evaluate the trained policy
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, deterministic=True
    )
    print(f"Evaluation over 10 episodes: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Save model
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {save_path}")

    return model, mean_reward, std_reward


def load_and_evaluate(
    model_path: Path = Path("models/maskable_ppo_bjj"),
    n_eval_episodes: int = 10,
    deterministic: bool = True,
) -> tuple[float, float]:
    """Load a saved MaskablePPO model and evaluate it.

    Args:
        model_path: Path to the saved model (without .zip extension).
        n_eval_episodes: Number of episodes to evaluate over.
        deterministic: Whether to use deterministic actions during evaluation.

    Returns:
        Tuple of (mean_reward, std_reward).
    """
    env = BJJEnv()
    model = MaskablePPO.load(str(model_path), env=env)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, deterministic=deterministic
    )
    print(
        f"Evaluation over {n_eval_episodes} episodes: "
        f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}"
    )

    return mean_reward, std_reward


if __name__ == "__main__":
    train()
