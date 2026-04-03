"""Training script for BJJEnv-v0. Supports MaskablePPO and MaskableRecurrentPPO."""

import argparse
from pathlib import Path
from typing import Any, ClassVar, Dict, Type, Union

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

import Game  # noqa: F401 – triggers BJJEnv-v0 registration
from Game.gym_env import BJJEnv
from Game.maskable_recurrent import MaskableRecurrentPPO


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "maskable_ppo": {
        "class": MaskablePPO,
        "policy": "MlpPolicy",
        "default_n_steps": 2048,
        "default_batch_size": 64,
    },
    "recurrent_ppo": {
        "class": MaskableRecurrentPPO,
        "policy": "MlpLstmPolicy",
        "default_n_steps": 128,
        "default_batch_size": 128,
    },
}


# ---------------------------------------------------------------------------
# Metrics callback
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    algorithm: str = "maskable_ppo",
    total_timesteps: int = 50_000,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int | None = None,
    batch_size: int | None = None,
    gamma: float = 0.99,
    verbose: int = 1,
    save_path: Path | None = None,
    tensorboard_log: Path | None = None,
    eval_freq: int = 5_000,
    n_eval_episodes: int = 10,
    eval_render_mode: str | None = None,
    eval_gameplay_log_path: Path | None = None,
    checkpoint_freq: int = 10_000,
    checkpoint_dir: Path | None = None,
    best_model_dir: Path | None = None,
) -> tuple[Union[MaskablePPO, MaskableRecurrentPPO], float, float]:
    """Train an agent on BJJEnv-v0.

    Args:
        algorithm: One of the keys in ALGORITHMS (``"maskable_ppo"`` or ``"recurrent_ppo"``).
        total_timesteps: Number of environment steps to train for.
        seed: Random seed for reproducibility.
        learning_rate: PPO learning rate.
        n_steps: Rollout length per update. Uses algorithm default when None.
        batch_size: Minibatch size for PPO updates. Uses algorithm default when None.
        gamma: Discount factor.
        verbose: Verbosity level (0 = silent, 1 = info).
        save_path: Where to save the trained model. Defaults to ``models/{algorithm}_bjj``.
        tensorboard_log: Directory for TensorBoard logs. None disables logging.
        eval_freq: Evaluate every this many timesteps.
        n_eval_episodes: Episodes per evaluation.
        eval_render_mode: Render mode used by the evaluation environment.
            Any non-None value enables gameplay logs to stdout.
        eval_gameplay_log_path: Optional path to write evaluation gameplay traces.
        checkpoint_freq: Save a checkpoint every this many timesteps.
        checkpoint_dir: Directory for periodic checkpoint saves.
        best_model_dir: Directory to save the best model found during eval.

    Returns:
        Tuple of (model, mean_reward, std_reward).
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}. Choose from: {list(ALGORITHMS)}")

    cfg = ALGORITHMS[algorithm]
    algo_class = cfg["class"]
    policy_str = cfg["policy"]
    resolved_n_steps = n_steps if n_steps is not None else cfg["default_n_steps"]
    resolved_batch_size = batch_size if batch_size is not None else cfg["default_batch_size"]
    resolved_save = Path(save_path) if save_path is not None else Path(f"models/{algorithm}_bjj")
    resolved_tb = Path(tensorboard_log) if tensorboard_log is not None else Path(f"logs/{algorithm}_bjj")
    resolved_checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path("models/checkpoints")
    resolved_best_model_dir = Path(best_model_dir) if best_model_dir is not None else Path("models/best_model")

    env = BJJEnv(render_mode=None)
    env.reset(seed=seed)

    model = algo_class(
        policy_str,
        env,
        learning_rate=learning_rate,
        n_steps=resolved_n_steps,
        batch_size=resolved_batch_size,
        gamma=gamma,
        seed=seed,
        verbose=verbose,
        tensorboard_log=str(resolved_tb),
    )

    eval_env = BJJEnv(
        render_mode=eval_render_mode,
        gameplay_log_path=eval_gameplay_log_path,
    )
    callbacks: list[Any] = [
        MaskableEvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(resolved_best_model_dir),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            use_masking=True,
            verbose=verbose,
        ),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(resolved_checkpoint_dir),
            name_prefix=f"{algorithm}_bjj",
            verbose=verbose,
        ),
        BJJMetricsCallback(verbose=verbose),
    ]

    model.learn(
        total_timesteps=total_timesteps,
        use_masking=True,
        callback=CallbackList(callbacks),
    )

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Evaluation over 10 episodes: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    resolved_save.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(resolved_save))
    print(f"Model saved to {resolved_save}")

    return model, mean_reward, std_reward


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def load_and_evaluate(
    model_path: Path = Path("models/maskable_ppo_bjj"),
    algorithm: str = "maskable_ppo",
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render_mode: str | None = None,
    gameplay_log_path: Path | None = None,
) -> tuple[float, float]:
    """Load a saved model and evaluate it.

    Args:
        model_path: Path to the saved model (without .zip extension).
        algorithm: Algorithm key used to look up the correct class for ``.load()``.
        n_eval_episodes: Number of episodes to evaluate over.
        deterministic: Whether to use deterministic actions during evaluation.
        render_mode: Render mode for evaluation. Any non-None value enables console traces.
        gameplay_log_path: Optional path to write gameplay traces during evaluation.

    Returns:
        Tuple of (mean_reward, std_reward).
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}. Choose from: {list(ALGORITHMS)}")

    algo_class = ALGORITHMS[algorithm]["class"]
    env = BJJEnv(render_mode=render_mode, gameplay_log_path=gameplay_log_path)
    model = algo_class.load(str(model_path), env=env)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, deterministic=deterministic
    )
    print(
        f"Evaluation over {n_eval_episodes} episodes: "
        f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}"
    )
    return mean_reward, std_reward


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train an RL agent on BJJEnv-v0.")
    parser.add_argument(
        "--algorithm",
        choices=list(ALGORITHMS),
        default="maskable_ppo",
        help="Algorithm to use for training (default: maskable_ppo).",
    )
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument(
        "--eval-render-mode",
        type=str,
        default=None,
        help="Evaluation render mode. Any non-None value enables gameplay logs to stdout.",
    )
    parser.add_argument(
        "--eval-gameplay-log-path",
        type=Path,
        default=None,
        help="Optional file path to write evaluation gameplay traces.",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        verbose=args.verbose,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        eval_render_mode=args.eval_render_mode,
        eval_gameplay_log_path=args.eval_gameplay_log_path,
        checkpoint_freq=args.checkpoint_freq,
    )
