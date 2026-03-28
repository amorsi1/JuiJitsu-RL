"""MaskablePPO training script for BJJEnv-v0."""

from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

import Game  # noqa: F401 – triggers BJJEnv-v0 registration
from Game.gym_env import BJJEnv


def train(
    total_timesteps: int = 50_000,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    verbose: int = 1,
    save_path: Path = Path("models/maskable_ppo_bjj"),
) -> MaskablePPO:
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

    Returns:
        The trained MaskablePPO model.
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
    )

    model.learn(total_timesteps=total_timesteps, use_masking=True)

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

    return model


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
