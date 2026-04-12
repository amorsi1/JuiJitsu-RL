#!/usr/bin/env python
"""Smoke test for MaskableRecurrentPPO on BJJEnv-v0.

Runs a short training session, evaluates the trained policy, and plays a
full episode using greedy masked prediction. Designed to be a quick sanity
check you can run from the command line.

Usage:
    uv run python scripts/smoke_test_recurrent.py
    uv run python scripts/smoke_test_recurrent.py --timesteps 2000 --episodes 5
"""

import argparse
import time

import numpy as np

import Game  # noqa: F401 — registers BJJEnv-v0
from Game.gym_env import BJJEnv
from Game.maskable_recurrent import MaskableRecurrentPPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for MaskableRecurrentPPO")
    parser.add_argument("--timesteps", type=int, default=512, help="Training timesteps (default: 512)")
    parser.add_argument("--episodes", type=int, default=3, help="Eval episodes after training (default: 3)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    return parser.parse_args()


def train_model(timesteps: int, seed: int) -> MaskableRecurrentPPO:
    print(f"\n{'='*60}")
    print(f"Training MaskableRecurrentPPO for {timesteps} timesteps (seed={seed})")
    print(f"{'='*60}")

    env = BJJEnv()
    env.reset(seed=seed)

    model = MaskableRecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=128,
        batch_size=128,
        verbose=1,
        seed=seed,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps, use_masking=True)
    elapsed = time.perf_counter() - t0

    print(f"\nTraining complete in {elapsed:.1f}s")
    return model


def evaluate_model(model: MaskableRecurrentPPO, n_episodes: int) -> None:
    print(f"\n{'='*60}")
    print(f"Evaluating over {n_episodes} episodes")
    print(f"{'='*60}")

    eval_env = BJJEnv()
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_episodes, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")


def play_episode(model: MaskableRecurrentPPO) -> None:
    """Play one full episode using greedy masked prediction and print each step."""
    print(f"\n{'='*60}")
    print("Playing one full episode (deterministic + masked)")
    print(f"{'='*60}")

    env = BJJEnv()
    obs, info = env.reset()
    lstm_state = None
    episode_start = np.array([True])

    step = 0
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        masks = env.action_masks()
        action, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=episode_start,
            deterministic=True,
            action_masks=masks,
        )
        episode_start = np.array([False])

        assert masks[int(action)], f"Step {step}: predicted illegal action {action}"

        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        step += 1

        if step <= 5 or terminated or truncated:
            status = ""
            if terminated:
                status = "  [TERMINATED]"
            elif truncated:
                status = "  [TRUNCATED]"
            print(f"  step {step:3d} | action={int(action):4d} | reward={float(reward):+7.1f} | total={total_reward:+8.1f}{status}")
        elif step == 6:
            print("  ...")

    result = "win" if info.get("is_win") else ("loss" if info.get("is_loss") else "draw/truncated")
    print(f"\nEpisode finished: {step} steps | result={result} | point_diff={info.get('point_diff', 'N/A')}")


def check_action_masks_legality(model: MaskableRecurrentPPO, n_checks: int = 20) -> None:
    """Verify predict() never returns a masked (illegal) action."""
    print(f"\n{'='*60}")
    print(f"Checking that predict() always returns legal actions ({n_checks} resets)")
    print(f"{'='*60}")

    env = BJJEnv()
    violations = 0

    for i in range(n_checks):
        obs, _ = env.reset()
        masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        if not masks[int(action)]:
            violations += 1
            print(f"  VIOLATION at reset {i}: action={action} is masked")

    if violations == 0:
        print(f"  OK — 0 violations across {n_checks} resets")
    else:
        print(f"  FAILED — {violations}/{n_checks} illegal actions predicted")


def main() -> None:
    args = parse_args()

    model = train_model(args.timesteps, args.seed)
    evaluate_model(model, args.episodes)
    check_action_masks_legality(model)
    play_episode(model)

    print(f"\n{'='*60}")
    print("Smoke test complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
