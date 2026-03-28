import gymnasium

if "BJJEnv-v0" not in gymnasium.registry:
    gymnasium.register(
        id="BJJEnv-v0",
        entry_point="Game.gym_env:BJJEnv",
        max_episode_steps=None,
        disable_env_checker=True,
    )
