from gym.envs.registration import register

register(
    id='parking_env-v0',
    entry_point='parking_env.env:CustomEnv',
)
