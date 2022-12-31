import gym
import highway_env
import numpy as np

from gym.wrappers import RecordVideo
from stable_baselines3 import HerReplayBuffer, SAC

env = gym.make("parking-v0")
env = RecordVideo(env, video_folder="run", episode_trigger=lambda e: True)  # record all episodes
env.unwrapped.set_record_video_wrapper(env)

model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
        max_episode_length=100,
        online_sampling=True,
    ),
    verbose=1,
    learning_rate=1e-3,
    gamma=0.95,
    batch_size=256,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
)

model.learn(int(1e5))
model.save("her_sac_highway")

model = SAC.load("her_sac_highway", env=env)
obs = env.reset()

episode_reward = 0
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    env.render()
    if done:
        break
