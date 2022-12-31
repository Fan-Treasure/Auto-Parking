import argparse
import datetime
import os
import requests
import gym
import parking_env
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="parking_env-v0", help='name of the environment to run')
parser.add_argument('--render', type=bool, default=True, help='render the environment')
parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint path')
parser.add_argument('--mode', type=str, default='1', choices=['1', '2', '3', '4', '5', '6', '7', '8'], help='mode')

args = parser.parse_args()

args.ckpt_path = 'log/DQN_1_1231_0413/dqn_agent.zip'


# Evaluation
env = gym.make(args.env, render=args.render, mode=args.mode)
obs = env.reset()
model = DQN.load(args.ckpt_path, env=env)

episode_return = 0
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    if action==0:
        requests.post('******/forward', data={signal:1})
    if action==1:
        requests.post('******/backward', data={signal:1})
    if action==2:
        requests.post('******/left', data={signal:1})
    if action==3:
        requests.post('******/right', data={signal:1})
    obs, reward, done, info = env.step(action)
    episode_return += reward
    if done:
        for j in range(10000000):
            reward += 0.0001
        break

env.close()
print(f'episode return: {episode_return}')
