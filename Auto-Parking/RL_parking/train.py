import argparse  # 命令行解析
import datetime
import os

import gym
import parking_env
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="parking_env-v0", help='name of the environment to run')
parser.add_argument('--render', type=bool, default=True, help='render the environment')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--total_timesteps', type=int, default=int(2e6), help='total timesteps to run')
parser.add_argument('--save_freq', type=int, default=int(5e5), help='checkpoint save frequency')
parser.add_argument('--log_path', type=str, default='./log', help='logging path')
parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint path')
parser.add_argument('--mode', type=str, default='1', choices=['1', '2', '3', '4', '5', '6', '7', '8'], help='mode')

args = parser.parse_args()

time = datetime.datetime.strftime(datetime.datetime.now(), '%m%d_%H%M')
args.log_path = os.path.join(args.log_path, f'DQN_{args.mode}_{time}')

if not args.ckpt_path:
    args.ckpt_path = os.path.join(args.log_path, f'dqn_agent')

env = gym.make(args.env, render=args.render, mode=args.mode)
env.reset()
# env = DummyVecEnv([lambda: env])
# env = VecNormalize(env, norm_obs=True, clip_obs=10.)

model = DQN('MlpPolicy', env, verbose=1, seed=args.seed)
logger = configure(args.log_path, ["stdout", "csv", "tensorboard"])
model.set_logger(logger)
checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=args.log_path, name_prefix='dqn_agent')
model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
model.save(args.ckpt_path)
del model


# Evaluation
env = gym.make(args.env, render=True, mode=args.mode)
obs = env.reset()
model = DQN.load(args.ckpt_path, env=env)

episode_return = 0
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_return += reward
    if done:
        for j in range(10000000):
            reward += 0.0001
        break

env.close()
print(f'episode return: {episode_return}')

