import argparse
import datetime
import os

import gym
import parking_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="parking_env-v0", help='name of the environment to run')
parser.add_argument('--render', type=bool, default=False, help='render the environment')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--total_timesteps', type=int, default=int(5e6), help='total timesteps to run')
parser.add_argument('--save_freq', type=int, default=int(5e5), help='checkpoint save frequency')
parser.add_argument('--log_path', type=str, default='./log/PPO', help='logging path')
parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint path')
parser.add_argument('--mode', type=str, default='5', choices=['1', '2', '3', '4', '5'], help='mode')

args = parser.parse_args()

time = datetime.datetime.strftime(datetime.datetime.now(), '%m%d_%H%M')
args.log_path = os.path.join(args.log_path, f'PPO_{args.mode}_{time}')
# args.ckpt_path = 'log/PPO/PPO_5_0808_1741/ppo_agent_1000000_steps.zip'
if not args.ckpt_path:
    args.ckpt_path = os.path.join(args.log_path, f'ppo_agent')

env = gym.make(args.env, render=args.render, mode=args.mode)
env.reset()
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = PPO('MlpPolicy', env, verbose=1, seed=args.seed)
logger = configure(args.log_path, ["stdout", "csv", "tensorboard"])
model.set_logger(logger)
checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=args.log_path, name_prefix='ppo_agent')
model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
model.save(args.ckpt_path)
del model


# Evaluation
env = gym.make(args.env, render=True, mode=args.mode)
obs = env.reset()
model = PPO.load(args.ckpt_path, env=env, print_system_info=True)
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print(f'mean reward: {mean_reward}, std reward: {std_reward}')

episode_return = 0
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_return += reward
    if done:
        break

print(f'episode return: {episode_return}')
