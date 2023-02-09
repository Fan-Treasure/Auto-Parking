import argparse
import datetime
import os
import time

import gym
import parking_env
from stable_baselines3 import DQN
import pybullet as p


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="parking_env-v0", help='name of the environment to run')
parser.add_argument('--render', type=bool, default=False, help='render the environment')
parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint path')
parser.add_argument('--mode', type=str, default='6', choices=['1', '2', '3', '4', '5', '6', '7', '8'], help='mode')

args = parser.parse_args()

if args.mode in ['1', '2', '3', '6']:
    cameraYaw = 0
else:
    cameraYaw = 180

for step in range(500000, 2000000, 50000):
    # args.ckpt_path = f'log/DQN_3_0809_1123/dqn_agent_{step}_steps.zip'
    args.ckpt_path = f'log/DQN_1_1227_1112/dqn_agent_{step}_steps.zip'
    # Evaluation
    env = gym.make(args.env, render=True, mode=args.mode, render_video=True)
    obs = env.reset()
    model = DQN.load(args.ckpt_path, env=env)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    p.resetDebugVisualizerCamera(
        cameraDistance=4,
        cameraYaw=cameraYaw,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0]
    )
    p.addUserDebugText(
        text=f"training step: {step}",
        textPosition=[0, 0, 2],
        textColorRGB=[1, 0, 0],
        textSize=2.5
    )

    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, f"log/training/DQN_{args.mode}_{step}.mp4")
    episode_return = 0
    for i in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        position, _ = p.getBasePositionAndOrientation(env.car)
        episode_return += reward
        if done:
            break

    p.stopStateLogging(log_id)

    env.close()
    print(f'step: {step}, episode return: {episode_return}')

    break
