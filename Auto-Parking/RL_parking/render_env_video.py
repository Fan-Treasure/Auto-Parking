import argparse
import datetime
import os
import time

import gym
import parking_env
import pybullet as p
import moviepy.editor as mpy


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="parking_env-v0", help='name of the environment to run')
parser.add_argument('--render', type=bool, default=False, help='render the environment')
parser.add_argument('--mode', type=str, default='1', choices=['1', '2', '3', '4', '5', '6', '7', '8'], help='mode')

args = parser.parse_args()

if args.mode in ['1', '2', '3', '6']:
    cameraYaw = 0
else:
    cameraYaw = 180

# Evaluation
env = gym.make(args.env, render=True, mode=args.mode, render_video=True)
obs = env.reset()

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

filepath = f"log/gif/DQN_env_{args.mode}.mp4"
log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, filepath)
# log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, f"log/env_720.mp4")

position = [0, 0, 0]
if args.mode == '1':
    position = [1.9, 0.95, 0]
elif args.mode == '2':
    position = [0.95, 1.05, 0]
elif args.mode == '3':
    position = [-0.95, 1.05, 0]
elif args.mode == '4':
    position = [-0.95, -1.05, 0]
elif args.mode == '5':
    position = [1.13, -1.14, 0]
elif args.mode == '6':
    position = [-0.95, 1.05, 0]

for cameraYaw in range(362):
    p.resetDebugVisualizerCamera(
        cameraDistance=4,
        cameraYaw=cameraYaw,
        cameraPitch=-45,
        cameraTargetPosition=position
    )
    time.sleep(1 / 240)

# for cameraYaw in range(720):
#     position, _ = p.getBasePositionAndOrientation(env.car)
#     p.resetDebugVisualizerCamera(
#         cameraDistance=6,
#         cameraYaw=cameraYaw,
#         cameraPitch=-40,
#         cameraTargetPosition=position
#     )
#     time.sleep(1 / 240)

p.stopStateLogging(log_id)

env.close()

'''video = mpy.VideoFileClip(filepath).subclip(0.02)
video.write_videofile(filepath.replace('.mp4', '2.mp4'))'''
