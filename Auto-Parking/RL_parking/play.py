import argparse
import gym
import parking_env
import pybullet as p


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="parking_env-v0", help='name of the environment to run')
parser.add_argument('--render', type=bool, default=True, help='render the environment')
parser.add_argument('--mode', type=str, default='1', choices=['1', '2', '3', '4', '5', '6', '7', '8'], help='mode')

args = parser.parse_args()

env = gym.make(args.env, render=args.render, manual=True, mode=args.mode)
env.reset()

done = False
while True:
    keys = p.getKeyboardEvents()
    for k, v in keys.items():
        if k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN):
            action = 2
        if k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN):
            action = 3
        if k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN):
            action = 0
        if k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN):
            action = 1
        if (k == p.B3G_LEFT_ARROW or k == p.B3G_RIGHT_ARROW or k == p.B3G_UP_ARROW or k == p.B3G_DOWN_ARROW) and (v & p.KEY_WAS_RELEASED):
            action = 4
        else:
            pass
        next_state, reward, done, _ = env.step(action)
