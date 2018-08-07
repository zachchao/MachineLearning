import gym
import numpy as np


env = gym.make('MountainCar-v0')

for t in range(200):
    observation = env.reset()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    print(round(observation[0], 3), round(observation[1], 3), reward, done)
