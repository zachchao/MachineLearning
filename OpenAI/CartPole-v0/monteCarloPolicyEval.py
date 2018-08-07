import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import matplotlib.pyplot as plt


env = gym.make('CartPole-v0')

# dataSize is how much data we want
def generateTrainingData(episodes):
    logits = []
    for episode in range(episodes):
        observation = env.reset()
        for t in range(200):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            # State action reward
            SAR = np.append(np.append(observation, action), reward)
            logits.append(SAR)
            if done:
                break
    return np.array(logits)
    

monteCarlo = generateTrainingData(10)
print(monteCarlo[0], len(monteCarlo))
