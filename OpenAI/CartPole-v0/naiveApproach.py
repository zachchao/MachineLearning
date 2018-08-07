import gym
import numpy as np


env = gym.make('CartPole-v0')

best_weights = np.zeros(4)
best_score = 0

def determine_action(weights, observation):
    dot_P = np.dot(weights, observation)
    if dot_P >= 0:
        return 1
    return 0


episodes = 0
while best_score != 200:
    episodes += 1
    weights = np.random.uniform(-1.0, 1.0, 4)
    observation = env.reset()
    for t in range(200):
        action = determine_action(weights, observation)
        observation, reward, done, info = env.step(action)
        if done:
            if t + 1 > best_score:
                best_score = t + 1
                best_weights = weights
            break

print(episodes)

for t in range(200):
    observation = env.reset()
    action = determine_action(best_weights, observation)
    observation, reward, done, info = env.step(action)
    env.render()
