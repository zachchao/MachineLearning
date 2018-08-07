import gym
import numpy as np


env = gym.make('MountainCar-v0')



class Experience(object):
	# S A R S'
	def __init__(self, state, action, reward, next_state):
		self.state = state
		self.action = action
		self.reward = reward
		self.next_state = next_state
		self.q_value = None

	def __str__(self):
		return "{0:+.2f} {1:+.5f} {2} {3} {4:+.2f} {5:+.5f}".format(
			self.state[0], self.state[1], self.action,
			self.reward, self.next_state[0], self.next_state[1])

class Memory(object):
	def __init__(self):
		self.experiences = []
		self.i = 0
		self.gamma = 0.9

	def add(self, experience):
		self.experiences.append(experience)

	def calc_q_values(self):
		print("Calculating Q_{}".format(self.i))
		# DP approach, go backwards and calculate
		# Q_{i + 1}(s,a) = E_{s'}[r + gamma max Q_i(s', a') | s, a]
		# Skip the last index as it is a terminal state, there exists no s'
		for i in range(len(self.experiences) - 2, -1, -1):
			# If the next state in memory is the next_state
			# if it isnt, this is a terminal state
			if self.experiences[i].next_state == self.experiences[i + 1]:
				self.experiences[i].q_value = self.experiences[i + 1].reward +
					self.gamma * 
		
		
class Agent(object):
	def __init__(self, epsilon=1, policy=None):
		self.policy = policy
		self.epsilon = epsilon

	def greedy_e(self):
		pass	

	def act(self, state):
		if self.policy == None:
			return greedy_e(state)
		return self.policy(state)	


memory = Memory()


for t in range(200):
    observation = env.reset()
    action = env.action_space.sample()
    next_observation, reward, done, info = env.step(action)
    experience = Experience(observation, action, reward, next_observation)
    memory.add(experience)
    observation = next_observation

    print(experience)
