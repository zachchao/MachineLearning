import random


class ActionSpace:
	def __init__(self, discrete_range):
		self.space = list(range(discrete_range))

	def sample(self):
		return random.choice(self.space)


# States are the agent's position
class Enviornment:
	def __init__(self):
		# 1 represents an obstacle, 0 is a path
		# Rewards : -1 per time-step
		# Actions : N, E, S, W
		# States : Agent's location
		self.maze = [
			[1, 1, 1, 1, 1, 1, 1 ,1],
			[1, 0, 0, 0, 0, 0, 0, 1],
			[0, 0, 1, 1, 0, 1, 0, 1],
			[1, 0, 0, 1, 1, 0, 0, 1],
			[1, 1, 0, 0, 1, 0, 1, 1],
			[1, 0, 1, 0, 1, 0, 0, 1],
			[1, 0, 0, 0, 0, 1, 0, 0]
		]
		self.start_pos = [2, 0]
		self.end_pos = [6, 7]
		self.agent_pos = self.start_pos[:]
		self.action_space = ActionSpace(4)

	def reset(self):
		self.agent_pos = self.start_pos[:]
		return self.start_pos

	# Return observation, reward, done, info
	# if you step out of bounds or into a wall you get -100
	def step(self, action):
		# 0 -> North | 1 -> East | 2 -> South | 3 -> West
		movements = [[-1, 0], [0, 1], [1, 0], [0, -1]]
		self.agent_pos = [x + y for x, y in zip(self.agent_pos, movements[action])]
		# -1 for movements, -100 for out of bounds
		try:
			if self.maze[self.agent_pos[0]][self.agent_pos[1]] == 0:
				reward = -1
				done = False
			else:
				reward = -100
				done = True
		# Out of bounds
		except IndexError:
			reward = -100
			done = True
		if self.agent_pos[0] < 0 or self.agent_pos[1] <0:
			reward = -100
			done = True
		# If finished maze
		if self.agent_pos == self.end_pos:
			done = True
			print("FINISHED!")

		return self.agent_pos, reward, done, {}


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
		#self.experiences = []
		# experiences[state[0]][state[1]][action] -> [reward, next_state]
		self.experiences = {}
		self.gamma = 0.9

	def remember(self, state, action, reward, next_state):
		experiences = self.experiences
		# Traverse dict until we get to the action dict
		for parameter in state:
			if parameter not in experiences.keys():
				experiences[parameter] = {}
			experiences = experiences[parameter]
		# We are now in the action dict
		if action not in experiences.keys():
			experiences[action] = []
		experiences[action] = [reward, next_state]

	def __getitem__(self, key):
		return self.experiences[key]

	def __str__(self):
		return str(self.experiences)

	'''
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
	'''
		
		
class Agent(object):
	def __init__(self, enviornment, epsilon=1, policy=None):
		self.enviornment = enviornment
		self.policy = policy
		self.epsilon = epsilon
		# State -> action -> value
		self.value_func = {}
		self.default_value = -1
		self.memory = Memory()

	def remember(self, state, action, reward, next_state):
		return self.memory.remember(state, action, reward, next_state)

	def greedy_e(self, state):
		# Greedy with probability (1 - epsilon) + (epsilon / k)
		# Random with probability (epsilon / k)
		chance = random.uniform(0, 1)
		# Greedy
		if change < (1 - self.epsilon) + (self.epsilon / len(self.enviornment.action_space.space)):
			pass

	# Very crude monte carlo verison
	def make_value_func(self):
		d = self.memory.experiences
		self.value_func = {
			state0 : {
				state1 : {
					action : None
					for action in d[state0][state1]
				}
				for state1 in d[state0]
			}
			for state0 in d
		}
		for state0 in d:
			for state1 in d[state0]:
				for action in d[state0][state1]:
					self.value_func[state0][state1][action] = self.value_of(state0, state1, action)
		print(self.value_func)

	# Accesses dict for value of a state
	def value_of(self, state0, state1, action, depth=4):
		if self.value_func[state0][state1][action] != None:
			return self.value_func[state0][state1][action]
		if depth == 0:
			return 0
		reward, next_state = self.memory[state0][state1][action]
		#print(state0, state1, action, reward, next_state)
		if reward == -100:
			self.value_func[state0][state1][action] = -100
			return -100
		next_actions = self.memory[next_state[0]][next_state[1]].keys()
		#print(state0, state1, action, reward, next_state, next_actions)
		maxVal = max([self.value_of(next_state[0], next_state[1], a, depth - 1)
			if a != action else -1000 for a in next_actions] )
		self.value_func[state0][state1][action] = maxVal
		return 0

	def act(self, state):
		if self.policy == None:
			return greedy_e(state)
		return self.policy(state)

	

env = Enviornment()
observation = env.reset()
agent = Agent(env)

for i in range(10):
	done = False
	observation = env.reset()
	while not done:
		action = env.action_space.sample()
		next_observation, reward, done, info = env.step(action)
		agent.remember(observation, action, reward, next_observation)
		#print(observation, action, reward, next_observation)
		observation = next_observation

print(agent.memory)
print()
#print(agent.memory.experiences.keys())
#print(agent.memory[2].keys())


agent.make_value_func()










