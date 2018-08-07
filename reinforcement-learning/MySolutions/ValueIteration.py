import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv


pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    def max_action(V, s):
    	best_action = 0
    	best_value = -1 * float('inf')
    	for a in range(env.nA):
    		prob, next_state, reward, done = env.P[s][a][0]
    		value = reward + discount_factor * prob * V[next_state]
    		if value > best_value:
    			best_action = a
    			best_value = value
    	return best_action, best_value

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    
    # Implement!
    while True:
    	stable = True
    	new_V = np.copy(V)
    	for s in range(env.nS):
    		best_action, best_value = max_action(V, s)
    		policy[s] = np.zeros(env.nA)
    		policy[s][best_action] = 1
    		if abs(V[s] - best_value) > theta:
    			stable = False
    		new_V[s] = best_value
    	V = new_V
    	if stable:
    		break

    return policy, V

policy, v = value_iteration(env)

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
