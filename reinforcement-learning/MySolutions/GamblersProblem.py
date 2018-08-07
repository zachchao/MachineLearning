import numpy as np
import sys
import matplotlib.pyplot as plt
if "../" not in sys.path:
    sys.path.append("../")



class enviornment:
    '''
    env: OpenAI env. env.P represents the transition probabilities of the environment.
        env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
        env.nS is a number of states in the environment. 
        env.nA is a number of actions in the environment.
    '''
    def __init__(self, p_h, n_states):
        self.p_h = p_h
        self.p_t = 1 - p_h
        self.nS = n_states
        self.P = {}
        self.generateP()

    def generateP(self):
        for state in range(1, self.nS + 1):
            state_outcomes = {}
            for stake in range(min(state, 100 - state) + 1):
                state_outcomes[stake] = []
                # Lost our stake
                # If we lost all our money
                if state == stake:
                    state_outcomes[stake].append([self.p_t, state - stake, 0, True])
                else:
                    state_outcomes[stake].append([self.p_t, state - stake, 0, False])
                # Won our stake
                # If we reached our goal
                if state + stake >= self.nS:
                    state_outcomes[stake].append([self.p_h, state + stake, 1, True])
                else:
                    state_outcomes[stake].append([self.p_h, state + stake, 0, False])
            self.P[state] = state_outcomes


def value_iteration_for_gamblers(p_h, theta=0.0000001, discount_factor=1.0):
    """
    Args:
        p_h: Probability of the coin coming up heads
    """
    def max_action(V, s):
        best_action = 1
        best_value = -1 * float('inf')
        for a in env.P[s]:
            value = 0
            for prob, next_state, reward, done in env.P[s][a]:
                value += prob * (reward + discount_factor * V[next_state])
            if value >= best_value:
                best_action = a
                best_value = value
        return best_action, best_value
    

    max_stake = 100
    env = enviornment(p_h, max_stake)
    # Initial value_func
    V = [0 for capital in range(env.nS + 1)]
    V = np.asarray(V, dtype=np.float32)
    while True:
        stable = True
        new_V = np.copy(V)
        for s in range(1, 100):
            value = 0
            best_action, best_value = max_action(V, s)
            #A = one_step_lookahead(s, V)
            #best_value = np.max(A)
            if abs(V[s] - best_value) > theta:
                stable = False
            new_V[s] = best_value
        V = new_V
        if stable:
            break

    policy = np.zeros(100)
    for s in range(1, 100):
        #A = one_step_lookahead(s, V)
        #A[0] = 0
        #best_action = np.argmax(A)
        best_action, best_value = max_action(V, s)
        policy[s] = best_action
    return policy, V

policy, v = value_iteration_for_gamblers(0.25)

print("Optimized Policy:")
print(policy)
print("")

print("Optimized Value Function:")
print(v)
print("")

# Plotting Final Policy (action stake) vs State (Capital)

# x axis values
x = range(100)
# corresponding y axis values
y = v[:100]
 
# plotting the points 
plt.plot(x, y)
 
# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Value Estimates')
 
# giving a title to the graph
plt.title('Final Policy (action stake) vs State (Capital)')
 
# function to show the plot
plt.show()


# Plotting Capital vs Final Policy

# x axis values
x = range(100)
# corresponding y axis values
y = policy
 
# plotting the bars
plt.bar(x, y, align='center', alpha=0.5)
 
# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Final policy (stake)')
 
# giving a title to the graph
plt.title('Capital vs Final Policy')
 
# function to show the plot
plt.show()