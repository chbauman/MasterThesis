
# Simple RL test example as a first try to use FQI
# Fixed each day but varying during day electricity prices, goal is 
# to charge the ev, s.t. it is full in the evening.

# General parameter
import numpy as np
import random

state_dim = 2
nb_actions = 2

# Specific parameters
t_max = 9
mac_charge = 3
p_d = 1
p_n = 2
p_e = 3
pen = 10
el_prices = np.ones(9) * p_d
el_prices[3:4] = p_n
el_prices[8] = p_e
# Initial state: time = 0, b = 0 
init_state = np.zeros(2, dtype = np.int)

def state_transition(s_t, a_t):
    """
    State transition function, returns the next state when 
    choosing action a_t in state s_t.
    """
    t = s_t[0]
    if t == t_max:
        return init_state
    s_tp1 = s_t
    s_tp1[0] += 1
    s_tp1[1] += a_t
    return s_tp1

def reward_function(s_t, a_t):
    """
    Reward function.
    """
    t = s_t[0]
    if t < t_max:
        return -el_prices[t] * a_t
    else:
        # End of day
        return 3 - s_t[1]

def random_action_policy(s_t):
    """
    Policy that always chooses a random action.
    """
    b_t = s_t[1]
    if b_t < mac_charge:
        return random.randint(0,1)
    else:
        return 0
    return 0

def get_transition_tuples(policy = random_action_policy, n_tuples = 10000):
    """
    Creates n_tuples tuples of the form (s_t, a_t, r_t, s_tp1)
    for training of FQI.
    """

    # Initialize containers
    s_t = np.empty((n_tuples, state_dim), dtype = np.int)
    s_tp1 = np.empty((n_tuples, state_dim), dtype = np.int)
    a_t = np.empty((n_tuples), dtype = np.int)
    r_t = np.empty((n_tuples), dtype = np.int)

    curr_state = init_state
    for k in range(n_tuples):
        
        # Apply policy
        action = policy(curr_state)
        reward = reward_function(curr_state, action)
        next_state = state_transition(curr_state, action)

        # Save data
        s_t[k,:] = curr_state
        s_tp1[k,:] = next_state
        a_t[k] = action
        r_t[k] = reward
        
        # Update state
        curr_state = next_state

    return [s_t, a_t, r_t, s_tp1]


