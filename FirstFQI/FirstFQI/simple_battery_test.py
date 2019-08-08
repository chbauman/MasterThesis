
# Simple RL test example as a first try to use FQI
# Fixed each day but varying during day electricity prices, goal is 
# to charge the ev, s.t. it is full in the evening.

# General parameter
import numpy as np
import random


class SimpleBatteryTest:
    def __init__(self):


        self.state_dim = 2
        self.nb_actions = 2

        # Specific parameters
        self.t_max = 9
        self.max_charge = 3
        self.p_d = 1
        self.p_n = 2
        self.p_e = 3
        self.pen = 10

        # Price vector
        self.el_prices = np.ones(9) * self.p_d
        self.el_prices[3:4] = self.p_n
        self.el_prices[8] = self.p_e

        # Initial state: time = 0, b = 0 
        self.init_state = np.zeros(2, dtype = np.int)

    def state_transition(self, s_t, a_t):
        """
        State transition function, returns the next state when 
        choosing action a_t in state s_t.
        """
        t = s_t[0]
        if t == self.t_max:
            return self.init_state
        s_tp1 = s_t
        s_tp1[0] += 1
        s_tp1[1] += a_t
        return s_tp1

    def reward_function(self, s_t, a_t):
        """
        Reward function.
        """
        t = s_t[0]
        if t < self.t_max:
            return -self.el_prices[t] * a_t
        else:
            # End of day
            return self.max_charge - s_t[1]

    def random_action_policy(self, s_t):
        """
        Policy that always chooses a random action.
        """
        b_t = s_t[1]
        if b_t < self.max_charge:
            return random.randint(0,1)
        else:
            return 0
        return 0

    def get_transition_tuples(self, n_tuples = 10000):
        """
        Creates n_tuples tuples of the form (s_t, a_t, r_t, s_tp1)
        for training of FQI.
        """

        # Initialize containers
        s_t = np.empty((n_tuples, self.state_dim), dtype = np.int)
        s_tp1 = np.empty((n_tuples, self.state_dim), dtype = np.int)
        a_t = np.empty((n_tuples), dtype = np.int)
        r_t = np.empty((n_tuples), dtype = np.int)

        curr_state = self.init_state
        for k in range(n_tuples):
        
            # Apply policy
            action = self.random_action_policy(curr_state)
            reward = self.reward_function(curr_state, action)
            next_state = self.state_transition(curr_state, action)

            # Save data
            s_t[k,:] = curr_state
            s_tp1[k,:] = next_state
            a_t[k] = action
            r_t[k] = reward
        
            # Update state
            curr_state = next_state

        return [s_t, a_t, r_t, s_tp1]


    def eval_policy(policy):
        """
        Evaluates a given policy.
        """

        tot_rew = 0

        curr_state = self.init_state
        for k in range(self.t_max):
            # select action
            a_t = policy(curr_state)
            tot_rew += self.reward_function(curr_state, a_t)
            curr_state = self.state_transition(curr_state, a_t)

        return tot_rew
