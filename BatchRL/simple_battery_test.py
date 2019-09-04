
# Simple RL test example as a first try to use FQI
# Fixed each day but varying during day electricity prices, goal is 
# to charge the ev, s.t. it is full in the evening.

# General parameter
import numpy as np
import random


class SimpleBatteryTest:
    def __init__(self, bidirectional = False):

        self.bidir = bidirectional
        self.state_dim = 2
        self.nb_actions = 2 + bidirectional

        # Specific parameters
        self.t_max = 9
        self.max_charge = 3
        self.p_d = 1
        self.p_n = 2
        self.p_e = 3
        self.pen = 4

        # Price vector
        self.el_prices = np.ones(9) * self.p_d
        self.el_prices[3:5] = self.p_n
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
            return np.copy(self.init_state)
        
        s_tp1 = s_t
        s_tp1[0] += 1
        if a_t == 1 and s_t[1] < self.max_charge:
            s_tp1[1] += a_t
        if self.bidir and a_t == 2 and s_t[1] > 0:
            s_tp1[1] -= 1
        return s_tp1

    def reward_function(self, s_t, a_t):
        """
        Reward function.
        """
        t = s_t[0]
        if t < self.t_max:
            if a_t == 1 and s_t[1] < self.max_charge:
                return -self.el_prices[t]
            if self.bidir and a_t == 2 and s_t[1] > 0:
                return self.el_prices[t]
            return 0
        else:
            # End of day
            return self.pen * s_t[1]

    def random_action_policy(self, s_t):
        """
        Policy that always chooses a random action.
        """
        b_t = s_t[1]
        if b_t < self.max_charge:
            if self.bidir and b_t > 0:
                return random.randint(0, 2)
            else:
                return random.randint(0, 1)
        elif self.bidir:
            return 2 * random.randint(0, 1)
        return 0

    def get_transition_tuples(self, n_tuples = 200000):
        """
        Creates n_tuples tuples of the form (s_t, a_t, r_t, s_tp1)
        for training of FQI.
        """

        # Initialize containers
        s_t = np.empty((n_tuples, self.state_dim), dtype = np.int)
        s_tp1 = np.empty((n_tuples, self.state_dim), dtype = np.int)
        a_t = np.empty((n_tuples), dtype = np.int)
        r_t = np.empty((n_tuples), dtype = np.int)

        curr_state = np.copy(self.init_state)
        for k in range(n_tuples):
        
            # Apply policy
            action = self.random_action_policy(curr_state)
            reward = self.reward_function(curr_state, action)
            next_state = self.state_transition(np.copy(curr_state), action)

            # Save data
            s_t[k,:] = np.copy(curr_state)
            s_tp1[k,:] = np.copy(next_state)
            a_t[k] = action
            r_t[k] = reward
        
            # Update state
            curr_state = next_state

        return [s_t, a_t, r_t, s_tp1]


    def eval_policy(self, policy):
        """
        Evaluates a given policy.
        """

        tot_rew = 0

        curr_state = np.copy(self.init_state)
        for k in range(self.t_max + 1):
            # select action            
            a_t = policy(curr_state)
            #if curr_state[1] == self.max_charge:
            #    a_t = 0
            r_t = self.reward_function(curr_state, a_t)
            tot_rew += r_t
            print("State:", k, curr_state, ", action:", a_t, ", reward:", int(r_t))
            curr_state = self.state_transition(curr_state, a_t)

        print("Reward should be:", (self.pen - self.p_d) * self.max_charge)
        print("Reward is:", int(tot_rew))
        return tot_rew