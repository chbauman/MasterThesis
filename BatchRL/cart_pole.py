import gym
from gym import spaces

import numpy as np

class CartPole:


    def __init__(self):

        self.env = gym.make('CartPole-v0')
        self.env.reset()

        a_s = self.env.action_space
        o_s = self.env.observation_space

        self.state_dim = o_s.shape[0]
        self.nb_actions = a_s.n
        print(o_s, "State dim")
        pass

    def get_transition_tuples(self, n_tuples = 200000):
        """
        Creates n_tuples tuples of the form (s_t, a_t, r_t, s_tp1)
        for training of batch RL algorithm.
        """
                
        # Initialize containers
        s_t = np.empty((n_tuples, self.state_dim), dtype = np.float32)
        s_tp1 = np.empty((n_tuples, self.state_dim), dtype = np.float32)
        a_t = np.empty((n_tuples), dtype = np.int)
        r_t = np.empty((n_tuples), dtype = np.float32)

        ct = 0
        while ct < n_tuples:
            observation = self.env.reset()
            last_obs = observation
            for t in range(100):
                #self.env.render()
                #print(observation)
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
                
                if ct == n_tuples:
                    break

                # Save tuple
                r_t[ct] = reward
                a_t[ct] = action
                s_tp1[ct,:] = observation
                s_t[ct,:] = last_obs

                last_obs = observation
                ct += 1


        self.env.close()
        return [s_t, a_t, r_t, s_tp1]

    def eval_policy(self, policy):
        """
        Evaluate policy by applying it and observing the system.
        """
        observation = self.env.reset()
        for i in range(500):

            self.env.render()
            action = policy(observation)
            observation, reward, done, info = self.env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(i+1))
                #break
