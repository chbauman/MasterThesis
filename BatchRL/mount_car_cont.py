
import gym
from gym import spaces

import random

import numpy as np

class MountCarCont:
    def __init__(self):

        self.env = gym.make('MountainCarContinuous-v0')
        self.env.reset()

        a_s = self.env.action_space
        o_s = self.env.observation_space

        self.state_dim = o_s.shape[0]
        self.nb_actions = a_s.shape[0]
        pass

    def get_transition_tuples(self, n_tuples = 200000):
        """
        Creates n_tuples tuples of the form (s_t, a_t, r_t, s_tp1)
        for training of batch RL algorithm.
        """
                
        # Initialize containers
        s_t = np.empty((n_tuples, self.state_dim), dtype = np.float32)
        s_tp1 = np.empty((n_tuples, self.state_dim), dtype = np.float32)
        a_t = np.empty((n_tuples), dtype = np.float32)
        r_t = np.empty((n_tuples), dtype = np.float32)

        ct = 0
        while ct < n_tuples:
            observation = self.env.reset()
            last_obs = observation
            curr_action = self.env.action_space.sample()
            for t in range(1000000):
                self.env.render()
                if random.randint(0, 1):
                    action = self.env.action_space.sample()
                else:
                    action = curr_action
                curr_action = action
                observation, reward, done, info = self.env.step(action)
                
                if ct == n_tuples:
                    break

                # Save tuple
                # Use a smooth loss here
                r_t[ct] = reward + (observation[0] - 0.5) ** 4
                a_t[ct] = action
                s_tp1[ct,:] = observation
                s_t[ct,:] = last_obs

                last_obs = observation
                ct += 1

                if done:
                    print("Yay", r_t[ct - 1], "using", t, "steps")
                    #r_t[ct - 1] = -10
                    #print("Episode finished after {} timesteps".format(t+1))
                    break


        self.env.close()
        return [s_t, a_t, r_t, s_tp1]

    def eval_policy(self, policy):
        """
        Evaluate policy by applying it and observing the system.
        """
        tot_rew = 0
        num_trials = 100
        for k in range(num_trials):
            observation = self.env.reset()            
            for i in range(200):
                if k < 10:
                    self.env.render()                
                action = policy(observation)
                observation, reward, done, info = self.env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(i+1))
                    tot_rew += i + 1
                    break
        avg_rev = tot_rew / num_trials
        print("Average reward over", num_trials, "trials is", avg_rev)