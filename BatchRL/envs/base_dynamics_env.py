"""Dynamics model environment base class.

Use this class if you want to build an environment
based on a model of class `BaseDynamicsModel`.
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, Union, List, Tuple

import gym
import numpy as np

from agents import base_agent
from dynamics.base_model import BaseDynamicsModel
from util.numerics import npf32
from util.util import Arr, create_dir, make_param_ext
from util.visualize import rl_plot_path, plot_env_evaluation, plot_reward_details

Agents = Union[List, base_agent.AgentBase]


class DynEnv(ABC, gym.Env):
    """The environment wrapper class for `BaseDynamicsModel`.

    Takes an instance of `BaseDynamicsModel` and adds
    all the functionality needed to turn it into an environment
    to be used for reinforcement learning.
    """
    # Fix data
    m: BaseDynamicsModel  #: Prediction model.
    act_dim: int  #: The dimension of the action space.
    state_dim: int  #: The dimension of the state space.
    n_ts_per_eps: int  #: The maximum number of timesteps per episode.

    # State data, might change if `step` is called.
    n_ts: int = 0  #: The current number of timesteps.
    hist: np.ndarray  #: 2D array with current state.

    # The current data to sample initial conditions from.
    train_data: np.ndarray  #: The training data.
    train_indices: np.ndarray  #: The indices corresponding to `train_data`.
    n_start_data: int  #: The number of possible initializations using the training data.

    # Info about the current episode.
    use_noise: bool = True  #: Whether to add noise when simulating.

    orig_actions: np.ndarray  #: Array with fallback actions

    # The one day data
    day_ind: int = 0  #: The index of the day in `train_days`.
    n_train_days: int  #: Number of training days
    n_ts_per_day: int  #: Number of time steps in a day
    train_days: List  #: The data of all days, where all data is available.
    day_inds: np.ndarray  #: Index vector storing the timestep offsets to the days

    info: Dict = None  #: A dict with info about the current agent.
    do_scaling: bool = False
    a_scaling_pars: Tuple[np.ndarray, np.ndarray] = None

    def __init__(self, m: BaseDynamicsModel, name: str = None, max_eps: int = None,
                 disturb_fac: float = 1.0):
        """Initialize the environment.

        Args:
            m: Full model predicting all the non-control features.
            max_eps: Number of continuous predictions in an episode.
        """
        m.model_disturbance()
        self.m = m
        if name is not None:
            dist_ex = make_param_ext([("DF", disturb_fac)])
            self.name = name + dist_ex + "_DATA_" + m.name
        else:
            self.name = "RLEnv_" + m.name
        self.plot_path = os.path.join(rl_plot_path, self.name)

        # Set attributes.
        self.disturb_fac = disturb_fac
        self.act_dim = m.data.n_c
        self.state_dim = m.data.d
        self.n_ts_per_eps = 100 if max_eps is None else max_eps

        # Set data and initialize env.
        self._set_data("train")
        self.reset()

    def set_agent(self, a: base_agent.AgentBase):
        self.info = a.get_info()
        scaling = self.info.get('action_scaled_01')
        self.do_scaling = scaling is not None
        if self.do_scaling:
            p1 = np.array([i[0] for i in scaling], dtype=np.float32)
            p2 = np.array([i[1] - i[0] for i in scaling], dtype=np.float32)
            self.a_scaling_pars = p1, p2

    def _set_data(self, part_str: str = "train") -> None:
        """Sets the data to use in the env.

        Args:
            part_str: The string specifying the part of the data.
        """
        self.m.data.check_part(part_str)
        self.train_data, _, self.train_indices = self.m.data.get_split(part_str)
        self.n_start_data = len(self.train_data)

    def get_plt_path(self, name: str) -> str:
        """Specifies the path of the plot with name 'name' where it should be saved.

        If there is not a directory
        for the current model, it is created.

        Args:
            name: Name of the plot.

        Returns:
            Full path of the plot file.
        """
        dir_name = self.plot_path
        create_dir(dir_name)
        return os.path.join(dir_name, name)

    reward_descs: List = []  #: The description of the detailed reward.

    @abstractmethod
    def detailed_reward(self, curr_pred: np.ndarray, action: Arr) -> np.ndarray:
        """Computes the different components of the reward and returns them all.

        Args:
            curr_pred: The current predictions.
            action: The most recent action taken.

        Returns:
            The reward components in an 1d array.
        """
        pass

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:
        """Computes the reward to be maximized during the RL training.

        Args:
            curr_pred: The current predictions.
            action: The most recent action taken.

        Returns:
            Value of the reward of that outcome.
        """
        # The base implementation just sums up the different rewards.
        return np.sum(self.detailed_reward(curr_pred, action)).item()

    @abstractmethod
    def episode_over(self, curr_pred: np.ndarray) -> bool:
        """Defines criterion for episode to be over.

        Args:
            curr_pred: The next predicted state.

        Returns:
            True if episode is over, else false
        """
        return False

    def scale_actions(self, actions):
        """Scales the actions to the correct range.

        Deprecated!!
        """
        return actions

    def scale_action_for_step(self, action: Arr):
        return action

    def step(self, action: Arr) -> Tuple[np.ndarray, float, bool, Dict]:
        """Evolve the model with the given control input `action`.

        This function should not be overridden, instead override
        `scale_action_for_step`!

        Args:
            action: The control input (action).

        Returns:
            The next state, the reward of having chosen that action and a bool
            determining if the episode is over. (And an empty dict)
        """
        # Scale the action
        action = self.scale_action_for_step(action)

        # Add the control to the history
        self.hist[-1, -self.act_dim:] = action

        # Predict using the model
        hist_res = np.copy(self.hist).reshape((1, -1, self.state_dim))
        curr_pred = self.m.predict(hist_res)[0]

        # Add noise
        if self.use_noise:
            curr_pred += self.disturb_fac * self.m.disturb()

        # Save the chosen action
        self.orig_actions[self.n_ts, :] = action

        # Update history
        self.hist[:-1, :] = self.hist[1:, :]
        self.hist[-1, :-self.act_dim] = curr_pred
        self.n_ts += 1

        # Compute reward and episode termination
        r = np.array(self.compute_reward(curr_pred, action)).item()
        ep_over = self.n_ts == self.n_ts_per_eps or self.episode_over(curr_pred)
        return curr_pred, r, ep_over, {}

    def get_curr_state(self):
        """Returns the current state."""
        return self.hist[-1, :-self.act_dim]

    def reset(self, start_ind: int = None, use_noise: bool = True) -> np.ndarray:
        """Resets the environment.

        Needs to be called if the episode is over.

        Returns:
            A new initial state.
        """
        # Reset time step and disturbance
        self.use_noise = use_noise
        self.n_ts = 0
        self.m.reset_disturbance()

        # Reset original actions (necessary?)
        self.orig_actions = np.empty((self.n_ts_per_eps, self.act_dim), dtype=np.float32)

        # Select new start data
        if start_ind is None:
            start_ind = np.random.randint(self.n_start_data)
        else:
            if self.n_start_data <= start_ind:
                raise ValueError("start_ind is too fucking large!")

        self.hist = np.copy(self.train_data[start_ind])
        return np.copy(self.hist[-1, :-self.act_dim])

    def render(self, mode='human'):
        print("Rendering not implemented!")

    def _to_scaled(self, action: Arr, to_original: bool = False) -> np.ndarray:
        """Converts actions to the right range."""
        raise NotImplementedError("Implement this!")

    def analyze_agents_visually(self, agents: Union[List, base_agent.AgentBase],
                                fitted: bool = True,
                                use_noise: bool = False,
                                start_ind: int = None,
                                max_steps: int = None) -> None:
        """Analyzes and compares a set of agents / control strategies.

        Args:
            agents: A list of agents or a single agent.
            fitted: Whether the agents are already fitted.
            use_noise: Whether to use noise in the predictions.
            start_ind: Index of initial configuration, random if None.
            max_steps: The maximum number of steps of an episode.
        """
        # Make function compatible for single agent input
        if not isinstance(agents, list):
            agents = [agents]

        if max_steps is None:
            max_steps = 100000

        # Define arrays to save trajectories
        n_non_c_states = self.state_dim - self.act_dim
        n_agents = len(agents)
        action_sequences = np.empty((n_agents, self.n_ts_per_eps, self.act_dim), dtype=np.float32)
        action_sequences.fill(np.nan)
        clipped_action_sequences = np.empty((n_agents, self.n_ts_per_eps, self.act_dim), dtype=np.float32)
        clipped_action_sequences.fill(np.nan)
        trajectories = np.empty((n_agents, self.n_ts_per_eps, n_non_c_states), dtype=np.float32)
        trajectories.fill(np.nan)
        rewards = np.empty((n_agents, self.n_ts_per_eps), dtype=np.float32)
        rewards.fill(np.nan)

        # Choose same random start for all agents
        if start_ind is None:
            start_ind = np.random.randint(self.n_start_data)
        elif start_ind >= self.n_start_data:
            raise ValueError("start_ind is too large!")

        for a_id, a in enumerate(agents):
            # Check that agent references this environment
            if not a.env == self:
                raise ValueError(f"Agent {a_id} was not assigned to this env!")

            # Set agent
            self.set_agent(a)

            # Fit agent if not already fitted
            if not fitted:
                a.fit()

            # Reset env
            curr_state = self.reset(start_ind=start_ind, use_noise=use_noise)
            episode_over = False
            count = 0

            # Evaluate agent and save states, actions and reward
            while not episode_over and count < max_steps:
                curr_action = a.get_action(curr_state)
                curr_state, rew, episode_over, extra = self.step(curr_action)
                action_sequences[a_id, count, :] = curr_action

                trajectories[a_id, count, :] = np.copy(curr_state)
                rewards[a_id, count] = rew
                count += 1

            # Get original actions
            clipped_action_sequences[a_id, :self.n_ts, :] = self.orig_actions[:self.n_ts, :]

        # Scale the data to the right values
        trajectories = self.m.rescale_output(trajectories, out_put=True)
        s_ac = clipped_action_sequences.shape
        for k in range(s_ac[0]):
            for i in range(s_ac[1]):
                clipped_action_sequences[k, i] = self._to_scaled(clipped_action_sequences[k, i], to_original=True)
        if np.allclose(clipped_action_sequences, action_sequences):
            clipped_action_sequences = None

        # Plot all the things
        name_list = [a.get_short_name() for a in agents]
        analysis_plot_path = self._construct_plot_name("AgentAnalysis", start_ind, agents)
        plot_env_evaluation(action_sequences, trajectories, rewards, self.m.data,
                            name_list, analysis_plot_path, clipped_action_sequences)

    def _construct_plot_name(self, base_name: str, start_ind: int, agent_list: List):
        name_list = [a.get_short_name() for a in agent_list]
        agent_names = '_'.join(name_list)
        return self.get_plt_path(base_name + "_" + str(start_ind) + "_" + agent_names)

    def eval_agents(self, agent_list: Agents, n_steps: int = 100) -> np.ndarray:

        # Make function compatible for single agent input.
        if not isinstance(agent_list, list):
            agent_list = [agent_list]

        # Init scores.
        n_agents = len(agent_list)
        scores = np.empty((n_agents,), dtype=np.float32)

        for a_id, a in enumerate(agent_list):
            # Check that agent references this environment
            if not a.env == self:
                raise ValueError(f"Agent {a_id} was not assigned to this env!")

            # Evaluate agent.
            scores[a_id] = a.eval(n_steps, reset_seed=True)

        return scores

    def detailed_eval_agents(self, agent_list: Agents, n_steps: int = 100, use_noise: bool = False) -> np.ndarray:
        """Evaluates the given agents for this environment.

        Plots the mean rewards and returns all rewards.
        """
        # Make function compatible for single agent input.
        if not isinstance(agent_list, list):
            agent_list = [agent_list]

        # Init scores.
        n_agents = len(agent_list)
        n_extra_rewards = len(self.reward_descs)
        n_tot_rewards = n_extra_rewards + 1
        all_rewards = npf32((n_agents, n_steps, n_tot_rewards))

        for a_id, a in enumerate(agent_list):

            # Set agent
            self.set_agent(a)

            # Check that agent references this environment
            if not a.env == self:
                raise ValueError(f"Agent {a_id} was not assigned to this env!")

            # Evaluate agent.
            rew, ex_rew = a.eval(n_steps, reset_seed=True, detailed=True, use_noise=use_noise)
            all_rewards[a_id, :, 0] = rew
            all_rewards[a_id, :, 1:] = ex_rew

        # Plot
        p_name = self._construct_plot_name("DetailAnalysis", n_steps, agent_list)
        plot_reward_details(agent_list, all_rewards, p_name,
                            self.reward_descs,
                            title=f"Mean rewards for {n_steps} steps (Timestep: {self.m.data.dt} min)")
        return all_rewards
