"""Dynamics model environment base class.

Use this class if you want to build an environment
based on a model of class `BaseDynamicsModel`.
"""

from abc import ABC, abstractmethod

import gym

import agents_heuristic
import base_agent
from base_dynamics_model import BaseDynamicsModel, TestModel, construct_test_ds
from util import *
from visualize import rl_plot_path, plot_env_evaluation


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

    train_data: np.ndarray  #: The training data.
    train_indices: np.ndarray  #: The indices corresponding to `train_data`.

    day_ind: int = 0  #: The index of the day in `train_days`.
    use_noise: bool = True  #: Whether to add noise when simulating.

    # The one day data
    n_train_days: int  #: Number of training days
    n_ts_per_day: int  #: Number of time steps in a day
    train_days: List  #: The data of all days, where all data is available.
    day_inds: np.ndarray  #: Index vector storing the timestep offsets to the days

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
            self.name = name
        else:
            self.name = "RLEnv_" + m.name
        self.plot_path = os.path.join(rl_plot_path, self.name)
        create_dir(self.plot_path)

        self.disturb_fac = disturb_fac
        self.act_dim = m.data.n_c
        self.state_dim = m.data.d
        self.n_ts_per_eps = 100 if max_eps is None else max_eps
        self.train_data, _, self.train_indices = m.data.get_split("train")
        self.n_start_data = len(self.train_data)
        self.reset()

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
        return os.path.join(dir_name, name)

    @abstractmethod
    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:
        """Computes the reward to be maximized during the RL training.

        Args:
            curr_pred: The current predictions.
            action: The most recent action taken.

        Returns:
            Value of the reward of that outcome.
        """
        pass

    @abstractmethod
    def episode_over(self, curr_pred: np.ndarray) -> bool:
        """Defines criterion for episode to be over.

        Args:
            curr_pred: The next predicted state.

        Returns:
            True if episode is over, else false
        """
        return False

    def step(self, action: Arr) -> Tuple[np.ndarray, float, bool, Any]:
        """Evolve the model with the given control input `action`.

        Args:
            action: The control input (action).

        Returns:
            The reward of having chosen that action and a bool
            determining if the episode is over.
        """
        self.hist[-1, -self.act_dim:] = action
        pred_sh = (1, -1, self.state_dim)
        curr_pred = self.m.predict(self.hist.reshape(pred_sh))[0]
        if self.use_noise:
            curr_pred += self.disturb_fac * self.m.disturb()
        self.hist[:-1, :] = self.hist[1:, :]
        self.hist[-1, :-self.act_dim] = curr_pred
        self.n_ts += 1

        r = self.compute_reward(curr_pred, action)
        ep_over = self.n_ts == self.n_ts_per_eps or self.episode_over(curr_pred)
        return curr_pred, r, ep_over, {}

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

        # Select new start data
        if start_ind is None:
            start_ind = np.random.randint(self.n_start_data)
        else:
            if self.n_start_data <= start_ind:
                raise ValueError("start_ind is too fucking large!")

        self.hist = self.train_data[start_ind]
        return self.hist[-1, :-self.act_dim]

    def render(self, mode='human'):
        print("Rendering not implemented!")

    def analyze_agent(self, agents: Union[List, base_agent.AgentBase],
                      fitted: bool = True,
                      use_noise: bool = False) -> None:
        """Analyzes and compares a set of agents / control strategies.

        Args:
            agents: A list of agents or a single agent.
            fitted: Whether the agents are already fitted.
            use_noise: Whether to use noise in the predictions.
        """
        # Make function compatible for single agent input
        if not isinstance(agents, list):
            agents = [agents]

        # Define arrays to save trajectories
        n_non_c_states = self.state_dim - self.act_dim
        n_agents = len(agents)
        action_sequences = np.empty((n_agents, self.n_ts_per_eps, self.act_dim), dtype=np.float32)
        action_sequences.fill(np.nan)
        trajectories = np.empty((n_agents, self.n_ts_per_eps, n_non_c_states), dtype=np.float32)
        trajectories.fill(np.nan)
        rewards = np.empty((n_agents, self.n_ts_per_eps), dtype=np.float32)
        rewards.fill(np.nan)

        # Choose same random start for all agents
        start_ind = np.random.randint(self.n_start_data)

        for a_id, a in enumerate(agents):
            # Check that agent references this environment
            if not a.env == self:
                raise ValueError(f"Agent {a_id} was not assigned this env!")

            # Fit agent if not already fitted
            if not fitted:
                a.fit()

            # Reset env
            curr_state = self.reset(start_ind=start_ind, use_noise=use_noise)
            episode_over = False
            count = 0

            # Evaluate agent and save states, actions and reward
            while not episode_over:
                curr_action = a.get_action(curr_state)
                curr_state, rew, episode_over, _ = self.step(curr_action)
                action_sequences[a_id, count, :] = curr_action
                trajectories[a_id, count, :] = curr_state
                rewards[a_id, count] = rew
                count += 1

        # Scale the data to the right values
        trajectories = self.m.rescale_output(trajectories, out_put=True)

        # Plot all the things
        analysis_plot_path = self.get_plt_path("AgentAnalysis")
        plot_env_evaluation(action_sequences, trajectories, rewards,
                            analysis_plot_path)


##########################################################################
# Testing stuff

class TestDynEnv(DynEnv):

    def __init__(self, m: BaseDynamicsModel, max_eps: int = None):
        super(TestDynEnv, self).__init__(m, "TestEnv", max_eps)
        d = m.data
        assert d.n_c == 1 and d.d == 4, "Dataset needs 4 series of which one is controllable!!"

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:
        assert curr_pred.shape == (3,), "Shape of prediction not correct!"
        return curr_pred[2] * action

    def episode_over(self, curr_pred: np.ndarray) -> bool:
        return False


# @TestDecoratorFactory("ModelEnvironment")
def test_test_env():
    n = 201
    test_ds = construct_test_ds(n)
    test_mod = TestModel(test_ds)
    n_ts_per_episode = 10
    test_env = TestDynEnv(test_mod, 10)

    for k in range(30):
        next_state, r, over, _ = test_env.step(0.0)
        assert next_state.shape == (3,), "Prediction does not have the right shape!"
        if (k + 1) % n_ts_per_episode == 0 and not over:
            raise AssertionError("Episode should be over!!")
        if over:
            init_state = test_env.reset()
            assert init_state.shape == (3,), "Prediction does not have the right shape!"

    const_ag_1 = agents_heuristic.ConstHeating(test_env, 0.0)
    const_ag_2 = agents_heuristic.ConstHeating(test_env, 1.0)
    test_env.analyze_agent([const_ag_1, const_ag_2])
    print("Model environment test passed :)")
