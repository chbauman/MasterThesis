"""Dynamics model environment base class.

Use this class if you want to build an environment
based on a model of class `BaseDynamicsModel`.

Todo:
    * Reset with random day sampling.
"""

from abc import ABC, abstractmethod

from base_dynamics_model import BaseDynamicsModel
from util import *


class DynEnv(ABC):
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

    # The one day data
    n_train_days: int  #: Number of training days
    n_ts_per_day: int  #: Number of time steps in a day
    train_days: List  #: The data of all days, where all data is available.
    day_inds: np.ndarray  #: Index vector storing the timestep offsets to the days

    def __init__(self, m: BaseDynamicsModel, max_eps: int = None):
        """Initialize the environment.

        Args:
            m: Full model predicting all the non-control features.
            max_eps: Number of continuous predictions in an episode.
        """
        self.m = m
        self.act_dim = m.data.n_c
        self.state_dim = m.data.d
        self.n_ts_per_eps = 100 if max_eps is None else max_eps
        self.hist = np.empty((m.data.seq_len, self.state_dim), dtype=np.float32)
        self.train_data, _, self.train_indices = m.data.get_split("train")
        self.n_start_data = len(self.train_data)

    @abstractmethod
    def compute_reward(self, curr_pred: np.ndarray, action: np.ndarray) -> float:
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
    
    def step(self, action) -> Tuple[float, bool]:
        """Evolve the model with the given control input `action`.

        Args:
            action: The control input (action).

        Returns:
            The reward of having chosen that action and a bool
            determining if the episode is over.
        """
        self.hist[-1, -self.act_dim:] = action
        curr_pred = self.m.predict(self.hist)
        curr_pred += self.m.disturb()
        self.hist[:-1, :] = self.hist[1:, :]
        self.hist[-1, :-self.act_dim] = curr_pred
        self.n_ts += 1

        r = self.compute_reward(curr_pred, action)
        ep_over = self.n_ts == self.n_ts_per_eps or self.episode_over(curr_pred)
        return r, ep_over

    def reset(self) -> None:
        """Resets the environment.

        Needs to be called if the episode is over.
        TODO: Reset noise for the noisy predictions.
        TODO: Sample new day.
        """
        self.n_ts = 0
