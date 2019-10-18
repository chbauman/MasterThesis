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
    m: BaseDynamicsModel  #: Prediction model.
    hist: np.ndarray  #: 2D array with current state.
    act_dim: int  #: The dimension of the action space.
    state_dim: int  #: The dimension of the state space.
    n_ts: int = 0  #: The current number of timesteps.
    n_ts_per_eps: int  #: The number of timesteps per episode.

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
        self.hist[-1, self.act_dim:] = curr_pred
        self.n_ts += 1

        r = self.compute_reward(curr_pred, action)
        ep_over = self.n_ts == self.n_ts_per_eps
        return r, ep_over

    def reset(self) -> None:
        """Resets the environment.

        Needs to be called if the episode is over.
        TODO: Reset noise for the noisy predictions.
        """
        self.n_ts = 0
