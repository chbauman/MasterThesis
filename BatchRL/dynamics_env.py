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
    m: BaseDynamicsModel  #: Prediction model.
    hist: np.ndarray  #: 2D array with current state.
    act_dim: int  #: The dimension of the action space.
    state_dim: int  #: The dimension of the state space.
    n_ts: int = 0  #: The current number of timesteps.
    n_ts_per_eps: int  #: The number of timesteps per episode.

    def __init__(self, m: BaseDynamicsModel, max_eps: int = None):
        self.m = m
        self.act_dim = m.data.n_c
        self.state_dim = m.data.d
        self.n_ts_per_eps = 100 if max_eps is None else max_eps

    @abstractmethod
    def compute_reward(self, curr_pred: np.ndarray, action: np.ndarray) -> float:
        pass

    def step(self, action) -> Tuple[float, bool]:
        self.hist[-1, -self.act_dim:] = action
        curr_pred = self.m.predict(self.hist)
        curr_pred += self.m.disturb()
        self.hist[:-1, :] = self.hist[1:, :]
        self.hist[-1, self.act_dim:] = curr_pred
        self.n_ts += 1

        r = self.compute_reward(curr_pred, action)
        ep_over = self.n_ts == self.n_ts_per_eps
        return r, ep_over

    def reset(self):
        self.n_ts = 0
