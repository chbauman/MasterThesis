from base_dynamics_env import DynEnv
from base_dynamics_model import BaseDynamicsModel
from data import Dataset
from util import *


class FullRoomEnv(DynEnv):

    alpha: float = 1.0  #: Weight factor for reward.
    temp_bounds: Sequence = (22.0, 26.0)  #: The requested temperature range.
    bound_violation_penalty: float = 10.0  #: The penalty in the reward for temperatures out of bound.
    scaling: np.ndarray = None

    def __init__(self, m: BaseDynamicsModel, temp_bounds: Sequence = None, n_disc_actions: int = 11):
        max_eps = 24 * 60 // m.data.dt
        super(FullRoomEnv, self).__init__(m, max_eps)
        d = m.data
        if temp_bounds is not None:
            self.temp_bounds = temp_bounds
        self.nb_actions = n_disc_actions
        self.c_ind = d.c_inds[0]
        if np.all(d.is_scaled):
            self.scaling = d.scaling

        # Check model
        assert len(m.out_inds) == d.d - d.n_c, "Model not suited for this environment!!"

        # Check underlying dataset
        assert d.d == 8 and d.n_c == 1, "Not the correct number of series in dataset!"

    def _to_continuous(self, action):
        zero_one_action = action / (self.nb_actions - 1)
        if self.scaling is None:
            return zero_one_action
        return rem_mean_and_std(zero_one_action, self.scaling[self.c_ind])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:

        return super(FullRoomEnv, self).step(self._to_continuous(action))

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:

        # Compute energy used
        action_rescaled = action
        if self.scaling is not None:
            action_rescaled = add_mean_and_std(action, self.scaling[self.c_ind])
        d_temp = np.abs(curr_pred[2] - curr_pred[3])
        energy_used = action_rescaled * d_temp * self.alpha
        assert 0.0 <= action_rescaled <= 1.0, "Fucking wrong"

        # Penalty for constraint violation
        r_temp = curr_pred[4]
        if self.scaling is not None:
            r_temp = add_mean_and_std(r_temp, self.scaling[5])
        t_bounds = self.temp_bounds
        too_low_penalty = 0.0 if r_temp > t_bounds[0] else t_bounds[0] - r_temp
        too_high_penalty = 0.0 if r_temp < t_bounds[1] else r_temp - t_bounds[1]
        bound_pen = self.bound_violation_penalty * (too_low_penalty + too_high_penalty)
        return -energy_used - bound_pen

    def episode_over(self, curr_pred: np.ndarray) -> bool:

        # TODO: Return true if constraints are not met.

        return False
