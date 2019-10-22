from base_dynamics_env import DynEnv
from base_dynamics_model import BaseDynamicsModel
from data import Dataset
from util import *


class FullRoomEnv(DynEnv):

    alpha: float = 1.0  #: Weight factor for reward.
    temp_bounds: Sequence = (22.0, 26.0)  #: The requested temperature range.
    bound_violation_penalty: float = 10.0  #: The penalty in the reward for temperatures out of bound.
    control_mas: np.ndarray = None

    def __init__(self, m: BaseDynamicsModel, temp_bounds: Sequence = None, n_disc_actions: int = 11):
        max_eps = 24 * 60 // m.data.dt
        super(FullRoomEnv, self).__init__(m, max_eps)
        d = m.data
        if temp_bounds is not None:
            self.temp_bounds = temp_bounds
        self.nb_actions = n_disc_actions
        c_ind = d.c_inds[0]
        if d.is_scaled[c_ind]:
            self.control_mas = d.scaling[c_ind]

        # Check model
        assert len(m.out_inds) == d.d - d.n_c, "Model not suited for this environment!!"

        # Check underlying dataset
        assert d.d == 8 and d.n_c == 1, "Not the correct number of series in dataset!"

    def _to_continuous(self, action):
        zero_one_action = action / (self.nb_actions - 1)
        if self.control_mas is None:
            return zero_one_action
        return rem_mean_and_std(zero_one_action, self.control_mas)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:

        return super(FullRoomEnv, self).step(self._to_continuous(action))

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:

        # Compute energy used
        d_temp = np.abs(curr_pred[2] - curr_pred[3])
        energy_used = self._to_continuous(action) * d_temp * self.alpha

        # Penalty for constraint violation
        r_temp = curr_pred[4]
        t_bounds = self.temp_bounds
        too_low_penalty = 0.0 if r_temp > t_bounds[0] else t_bounds[0] - r_temp
        too_high_penalty = 0.0 if r_temp < t_bounds[1] else r_temp - t_bounds[1]
        bound_pen = self.bound_violation_penalty * (too_low_penalty + too_high_penalty)
        return -energy_used - bound_pen

    def episode_over(self, curr_pred: np.ndarray) -> bool:

        # TODO: Return true if constraints are not met.

        return False
