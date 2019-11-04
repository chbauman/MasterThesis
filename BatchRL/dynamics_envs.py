from base_dynamics_env import DynEnv
from base_dynamics_model import BaseDynamicsModel
from util import *


class FullRoomEnv(DynEnv):
    alpha: float = 1.0  #: Weight factor for reward.
    temp_bounds: Sequence = (22.0, 26.0)  #: The requested temperature range.
    bound_violation_penalty: float = 2.0  #: The penalty in the reward for temperatures out of bound.
    scaling: np.ndarray = None

    def __init__(self, m: BaseDynamicsModel, temp_bounds: Sequence = None,
                 n_disc_actions: int = 11,
                 **kwargs):
        max_eps = 2 * 24 * 60 // m.data.dt // 1  # max predictions length
        super(FullRoomEnv, self).__init__(m, "FullRoom", max_eps, **kwargs)
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
        """Converts discrete actions to the right range."""
        zero_one_action = check_and_scale(action, self.nb_actions, [0.0, 1.0])
        if self.scaling is None:
            return zero_one_action
        return rem_mean_and_std(zero_one_action, self.scaling[self.c_ind])

    def scale_actions(self, actions):
        """Scales the actions to the correct range."""
        zero_one_action = check_and_scale(actions, self.nb_actions, [0.0, 1.0])
        return zero_one_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:
        print(f"Full room step, action: {action}")
        return super(FullRoomEnv, self).step(self._to_continuous(action))

    def get_r_temp(self, curr_pred: np.ndarray) -> float:
        r_temp = curr_pred[4]
        if self.scaling is not None:
            r_temp = add_mean_and_std(r_temp, self.scaling[5])
        return r_temp

    def get_w_temp(self, curr_pred: np.ndarray):
        w_inds = 2, 3
        w = [curr_pred[i] for i in w_inds]
        if self.scaling is not None:
            for k in range(2):
                w[k] = add_mean_and_std(w[k], self.scaling[w_inds[k]])
        return w

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:

        # Compute energy used
        action_rescaled = action
        if self.scaling is not None:
            action_rescaled = add_mean_and_std(action, self.scaling[self.c_ind])
        w_temps = self.get_w_temp(curr_pred)
        d_temp = np.abs(w_temps[0] - w_temps[1])
        energy_used = action_rescaled * d_temp * self.alpha
        assert 0.0 <= action_rescaled <= 1.0, "Fucking wrong"
        assert 10.0 <= w_temps[0] <= 50.0, "Water temperature scaled incorrectly!"

        # Penalty for constraint violation
        r_temp = self.get_r_temp(curr_pred)
        t_bounds = self.temp_bounds
        too_low_penalty = 0.0 if r_temp >= t_bounds[0] else t_bounds[0] - r_temp
        too_high_penalty = 0.0 if r_temp <= t_bounds[1] else r_temp - t_bounds[1]
        bound_pen = self.bound_violation_penalty * (too_low_penalty + too_high_penalty)
        return -energy_used - bound_pen

    def episode_over(self, curr_pred: np.ndarray) -> bool:

        r_temp = self.get_r_temp(curr_pred)
        thresh = 5.0
        t_bounds = self.temp_bounds
        if r_temp > t_bounds[1] + thresh or r_temp < t_bounds[0] - thresh:
            print("Diverging...")
            return True
        return False


class BatteryEnv(DynEnv):
    """The environment for the battery model.


    """

    alpha: float = 1.0  #: Weight factor for reward.
    action_range: Sequence = (-100, 100)  #: The requested active power range.
    soc_bound: Sequence = (20, 80)  #: The requested state-of-charge range.
    scaling: np.ndarray = None

    def __init__(self, m: BaseDynamicsModel,
                 n_disc_actions: int = 11,
                 **kwargs):
        max_eps = 24 * 60 // m.data.dt // 2  # max predictions length
        super().__init__(m, "Battery", max_eps, **kwargs)
        d = m.data
        self.nb_actions = n_disc_actions
        self.c_ind = d.c_inds[0]
        if np.all(d.is_scaled):
            self.scaling = d.scaling

        # Check model
        assert len(m.out_inds) == 1, "Model not suited for this environment!!"

        # Check underlying dataset
        assert d.d == 2 and d.n_c == 1, "Not the correct number of series in dataset!"
        assert d.c_inds[0] == 1, "Second series needs to be controllable!"

    def _to_continuous(self, action):
        """Converts discrete actions to the right range."""
        cont_action = check_and_scale(action, self.nb_actions, self.action_range)
        if self.scaling is None:
            return cont_action
        return rem_mean_and_std(cont_action, self.scaling[self.c_ind])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:
        return super().step(self._to_continuous(action))

    def scale_actions(self, actions):
        """Scales the actions to the correct range."""
        zero_one_action = check_and_scale(actions, self.nb_actions, self.action_range)
        return zero_one_action

    def _get_scaled_soc(self, unscaled_soc):
        if self.scaling is not None:
            return add_mean_and_std(unscaled_soc, self.scaling[0])
        return unscaled_soc

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:

        # Compute energy used
        action_rescaled = action
        if self.scaling is not None:
            action_rescaled = add_mean_and_std(action, self.scaling[self.c_ind])
        curr_pred = self._get_scaled_soc(curr_pred)

        energy_used = action_rescaled * self.alpha
        bound_pen = linear_oob_penalty(curr_pred, self.soc_bound)
        if self.n_ts > self.n_ts_per_eps - 2:
            bound_pen += 10.0

        # Penalty for constraint violation
        return -energy_used - bound_pen

    def episode_over(self, curr_pred: np.ndarray) -> bool:
        thresh = 10
        b = self.soc_bound
        scaled_soc = self._get_scaled_soc(curr_pred)
        if scaled_soc > b[1] + thresh or scaled_soc < b[0] - thresh:
            print("Diverging...")
            return True
        return False
