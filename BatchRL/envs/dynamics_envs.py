from abc import ABC, abstractmethod
from typing import Sequence, Tuple, List

import numpy as np

from dynamics.base_model import BaseDynamicsModel
from dynamics.battery_model import BatteryModel
from envs.base_dynamics_env import DynEnv
from util.numerics import trf_mean_and_std, add_mean_and_std, rem_mean_and_std
from util.util import make_param_ext, Arr, linear_oob_penalty, LOrEl, Num, to_list

RangeT = Tuple[Num, Num]
InRangeT = LOrEl[RangeT]  #: The type of action ranges.
RangeListT = List[RangeT]


class RLDynEnv(DynEnv, ABC):
    """The base class for RL environments based on `BaseDynamicsModel`.

    Only working for one control input.
    """
    action_range: RangeListT  #: The range of the actions.
    action_range_scaled: np.ndarray  #: The range scaled to the whitened actions.
    scaling: np.ndarray = None  #: Whether the underlying `Dataset` was scaled.
    nb_actions: int  #: Number of actions if discrete else action space dim.

    def __init__(self, m: BaseDynamicsModel,
                 max_eps: int,
                 action_range: Sequence = (0, 1),
                 cont_actions: bool = True,
                 n_disc_actions: int = 11,
                 n_cont_actions: int = 1,
                 **kwargs):
        """Constructor."""
        super().__init__(m, max_eps=max_eps, **kwargs)

        if cont_actions and n_cont_actions is None:
            raise ValueError("Need to specify action space dimensionality!")
        if not cont_actions:
            raise NotImplementedError("Discrete actions are deprecated!")

        # Save info
        self.action_range = to_list(action_range)
        assert len(self.action_range) == n_cont_actions, "False amount of action ranges!"

        self.nb_actions = n_disc_actions if not cont_actions else n_cont_actions
        self.cont_actions = cont_actions

        # Initialize fallback actions
        self.fb_actions = np.empty((max_eps, n_cont_actions), dtype=np.float32)

        d = m.data
        self.dt_h = d.dt / 60
        self.c_ind = d.c_inds
        if np.all(d.is_scaled):
            self.scaling = d.scaling

        # Scaled action range
        ac_r_scales = np.empty((n_cont_actions, 2), dtype=np.float32)
        for k in range(n_cont_actions):
            ac_r_scales[k] = np.array(self.action_range[k])
            if self.scaling is not None:
                ac_r_scales[k] = rem_mean_and_std(ac_r_scales[k],
                                                  self.scaling[self.c_ind[k]])

    def _to_scaled(self, action: Arr, to_original: bool = False) -> np.ndarray:
        """Converts actions to the right range."""
        if np.array(action).shape == ():
            assert self.nb_actions == 1, "Ambiguous for more than one action!"
            action = np.array([action])
        else:
            assert len(action) == self.nb_actions, "Not the right amount of actions!"
        cont_action = self.scale_actions(action)
        if self.scaling is None:
            return cont_action
        c_actions_scaled = np.empty_like(cont_action)
        for k in range(self.nb_actions):
            c_actions_scaled[k] = trf_mean_and_std(cont_action[k], self.scaling[self.c_ind[k]], not to_original)
        return c_actions_scaled

    def scale_action_for_step(self, action: Arr):
        return self._to_scaled(action)


class FullRoomEnv(RLDynEnv):
    """The environment modeling one room only."""
    alpha: float = 1.0  #: Weight factor for reward.
    temp_bounds: Sequence = (22.0, 26.0)  #: The requested temperature range.
    bound_violation_penalty: float = 2.0  #: The penalty in the reward for temperatures out of bound.

    def __init__(self, m: BaseDynamicsModel,
                 max_eps: int = 48,
                 temp_bounds: Sequence = None,
                 **kwargs):
        # Define name
        ext = make_param_ext([("NEP", max_eps), ("TBD", temp_bounds)])
        name = "FullRoom" + ext

        # Initialize super class
        super(FullRoomEnv, self).__init__(m, max_eps, name=name, **kwargs)

        # Save parameters
        d = m.data
        if temp_bounds is not None:
            self.temp_bounds = temp_bounds
        if np.all(d.is_scaled):
            self.scaling = d.scaling

        # Check model and dataset
        assert len(m.out_inds) == d.d - d.n_c, "Model not suited for this environment!!"
        assert d.d == 8 and d.n_c == 1, "Not the correct number of series in dataset!"

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

    reward_descs = ["Energy Consumption []",
                    "Temperature Bound Violation [°Kh]"]

    def detailed_reward(self, curr_pred: np.ndarray, action: Arr) -> np.ndarray:

        # Compute energy used
        action_rescaled = self._to_scaled(action, True)[0]
        w_temps = self.get_w_temp(curr_pred)
        d_temp = np.abs(w_temps[0] - w_temps[1])
        energy_used = np.clip(action_rescaled, 0.0, 1.0) * d_temp * self.dt_h
        # assert 10.0 <= w_temps[0] <= 50.0, "Water temperature scaled incorrectly!"

        # Check for actions out of range
        action_penalty = linear_oob_penalty(action_rescaled, [0.0, 1.0])
        assert action_penalty <= 0.0, "Actions scaled wrongly!"

        # Penalty for constraint violation
        r_temp = self.get_r_temp(curr_pred)
        temp_pen = self.dt_h * linear_oob_penalty(r_temp, self.temp_bounds)
        return np.array([energy_used, temp_pen])

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:
        """Computes the total reward from the individual components."""
        return -np.sum(self.detailed_reward(curr_pred, action)).item()

    def episode_over(self, curr_pred: np.ndarray) -> bool:

        r_temp = self.get_r_temp(curr_pred)
        thresh = 10.0
        t_bounds = self.temp_bounds
        if r_temp > t_bounds[1] + thresh or r_temp < t_bounds[0] - thresh:
            return True
        return False


class CProf(ABC):
    name: str

    @abstractmethod
    def __call__(self, t: int) -> float:
        """Returns the cost at timestep `t`."""
        pass


class ConstProfile(CProf):
    """Constant price profile."""

    def __init__(self, p: float):
        self.p = p
        self.name = f"PConst_{p}"

    def __call__(self, t: int) -> float:
        return self.p


class PWProfile(CProf):
    """Some example price profile."""

    name = "PW_Profile"

    def __call__(self, t: int) -> float:
        if t < 5:
            return 1.0
        elif t < 10:
            return 2.0
        elif t < 20:
            return 1.0
        elif t < 25:
            return 3.0
        elif t < 30:
            return 2.0
        else:
            return 1.0


class BatteryEnv(RLDynEnv):
    """The environment for the battery model.

    """

    alpha: float = 1.0  #: Reward scaling factor.
    action_range: RangeListT = [(-100, 100)]  #: The requested active power range.
    soc_bound: Sequence = (20, 80)  #: The requested state-of-charge range.
    scaled_soc_bd: np.ndarray = None  #: `soc_bound` scaled to the model space.
    req_soc: float = 60.0  #: Required SoC at end of episode.
    prev_pred: np.ndarray  #: The previous prediction.
    m: BatteryModel  #: The battery model.
    p: CProf = None  #: The cost profile.

    def __init__(self, m: BatteryModel, p: CProf = None, **kwargs):

        d = m.data

        # Add max predictions length to kwargs if not there yet.
        ep_key = 'max_eps'
        kwargs[ep_key] = kwargs.get(ep_key, 24 * 60 // d.dt // 2)

        # Define name
        ext = "_" + p.name if p is not None else ""
        name = "Battery" + ext

        # Init base class.
        super().__init__(m, name=name, action_range=[(-100, 100)], **kwargs)

        self.p = p
        assert p is None, "Cost profile does not make sense here!"
        # TODO: Remove cost profile from model!

        self.scaled_soc_bd = rem_mean_and_std(np.array(self.soc_bound), self.scaling[0])

        # Check model
        assert len(m.out_inds) == 1, "Model not suited for this environment!!"
        assert self.action_range == [(-100, 100)], "action_range value was overridden!"

        # Check underlying dataset
        assert d.d == 2 and d.n_c == 1, "Not the correct number of series in dataset!"
        assert d.c_inds[0] == 1, "Second series needs to be controllable!"

    def _get_scaled_soc(self, unscaled_soc, remove_mean: bool = False):
        """Scales the state-of-charge."""
        if self.scaling is not None:
            return trf_mean_and_std(unscaled_soc, self.scaling[0], remove=remove_mean)
        return unscaled_soc

    reward_descs = ["Energy Consumption [kWh]"]  #: Description of the detailed reward.

    def detailed_reward(self, curr_pred: np.ndarray, action: Arr) -> np.ndarray:
        """Computes the energy used by dis- / charging the battery."""

        action_rescaled = self._to_scaled(action, True)[0]
        assert linear_oob_penalty(action_rescaled, self.action_range[0]) <= 0.0001, "WTF"

        # Compute energy used
        energy_used = action_rescaled * self.dt_h

        # Compute costs (Deprecated!)
        if self.p is not None:
            energy_used *= self.p(self.n_ts)
            assert False, "No pricing for battery only!"

        # Check constraint violation
        curr_pred = self._get_scaled_soc(curr_pred)
        assert linear_oob_penalty(curr_pred, self.soc_bound) <= 0.001, "WTF2"

        # Penalty for not having charged enough at the end of the episode.
        if self.n_ts > self.n_ts_per_eps - 1 and curr_pred < self.req_soc - 0.001:
            assert linear_oob_penalty(curr_pred, [self.req_soc, 100]) <= 0.0, "Model not working"

        # Total reward is minus the energy used.
        return np.array([energy_used])

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:
        """Compute the reward for choosing action `action`.

        The reward takes into account the energy used.
        """
        # Return minus the energy used.
        e_used = self.detailed_reward(curr_pred, action)
        return -e_used.item() * self.alpha

    def episode_over(self, curr_pred: np.ndarray) -> bool:
        """Declare the episode as over if the SoC lies too far without bounds."""
        thresh = 10
        b = self.soc_bound
        scaled_soc = self._get_scaled_soc(curr_pred)
        if scaled_soc > b[1] + thresh or scaled_soc < b[0] - thresh:
            raise AssertionError("Battery model wrong!!")
            # return True
        return False

    def scale_action_for_step(self, action: Arr):
        # Do scaling if agent wants it
        if self.do_scaling:
            action = self.a_scaling_pars[0] + action * self.a_scaling_pars[1]

        # Get default min and max actions from bounds
        scaled_ac = self._to_scaled(np.array(self.action_range, dtype=np.float32))
        assert len(scaled_ac) == 1 and len(scaled_ac[0]) == 2, "Shape mismatch!"
        min_ac, max_ac = scaled_ac[0]

        # Find the minimum and maximum action satisfying SoC constraints.
        curr_state = self.get_curr_state()
        soc_bound_arr = np.array(self.soc_bound, copy=True)
        s_min_scaled, s_max_scaled = self._get_scaled_soc(soc_bound_arr, remove_mean=True)
        b, c_min, gam = self.m.params
        c_max = c_min + gam

        # SoC bounds
        min_goal_soc = self._get_scaled_soc(self.req_soc, remove_mean=True)
        n_remain_steps = self.n_ts_per_eps - self.n_ts
        max_ds = b + max_ac * c_max
        next_d_soc_min = np.maximum(s_min_scaled, min_goal_soc - (n_remain_steps - 1) * max_ds) - b - curr_state
        if next_d_soc_min < 0:
            ac_min = np.maximum(next_d_soc_min / c_min, min_ac)
        else:
            ac_min = np.maximum(next_d_soc_min / c_max, min_ac)
        next_d_soc_max = s_max_scaled - b - curr_state
        ac_max = np.minimum(next_d_soc_max / c_max, max_ac)

        # Clip the actions.
        scaled_action = self._to_scaled(action)
        chosen_action = np.clip(scaled_action, ac_min, ac_max)
        if self.scaling is not None:
            assert not np.array_equal(action, chosen_action)

        return chosen_action

    def reset(self, *args, **kwargs) -> np.ndarray:
        super().reset(*args, **kwargs)

        # Clip the values to the valid SoC range!
        if self.scaled_soc_bd is None:
            self.scaled_soc_bd = rem_mean_and_std(np.array(self.soc_bound), self.m.data.scaling[0])
        self.hist[:, 0] = np.clip(self.hist[:, 0],
                                  self.scaled_soc_bd[0],
                                  self.scaled_soc_bd[1])
        return np.copy(self.hist[-1, :-self.act_dim])
