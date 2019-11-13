from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Any, List

import numpy as np

from dynamics.base_model import BaseDynamicsModel
from dynamics.battery_model import BatteryModel
from envs.base_dynamics_env import DynEnv
from util.numerics import trf_mean_and_std, add_mean_and_std
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
                 cont_actions: bool = False,
                 n_disc_actions: int = 11,
                 n_cont_actions: int = None,
                 **kwargs):
        """Constructor."""
        super().__init__(m, max_eps=max_eps, **kwargs)

        if cont_actions and n_cont_actions is None:
            raise ValueError("Need to specify action space dimensionality!")
        if not cont_actions:
            raise NotImplementedError("This is deprecated!")

        # Save info
        self.action_range = to_list(action_range)
        assert len(self.action_range) == n_cont_actions, "False amount of action ranges!"

        self.nb_actions = n_disc_actions if not cont_actions else n_cont_actions
        self.cont_actions = cont_actions

        d = m.data
        self.c_ind = d.c_inds
        if np.all(d.is_scaled):
            self.scaling = d.scaling

    def _to_scaled(self, action: Arr, to_original: bool = False) -> np.ndarray:
        """Converts actions to the right range."""
        if np.array(action).shape == ():
            action = np.array([action])
        cont_action = self.scale_actions(action)
        if self.scaling is None:
            return cont_action
        c_actions_scaled = np.empty_like(cont_action)
        for k in range(self.nb_actions):
            c_actions_scaled[k] = trf_mean_and_std(cont_action[k], self.scaling[self.c_ind[k]], not to_original)
        return c_actions_scaled

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:
        return super().step(self._to_scaled(action))


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

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:

        # Compute energy used
        action_rescaled = self._to_scaled(action, True)[0]
        w_temps = self.get_w_temp(curr_pred)
        d_temp = np.abs(w_temps[0] - w_temps[1])
        energy_used = np.clip(action_rescaled, 0.0, 1.0) * d_temp * self.alpha
        # assert 0.0 <= action_rescaled <= 1.0, "Fucking wrong"
        # assert 10.0 <= w_temps[0] <= 50.0, "Water temperature scaled incorrectly!"

        # Penalty for actions out of the range
        action_penalty = 100 * linear_oob_penalty(action_rescaled, [0.0, 1.0])

        # Penalty for constraint violation
        r_temp = self.get_r_temp(curr_pred)
        temp_pen = self.bound_violation_penalty * linear_oob_penalty(r_temp, self.temp_bounds)
        return (-energy_used - temp_pen - action_penalty) / 10

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
    alpha: float = 1.0  #: Weight factor for reward.
    action_range: Sequence = (-100, 100)  #: The requested active power range.
    soc_bound: Sequence = (20, 80)  #: The requested state-of-charge range.
    req_soc: float = 60.0  #: Required SoC at end of episode.
    prev_pred: np.ndarray  #: The previous prediction.
    m: BatteryModel  #: The battery model.
    p: CProf = None  #: The cost profile.

    def __init__(self, m: BatteryModel, p: CProf = None, **kwargs):
        d = m.data
        max_eps = 24 * 60 // d.dt // 2  # max predictions length
        ext = "_" + p.name if p is not None else ""
        name = "Battery" + ext
        super().__init__(m, max_eps, name=name, action_range=(-100, 100), **kwargs)

        self.p = p
        assert p is None, "Cost profile does not make sense here!"
        # TODO: Remove cost profile from model!

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

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:
        """Compute the reward for choosing action `action`.

        The reward takes into account the energy used, whether
        the bounds are satisfied and whether the SoC is high enough
        at the end of the episode.
        """
        # Compute energy used
        action_rescaled = self._to_scaled(action, True)[0]
        action_pen = 1000 * linear_oob_penalty(action_rescaled, self.action_range[0])
        assert action_pen <= 0.0, "WTF"
        energy_used = action_rescaled * self.alpha
        if self.p is not None:
            energy_used *= self.p(self.n_ts)

        # Penalty for constraint violation
        curr_pred = self._get_scaled_soc(curr_pred)
        bound_pen = 1000 * linear_oob_penalty(curr_pred, self.soc_bound)
        if self.n_ts > 1:
            assert bound_pen <= 0.0, "WTF2"

        # Penalty for not having charged enough at the end of the episode.
        if self.n_ts > self.n_ts_per_eps - 1 and curr_pred < self.req_soc:
            not_enough = 2000 * linear_oob_penalty(curr_pred, [self.req_soc, 100])
            bound_pen += not_enough
            assert not_enough <= 0.0, "Model not working"

        # Total reward is the negative penalty minus energy used.
        tot_rew = -energy_used - bound_pen - action_pen
        return np.array(tot_rew).item() / 300

    def episode_over(self, curr_pred: np.ndarray) -> bool:
        """Declare the episode as over if the SoC lies too far without bounds."""
        thresh = 10
        b = self.soc_bound
        scaled_soc = self._get_scaled_soc(curr_pred)
        if scaled_soc > b[1] + thresh or scaled_soc < b[0] - thresh:
            raise AssertionError("Battery model wrong!!")
            # return True
        return False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Any]:
        """Step function for battery environment.

        If the chosen action would result in a SoC outside the bounds,
        it is clipped, s.t. the bound constraints are always fulfilled.
        """
        # Get default min and max actions from bounds
        min_ac, max_ac = self._to_scaled(np.array(self.action_range))[0]

        # Find the minimum and maximum action satisfying SoC constraints.
        curr_state = self.get_curr_state()
        soc_bound_arr = np.array(self.soc_bound, copy=True)
        s_min_scaled, s_max_scaled = self._get_scaled_soc(soc_bound_arr, remove_mean=True)
        b, c_min, gam = self.m.params
        c_max = c_min + gam
        ac_min = np.maximum((s_min_scaled - b - curr_state) / c_min, min_ac)
        ac_max = np.minimum((s_max_scaled - b - curr_state) / c_max, max_ac)

        # Compute minimal action to reach SoC goal
        n_remain_steps = self.n_ts_per_eps - self.n_ts
        min_goal_soc = self._get_scaled_soc(self.req_soc, remove_mean=True)
        ds_remain = min_goal_soc - curr_state
        if ds_remain > 0:
            max_ds = b + max_ac * c_max
            n_ts_needed_min = np.ceil(ds_remain / max_ds)
            if n_ts_needed_min >= n_remain_steps:
                min_ds_now = ds_remain - (n_ts_needed_min - 1) * max_ds
                ac_min = (min_ds_now - b) / c_max

        # Clip the actions.
        chosen_action = self._to_scaled(action)
        action = np.clip(chosen_action, ac_min, ac_max)

        # Call the step function of DynEnv to avoid another scaling.
        # TODO: Return info about action clipping. (Fallback control!)
        return DynEnv.step(self, action)
