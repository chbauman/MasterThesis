import warnings
from typing import Sequence, Union, Tuple

import numpy as np

from agents.base_agent import AgentBase
from envs.dynamics_envs import FullRoomEnv, RoomBatteryEnv, BatteryEnv
from util.util import Arr, Num


def get_const_agents(env: Union[FullRoomEnv, RoomBatteryEnv, BatteryEnv]
                     ) -> Tuple['ConstActionAgent', 'ConstActionAgent']:
    """Defines two constant agents that can be used for analysis.

    Args:
        env: The environment.

    Returns:
        Tuple with two ConstActionAgent
    """
    n_agents = 2

    heat_pars = (0.0, 1.0)
    bat_pars = (6.0, -3.0)

    # Define constant action based on env.
    if isinstance(env, FullRoomEnv):
        c = [np.array(heat_pars[i]) for i in range(n_agents)]
    elif isinstance(env, BatteryEnv):
        c = [np.array(bat_pars[i]) for i in range(n_agents)]
    elif isinstance(env, RoomBatteryEnv):
        c = [np.array([heat_pars[i], bat_pars[i]]) for i in range(n_agents)]
    else:
        raise TypeError(f"Env: {env} not supported!")

    return ConstActionAgent(env, c[0]), ConstActionAgent(env, c[1])


class RuleBasedAgent(AgentBase):
    """Agent applying rule-based heating control.

    """
    bounds: Sequence  #: The sequence specifying the rule for control.
    const_charge_rate: Num
    env: Union[FullRoomEnv, RoomBatteryEnv]

    w_inds_orig = [2, 3]
    r_temp_ind_orig = 5

    def __init__(self, env: Union[FullRoomEnv, RoomBatteryEnv],
                 rule: Sequence,
                 const_charge_rate: Num = None,
                 strict: bool = False):
        """Initializer.

        Args:
            env: The RL environment.
            rule: The temperature bounds.
            const_charge_rate: The charging rate if the env includes the battery.
            strict: Whether to apply strict heating / cooling, start as soon as the
                room temperature deviates from the midpoint of the temperature bounds.
        """
        name = "RuleBasedHeating"
        super().__init__(env, name=name)

        # Check input
        assert len(rule) == 2, "Rule needs to consist of two values!"
        if isinstance(env, RoomBatteryEnv):
            assert const_charge_rate is not None, "Need to specify charging rate!"
        elif const_charge_rate is not None:
            warnings.warn("Ignored charging rate!")
            const_charge_rate = None

        # Store parameters
        self.const_charge_rate = const_charge_rate
        if strict:
            mid = 0.5 * (rule[0] + rule[1])
            self.bounds = (mid, mid)
        else:
            self.bounds = rule

    def get_action(self, state) -> Arr:
        """Defines the control strategy.

        Args:
            state: The current state.

        Returns:
            Next control action.
        """
        # Find water and room temperatures
        w_in_temp, _ = self.env.get_w_temp(state)
        r_temp = self.env.get_r_temp(state)

        # Determine if you want to do heating / cooling or not
        heat_action = 0.0
        if r_temp < self.bounds[0] and w_in_temp > r_temp:
            # Heating
            heat_action = 1.0
        if r_temp > self.bounds[1] and w_in_temp < r_temp:
            # Cooling
            heat_action = 1.0
        # Return
        final_action = heat_action
        if self.const_charge_rate is not None:
            final_action = np.array([heat_action, self.const_charge_rate])
        return final_action


class ConstActionAgent(AgentBase):
    """Applies a constant control input.

    Can be used for comparison, e.g. if you want
    to compare an agent to always heating or never heating.
    Does not really need the environment.
    """
    rule: Arr  #: The constant control input / action.
    out_num: int  #: The dimensionality of the action space.

    def __init__(self, env, rule: Arr):
        super().__init__(env, name=f"Const_{rule}")

        self.out_num = env.nb_actions
        self.rule = rule

        # Check rule
        if isinstance(rule, (np.ndarray, np.generic)):
            r_s, n_out = rule.shape, self.out_num
            if self.out_num > 1:
                assert r_s == (n_out,), f"Rule shape: {r_s} incompatible!"
            else:
                assert r_s == (n_out,) or r_s == (), f"Rule shape: {r_s} incompatible!"

    def get_action(self, state) -> Arr:
        """Defines the control strategy.

        Using broadcasting it can handle numpy array rules
        of shape (`out_num`, )

        Args:
            state: The current state.

        Returns:
            Next control action.
        """
        return self.rule * np.ones((self.out_num,), dtype=np.float32)
