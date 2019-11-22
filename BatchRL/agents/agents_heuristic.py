from typing import Sequence

import numpy as np

from agents.base_agent import AgentBase
from envs.dynamics_envs import FullRoomEnv
from util.util import Arr


class RuleBasedHeating(AgentBase):
    """Agent applying rule-based heating control.

    """
    rule: Sequence  #: The sequence specifying the rule for control.

    def __init__(self, env: FullRoomEnv, rule: Sequence):
        name = "RuleBasedHeating"
        super().__init__(env, name=name)

        assert len(rule) == 2, "Rule needs to consist of two values!"
        self.rule = rule

    def get_action(self, state) -> Arr:
        """Defines the control strategy.

        Args:
            state: The current state.

        Returns:
            Next control action.
        """
        w_in_temp, _ = self.env.get_w_temp(state)
        r_temp = self.env.get_r_temp(state)
        if r_temp < self.rule[0] and w_in_temp > r_temp:
            # Heating
            return 1.0
        if r_temp > self.rule[1] and w_in_temp < r_temp:
            # Cooling
            return 1.0
        # Do nothing
        return 0.0


class ConstActionAgent(AgentBase):
    """Applies a constant control input.

    Can be used for comparison, e.g. if you want
    to compare an agent to always heating or never heating.
    Does not really need the environment.
    """
    rule: float  #: The constant control input / action.
    out_num: int  #: The dimensionality of the action space.

    def __init__(self, env, rule: float):
        super().__init__(env, name=f"Const_{rule}")

        self.out_num = env.nb_actions
        self.rule = rule

    def get_action(self, state) -> Arr:
        """Defines the control strategy.

        Args:
            state: The current state.

        Returns:
            Next control action.
        """
        return self.rule * np.ones((self.out_num,), dtype=np.float32)
