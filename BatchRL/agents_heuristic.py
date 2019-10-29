from base_agent import AgentBase
from dynamics_envs import FullRoomEnv
from util import *


class RuleBasedHeating(AgentBase):
    """Agent applying rule-based heating control.

    """
    rule: Sequence  # The sequence specifying the rule for control.

    def __init__(self, env: FullRoomEnv, rule: Sequence):
        super().__init__(env)

        assert len(rule) == 2, "Rule needs to consist of two values!"
        self.rule = rule

    def fit(self) -> None:
        """No need to fit anything."""
        pass

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
