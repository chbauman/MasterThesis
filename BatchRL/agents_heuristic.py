from base_agent import AgentBase
from dynamics_envs import FullRoomEnv


class RuleBasedHeating(AgentBase):
    """Agent applying rule-based heating control.

    """

    def __init__(self, env: FullRoomEnv, rule):
        super().__init__(env)

        self.rule = rule

    def fit(self) -> None:
        """No need to fit anything."""
        pass

    def get_action(self, state):

        pass
