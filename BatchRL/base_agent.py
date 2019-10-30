from abc import ABC, abstractmethod

# import base_dynamics_env
from util import *


class AgentBase(ABC):
    """Base class for an agent / control strategy.

    Might be specific for a certain environment accessible
    by attribute `env`.
    """
    env: Any  #: The corresponding environment
    name: str  #: The name of the Agent / control strategy

    def __init__(self, env: 'base_dynamics_env.DynEnv', name: str = "Abstract Agent"):
        self.env = env
        self.name = name

    def fit(self) -> None:
        """No fitting needed."""
        pass

    @abstractmethod
    def get_action(self, state) -> Arr:
        """Defines the control strategy.

        Args:
            state: The current state.

        Returns:
            Next control action.
        """
        pass
