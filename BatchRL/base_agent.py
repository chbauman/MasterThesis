from abc import ABC, abstractmethod

from base_dynamics_env import DynEnv
from util import *


class AgentBase(ABC):
    """Base class for an agent / control strategy.

    Might be specific for a certain environment accessible
    by attribute `env`.
    """
    env: Any  #: The corresponding environment
    name: str  #: The name of the Agent / control strategy

    def __init__(self, env: DynEnv, name: str = "Abstract Agent"):
        self.env = env
        self.name = name

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def get_action(self, state) -> Arr:
        pass

    pass
