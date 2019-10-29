from abc import ABC, abstractmethod

from base_dynamics_env import DynEnv, Any

from util import *


class AgentBase(ABC):
    env: Any  # The corresponding environment

    def __init__(self, env: DynEnv):
        self.env = env

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def get_action(self, state) -> Arr:
        pass

    pass
