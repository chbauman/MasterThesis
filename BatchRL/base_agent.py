from abc import ABC, abstractmethod

from base_dynamics_env import DynEnv


class AgentBase(ABC):

    env: DynEnv  # The corresponding environment

    def __init__(self, env: DynEnv):
        self.env = env

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def get_action(self, state):
        pass
    pass
