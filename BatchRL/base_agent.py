from abc import ABC, abstractmethod

import gym


class AgentBase(ABC):

    env: gym.Env  # The corresponding environment

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def get_action(self, state):
        pass
    pass
