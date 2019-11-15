from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from util.util import Arr, fix_seed


class AgentBase(ABC):
    """Base class for an agent / control strategy.

    Might be specific for a certain environment accessible
    by attribute `env`.
    """
    env: Any  #: The corresponding environment
    name: str  #: The name of the Agent / control strategy

    def __init__(self, env: 'DynEnv', name: str = "Abstract Agent"):
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

    def get_short_name(self):
        return self.name

    def get_info(self) -> Dict:
        return {}

    def eval(self, n_steps: int = 100, reset_seed: bool = False, detailed: bool = False):
        """Evaluates the agent for a given number of steps.

        Args:
            n_steps: Number of steps.
            reset_seed: Whether to reset the seed at start.
            detailed: Whether to return all parts of the reward.

        Returns:
            The mean received reward.
        """
        # Fix seed if needed.
        if reset_seed:
            fix_seed()

        # Initialize env and reward.
        s_curr = self.env.reset()
        curr_cum_reward = 0.0

        # Detailed stuff
        if detailed:
            n_det = len(self.env.reward_descs)
            det_rewards = np.empty((n_steps, n_det))

        # Evaluate for `n_steps` steps.
        for k in range(n_steps):

            # Determine action
            a = self.get_action(s_curr)


            s_curr, r, fin, _ = self.env.step(a)
            curr_cum_reward += r

            if detailed:
                det_rewards[k, :] = self.env.detailed_reward()

            # Reset env if episode is over.
            if fin:
                s_curr = self.env.reset()

        # Return mean reward.
        return curr_cum_reward / n_steps
