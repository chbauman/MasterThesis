from abc import ABC, abstractmethod

from util.util import *


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

    def eval(self, n_steps: int = 100, reset_seed: bool = False):
        """Evaluates the agent for a given number of steps.

        Args:
            n_steps: Number of steps.
            reset_seed: Whether to reset the seed at start.

        Returns:
            The mean received reward.
        """
        # Fix seed if needed.
        if reset_seed:
            fix_seed()

        # Initialize env and reward.
        s_curr = self.env.reset()
        curr_cum_reward = 0.0

        # Evaluate for `n_steps` steps.
        for k in range(n_steps):
            a = self.get_action(s_curr)
            s_curr, r, fin, _ = self.env.step(a)
            curr_cum_reward += r

            # Reset env if episode is over.
            if fin:
                s_curr = self.env.reset()

        # Return mean reward.
        return curr_cum_reward / n_steps
