from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from util.numerics import npf32
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

    def eval(self, n_steps: int = 100, reset_seed: bool = False, detailed: bool = False,
             use_noise: bool = False):
        """Evaluates the agent for a given number of steps.

        Args:
            n_steps: Number of steps.
            reset_seed: Whether to reset the seed at start.
            detailed: Whether to return all parts of the reward.
            use_noise: Whether to use noise during the evaluation.

        Returns:
            The mean received reward.
        """
        # Fix seed if needed.
        if reset_seed:
            fix_seed()

        # Initialize env and reward.
        s_curr = self.env.reset(use_noise=use_noise)
        all_rewards = npf32((n_steps,))

        # Detailed stuff
        det_rewards = None
        if detailed:
            n_det = len(self.env.reward_descs)
            det_rewards = np.empty((n_steps, n_det), dtype=np.float32)

        # Evaluate for `n_steps` steps.
        for k in range(n_steps):

            # Determine action
            a = self.get_action(s_curr)
            scaled_a = self.env.scale_action_for_step(a)

            # Execute step
            s_curr, r, fin, _ = self.env.step(a)

            # Store rewards
            all_rewards[k] = r
            if det_rewards is not None:
                det_rew = self.env.detailed_reward(s_curr, scaled_a)
                det_rewards[k, :] = det_rew

            # Reset env if episode is over.
            if fin:
                s_curr = self.env.reset()

        # Return all rewards
        if detailed:
            return all_rewards, det_rewards

        # Return mean reward.
        return np.sum(all_rewards) / n_steps
