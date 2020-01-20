import os
import warnings
from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING

import numpy as np

from util.numerics import npf32
from util.util import Arr, fix_seed, model_dir, create_dir

if TYPE_CHECKING:
    from envs.base_dynamics_env import DynEnv

# Define directory for agent models
rl_model_dir = os.path.join(model_dir, "RL")
create_dir(rl_model_dir)


class AbstractAgent(ABC):
    @abstractmethod
    def get_action(self, state) -> Arr:
        """Defines the control strategy.

        Args:
            state: The current state.

        Returns:
            Next control action.
        """
        pass


class AgentBase(AbstractAgent, ABC):
    """Base class for an agent / control strategy.

    Might be specific for a certain environment accessible
    by attribute `env`.
    """
    env: 'DynEnv'  #: The corresponding environment
    name: str  #: The name of the Agent / control strategy

    def __init__(self, env: 'DynEnv', name: str = "Abstract Agent"):
        self.env = env
        self.name = name

    def fit(self, verbose: int = 0) -> None:
        """No fitting needed."""
        pass

    def get_short_name(self) -> str:
        return self.name

    def get_info(self) -> Dict:
        return {}

    def eval(self, n_steps: int = 100, reset_seed: bool = False, detailed: bool = False,
             use_noise: bool = False, scale_states: bool = False):
        """Evaluates the agent for a given number of steps.

        If the number is greater than the number of steps in an episode, the
        env is reset and a new episode is started.

        Args:
            n_steps: Number of steps.
            reset_seed: Whether to reset the seed at start.
            detailed: Whether to return all parts of the reward.
            use_noise: Whether to use noise during the evaluation.
            scale_states: Whether to scale the state trajectory to
                original values, only used if `detailed` is True.

        Returns:
            The mean received reward if `detailed` is False, else
            all the rewards for all steps.
        """
        # Fix seed if needed.
        if reset_seed:
            fix_seed()

        # Initialize env and reward.
        s_curr = self.env.reset(use_noise=use_noise)
        all_rewards = npf32((n_steps,))

        # Detailed stuff
        det_rewards, state_t = None, None
        if detailed:
            n_det = len(self.env.reward_descs)
            n_states = self.env.state_dim
            det_rewards = npf32((n_steps, n_det), fill=np.nan)
            state_t = npf32((n_steps, n_states), fill=np.nan)
        elif scale_states:
            warnings.warn(f"Argument: {scale_states} ignored!")

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
                state_t[k, :] = s_curr

            # Reset env if episode is over.
            if fin:
                s_curr = self.env.reset()

        # Return all rewards
        if detailed:
            if scale_states:
                state_t = self.env.scale_state(state_t, remove_mean=False)
            return all_rewards, det_rewards, state_t

        # Return mean reward.
        return np.sum(all_rewards) / n_steps
