import os
import warnings
from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING, Callable

import numpy as np

from util.numerics import npf32
from util.share_data import upload_folder_zipped, download_and_extract_zipped_folder
from util.util import Arr, fix_seed, MODEL_DIR, create_dir

if TYPE_CHECKING:
    from envs.base_dynamics_env import DynEnv

# Define directory for agent models
RL_MODEL_DIR = os.path.join(MODEL_DIR, "RL")
create_dir(RL_MODEL_DIR)


def upload_trained_agents(verbose: int = 1):
    if verbose:
        print("Uploading agent neural network parameters to Google Drive.")
    upload_folder_zipped(RL_MODEL_DIR)


def download_trained_agents(verbose: int = 1):
    if verbose:
        print("Downloading agent neural network parameters from Google Drive.")
    download_and_extract_zipped_folder("RL", RL_MODEL_DIR)


def remove_agents(min_steps: int = 1000, verbose: int = 5) -> None:
    """Removes all agents that were trained for less than `min_steps` steps.

    Args:
        min_steps: Minimum number of training steps for an agent not to be
            deleted.
        verbose: Whether to print infos.
    """
    for sub_dir in os.listdir(RL_MODEL_DIR):
        # Get full path
        full_sub_path = os.path.join(RL_MODEL_DIR, sub_dir)

        # Check if it is a file instead of a folder
        if os.path.isfile(full_sub_path):
            if verbose:
                print(f"Found unexpected file: {full_sub_path}")
            continue

        # Find sub files (and folders)
        sub_files = os.listdir(full_sub_path)

        # Delete folder if empty
        if len(sub_files) == 0:
            print(f"Removing folder: {sub_dir}")
            os.rmdir(full_sub_path)

        # Iterate over files in sub-folder
        for f in sub_files:
            f_path = os.path.join(full_sub_path, f)

            # Check if it is actually a folder
            if os.path.isdir(f):
                if verbose:
                    print(f"Found unexpected folder: {f} in {full_sub_path}")
                continue

            # Decide if it will be removed
            rem_file = False
            try:
                n_ep = int(f.split("_")[1][3:])
                rem_file = n_ep < min_steps
            except (IndexError, ValueError):
                if verbose:
                    print(f"Invalid file name: {f}")

            # Remove
            if rem_file:
                if verbose:
                    print(f"Removing file: {f}")
                os.remove(f_path)


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
             use_noise: bool = False, scale_states: bool = False,
             episode_marker: Callable = None):
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
            episode_marker: Function mapping from state to natural numbers.

        Returns:
            The mean received reward if `detailed` is False, else
            all the rewards for all steps.
        """
        # Fix seed if needed.
        if reset_seed:
            fix_seed()

        # Initialize env and reward.
        s_curr = self.env.reset(use_noise=use_noise)
        ep_mark = 0 if episode_marker is None else episode_marker(s_curr)
        all_rewards = npf32((n_steps,))

        # Detailed stuff
        det_rewards, state_t, ep_marks = None, None, None
        if detailed:
            n_det = len(self.env.reward_descs)
            n_states = self.env.state_dim - self.env.act_dim
            det_rewards = npf32((n_steps, n_det), fill=np.nan)
            state_t = npf32((n_steps, n_states), fill=np.nan)
            ep_marks = npf32((n_steps, ), fill=np.nan)
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
                ep_marks[k] = ep_mark

            # Reset env if episode is over.
            if fin:
                s_curr = self.env.reset()
                ep_mark = 0 if episode_marker is None else episode_marker(s_curr)

        # Return all rewards
        if detailed:
            if scale_states:
                state_t = self.env.scale_state(state_t, remove_mean=False)
            return all_rewards, det_rewards, state_t, ep_marks

        # Return mean reward.
        return np.sum(all_rewards) / n_steps
