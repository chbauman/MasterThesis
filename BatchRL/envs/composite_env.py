from typing import List, Tuple, Dict

import numpy as np

from envs.dynamics_envs import RLDynEnv
from util.util import Arr


class CompositeRLEnv(RLDynEnv):

    def __init__(self, env_list: List[RLDynEnv]):
        self.env_list = env_list
        pass

    def detailed_reward(self, curr_pred: np.ndarray, action: Arr) -> np.ndarray:
        pass

    def episode_over(self, curr_pred: np.ndarray) -> bool:
        pass

    def step(self, action: Arr) -> Tuple[np.ndarray, float, bool, Dict]:
        pass

    def get_curr_state(self):
        pass

    def reset(self, start_ind: int = None, use_noise: bool = True) -> np.ndarray:
        pass

    pass
