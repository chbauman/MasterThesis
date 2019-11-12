from unittest import TestCase

import numpy
import numpy as np

from agents import agents_heuristic
from dynamics.base_model import construct_test_ds, BaseDynamicsModel
from envs.base_dynamics_env import DynEnv
from tests.test_dynamics import TestModel
from util.util import Arr


class TestDynEnv(DynEnv):
    """The test environment."""
    def __init__(self, m: BaseDynamicsModel, max_eps: int = None):
        super(TestDynEnv, self).__init__(m, "TestEnv", max_eps)
        d = m.data
        self.n_pred = 3
        assert d.n_c == 1 and d.d == 4, "Dataset needs 4 series of which one is controllable!!"

    def compute_reward(self, curr_pred: np.ndarray, action: Arr) -> float:
        self._assert_pred_shape(curr_pred)
        return curr_pred[2] * action

    def episode_over(self, curr_pred: np.ndarray) -> bool:
        self._assert_pred_shape(curr_pred)
        return False

    def _assert_pred_shape(self, curr_pred):
        assert curr_pred.shape == (self.n_pred,), "Shape of prediction not correct!"


class TestEnvs(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define dataset
        self.n = 201
        self.test_ds = construct_test_ds(self.n)
        self.test_mod = TestModel(self.test_ds)
        self.n_ts_per_episode = 10
        self.test_env = TestDynEnv(self.test_mod, 10)

    def test_shapes(self):
        # Test shapes
        for k in range(30):
            next_state, r, over, _ = self.test_env.step(0.0)
            assert next_state.shape == (3,), "Prediction does not have the right shape!"
            if (k + 1) % self.n_ts_per_episode == 0 and not over:
                raise AssertionError("Episode should be over!!")
            if over:
                init_state = self.test_env.reset()
                self.assertEqual(init_state.shape, (3,), "Prediction does not have the right shape!")

    def test_agent_analysis(self):
        # Test agent analysis
        const_ag_1 = agents_heuristic.ConstHeating(self.test_env, 0.0)
        const_ag_2 = agents_heuristic.ConstHeating(self.test_env, 1.0)
        self.test_env.analyze_agent([const_ag_1, const_ag_2])

    def test_reset(self):
        # Test deterministic reset
        const_control = 0.0
        max_ind = self.test_env.n_start_data
        rand_int = np.random.randint(max_ind)
        self.test_env.reset(start_ind=rand_int, use_noise=False)
        first_out = self.test_env.step(const_control)
        for k in range(5):
            self.test_env.step(const_control)
        self.test_env.reset(start_ind=rand_int, use_noise=False)
        sec_first_out = self.test_env.step(const_control)
        assert np.allclose(first_out[0], sec_first_out[0]), "State output not correct!"
        assert first_out[1] == sec_first_out[1], "Rewards not correct!"
        assert first_out[2] == sec_first_out[2], "Episode termination not correct!"
