from unittest import TestCase

import numpy as np

from agents import agents_heuristic
from dynamics.base_model import construct_test_ds, BaseDynamicsModel
from envs.base_dynamics_env import DynEnv
from tests.test_dynamics import TestModel, ConstTestModelControlled
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
    """Tests the RL environments.

    TODO: Remove files after testing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define dataset
        self.n = 201
        self.test_ds = construct_test_ds(self.n)
        self.test_ds.c_inds = np.array([1])
        sh = self.test_ds.data.shape
        self.test_mod = TestModel(self.test_ds)
        self.n_ts_per_episode = 10
        self.test_env = TestDynEnv(self.test_mod, 10)

        # Another one
        self.test_ds2 = construct_test_ds(self.n)
        self.test_ds2.c_inds = np.array([1])
        self.test_ds2.data = np.arange(sh[0]).reshape((-1, 1)) * np.ones(sh, dtype=np.float32)
        self.test_ds2.split_data()
        self.model_2 = ConstTestModelControlled(self.test_ds2)
        self.test_env2 = TestDynEnv(self.model_2, 10)

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

    def test_step(self):
        for action in [1.0, 0.0]:
            init_state = self.test_env2.reset(0)
            next_state, rew, ep_over, _ = self.test_env2.step(action)
            self.assertTrue(np.allclose(init_state + action, next_state), "Step contains a bug!")
            for k in range(3):
                prev_state = np.copy(next_state)
                next_state, rew, ep_over, _ = self.test_env2.step(action)
                self.assertTrue(np.allclose(prev_state + action, next_state), "Step contains a bug!")

    def test_agent_analysis(self):
        # Test agent analysis
        const_ag_1 = agents_heuristic.ConstHeating(self.test_env, 0.0)
        const_ag_2 = agents_heuristic.ConstHeating(self.test_env, 1.0)
        self.test_env.analyze_agents_visually([const_ag_1, const_ag_2])

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
