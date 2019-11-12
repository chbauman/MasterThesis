from typing import Dict
from unittest import TestCase

import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope as ho_scope

from data_processing.data import Dataset
from dynamics.base_hyperopt import HyperOptimizableModel
from dynamics.base_model import BaseDynamicsModel, construct_test_ds
from util.numerics import copy_arr_list


class TestModel(BaseDynamicsModel):
    """Dummy dynamics model class for testing.

    Does not fit anything. Works only with datasets
    that have exactly 3 series to predict and one
    control variable series.
    """
    n_prediction: int = 0
    n_pred: int = 3

    def __init__(self,
                 ds: Dataset):
        name = "TestModel"
        super(TestModel, self).__init__(ds, name)
        self.use_AR = False

        if ds.n_c != 1:
            raise ValueError("Only one control variable allowed!!")
        if ds.d - 1 != self.n_pred:
            raise ValueError("Dataset needs exactly 3 non-controllable state variables!")

    def fit(self) -> None:
        pass

    def predict(self, in_data: np.ndarray) -> np.ndarray:

        rel_out_dat = -0.9 * in_data[:, -1, :self.n_pred]

        self.n_prediction += 1
        return rel_out_dat


class TestBaseDynamics(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define dataset
        self.n = 151
        self.ds = construct_test_ds(self.n)
        self.ds_1 = construct_test_ds(self.n, 1)

        # Define models
        from dynamics.const import ConstSeriesTestModel, ConstModel
        self.test_mod = TestModel(self.ds)
        self.test_model_2 = ConstSeriesTestModel(self.ds_1,
                                                 pred_val_list=[0.0, 2.0],
                                                 out_indices=np.array([0, 2], dtype=np.int32),
                                                 in_indices=np.array([1, 2, 3], dtype=np.int32))
        self.test_model_3 = ConstModel(self.ds_1)

        # Compute sizes
        self.n_val = self.n - int((1.0 - self.ds.val_percent) * self.n)
        self.n_train = self.n - 2 * self.n_val
        self.n_train_seqs = self.n_train - self.ds.seq_len + 1
        self.n_streak = 7 * 24 * 60 // self.ds.dt
        self.n_streak_offset = self.n_train_seqs - self.n_streak

        self.dat_in_train, self.dat_out_train, _ = self.ds.get_split("train")

    def test_sizes(self):
        # Check sizes
        self.assertEqual(len(self.dat_in_train), self.n_train_seqs, "Something baaaaad!")
        dat_in, dat_out, n_str = self.ds.get_streak("train")
        self.assertEqual(self.n_streak_offset, n_str, "Streak offset is wrong!!")

    def test_indices(self):
        # Check indices
        self.assertTrue(np.array_equal(np.array([0, 2, 3]), self.test_model_3.out_inds),
                        "Out indices incorrect!")
        self.assertTrue(np.array_equal(np.array([0, 1, 2]), self.test_model_3.p_out_inds),
                        "Prepared out indices incorrect!")
        self.assertTrue(np.array_equal(np.array([0, 1, 2, 3]), self.test_model_3.in_indices),
                        "In indices incorrect!")
        self.assertTrue(np.array_equal(np.array([0, 3, 1, 2]), self.test_model_3.p_in_indices),
                        "Prepared in indices incorrect!")

    def test_analysis(self):
        # Test model analysis
        # TODO: This is slow (~4s), make it faster.
        base_data = np.copy(self.ds_1.data)
        streak = copy_arr_list(self.ds_1.get_streak("train"))
        self.test_model_2.analyze(plot_acf=False, n_steps=(2,))
        streak_after = self.ds_1.get_streak("train")
        self.assertTrue(np.array_equal(base_data, self.ds_1.data), "Data was changed during analysis!")
        for k in range(3):
            self.assertTrue(np.array_equal(streak_after[k], streak[k]), "Streak data was changed during analysis!")


class TestHopTable(HyperOptimizableModel):
    """Example hyperopt class that does not need fitting."""
    name: str = "TestHop"

    def __init__(self, ds: Dataset, base_param: int = 5, h_param_1: int = 0):
        super().__init__(ds, self.name)
        self.bp = base_param
        self.h_param = h_param_1

        self.base_name = self.get_base_name(base_param=base_param)

    def get_space(self) -> Dict:
        hp_space = {
            'h_param_1': ho_scope.int(hp.quniform('n_layers', low=0, high=20, q=1)),
        }
        return hp_space

    @classmethod
    def get_base_name(cls, **kwargs):
        return cls.name + "_" + str(kwargs['base_param'])

    def conf_model(self, hp_sample: Dict) -> 'HyperOptimizableModel':
        new_mod = TestHopTable(self.data, self.bp, **hp_sample)
        return new_mod

    def hyper_objective(self) -> float:
        x = self.h_param
        return -x * x + 12 * x + 15

    def fit(self) -> None:
        pass

    def predict(self, in_data):
        pass


class TestHop(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define dataset
        self.ds = construct_test_ds(20)
        self.n = 3

    def test_save_and_load(self):
        # Init model
        test_hop_mod_4 = TestHopTable(self.ds, 4, 6)
        test_hop_mod_4.optimize(self.n)
        assert len(test_hop_mod_4.param_list) == self.n, "Parameter logging incorrect!"
        best_mod_4 = TestHopTable.from_best_hp(ds=self.ds, base_param=4)
        best_mod_4.optimize(self.n)