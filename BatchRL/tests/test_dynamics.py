from typing import Dict, Union, List
from unittest import TestCase

import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope as ho_scope

from data_processing.dataset import Dataset
from dynamics.base_hyperopt import HyperOptimizableModel
from dynamics.base_model import BaseDynamicsModel, construct_test_ds
from dynamics.battery_model import BatteryModel
from dynamics.composite import CompositeModel
from dynamics.const import NoDisturbanceModel, ConstModel
from util.numerics import copy_arr_list
from util.util import Num, tot_size


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

    def fit(self, verbose: int = 0) -> None:
        if verbose:
            print("Not fitting anything!")

    def predict(self, in_data: np.ndarray) -> np.ndarray:

        rel_out_dat = -0.9 * in_data[:, -1, :self.n_pred] + in_data[:, -1, -1].reshape(-1, 1)

        self.n_prediction += 1
        return rel_out_dat


class ConstSeriesTestModel(NoDisturbanceModel):
    """Test model that predicts a possibly different constant value for each series.

    Can take any number of input series and output series.
    """

    def __init__(self,
                 ds: Dataset,
                 pred_val_list: Union[Num, List[Num]],
                 predict_input_check: np.ndarray = None,
                 **kwargs):
        """Constructor

        If `pred_val_list` is a number, it will be used as prediction
        for all series that are predicted.

        Args:
            ds: The dataset.
            pred_val_list: The list with the values that will be predicted.
            predict_input_check: If not None, the input series have to have these values.
            **kwargs: Kwargs for the base class, e.g. `out_indices` or `in_indices`.
        """
        name = f"ConstModel"
        super().__init__(ds, name, **kwargs)

        # Check if the list contains the right number of elements if it is a list.
        if isinstance(pred_val_list, List):
            if not len(pred_val_list) == self.n_pred:
                raise ValueError("Not the right number of values specified!")

        self.predict_input_check = predict_input_check
        self.values = np.array(pred_val_list)
        self.use_AR = False

    def fit(self, verbose: int = 0) -> None:
        if verbose:
            print("Not fitting anything!")

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Predicts a constant value for each series.

        Raises:
            AssertionError: If `predict_input_check` is not None and
                the input does not match.
        """
        in_sh = in_data.shape
        if self.predict_input_check is not None:
            pc = self.predict_input_check
            assert len(pc) == in_sh[-1], "Predictions do not have the right shape!"
            exp_inp = np.ones(in_sh) * pc
            assert np.array_equal(exp_inp, in_data), "Input incorrect!!!"
        preds = np.ones((in_sh[0], self.n_pred), dtype=np.float32) * self.values
        return preds


class ConstTestModel(ConstModel, NoDisturbanceModel):
    """Const Test model without a disturbance.

    Same as `ConstModel`, but no disturbance.
    """

    name: str = "NaiveNoDisturb"  #: Base name of model.


class ConstTestModelControlled(ConstTestModel):
    """Same as `ConstTestModel` but adds the control input to predictions."""

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        action = in_data[:, -1, -1]
        return in_data[:, -1, :self.n_pred] + action


class TestBaseDynamics(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define dataset
        self.n = 151
        self.ds = construct_test_ds(self.n)
        self.ds_1 = construct_test_ds(self.n, 1)

        # Define models
        from dynamics.const import ConstModel
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
        base_data = np.copy(self.ds_1.data)
        streak = copy_arr_list(self.ds_1.get_streak("train"))
        self.test_model_2.analyze(plot_acf=False, n_steps=(2,), verbose=False)
        streak_after = self.ds_1.get_streak("train")
        self.assertTrue(np.array_equal(base_data, self.ds_1.data), "Data was changed during analysis!")
        for k in range(3):
            self.assertTrue(np.array_equal(streak_after[k], streak[k]), "Streak data was changed during analysis!")

    def test_const_series_test_model(self):
        # Test the test model
        n_feat = self.ds_1.d - self.ds_1.n_c
        preds = list(range(n_feat))
        c_mod = ConstSeriesTestModel(self.ds_1, preds)
        exp_out = np.copy(preds).reshape((1, -1))
        self.assertTrue(np.array_equal(exp_out, c_mod.predict(exp_out)),
                        "ConstSeriesTestModel implemented incorrectly!")


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

    def fit(self, verbose: int = 0) -> None:
        if verbose:
            print("Not fitting anything!")

    def predict(self, in_data):
        pass


class TestHop(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define dataset
        self.ds = construct_test_ds(20)
        self.n = 2

    def test_save_and_load(self):
        # Init model
        test_hop_mod_4 = TestHopTable(self.ds, 4, 6)
        test_hop_mod_4.optimize(self.n, verbose=0)
        assert len(test_hop_mod_4.param_list) == self.n, "Parameter logging incorrect!"
        best_mod_4 = TestHopTable.from_best_hp(ds=self.ds, base_param=4)
        best_mod_4.optimize(self.n, verbose=0)


class TestComposite(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define datasets
        self.n = 200
        self.dataset = construct_test_ds(self.n)
        self.dataset.data = np.ones((self.n, 4)) * np.arange(4)
        self.dataset.split_data()
        self.ds_1 = construct_test_ds(self.n, c_series=1)
        self.ds_1.data = np.ones((self.n, 4)) * np.arange(4)
        self.ds_1.split_data()

        # Define indices
        self.inds_1 = np.array([1], dtype=np.int32)
        self.inds_2 = np.array([2], dtype=np.int32)
        self.inds_02 = np.array([0, 2], dtype=np.int32)
        self.inds_123 = np.array([1, 2, 3], dtype=np.int32)

        # Define output data
        self.sh = (2, 5, 4)
        self.sh_out = (2, 3)
        self.sh_tot = tot_size(self.sh)
        self.in_data = np.reshape(np.arange(self.sh_tot), self.sh)
        self.in_data_2 = np.ones(self.sh, dtype=np.float32) * np.arange(4)

    def test_first(self):
        m1 = ConstSeriesTestModel(self.dataset, pred_val_list=1.0,
                                  out_indices=self.inds_02)
        m2 = ConstSeriesTestModel(self.dataset, pred_val_list=2.0, out_indices=self.inds_1)
        mc1 = CompositeModel(self.dataset, [m1, m2], new_name="CompositeTest1")
        exp_out_data = np.ones((2, 3), dtype=np.float32)
        exp_out_data[:, 1] = 2.0
        assert np.array_equal(exp_out_data, mc1.predict(self.in_data)), "Composite model prediction wrong!"

    def test_second(self):
        m3 = ConstSeriesTestModel(self.dataset,
                                  pred_val_list=[1.0],
                                  out_indices=self.inds_1,
                                  in_indices=self.inds_1)
        m4 = ConstSeriesTestModel(self.dataset,
                                  pred_val_list=[0.0, 2.0],
                                  out_indices=self.inds_02,
                                  in_indices=self.inds_123)
        mc2 = CompositeModel(self.dataset, [m3, m4], new_name="CompositeTest2")

        exp_out_2 = np.ones((2, 3), dtype=np.float32) * np.arange(3.0)
        assert np.array_equal(exp_out_2, mc2.predict(self.in_data)), "Composite model prediction wrong!"
        assert np.array_equal(exp_out_2, mc2.predict(self.in_data_2)), "Composite model prediction wrong!"

    def test_third(self):
        m5 = ConstTestModel(self.dataset, pred_inds=self.inds_1)
        m6 = ConstTestModel(self.dataset,
                            pred_inds=self.inds_02,
                            in_indices=self.inds_123)
        mc3 = CompositeModel(self.dataset, [m5, m6], new_name="CompositeTest3")

        exp_out_3 = np.ones((2, 3), dtype=np.float32) * np.arange(3.0)
        exp_out_3[:, 0] = 1.0
        assert np.array_equal(exp_out_3, mc3.predict(self.in_data_2)), "Composite model prediction wrong!"

    def test_fourth(self):
        m7 = ConstTestModel(self.ds_1,
                            pred_inds=self.inds_02,
                            in_indices=np.array([1, 2], dtype=np.int32))
        m8 = ConstTestModel(self.ds_1,
                            pred_inds=np.array([3], dtype=np.int32),
                            in_indices=self.inds_1)
        mc4 = CompositeModel(self.ds_1, [m7, m8], new_name="CompositeTest4")

        exp_out_4 = np.ones(self.sh_out) * np.array([3, 1, 3])
        act_out = mc4.predict(self.in_data_2)
        assert np.array_equal(exp_out_4, act_out), "Composite model prediction wrong!"

    def test_analyze(self):
        # Test analyze()
        m9 = ConstSeriesTestModel(self.ds_1,
                                  pred_val_list=[2.0],
                                  predict_input_check=self.inds_2,
                                  out_indices=self.inds_2,
                                  in_indices=self.inds_2)
        m10 = ConstSeriesTestModel(self.ds_1,
                                   pred_val_list=[0.0, 3.0],
                                   predict_input_check=self.inds_123,
                                   out_indices=np.array([0, 3], dtype=np.int32),
                                   in_indices=self.inds_123)
        mc5 = CompositeModel(self.ds_1, [m9, m10], new_name="CompositeTest4")
        mc5.analyze(plot_acf=False, verbose=False, n_steps=(2,))


def get_test_battery_model(n: int = 150,
                           noise: float = 1.0,
                           linear: bool = True):
    """Constructs a fitted battery model."""
    data = np.ones((n, 2)) * np.array([50, 0.0])
    ds2 = construct_test_ds(n, c_series=1, n_feats=2)
    ds2.data = data
    if noise > 0.0:
        ds2.data += np.random.normal(0.0, noise, (n, 2))
        ds2.standardize()
    ds2.split_data()
    m2 = BatteryModel(ds2)
    m2.name = "BatteryTest"
    par3 = 0 if linear else 1
    m2.params = np.array([0, 1, par3])
    return m2


class TestBattery(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define models
        n = 150
        self.m = get_test_battery_model(n, noise=0.0, linear=True)
        self.m2 = get_test_battery_model(n, noise=1.0, linear=False)

    def test_eval(self):
        val = 5.0
        self.assertAlmostEqual(self.m._eval_at(val), val, msg="_eval_at sucks!")
        self.assertAlmostEqual(self.m2._eval_at(val), 2 * val, msg="_eval_at sucks!")
        self.assertAlmostEqual(self.m2._eval_at(-val), -val, msg="_eval_at sucks!")

    def test_battery(self):
        n = 10
        in_data = np.ones((n, 1, 2))
        soc = np.arange(n) + 30
        in_data[:, 0, 0] = soc
        p = self.m.predict(in_data)
        self.assertTrue(np.array_equal(p.reshape(n), soc + 1))

    def test_nonlinear_battery(self):
        n = 5
        in_data = np.ones((n, 1, 2))
        soc = np.arange(n) + 30
        in_data[:, 0, 0] = soc
        p = self.m2.predict(in_data)
        self.assertTrue(np.array_equal(p.reshape(n), soc + 2))

    pass
