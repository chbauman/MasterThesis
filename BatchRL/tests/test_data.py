from unittest import TestCase

import numpy as np

from data_processing.dataset import ModelDataView, SeriesConstraint, Dataset
from util.numerics import nan_array_equal
from util.visualize import plot_dataset


def get_test_ds(dat: np.ndarray, c_inds: np.ndarray,
                name: str = "SyntheticTest",
                dt: int = 60 * 12,
                t_init: str = '2019-01-01 12:00:00'):
    """Constructs a test dataset.

    Args:
        dat: The data array.
        c_inds: The control indices.
        name: The name of the dataset.
        dt: Timestep in minutes.
        t_init: Initial time.

    Returns:
        New dataset with dummy descriptions and unscaled data.
    """
    n_series = dat.shape[1]
    descs = np.array([f"Series {i} [unit{i}]" for i in range(n_series)])
    is_sc = np.array([False for _ in range(n_series)])
    sc = np.empty((n_series, 2), dtype=np.float32)
    ds = Dataset(dat, dt, t_init, sc, is_sc, descs, c_inds, name=name)
    return ds


class TestFullRoomEnv(TestCase):
    """Tests the room RL environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define dataset
        dat = np.array([0, 2, 3, 7, 8,
                        1, 3, 4, 8, 9,
                        1, 4, 5, 7, 8,
                        2, 5, 6, 7, 9], dtype=np.float32).reshape((4, -1))
        self.c_inds = np.array([1, 3])
        self.ds = get_test_ds(dat, self.c_inds)
        self.ds.standardize()

        # Dataset containing nans
        self.dat_nan = np.array([0, 2, 3, 7, 8,
                                 1, 3, 4, 8, np.nan,
                                 2, 3, 4, 8, np.nan,
                                 3, 3, 4, 8, np.nan,
                                 4, 4, 5, 7, 8,
                                 5, 4, 5, 7, 8,
                                 6, 4, 5, 7, 8,
                                 7, 4, 5, 7, 8,
                                 8, 4, np.nan, 7, 8,
                                 9, 4, np.nan, 7, 8,
                                 10, 5, 6, 7, 9], dtype=np.float32).reshape((-1, 5))
        ds_nan = get_test_ds(self.dat_nan, self.c_inds, name="SyntheticTest")
        ds_nan.seq_len = 2
        self.ds_nan = ds_nan
        self.ds_nan.val_percent = 0.33
        self.ds_nan.split_data()

        # Create MDV
        self.mdv = ModelDataView(self.ds_nan, "Test", 2, 7)

    def test_saving(self):
        self.ds.save()

    def test_plot(self):
        plot_dataset(self.ds, False, ["Test", "Fuck"])

    def test_model_data_view_1(self):
        mod_dat = self.mdv.get_rel_data()
        self.assertTrue(nan_array_equal(mod_dat, self.dat_nan[2:9]),
                        "Something's fucking wrong with model data view's get_rel_data!!")

    def test_model_data_view_2(self):
        self.mdv.extract_streak(3)
        str_dat, i = self.mdv.extract_streak(3)
        exp_dat = np.array([
            self.dat_nan[5:7],
            self.dat_nan[6:8],
        ])
        if not np.array_equal(str_dat, exp_dat) or not i == 3:
            raise AssertionError("Something in MDVs extract_streak is fucking wrong!!")

    def test_model_data_view_3(self):
        # Test disjoint streak extraction
        dis_dat, dis_inds = self.mdv.extract_disjoint_streaks(2, 1)
        exp_dis = np.array([[
            self.dat_nan[4:6],
            self.dat_nan[5:7],
        ]])
        if not np.array_equal(dis_dat, exp_dis) or not np.array_equal(dis_inds, np.array([2])):
            raise AssertionError("Something in extract_disjoint_streaks is fucking wrong!!")

    def test_split_data(self):
        # Test split_data
        test_dat = self.ds_nan.split_dict['test'].get_rel_data()
        val_dat = self.ds_nan.split_dict['val'].get_rel_data()
        if not nan_array_equal(test_dat, self.dat_nan[7:]):
            raise AssertionError("Something in split_data is fucking wrong!!")
        if not nan_array_equal(val_dat, self.dat_nan[3:7]):
            raise AssertionError("Something in split_data is fucking wrong!!")

    def test_get_day(self):
        # Test get_day
        day_dat, ns = self.ds_nan.get_days('val')
        exp_first_out_dat = np.array([
            [5, 5.0, 8.0],
            [6, 5.0, 8.0],
        ], dtype=np.float32)
        if ns[0] != 4 or not np.array_equal(day_dat[0][1], exp_first_out_dat):
            raise AssertionError("get_days not implemented correctly!!")

    def test_standardize(self):
        self.assertTrue(np.allclose(self.ds.scaling[0][0], 1.0), "Standardizing failed!")

    def test_get_scaling_mul(self):
        # Test get_scaling_mul
        scaling, is_sc = self.ds.get_scaling_mul(0, 3)
        is_sc_exp = np.array([True, True, True])
        sc_mean_exp = np.array([1.0, 1.0, 1.0])
        self.assertTrue(np.array_equal(is_sc_exp, is_sc), "get_scaling_mul failed!")
        self.assertTrue(np.allclose(sc_mean_exp, scaling[:, 0]), "get_scaling_mul failed!")

    def test_transform_c_list(self):
        c_list = [
            SeriesConstraint('interval', np.array([0.0, 1.0])),
            SeriesConstraint(),
            SeriesConstraint(),
            SeriesConstraint(),
            SeriesConstraint(),
        ]
        self.ds.transform_c_list(c_list)
        if not np.allclose(c_list[0].extra_dat[1], 0.0):
            raise AssertionError("Interval transformation failed!")

    def test_index_trafo(self):
        # Specify index tests
        test_list = [
            (np.array([2, 4], dtype=np.int32), np.array([1, 2], dtype=np.int32), self.ds.to_prepared),
            (np.array([2, 3], dtype=np.int32), np.array([1, 4], dtype=np.int32), self.ds.to_prepared),
            (np.array([0, 1, 2, 3, 4], dtype=np.int32), np.array([0, 3, 1, 4, 2], dtype=np.int32), self.ds.to_prepared),
            (np.array([0, 1, 2], dtype=np.int32), np.array([0, 2, 4], dtype=np.int32), self.ds.from_prepared),
            (np.array([2, 3, 4], dtype=np.int32), np.array([4, 1, 3], dtype=np.int32), self.ds.from_prepared),
        ]

        # Run index tests
        for t in test_list:
            inp, sol, fun = t
            out = fun(inp)
            if not np.array_equal(sol, out):
                print("Test failed :(")
                raise AssertionError("Function: {} with input: {} not giving: {} but: {}!!!".format(fun, inp, sol, out))

    pass
