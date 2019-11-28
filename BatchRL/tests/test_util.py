import os
import random
from unittest import TestCase

import numpy as np

from agents.agents_heuristic import ConstActionAgent
from data_processing.dataset import dataset_data_path
from dynamics.base_hyperopt import hop_path
from dynamics.recurrent import RNN_TEST_DATA_NAME
from tests.test_data import SYNTH_DATA_NAME
from util.numerics import has_duplicates, split_arr, move_inds_to_back, find_rows_with_nans, nan_array_equal, \
    extract_streak, cut_data, find_all_streaks, find_disjoint_streaks, prepare_supervised_control, npf32, align_ts, \
    num_nans, find_longest_streak, mse, save_performance, mae, max_abs_err, check_shape, save_performance_extended
from util.util import rem_first, tot_size, scale_to_range, linear_oob_penalty, make_param_ext, CacheDecoratorFactory, \
    np_dt_to_str, str_to_np_dt, day_offset_ts, fix_seed, to_list, rem_dirs, split_desc_units, create_dir, yeet, \
    dynamic_model_dir, get_metrics_eval_save_name_list
from util.visualize import plot_dir, plot_reward_details, model_plot_path, rl_plot_path


# Define and create directory for test files.
TEST_DIR = os.path.join(plot_dir, "Test")  #: Directory for test output.
create_dir(TEST_DIR)


class TestNumerics(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define some index arrays
        self.ind_arr = np.array([1, 2, 3, 4, 2, 3, 0], dtype=np.int32)
        self.ind_arr_no_dup = np.array([1, 2, 4, 3, 0], dtype=np.int32)

        # Define data arrays
        self.data_array = np.array([
            [1.0, 1.0, 2.0],
            [2.0, 2.0, 5.0],
            [2.0, 2.0, 5.0],
            [3.0, -1.0, 2.0]])
        self.data_array_with_nans = np.array([
            [1.0, np.nan, 2.0],
            [2.0, 2.0, 5.0],
            [2.0, 2.0, 5.0],
            [2.0, 2.0, np.nan],
            [2.0, np.nan, np.nan],
            [2.0, 2.0, 5.0],
            [2.0, 2.0, 5.0],
            [3.0, -1.0, 2.0]])

        # Define bool vectors
        self.bool_vec = np.array([True, False, False, False, True, False, False, True])
        self.bool_vec_2 = np.array([True, False, False, False, False, True, False, False, False, False, True])
        self.bool_vec_3 = np.array([True, False, False, False])

        # Sequence data
        self.sequences = np.array([
            [[1, 2, 3],
             [1, 2, 3],
             [2, 3, 4]],
            [[3, 2, 3],
             [1, 2, 3],
             [4, 3, 4]],
        ])
        self.c_inds = np.array([1])

    def test_shape_check(self):
        self.assertEqual(check_shape(self.bool_vec, (-1,)), True)
        self.assertEqual(check_shape(self.sequences, (2, 3, 3)), True)
        with self.assertRaises(ValueError):
            check_shape(self.sequences, (2, 5, 3), "test")

    def test_has_duplicates(self):
        self.assertTrue(has_duplicates(self.ind_arr) and not has_duplicates(self.ind_arr_no_dup),
                        "Implementation of has_duplicates contains errors!")

    def test_array_splitting(self):
        # Test array splitting
        d1, d2, n = split_arr(self.data_array, 0.1)
        d1_exp = self.data_array[:3]
        self.assertTrue(np.array_equal(d1, d1_exp) and n == 3,
                        "split_arr not working correctly!!")

    def test_find_nans(self):
        # Test finding rows with nans
        nans_bool_arr = find_rows_with_nans(self.data_array_with_nans)
        nans_exp = np.array([True, False, False, True, True, False, False, False])
        self.assertTrue(np.array_equal(nans_exp, nans_bool_arr),
                        "find_rows_with_nans not working correctly!!")

    def test_streak_extract(self):
        # Test last streak extraction
        d1, d2, n = extract_streak(self.data_array_with_nans, 1, 1)
        d2_exp = self.data_array_with_nans[6:8]
        d1_exp = self.data_array_with_nans[:6]
        if not nan_array_equal(d2, d2_exp) or n != 7 or not nan_array_equal(d1, d1_exp):
            raise AssertionError("extract_streak not working correctly!!")

    def test_longest_sequence(self):
        # Test find_longest_streak
        ex_1 = (0, 1)
        ls_first = find_longest_streak(self.bool_vec, last=False)
        ls_last = find_longest_streak(self.bool_vec, last=True)
        self.assertEqual(ls_first, ex_1, "find_longest_streak incorrect!")
        self.assertEqual(ls_last, (7, 8), "find_longest_streak incorrect!")
        another_bool = np.array([0, 1, 1, 1, 0, 1, 0], dtype=np.bool)
        ls_last = find_longest_streak(another_bool, last=True)
        self.assertEqual(ls_last, (1, 4), "find_longest_streak incorrect!")
        one_only = np.array([1, 1, 1, 1])
        with self.assertRaises(ValueError):
            find_longest_streak(one_only, last=True, seq_val=0)

    def test_seq_cutting(self):
        # Test sequence cutting
        cut_dat_exp = np.array([
            self.data_array_with_nans[1:3],
            self.data_array_with_nans[5:7],
            self.data_array_with_nans[6:8],
        ])
        c_dat, inds = cut_data(self.data_array_with_nans, 2)
        inds_exp = np.array([1, 5, 6])
        if not np.array_equal(c_dat, cut_dat_exp) or not np.array_equal(inds_exp, inds):
            raise AssertionError("cut_data not working correctly!!")

    def test_streak_finding(self):
        streaks = find_all_streaks(self.bool_vec, 2)
        s_exp = np.array([1, 2, 5])
        if not np.array_equal(s_exp, streaks):
            raise AssertionError("find_all_streaks not working correctly!!")

    def test_disjoint_streak_finding(self):
        # Test find_disjoint_streaks
        s_exp = np.array([1, 2, 5])
        dis_s = find_disjoint_streaks(self.bool_vec, 2, 1)
        if not np.array_equal(dis_s, s_exp):
            raise AssertionError("find_disjoint_streaks not working correctly!!")
        dis_s = find_disjoint_streaks(self.bool_vec_2, 2, 2, 1)
        if not np.array_equal(dis_s, np.array([2, 6])):
            raise AssertionError("find_disjoint_streaks not working correctly!!")
        dis_s = find_disjoint_streaks(self.bool_vec_2, 2, 2, 0)
        if not np.array_equal(dis_s, np.array([1, 7])):
            raise AssertionError("find_disjoint_streaks not working correctly!!")
        dis_s = find_disjoint_streaks(self.bool_vec_3, 2, 2, 0)
        if not np.array_equal(dis_s, np.array([1])):
            raise AssertionError("find_disjoint_streaks not working correctly!!")

    def test_supervision_prep(self):
        # Test prepare_supervised_control
        in_arr_exp = np.array([
            [[1, 3, 2],
             [1, 3, 3]],
            [[3, 3, 2],
             [1, 3, 3]],
        ])
        out_arr_exp = np.array([
            [2, 4],
            [4, 4],
        ])
        in_arr, out_arr = prepare_supervised_control(self.sequences, self.c_inds, False)
        if not np.array_equal(in_arr, in_arr_exp) or not np.array_equal(out_arr, out_arr_exp):
            raise AssertionError("Problems encountered in prepare_supervised_control")

    def test_move_inds(self):
        # Test move_inds_to_back
        arr = np.arange(5)
        inds = [1, 3]
        exp_res = np.array([0, 2, 4, 1, 3])
        np.array_equal(move_inds_to_back(arr, inds), exp_res), "move_inds_to_back not working!"

    def test_fix_seed(self):
        # Tests the seed fixing.
        max_int = 1000
        fix_seed()
        a = random.randint(-max_int, max_int)
        arr = np.random.normal(0.0, 1.0, 10)
        fix_seed()
        self.assertEqual(a, random.randint(-max_int, max_int))
        self.assertTrue(np.array_equal(arr, np.random.normal(0.0, 1.0, 10)))

    def test_npf32(self):
        sh = (2, 3)
        arr = npf32(sh)
        self.assertEqual(arr.shape, sh)
        val = 3.0
        arr2 = npf32(sh, fill=val)
        self.assertTrue(np.all(arr2 == val))

    def test_align(self):
        # Test data
        t_i1 = '2019-01-01 00:00:00'
        t_i2 = '2019-01-01 00:30:00'
        dt = 15
        ts_1 = np.array([1, 2, 2, 2, 3, 3], dtype=np.float32)
        ts_2 = np.array([2, 3, 3], dtype=np.float32)
        msg = "align_ts not correct!!"

        # Do tests
        test1, t_init1 = align_ts(ts_1, ts_2, t_i1, t_i2, dt)
        exp1 = npf32((6, 2), fill=np.nan)
        exp1[:, 0] = ts_1
        exp1[2:5, 1] = ts_2
        self.assertEqual(t_init1, t_i1, msg)
        self.assertTrue(nan_array_equal(exp1, test1), msg=msg)

        sh_large = (8, 2)
        test2, _ = align_ts(ts_2, ts_1, t_i1, t_i2, dt)
        self.assertEqual(test2.shape, sh_large)

        test3, _ = align_ts(ts_1, ts_1, t_i1, t_i2, dt)
        self.assertEqual(test3.shape, sh_large)
        exp3 = npf32(sh_large, fill=np.nan)
        exp3[:6, 0] = ts_1
        exp3[2:, 1] = ts_1
        self.assertTrue(nan_array_equal(exp3, test3), msg=msg)

        test4, _ = align_ts(ts_1, ts_1, t_i2, t_i1, dt)
        self.assertEqual(test4.shape, sh_large)
        exp4 = npf32(sh_large, fill=np.nan)
        exp4[:6, 1] = ts_1
        exp4[2:, 0] = ts_1
        self.assertTrue(nan_array_equal(exp4, test4), msg=msg)

    def test_num_nans(self):
        sh = (2, 3)
        arr1 = npf32(sh, fill=np.nan)
        self.assertEqual(num_nans(arr1), tot_size(sh), "num_nans incorrect!")

    def test_save_performance(self):
        n, n_f = 4, 2
        inds = range(n)
        np_arr = np.ones((n_f, 3, n))
        f_names = [os.path.join(TEST_DIR, f"tsp_{i}.txt") for i in range(n_f + 1)]
        save_performance(np_arr, inds, f_names)

    def test_save_performance_extended(self):
        n, n_f = 4, 2
        inds = range(n)
        met_list = ["met1", "met2"]
        n_metrics = len(met_list)
        np_arr = np.ones((n_f, 3, n_metrics, n))
        f_names = [os.path.join(TEST_DIR, f"test_extended_{i}.txt") for i in range(n_f + 1)]
        save_performance_extended(np_arr, inds, f_names, met_list)
    pass


class TestMetrics(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define some index arrays
        self.a1 = np.array([1, 2])
        self.a2 = np.array([1, 1])
        self.a3 = np.array([-1, -1])

    def test_mse(self):
        self.assertAlmostEqual(mse(self.a1, self.a2), 0.5, msg="mse incorrect")
        self.assertAlmostEqual(mse(self.a3, self.a2), 4.0, msg="mse incorrect")

    def test_max_abs_err(self):
        self.assertAlmostEqual(max_abs_err(self.a1, self.a2), 1.0, msg="max_abs_err incorrect")
        self.assertAlmostEqual(max_abs_err(self.a3, self.a2), 2.0, msg="mae incorrect")

    def test_mae(self):
        self.assertAlmostEqual(mae(self.a1, self.a2), 0.5, msg="mae incorrect")
        self.assertAlmostEqual(mae(self.a3, self.a2), 2.0, msg="mae incorrect")


class TestUtil(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define data
        self.dt1 = np.datetime64('2000-01-01T00:00', 'm')
        self.dt3 = np.datetime64('2000-01-01T22:45', 'm')
        self.n_mins = 15
        self.t_init_str = np_dt_to_str(self.dt3)

    class Dummy:
        def __init__(self):
            # self.mutable_fun = CacheDecoratorFactory()(self.mutable_fun)
            pass

        @CacheDecoratorFactory()
        def fun(self, n: int, k: int):
            return n + k * k

        @CacheDecoratorFactory()
        def mutable_fun(self, n: int, k: int):
            return [n, k]

    def test_cache_decorator(self):
        try:
            d = self.Dummy()
            assert d.fun(1, k=3) == 10
            assert d.fun(2, 3) == 11
            assert d.fun(1, k=4) == 10
            list_1_1 = d.mutable_fun(1, 1)
            assert d.mutable_fun(1, 2) == list_1_1
            list_1_1[0] = 0
            assert list_1_1 == d.mutable_fun(1, 5)
            # d2 = Dummy()
            # assert d2.mutable_fun(1, 2) == [1, 2]  # It fails here!
            assert d.fun(2, 7) == 11
            assert [4, 7] == d.mutable_fun(4, 7)
        except AssertionError as e:
            print("Cache Decorator Test failed!!")
            raise e
        except Exception as e:
            raise AssertionError("Some error happened: {}".format(e))

    def test_yeet(self):
        self.assertRaises(ValueError, yeet)

    def test_to_list(self):
        self.assertEqual([1], to_list(1))
        self.assertEqual([1], to_list([1]))
        self.assertEqual(["1"], to_list("1"))

    def test_rem_first(self):
        # Test rem_first
        self.assertEqual(rem_first((1, 2, 3)), (2, 3),
                         "rem_first not working correctly!")
        self.assertEqual(rem_first((1, 2)), (2,),
                         "rem_first not working correctly!")

    def test_tot_size(self):
        # Test tot_size
        msg = "tot_size not working!"
        self.assertEqual(tot_size((1, 2, 3)), 6, msg)
        self.assertEqual(tot_size((0, 1)), 0, msg)
        self.assertEqual(tot_size(()), 0, msg)

    def test_scale_to_range(self):
        # Test scale_to_range
        assert np.allclose(scale_to_range(1.0, 2.0, [-1.0, 1.0]), 0.0), "scale_to_range not working correctly!"
        assert np.allclose(scale_to_range(1.0, 2.0, [0.0, 2.0]), 1.0), "scale_to_range not working correctly!"

    def test_lin_oob_penalty(self):
        # Test linear_oob_penalty
        assert np.allclose(linear_oob_penalty(1.0, [-1.0, 1.0]), 0.0), "linear_oob_penalty not working correctly!"
        assert np.allclose(linear_oob_penalty(5.0, [0.0, 2.0]), 3.0), "linear_oob_penalty not working correctly!"
        assert np.allclose(linear_oob_penalty(-5.0, [0.0, 2.0]), 5.0), "linear_oob_penalty not working correctly!"

    def test_make_param_ext(self):
        # Test make_param_ext
        res1 = make_param_ext([("a", 4), ("b", [1, 2])])
        assert res1 == "_a4_b1-2", f"make_param_ext not implemented correctly: {res1}"
        assert make_param_ext(
            [("a", 4), ("b", None), ("c", False)]) == "_a4", "make_param_ext not implemented correctly!"
        res3 = make_param_ext([("a", 4.1111111), ("b", True)])
        assert res3 == "_a4.111_b", f"make_param_ext not implemented correctly: {res3}"

    def test_time_conversion(self):
        # Test time conversion
        assert str_to_np_dt(np_dt_to_str(self.dt1)) == self.dt1, "Time conversion not working"

    def test_day_offset_ts(self):
        # Test day_offset_ts
        n_ts = day_offset_ts(self.t_init_str, self.n_mins)
        assert n_ts == 5, "Fuck you!!"

    def test_file_and_dir_removal(self):
        id_str = "Test_Test_test_2519632984160348"

        # Create some files and dirs.
        f_name = id_str + ".txt"
        with open(f_name, "w") as f:
            f.write("Test")
        d_name1 = id_str + "_dir"
        d_name2 = "Test_" + id_str + "_dir"
        os.mkdir(d_name1)
        os.mkdir(d_name2)

        # Test removal
        rem_dirs(".", pat=id_str)
        self.assertFalse(os.path.isfile(f_name))
        self.assertFalse(os.path.isdir(d_name1))
        self.assertTrue(os.path.isdir(d_name2))
        rem_dirs(".", pat=id_str, anywhere=True)
        self.assertFalse(os.path.isdir(d_name2))

    def test_desc_split(self):
        d1 = "desc [1]"
        p1, p2 = split_desc_units(d1)
        self.assertEqual(p1, "desc ")
        self.assertEqual(p2, "[1]")
        self.assertEqual("hoi", split_desc_units("hoi")[0])

    def test_file_name_generation(self):
        lst = ["test", "foo"]
        dt = 100
        name_list = get_metrics_eval_save_name_list(lst, dt)
        self.assertEqual(len(name_list), len(lst) + 1)


class TestPlot(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.test_plot_dir = os.path.join(plot_dir, "Test")

    def get_test_path(self, base_name: str):
        create_dir(self.test_plot_dir)
        return os.path.join(self.test_plot_dir, base_name)

    def make_bar_plot(self, n_ag, n_rew, n_steps):
        descs = [f"Rew_{i}" for i in range(n_rew - 1)]

        class Dummy:
            nb_actions = 1

        agents = [ConstActionAgent(Dummy(), 1.0 * i) for i in range(n_ag)]
        rewards = np.random.normal(2.0, 1.0, (n_ag, n_steps, n_rew))
        test_path = self.get_test_path(f"test_reward_bar_{n_ag}_{n_rew}_{n_steps}")
        plot_reward_details(agents, rewards, test_path, descs)

    def test_reward_bar_plot(self):
        self.make_bar_plot(3, 4, 5)
        self.make_bar_plot(6, 2, 10)
        self.make_bar_plot(1, 8, 3)

    pass


def cleanup_test_data(verbose: int = 0):
    """Removes all test folders and files."""

    if verbose:
        print("Cleaning up some test files...")

    # Remove files
    rem_dirs(model_plot_path, SYNTH_DATA_NAME)
    rem_dirs(model_plot_path, RNN_TEST_DATA_NAME)
    rem_dirs(dynamic_model_dir, RNN_TEST_DATA_NAME)
    rem_dirs(hop_path, "TestHop", anywhere=True)
    rem_dirs(rl_plot_path, "TestEnv")
    rem_dirs(rl_plot_path, "BatteryTest", anywhere=True)
    rem_dirs(rl_plot_path, "FullTest", anywhere=True)
    rem_dirs(dataset_data_path, "Test", anywhere=True)
    rem_dirs(TEST_DIR, "")
