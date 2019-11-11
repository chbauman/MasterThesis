from unittest import TestCase

import numpy as np

from util.numerics import has_duplicates, split_arr, move_inds_to_back, find_rows_with_nans, nan_array_equal, \
    extract_streak, cut_data, find_all_streaks, find_disjoint_streaks, prepare_supervised_control


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

    pass
