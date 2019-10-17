import os
from typing import Union, Optional, List, Tuple, Any, Sequence

import numpy as np

import scipy.optimize.nnls

from datetime import datetime

#######################################################################################################
# Typing

# Type for general number
Num = Union[int, float]

# Type for general array, including 0-D ones, i.e. single numbers
Arr = Union[Num, np.ndarray]


#######################################################################################################
# Python stuff

def get_if_nnone(lst: Sequence, indx: int, default=None):
    """
    Returns a list element if list is not None,
    else the default value.

    :param lst: List of elements or None
    :param indx: List index.
    :param default: Default return value
    :return: List element at position indx if lst is not None, else default.
    """
    return default if lst is None else lst[indx]


def apply(list_or_el, fun):
    """
    Applies the function fun to each element of list_or_el
    if it is a list, else it is applied directly to list_or_el.

    :param list_or_el: List of elements or single element.
    :param fun: Function to apply to elements.
    :return: List or element with function applied.
    """
    if isinstance(list_or_el, list):
        return [fun(k) for k in list_or_el]
    else:
        return fun(list_or_el)


def repl(el, n: int) -> List:
    """
    Constructs a list with n equal elements 'el'.
    If el is not a primitive type, then it might
    give a list with views on el.

    :param el: Element to repeat.
    :param n: Number of times.
    :return: New list with elements.
    """
    return [el for _ in range(n)]


def b_cast(l_or_el, n: int) -> List:
    """
    Checks if 'l_or_el' is a list or not.
    If not returns a list with 'n' repeated elements 'l_or_el'.

    :param l_or_el: List of elements or element.
    :param n: Length of list.
    :return: list
    """
    if isinstance(l_or_el, list):
        if len(l_or_el) == n:
            return l_or_el
        raise ValueError("Broadcast failed!!!")
    return repl(l_or_el, n)


#######################################################################################################
# Numerical stuff

def has_duplicates(arr: np.ndarray) -> bool:
    """
    Returns true if arr contains duplicate values else
    False.

    :param arr: Array to check for duplicates.
    :return: Whether it has duplicates.
    """
    m = np.zeros_like(arr, dtype=bool)
    m[np.unique(arr, return_index=True)[1]] = True
    return np.sum(~m) > 0


def arr_eq(arr1: Arr, arr2: Arr) -> bool:
    """
    Tests if two arrays have all equal elements.
    DEPRECATED: Use np.array_equal

    :param arr1: First array.
    :param arr2: Array to compare with.
    :return: Whether all the elements are the same.
    """
    return np.sum(arr1 != arr2) == 0


def fit_linear_1d(x: np.ndarray, y: np.ndarray, x_new: Optional[np.ndarray] = None):
    """
    Fit a linear model y = c * x + m.
    Returns coefficients m and c. If x_new
    is not None, returns the evaluated linear
    fit at x_new.
    """

    n = x.shape[0]
    ls_mat = np.empty((n, 2), dtype=np.float32)
    ls_mat[:, 0] = 1
    ls_mat[:, 1] = x
    m, c = np.linalg.lstsq(ls_mat, y, rcond=None)[0]
    if x_new is None:
        return [m, c]
    else:
        return c * x_new + m


def fit_linear_bf_1d(x: np.ndarray, y: np.ndarray, b_fun, offset: bool = False) -> np.ndarray:
    """
    Fits a linear model y = \alpha^T f(x).
    TODO: implement with offset!
    """

    if offset:
        raise NotImplementedError("Not implemented with offset.")

    # Get shapes
    dummy = b_fun(0.0)
    d = dummy.shape[0]
    n = x.shape[0]

    # Fill matrix
    ls_mat = np.empty((n, d), dtype=np.float32)
    for ct, x_el in enumerate(x):
        ls_mat[ct, :] = b_fun(x_el)

    # Solve and return
    coeffs = np.linalg.lstsq(ls_mat, y, rcond=None)[0]
    return coeffs


def get_shape1(arr: np.ndarray) -> int:
    """
    Returns the shape of the second dimension
    of a matrix. If it is a vector returns 1.
    """
    s = arr.shape
    if len(s) < 2:
        return 1
    else:
        return s[1]


def align_ts(ts_1: np.ndarray, ts_2: np.ndarray, t_init1, t_init2, dt):
    """
    Aligns the two time series with given initial time
    and constant timestep by padding by np.nan.
    """

    # Get shapes
    n_1 = ts_1.shape[0]
    n_2 = ts_2.shape[0]
    d_1 = get_shape1(ts_1)
    d_2 = get_shape1(ts_2)

    # Compute relative offset
    interv = np.timedelta64(dt, 'm')
    ti1 = datetime_to_npdatetime(string_to_dt(t_init2))
    ti2 = datetime_to_npdatetime(string_to_dt(t_init1))

    # Ugly bug-fix
    if ti1 < ti2:
        dout, t = align_ts(ts_2, ts_1, t_init2, t_init1, dt)
        dout_real = np.copy(dout)
        dout_real[:, :d_1] = dout[:, d_2:]
        dout_real[:, d_1:] = dout[:, :d_2]
        return dout_real, t

    offset = np.int(np.round((ti2 - ti1) / interv))

    # Compute length
    out_len = np.maximum(n_2 - offset, n_1)
    start_s = offset <= 0
    out_len += offset if not start_s else 0
    out = np.empty((out_len, d_1 + d_2), dtype=ts_1.dtype)
    out.fill(np.nan)

    # Copy over
    t1_res = np.reshape(ts_1, (n_1, d_1))
    t2_res = np.reshape(ts_2, (n_2, d_2))
    if not start_s:
        out[:n_2, :d_2] = t2_res
        out[offset:(offset + n_1), d_2:] = t1_res
        t_init_out = t_init2
    else:
        out[:n_1, :d_1] = t1_res
        out[-offset:(-offset + n_2), d_1:] = t2_res
        t_init_out = t_init1

    return out, t_init_out


def trf_mean_and_std(ts: Arr, mean_and_std: Sequence, remove: bool = True) -> Arr:
    """
    Adds or removes mean and std from time series.

    Args:
        ts: The time series.
        mean_and_std: Mean and standard deviance.
        remove: Whether to remove or to add.

    Returns:
        New time series with mean and std removed or added.
    """

    if remove:
        return rem_mean_and_std(ts, mean_and_std)
    else:
        return add_mean_and_std(ts, mean_and_std)


def add_mean_and_std(ts: Arr, mean_and_std: Sequence) -> Arr:
    """
    Transforms the data back to having mean
    and std as specified.
    """
    if len(mean_and_std) < 2:
        raise ValueError("Invalid value for mean_and_std")
    return ts * mean_and_std[1] + mean_and_std[0]


def rem_mean_and_std(ts: Arr, mean_and_std: Sequence) -> Arr:
    """
    Whitens the data with known mean and standard deviation.

    :param ts: Data to be whitened.
    :param mean_and_std: Container of the mean and the std.
    :return: Whitened data.
    """
    if len(mean_and_std) < 2:
        raise ValueError("Invalid value for mean_and_std")
    if mean_and_std[1] == 0:
        raise ZeroDivisionError("Standard deviation cannot be 0")
    return (ts - mean_and_std[0]) / mean_and_std[1]


def check_in_range(arr: np.ndarray, low: Num, high: Num) -> bool:
    """
    Returns true if all elements in arr are in
    range [low, high) else false.
    """
    if arr.size == 0:
        return True
    return np.max(arr) < high and np.min(arr) >= low


def split_arr(arr: np.ndarray, frac2: float) -> Tuple[Any, Any, int]:
    """
    Splits an array along the first axis, s.t.
    in the second part a fraction of 'frac2'
    is contained.

    Args:
        arr: The array to split into two parts.
        frac2: The fraction of data contained in the second part.

    Returns:
        Both parts of the array and the index where the second part
        starts relative to the whole array.
    """
    n: int = arr.shape[0]
    n_1: int = int((1.0 - frac2) * n)
    return arr[:n_1], arr[n_1:], n_1


def copy_arr_list(arr_list: Sequence[Arr]) -> Sequence[Arr]:
    """
    Copies a list of numpy arrays.
    """
    copied_arr_list = [np.copy(a) for a in arr_list]
    return copied_arr_list


def solve_ls(a_mat: np.ndarray, b: np.ndarray, offset: bool = False, non_neg: bool = False, ret_fit: bool = False):
    """
    Solves the least squares problem min_x ||Ax = b||.
    If offset is true, then a bias term is added.
    If non_neg is true, then the regression coefficients are
    constrained to be positive.
    If ret_fit is true, then a tuple (params, fit_values)
    is returned.
    """

    def ls_fun(a_mat_temp, b_temp):
        if non_neg:
            return scipy.optimize.nnls(a_mat_temp, b_temp)[0]
        else:
            return np.linalg.lstsq(a_mat_temp, b_temp, rcond=None)[0]

    n, m = a_mat.shape
    if offset:
        # Add a bias regression term
        a_mat_off = np.empty((n, m + 1), dtype=a_mat.dtype)
        a_mat_off[:, 0] = 1.0
        a_mat_off[:, 1:] = a_mat
        a_mat = a_mat_off

    ret_val = ls_fun(a_mat, b)
    if ret_fit:
        fit_values = np.matmul(a_mat, ret_val)
        ret_val = (ret_val, fit_values)

    return ret_val


def make_periodic(arr_1d: np.ndarray, keep_start: bool = True, keep_min: bool = True):
    """
    Makes a data series periodic by scaling it by a linearly in / decreasing
    factor to in / decrease the values towards the end of the series to match
    the start.

    :param arr_1d: Series to make periodic.
    :param keep_start:
    :param keep_min: Whether to keep the minimum at the same level.
    :return: The periodic series.
    """

    n = len(arr_1d)
    if not keep_start:
        raise NotImplementedError("Fucking do it already!")
    if n < 2:
        raise ValueError("Too small fucking array!!")
    first = arr_1d[0]
    last = arr_1d[-1]
    d_last = last - arr_1d[-2]
    min_val = 0.0 if not keep_min else np.min(arr_1d)
    first_offs = first - min_val
    last_offs = last + d_last - min_val
    fac = first_offs / last_offs
    arr_01 = np.arange(n) / (n - 1)
    f = 1.0 * np.flip(arr_01) + fac * arr_01
    return (arr_1d - min_val) * f + min_val


def check_dim(a: np.ndarray, n: int) -> bool:
    """
    Check whether a is n-dimensional.

    Args:
        a: Numpy array.
        n: Number of dimensions.

    Returns:
        True if a is n-dim else False
    """
    return len(a.shape) == n


def find_rows_with_nans(all_data: np.ndarray) -> np.ndarray:
    """
    Returns a boolean vector indicating which
    rows of 'all_dat' contain NaNs.

    :param all_data: Numpy array with data series as columns.
    :return: 1D array of bool specifying rows containing nans.
    """

    n = all_data.shape[0]
    m = all_data.shape[1]
    col_has_nan = np.empty((n,), dtype=np.bool)
    col_has_nan.fill(False)

    for k in range(m):
        col_has_nan = np.logical_or(col_has_nan, np.isnan(all_data[:, k]))

    return col_has_nan

#######################################################################################################
# NEST stuff

def clean_desc(nest_desc: str) -> str:
    """
    Removes the measurement code from the string containing
    the description of the measurement series.
    """
    if nest_desc[:4] == "65NT":
        return nest_desc.split(" ", 1)[1]
    return nest_desc


def add_dt_and_tinit(m: Sequence, dt_mins, dt_init):
    """
    Adds dt and t_init to the metadata dictionary m.
    """
    for ct, e in enumerate(m):
        m[ct]['t_init'] = dt_to_string(npdatetime_to_datetime(dt_init))
        m[ct]['dt'] = dt_mins


#######################################################################################################
# Os functions

def create_dir(dirname: str) -> None:
    """
    Creates directory if it doesn't exist already.

    Args:
        dirname: The directory to create.

    Returns:
        None
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


#######################################################################################################
# Datetime conversions

def npdatetime_to_datetime(np_dt: np.datetime64) -> datetime:
    """
    Convert from numpy datetime to datetime.
    """
    ts = (np_dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    dt = datetime.utcfromtimestamp(ts)
    return dt


def datetime_to_npdatetime(dt: datetime) -> np.datetime64:
    """
    Convert from datetime to numpy datetime.
    """
    return np.datetime64(dt)


def dt_to_string(dt: datetime) -> str:
    """
    Convert datetime to string.
    """
    return str(dt)


def string_to_dt(s: str) -> datetime:
    """
    Convert string to datetime.
    """
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def str_to_np_dt(s: str) -> np.datetime64:
    """
    Convert string to numpy datetime64.

    Args:
        s: Date string.

    Returns:
        np.datetime64
    """
    dt = string_to_dt(s)
    return datetime_to_npdatetime(dt)


def np_dt_to_str(np_dt: np.datetime64) -> str:
    """
    Converts a single datetime64 to a string.

    Args:
        np_dt: np.datetime64

    Returns:
        String
    """

    dt = npdatetime_to_datetime(np_dt)
    return dt_to_string(dt)


def mins_to_str(mins: int) -> str:
    """
    Converts the integer 'mins' to a string.

    :param mins: Number of minutes.
    :return: String
    """
    return str(mins) + 'min' if mins < 60 else str(mins / 60) + 'h'


def floor_datetime_to_min(dt, mt: int) -> np.ndarray:
    """
    Rounds deltatime64 dt down to mt minutes.
    In a really fucking cumbersome way.

    :param dt: Original deltatime.
    :param mt: Number of minutes.
    :return: Floored deltatime.
    """
    assert 60 % mt == 0

    dt = np.array(dt, dtype='datetime64[s]')
    dt64 = np.datetime64(dt)
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    pdt = datetime.utcfromtimestamp(ts)
    minutes = pdt.minute
    minutes = minutes % mt
    secs = pdt.second
    dt -= np.timedelta64(secs, 's')
    dt -= np.timedelta64(minutes, 'm')
    return dt


def n_mins_to_np_dt(mins: int) -> np.timedelta64:
    return np.timedelta64(mins, 'm')


#######################################################################################################
# Tests

def test_numpy_functions() -> None:
    """
    Tests some of the numpy functions.
    Raises errors if the tests are not passed.

    :return: None
    :raises AssertionError: If a test fails.
    """

    # Define some index arrays
    ind_arr = np.array([1, 2, 3, 4, 2, 3, 0], dtype=np.int32)
    ind_arr_no_dup = np.array([1, 2, 4, 3, 0], dtype=np.int32)

    # Test index functions
    if not has_duplicates(ind_arr) or has_duplicates(ind_arr_no_dup):
        raise AssertionError("Implementation of has_duplicates contains errors!")

    if arr_eq(ind_arr, ind_arr_no_dup) or not arr_eq(ind_arr, ind_arr):
        raise AssertionError("Implementation of arr_eq(...) contains errors!")

    # Define data arrays
    data_array = np.array([
        [1.0, 1.0, 2.0],
        [2.0, 2.0, 5.0],
        [2.0, 2.0, 5.0],
        [3.0, -1.0, 2.0]])
    data_array_with_nans = np.array([
        [1.0, np.nan, 2.0],
        [2.0, 2.0, 5.0],
        [2.0, 2.0, np.nan],
        [2.0, np.nan, np.nan],
        [3.0, -1.0, 2.0]])

    # Test data functions
    d1, d2, n = split_arr(data_array, 0.1)
    d1_exp = data_array[:3]
    if not np.array_equal(d1, d1_exp) or not n == 1:
        raise AssertionError("split_arr not working correctly!!")

    nans_bool_arr = find_rows_with_nans(data_array_with_nans)
    nans_exp = np.array([True, False, True, True, False])
    if not np.array_equal(nans_exp, nans_bool_arr):
        raise AssertionError("find_rows_with_nans not working correctly!!")

    print("Numpy test passed :)")
