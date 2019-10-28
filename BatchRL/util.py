import os
from datetime import datetime
from typing import Union, List, Tuple, Any, Sequence

import numpy as np
import scipy.optimize.nnls

"""A few general functions with multiple use cases.

Includes a few general python functions, 
a lot of numpy transformations and also some tools
to handle the datetime of python and numpy. Also some
tests of these functions are included. 
"""

# Determine platform, assuming we are on Euler if it is not a windows platform
EULER = not os.name == 'nt'

#######################################################################################################
# Typing

# Type for general number
Num = Union[int, float]

# Type for general array, including 0-D ones, i.e. single numbers
Arr = Union[Num, np.ndarray]


#######################################################################################################
# Python stuff

def rem_first(t: Tuple) -> Tuple:
    """Removes first element from tuple.

    Args:
        t: Original tuple.

    Returns:
        New tuple without first value.
    """
    assert len(t) >= 1, "Tuple must have at least one element!"
    lis = [i for i in t]
    return tuple(lis[1:])


def get_if_not_none(lst: Sequence, indx: int, default=None):
    """Returns a list element if list is not None, else the default value.

    Args:
        lst: List of elements or None
        indx: List index.
        default: Default return value

    Returns:
        List element at position indx if lst is not None, else default.
    """
    return default if lst is None else lst[indx]


def apply(list_or_el, fun):
    """Applies the function fun to each element of `list_or_el`.

    If it is a list, else it is applied directly to `list_or_el`.

    Args:
        list_or_el: List of elements or single element.
        fun: Function to apply to elements.

    Returns:
        List or element with function applied.
    """
    if isinstance(list_or_el, list):
        return [fun(k) for k in list_or_el]
    else:
        return fun(list_or_el)


def repl(el, n: int) -> List:
    """Constructs a list with `n` equal elements 'el'.

    If el is not a primitive type, then it might
    give a list with views on el.

    Args:
        el: Element to repeat.
        n: Number of times.

    Returns:
        New list with `n` elements.
    """
    return [el for _ in range(n)]


def b_cast(l_or_el, n: int) -> List:
    """Returns a list with `n` repeated elements `l_or_el`.

    Checks if `l_or_el` is a list or not, if it is and
    it already has length `n`, it is returned.

    Args:
        l_or_el: List of elements or element.
        n: Length of list.

    Returns:
        list

    Raises:
        ValueError: If `l_or_el` is a list and does not have `n` elements.
    """
    if isinstance(l_or_el, list):
        if len(l_or_el) == n:
            return l_or_el
        raise ValueError("Broadcast failed!!!")
    return repl(l_or_el, n)


class CacheDecoratorFactory(object):
    """Decorator for caching results of a function.

    Function output and function input is stored in a list
    and returned if the same input is given to the decorated function.

    TODO: Test more, make it work for non-member functions!!
    """

    n: List  #: List of function arguments.
    d: List  #: List of function outputs.

    def __init__(self, n_list: List = None, data_list: List = None):
        """Initialize the decorator.

        If no lists are provided, the
        results are stored in this class.

        Args:
            n_list: List where the input is stored.
            data_list: List where the function output is stored.
        """
        self.n = [] if n_list is None else n_list
        self.d = [] if data_list is None else data_list

    def __call__(self, f):
        """Decorates the function `f`.

        Args:
            f: The function to be decorated.

        Returns:
            The decorated function.
        """

        def decorated(s, n: Union[Tuple, int], *args, **kwargs):
            """The actual decorator.

            Args:
                s: Self of the class whose member function is decorated.
                n: The unique input to the function.
                *args: Arguments.
                **kwargs: Keyword arguments.

            Returns:
                The decorated function.
            """
            if n in self.n:
                i = self.n.index(n)
                return self.d[i]
            else:
                dat = f(s, n, *args, **kwargs)
                self.n += [n]
                self.d += [dat]
                return dat

        return decorated


class TestDecoratorFactory(object):
    """Testing decorator.

    Prints different messages for AssertionErrors
    and other errors.
    This sucks because it fucks up the debugging.
    TODO: Solve this!
    """

    def __init__(self, msg: str = "Test failed!"):
        """Initialize the decorator.

        Args:
            msg: Error message .
        """
        self.m = msg

    def __call__(self, f):
        """Decorates the function `f`.

        Args:
            f: The function to be decorated.

        Returns:
            The decorated function.
        """

        def decorated(*args, **kwargs):

            try:
                f(*args, **kwargs)
            except AssertionError as a:
                print("{}-test failed!".format(self.m))
                raise a
            except Exception as e:
                print("Exception: {}".format(e))
                raise AssertionError("Unexpected error happened in test {}".format(self.m))

        return decorated


#######################################################################################################
# Numerical stuff

def has_duplicates(arr: np.ndarray) -> bool:
    """Checks if `arr` contains duplicates.

    Returns true if arr contains duplicate values else False.

    Args:
        arr: Array to check for duplicates.

    Returns:
        Whether it has duplicates.
    """
    m = np.zeros_like(arr, dtype=bool)
    m[np.unique(arr, return_index=True)[1]] = True
    return np.sum(~m) > 0


def fit_linear_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray = None):
    """Fit a linear model y = c * x + m.

    Returns coefficients m and c. If x_new
    is not None, returns the evaluated linear
    fit at x_new.

    Args:
        x_new: Where to evaluate the fitted model.
        y: Y values.
        x: X values.

    Returns:
        The parameters m and c if `x_new` is None, else
        the model evaluated at `x_new`.
    """
    n = x.shape[0]
    ls_mat = np.empty((n, 2), dtype=np.float32)
    ls_mat[:, 0] = 1
    ls_mat[:, 1] = x
    m, c = np.linalg.lstsq(ls_mat, y, rcond=None)[0]
    if x_new is None:
        return m, c
    else:
        return c * x_new + m


def fit_linear_bf_1d(x: np.ndarray, y: np.ndarray, b_fun, offset: bool = False) -> np.ndarray:
    """Fits a linear model y = alpha^T f(x).

    TODO: implement with offset!

    Args:
        x: The values on the x axis.
        y: The values to fit corresponding to x.
        b_fun: A function evaluating all basis function at the input.
        offset: Whether to add a bias term.

    Returns:
        The fitted linear parameters.
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
    """Save version of `arr`.shape[1]

    Returns:
        The shape of the second dimension
        of an array. If it is a vector returns 1.
    """
    s = arr.shape
    if len(s) < 2:
        return 1
    else:
        return s[1]


def align_ts(ts_1: np.ndarray, ts_2: np.ndarray, t_init1: str, t_init2: str, dt: int) -> Tuple[np.ndarray, str]:
    """Aligns two time series.

    Aligns the two time series with given initial time
    and constant timestep by padding by np.nan.

    Args:
        ts_1: First time series.
        ts_2: Second time series.
        t_init1: Initial time string of series 1.
        t_init2: Initial time string of series 2.
        dt: Number of minutes in a timestep.

    Returns:
        The combined data array and the new initial time string.
    """

    # Get shapes
    n_1 = ts_1.shape[0]
    n_2 = ts_2.shape[0]
    d_1 = get_shape1(ts_1)
    d_2 = get_shape1(ts_2)

    # Compute relative offset
    interval = np.timedelta64(dt, 'm')
    ti1 = datetime_to_np_datetime(string_to_dt(t_init2))
    ti2 = datetime_to_np_datetime(string_to_dt(t_init1))

    # Ugly bug-fix
    if ti1 < ti2:
        d_out, t = align_ts(ts_2, ts_1, t_init2, t_init1, dt)
        d_out_real = np.copy(d_out)
        d_out_real[:, :d_1] = d_out[:, d_2:]
        d_out_real[:, d_1:] = d_out[:, :d_2]
        return d_out_real, t

    offset = np.int(np.round((ti2 - ti1) / interval))

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
    """Adds or removes given  mean and std from time series.

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
    """Transforms the data back to having mean and std as specified.

    Args:
        ts: The series to add mean and std.
        mean_and_std: The mean and the std.

    Returns:
        New scaled series.
    """
    if len(mean_and_std) < 2:
        raise ValueError("Invalid value for mean_and_std")
    return ts * mean_and_std[1] + mean_and_std[0]


def rem_mean_and_std(ts: Arr, mean_and_std: Sequence) -> Arr:
    """Whitens the data with known mean and standard deviation.

    Args:
        ts: Data to be whitened.
        mean_and_std: Container of the mean and the std.

    Returns:
        Whitened data.
    """
    if len(mean_and_std) < 2:
        raise ValueError("Invalid value for mean_and_std")
    if mean_and_std[1] == 0:
        raise ZeroDivisionError("Standard deviation cannot be 0")
    return (ts - mean_and_std[0]) / mean_and_std[1]


def check_in_range(arr: np.ndarray, low: Num, high: Num) -> bool:
    """Checks if elements of an array are in specified range.

    Returns:
        True if all elements in arr are in
        range [low, high) else false.
    """
    if arr.size == 0:
        return True
    return np.max(arr) < high and np.min(arr) >= low


def split_arr(arr: np.ndarray, frac2: float) -> Tuple[Any, Any, int]:
    """Splits an array along the first axis.

    In the second part a fraction of 'frac2' elements
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
    """Copies a list of numpy arrays.

    Args:
        arr_list: The sequence of numpy arrays.

    Returns:
        A list with all the copied elements.
    """
    copied_arr_list = [np.copy(a) for a in arr_list]
    return copied_arr_list


def solve_ls(a_mat: np.ndarray, b: np.ndarray, offset: bool = False,
             non_neg: bool = False,
             ret_fit: bool = False):
    """Solves the least squares problem min_x ||Ax = b||.

    If offset is true, then a bias term is added.
    If non_neg is true, then the regression coefficients are
    constrained to be positive.
    If ret_fit is true, then a tuple (params, fit_values)
    is returned.

    Args:
        a_mat: The system matrix.
        b: The RHS vector.
        offset: Whether to include an offset.
        non_neg: Whether to use non-negative regression.
        ret_fit: Whether to additionally return the fitted values.

    Returns:
        The fitted parameters and optionally the fitted values.
    """
    # Choose least squares solver
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
        # Add fitted values to return value
        fit_values = np.matmul(a_mat, ret_val)
        ret_val = (ret_val, fit_values)

    return ret_val


def make_periodic(arr_1d: np.ndarray, keep_start: bool = True,
                  keep_min: bool = True) -> np.ndarray:
    """Makes a data series periodic.

    By scaling it by a linearly in- / decreasing
    factor to in- / decrease the values towards the end of the series to match
    the start.

    Args:
        arr_1d: Series to make periodic.
        keep_start: Whether to keep the beginning of the series fixed.
            If False keeps the end of the series fixed.
        keep_min: Whether to keep the minimum at the same level.

    Returns:
        The periodic series.
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
    """Check whether a is n-dimensional.

    Args:
        a: Numpy array.
        n: Number of dimensions.

    Returns:
        True if a is n-dim else False
    """
    return len(a.shape) == n


def find_rows_with_nans(all_data: np.ndarray) -> np.ndarray:
    """Finds nans in the data.

    Returns a boolean vector indicating which
    rows of 'all_dat' contain NaNs.

    Args:
        all_data: Numpy array with data series as columns.

    Returns:
        1D array of bool specifying rows containing nans.
    """
    n = all_data.shape[0]
    m = all_data.shape[1]
    row_has_nan = np.empty((n,), dtype=np.bool)
    row_has_nan.fill(False)

    for k in range(m):
        row_has_nan = np.logical_or(row_has_nan, np.isnan(all_data[:, k]))

    return row_has_nan


def extract_streak(all_data: np.ndarray, s_len: int, lag: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Extracts a streak where all data is available.

    Finds the last sequence where all data is available
    for at least `s_len` + `lag` timesteps. Then splits the
    data before that last sequence and returns both parts.

    Args:
        all_data: The data.
        s_len: The sequence length.
        lag: The number of sequences the streak should contain.

    Returns:
        The data before the streak, the streak data and the index
        pointing to the start of the streak data.

    Raises:
        IndexError: If there is no streak of specified length found.
    """
    tot_s_len = s_len + lag

    # Find last sequence of length tot_s_len
    inds = find_all_streaks(find_rows_with_nans(all_data), tot_s_len)
    if len(inds) < 1:
        raise IndexError("No fucking streak of length {} found!!!".format(tot_s_len))
    last_seq_start = inds[-1]

    # Extract
    first_dat = all_data[:last_seq_start, :]
    streak_dat = all_data[last_seq_start:(last_seq_start + tot_s_len), :]
    return first_dat, streak_dat, last_seq_start + lag


def nan_array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Analog for np.array_equal but this time ignoring nans.

    Args:
        a: First array.
        b: Second array to compare.

    Returns:
        True if the arrays contain the exact same elements else False.
    """
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


def find_all_streaks(col_has_nan: np.ndarray, s_len: int) -> np.ndarray:
    """Finds all streak of length `s_len`.

    Finds all sequences of length `s_len` where `col_has_nan`
    is never False. Then returns all indices of the start of
    these sequences in `col_has_nan`.

    Args:
        col_has_nan: Bool vector specifying where nans are.
        s_len: The length of the sequences.

    Returns:
        Index vector specifying the start of the sequences.
    """
    # Define True filter
    true_seq = np.empty((s_len,), dtype=np.int32)
    true_seq.fill(1)

    # Find sequences of length s_len
    tmp = np.convolve(np.logical_not(col_has_nan), true_seq, 'valid')
    inds = np.where(tmp == s_len)[0]
    return inds


def cut_data(all_data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cut the data into sequences.

    Cuts the data contained in `all_data` into sequences of length `seq_len` where
    there are no nans in the row of `all_data`.
    Also returns the indices where to find the sequences in `all_data`.

    Args:
        all_data: The 2D numpy array containing the data series.
        seq_len: The length of the sequences to extract.

    Returns:
        3D array with sequences and 1D int array with indices where the
        sequences start with respect to `all_data`.
    """
    # Check input
    if len(all_data.shape) > 2:
        raise ValueError("Data array has too many dimensions!")

    # Find sequences
    nans = find_rows_with_nans(all_data)
    all_inds = find_all_streaks(nans, seq_len)

    # Get shapes
    n_seqs = len(all_inds)
    n_feat = get_shape1(all_data)

    # Extract sequences
    out_dat = np.empty((n_seqs, seq_len, n_feat), dtype=np.float32)
    for ct, k in enumerate(all_inds):
        out_dat[ct] = all_data[k:(k + seq_len)]
    return out_dat, all_inds


def find_disjoint_streaks(nans: np.ndarray, seq_len: int, streak_len: int,
                          n_ts_offs: int = 0) -> np.ndarray:
    """Finds streaks that are only overlapping by `seq_len` - 1 steps.

    They will be a multiple of `streak_len` from each other relative
    to the `nans` vector.

    Args:
        nans: Boolean array indicating that there is a nan if entry is true.
        seq_len: The required sequence length.
        streak_len: The length of the streak that is disjoint.
        n_ts_offs: The number of timesteps that the start is offset.

    Returns:
        Indices pointing to the start of the disjoint streaks.
    """
    n = len(nans)
    tot_len = streak_len + seq_len - 1
    start = (n_ts_offs - seq_len + 1 + streak_len) % streak_len
    n_max = (n - start) // streak_len
    inds = np.empty((n_max,), dtype=np.int32)
    ct = 0
    for k in range(n_max):
        k_start = start + k * streak_len
        curr_dat = nans[k_start: (k_start + tot_len)]
        if not np.any(curr_dat):
            inds[ct] = k_start
            ct += 1
    return inds[:ct]


def prepare_supervised_control(sequences: np.ndarray,
                               c_inds: np.array,
                               sequence_pred: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for supervised learning.

    Transforms a batch of sequences of constant length to
    prepare it for supervised model training. Removes control indices
    to the end of the features and shifts them to the past for one
    time step.

    Args:
        sequences: Batch of sequences of same length.
        c_inds: Indices determining the features to control.
        sequence_pred: Whether to use sequence output.

    Returns:
        The prepared input and output data.
    """
    n_feat = sequences.shape[-1]

    # Get inverse mask
    mask = np.ones((n_feat,), np.bool)
    mask[c_inds] = False

    # Extract and concatenate input data
    arr_list = [sequences[:, :-1, mask],
                sequences[:, 1:, c_inds]]
    input_dat = np.concatenate(arr_list, axis=-1)

    # Extract output data
    if not sequence_pred:
        output_data = sequences[:, -1, mask]
    else:
        output_data = sequences[:, 1:, mask]

    # Return
    return input_dat, output_data


#######################################################################################################
# NEST stuff

def clean_desc(nest_desc: str) -> str:
    """Cleans a description string of the NEST database.

    Removes the measurement code from the string containing
    the description of the measurement series.

    Args:
        nest_desc: The description from the database.

    Returns:
        The clean description.
    """
    if nest_desc[:4] == "65NT":
        return nest_desc.split(" ", 1)[1]
    return nest_desc


def add_dt_and_t_init(m: Sequence, dt_mins: int, dt_init: np.datetime64) -> None:
    """Adds dt and t_init to each metadata dictionary in `m`.

    Args:
        m: List with all the metadata dictionaries.
        dt_mins: Number of minutes in a timestep.
        dt_init: Time of first timestep.
    """
    for ct, e in enumerate(m):
        m[ct]['t_init'] = dt_to_string(np_datetime_to_datetime(dt_init))
        m[ct]['dt'] = dt_mins


#######################################################################################################
# Os functions

def create_dir(dirname: str) -> None:
    """Creates directory if it doesn't exist already.

    Args:
        dirname: The directory to create.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


#######################################################################################################
# Datetime conversions

def np_datetime_to_datetime(np_dt: np.datetime64) -> datetime:
    """Convert from numpy datetime to datetime.

    Args:
        np_dt: Numpy datetime.

    Returns:
        Python datetime.
    """
    ts = (np_dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    dt = datetime.utcfromtimestamp(ts)
    return dt


def datetime_to_np_datetime(dt: datetime) -> np.datetime64:
    """Convert from datetime to numpy datetime.

    Args:
        dt: Python datetime.

    Returns:
        Numpy datetime object.
    """
    return np.datetime64(dt)


def dt_to_string(dt: datetime) -> str:
    """Convert datetime to string.
    """
    return str(dt)


def string_to_dt(s: str) -> datetime:
    """Convert string to datetime.

    Assumes smallest unit of time in string are seconds.
    """
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def str_to_np_dt(s: str) -> np.datetime64:
    """Convert string to numpy datetime64.

    Args:
        s: Date string.

    Returns:
        np.datetime64
    """
    dt = string_to_dt(s)
    return datetime_to_np_datetime(dt)


def np_dt_to_str(np_dt: np.datetime64) -> str:
    """
    Converts a single datetime64 to a string.

    Args:
        np_dt: np.datetime64

    Returns:
        String
    """

    dt = np_datetime_to_datetime(np_dt)
    return dt_to_string(dt)


def mins_to_str(mins: int) -> str:
    """Converts the integer `mins` to a string.

    Args:
        mins: Number of minutes.

    Returns:
        String representation.
    """
    return str(mins) + 'min' if mins < 60 else str(mins / 60) + 'h'


def floor_datetime_to_min(dt, mt: int) -> np.ndarray:
    """Rounds date- / deltatime64 `dt` down to `mt` minutes.

    In a really fucking cumbersome way!

    Args:
        dt: Original deltatime.
        mt: Number of minutes.

    Returns:
        Floored deltatime.
    """
    assert 60 % mt == 0, "Not implemented for more than 60 minutes!"

    # Convert to python datetime
    dt = np.array(dt, dtype='datetime64[s]')
    dt64 = np.datetime64(dt)
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    pdt = datetime.utcfromtimestamp(ts)

    # Subtract remainder minutes and seconds
    minutes = pdt.minute
    minutes = minutes % mt
    secs = pdt.second
    dt -= np.timedelta64(secs, 's')
    dt -= np.timedelta64(minutes, 'm')
    return dt


def n_mins_to_np_dt(mins: int) -> np.timedelta64:
    """Converts an int (assuming number of minutes) to a numpy deltatime object."""
    return np.timedelta64(mins, 'm')


def ts_per_day(n_min: int) -> int:
    """Computes the number of timesteps in a day.

    Returns the number of time steps in a day when
    one timestep is `n_min` minutes.

    Args:
        n_min: Length of timestep in minutes.

    Returns:
        Number of timesteps in a day.

    Raises:
        ValueError: If the result would be a float.
    """
    if (24 * 60) % n_min != 0:
        raise ValueError(f"Number of mins in a day not divisible by n_min: {n_min}")
    return 24 * 60 // n_min


def day_offset_ts(t_init: str, mins: int = 15) -> int:
    """Computes the number of timesteps of length `mins` minutes until the next day starts.

    Args:
        t_init: The reference time.
        mins: The number of minutes in a timestep.

    Returns:
        Number of timesteps until next day.
    """
    np_t_init = str_to_np_dt(t_init)
    t_0 = np.datetime64(np_t_init, 'D')
    dt_int = np.timedelta64(mins, 'm')
    n_ts_passed = int((np_t_init - t_0) / dt_int)
    tot_n_ts = int(np.timedelta64(1, 'D') / dt_int)
    return tot_n_ts - n_ts_passed


#######################################################################################################
# Tests

def test_numpy_functions() -> None:
    """Tests some of the numpy functions.

    Returns:
        None

    Raises:
        AssertionError: If a test fails.
    """
    # Define some index arrays
    ind_arr = np.array([1, 2, 3, 4, 2, 3, 0], dtype=np.int32)
    ind_arr_no_dup = np.array([1, 2, 4, 3, 0], dtype=np.int32)

    # Test index functions
    if not has_duplicates(ind_arr) or has_duplicates(ind_arr_no_dup):
        raise AssertionError("Implementation of has_duplicates contains errors!")

    if np.array_equal(ind_arr, ind_arr_no_dup) or not np.array_equal(ind_arr, ind_arr):
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
        [2.0, 2.0, 5.0],
        [2.0, 2.0, np.nan],
        [2.0, np.nan, np.nan],
        [2.0, 2.0, 5.0],
        [2.0, 2.0, 5.0],
        [3.0, -1.0, 2.0]])

    # Test array splitting
    d1, d2, n = split_arr(data_array, 0.1)
    d1_exp = data_array[:3]
    if not np.array_equal(d1, d1_exp) or n != 3:
        raise AssertionError("split_arr not working correctly!!")

    # Test finding rows with nans
    nans_bool_arr = find_rows_with_nans(data_array_with_nans)
    nans_exp = np.array([True, False, False, True, True, False, False, False])
    if not np.array_equal(nans_exp, nans_bool_arr):
        raise AssertionError("find_rows_with_nans not working correctly!!")

    # Test last streak extraction
    d1, d2, n = extract_streak(data_array_with_nans, 1, 1)
    d2_exp = data_array_with_nans[6:8]
    d1_exp = data_array_with_nans[:6]
    if not nan_array_equal(d2, d2_exp) or n != 7 or not nan_array_equal(d1, d1_exp):
        raise AssertionError("extract_streak not working correctly!!")

    # Test sequence cutting
    cut_dat_exp = np.array([
        data_array_with_nans[1:3],
        data_array_with_nans[5:7],
        data_array_with_nans[6:8],
    ])
    c_dat, inds = cut_data(data_array_with_nans, 2)
    inds_exp = np.array([1, 5, 6])
    if not np.array_equal(c_dat, cut_dat_exp) or not np.array_equal(inds_exp, inds):
        raise AssertionError("cut_data not working correctly!!")

    bool_vec = np.array([True, False, False, False, True, False, False, True])
    bool_vec_2 = np.array([True, False, False, False, False, True, False, False, False, False, True])
    bool_vec_3 = np.array([True, False, False, False])

    streaks = find_all_streaks(bool_vec, 2)
    s_exp = np.array([1, 2, 5])
    if not np.array_equal(s_exp, streaks):
        raise AssertionError("find_all_streaks not working correctly!!")

    # Test find_disjoint_streaks
    dis_s = find_disjoint_streaks(bool_vec, 2, 1)
    if not np.array_equal(dis_s, s_exp):
        raise AssertionError("find_disjoint_streaks not working correctly!!")
    dis_s = find_disjoint_streaks(bool_vec_2, 2, 2, 1)
    if not np.array_equal(dis_s, np.array([2, 6])):
        raise AssertionError("find_disjoint_streaks not working correctly!!")
    dis_s = find_disjoint_streaks(bool_vec_2, 2, 2, 0)
    if not np.array_equal(dis_s, np.array([1, 7])):
        raise AssertionError("find_disjoint_streaks not working correctly!!")
    dis_s = find_disjoint_streaks(bool_vec_3, 2, 2, 0)
    if not np.array_equal(dis_s, np.array([1])):
        raise AssertionError("find_disjoint_streaks not working correctly!!")

    # Sequence data
    sequences = np.array([
        [[1, 2, 3],
         [1, 2, 3],
         [2, 3, 4]],
        [[3, 2, 3],
         [1, 2, 3],
         [4, 3, 4]],
    ])
    c_inds = np.array([1])

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
    in_arr, out_arr = prepare_supervised_control(sequences, c_inds, False)
    if not np.array_equal(in_arr, in_arr_exp) or not np.array_equal(out_arr, out_arr_exp):
        raise AssertionError("Problems encountered in prepare_supervised_control")

    # Tests are done
    print("Numpy test passed :)")


def test_python_stuff() -> None:
    """Tests some of the python functions.

    Returns:
        None

    Raises:
        AssertionError: If a test fails.
    """
    # Test the caching decorator
    class Dummy:
        @CacheDecoratorFactory()
        def fun(self, n: int, k: int):
            return n + k * k

        @CacheDecoratorFactory()
        def mutable_fun(self, n: int, k: int):
            return [n, k]

    d = Dummy()
    try:
        assert d.fun(1, k=3) == 10
        assert d.fun(2, 3) == 11
        assert d.fun(1, k=4) == 10
        list_1_1 = d.mutable_fun(1, 1)
        assert d.mutable_fun(1, 2) == list_1_1
        list_1_1[0] = 0
        assert list_1_1 == d.mutable_fun(1, 5)
        assert d.fun(2, 7) == 11
        assert [4, 7] == d.mutable_fun(4, 7)
    except AssertionError as e:
        print("Cache Decorator Test failed!!")
        raise e
    except Exception as e:
        raise AssertionError("Some error happened: {}".format(e))

    assert rem_first((1, 2, 3)) == (2, 3), "rem_first not working correctly!"
    assert rem_first((1, 2)) == (2,), "rem_first not working correctly!"


def test_time_stuff() -> None:
    """Tests some of the functions concerned with datetime formats.

    Returns:
        None

    Raises:
        AssertionError: If a test fails.
    """
    # Define data
    dt1 = np.datetime64('2000-01-01T00:00', 'm')
    dt3 = np.datetime64('2000-01-01T22:45', 'm')
    n_mins = 15
    t_init_str = np_dt_to_str(dt3)

    # Test time conversion
    assert str_to_np_dt(np_dt_to_str(dt1)) == dt1, "Time conversion not working"

    # Test day_offset_ts
    n_ts = day_offset_ts(t_init_str, n_mins)
    assert n_ts == 5, "Fuck you!!"

    print("Time tests passed! :)")
