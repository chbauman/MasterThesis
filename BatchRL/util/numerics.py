from typing import Tuple, Sequence, Any

import numpy as np
import scipy.optimize

from util.util import datetime_to_np_datetime, string_to_dt, Arr, Num


def num_nans(arr: np.ndarray) -> int:
    """Computes the number of nans in an array."""
    return np.sum(np.isnan(arr)).item()


def npf32(sh: Tuple, fill: float = None) -> np.ndarray:
    """Returns an empty numpy float array of specified shape."""
    empty_arr = np.empty(sh, dtype=np.float32)
    if fill is not None:
        empty_arr.fill(fill)
    return empty_arr


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

    # Bug-fix: Ugly, but working :P
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


def move_inds_to_back(arr: np.ndarray, inds) -> np.ndarray:
    """Moves the series specified by the `inds` to the end of the array.

    Args:
        arr: The array to transform.
        inds: The indices specifying the series to move.

    Returns:
        New array with permuted features.
    """
    n_feat = arr.shape[-1]
    mask = np.ones((n_feat,), np.bool)
    mask[inds] = False
    input_dat = np.concatenate([arr[..., mask], arr[..., inds]], axis=-1)
    return input_dat


def prepare_supervised_control(sequences: np.ndarray,
                               c_inds: np.array,
                               sequence_pred: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for supervised learning.

    Transforms a batch of sequences of constant length to
    prepare it for supervised model training. Moves control series
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
