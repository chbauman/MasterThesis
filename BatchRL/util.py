
import os

import numpy as np

from datetime import datetime

#######################################################################################################
# Python stuff

def get_if_nnone(lst, indx, default = None):
    """
    Returns a list element if list is not None, 
    else the default value.
    """
    return default if lst is None else lst[indx]

def apply(list_or_el, fun):
    """
    Applies the function fun to each element of list_or_el
    if it is a list, else it is applied directly to list_or_el.
    """
    if isinstance(list_or_el, list):
        return [fun(k) for k in list_or_el]
    else:
        return fun(list_or_el)

#######################################################################################################
# Numerical stuff

def fit_linear_1d(x, y, x_new = None):
    """
    Fit a linear model y = c * x + m.
    Returns coefficients m and c. If x_new
    is not None, returns the evaluated linear
    fit at x_new.
    """

    n = x.shape[0]
    ls_mat = np.empty((n, 2), dtype = np.float32)
    ls_mat[:, 0] = 1
    ls_mat[:, 1] = x
    m, c = np.linalg.lstsq(ls_mat, y, rcond=None)[0]
    if x_new is None:
        return [m, c]
    else:
        return c * x_new + m

def fit_linear_bf_1d(x, y, b_fun):
    """
    Fits a linear model y = \alpha^T f(x).
    """
    raise NotImplementedError("Implement this fucking function!")

def get_shape1(arr):
    """
    Returns the shape of the second dimension
    of a matrix. If it is a vector returns 1.
    """
    s = arr.shape
    if len(s) < 2:
        return 1
    else:
        return s[1]
    return 0

def align_ts(ts_1, ts_2, t_init1, t_init2, dt):
    """
    Aligns the two timeseries with given initial time
    and constant timestep by padding by np.nan.
    """

    # Get shapes
    n_1 = ts_1.shape[0]
    n_2 = ts_2.shape[0]
    d_1 = get_shape1(ts_1)
    d_2 = get_shape1(ts_2)

    # Compute relative offset
    interv = np.timedelta64(dt, 'm')
    ti1 = datetime_to_npdatetime(string_to_dt(t_init1))
    ti2 = datetime_to_npdatetime(string_to_dt(t_init2))
    offset = np.int(np.round((ti2 - ti1) / interv))

    # Compute length
    out_len = np.maximum(n_2 - offset, n_1) 
    start_s = offset <= 0
    out_len += offset if not start_s else 0
    out = np.empty((out_len, d_1 + d_2), dtype = ts_1.dtype)
    out.fill(np.nan)

    # Copy over
    t1_res = np.reshape(ts_1, (n_1, d_1))
    t2_res = np.reshape(ts_2, (n_2, d_2))
    if not start_s:
        out[:n_2, :d_2] = t2_res
        out[offset:(offset + n_1), d_2:] = t1_res
        t_init_out = t_init1
    else:
        out[:n_1, :d_1] = t1_res
        out[-offset:(-offset + n_2), d_1:] = t2_res
        t_init_out = t_init2

    return out, t_init_out

def add_mean_and_std(ts, mean_and_std):
    """
    Transforms the data back to having mean
    and std as specified.
    """
    return ts * mean_and_std[1] + mean_and_std[0]

def check_in_range(arr, low, high):
    """
    Returns true if all elements in arr are in
    range [low, high) else false.
    """
    return np.max(arr < high) and np.min(arr >= low)

#######################################################################################################
# NEST stuff

def cleas_desc(nest_desc):
    """
    Removes the measurement code from the string containing
    the description of the measurement series.
    """
    if nest_desc[:4] == "65NT":
        return nest_desc.split(" ", 1)[1]
    return nest_desc

def add_dt_and_tinit(m, dt_mins, dt_init):
    """
    Adds dt and t_init to the metadata dictionnary m.
    """
    for ct, e in enumerate(m):
        m[ct]['t_init'] = dt_to_string(npdatetime_to_datetime(dt_init))
        m[ct]['dt'] = dt_mins

#######################################################################################################
# Os functions

def create_dir(dirname):
    """
    Creates directory if it doesn't exist already.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return

#######################################################################################################
# Datetime conversions

def npdatetime_to_datetime(npdt):
    """
    Convert from numpy datetime to datetime.
    """
    ts = (npdt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    dt = datetime.utcfromtimestamp(ts)
    return dt

def datetime_to_npdatetime(dt):
    """
    Convert from datetime to numpy datetime.
    """
    return np.datetime64(dt)

def dt_to_string(dt):
    """
    Convert datetime to string.
    """
    return str(dt)

def string_to_dt(s):
    """
    Convert string to datetime.
    """
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
