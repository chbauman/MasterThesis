
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
