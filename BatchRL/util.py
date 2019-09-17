
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
