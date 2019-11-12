"""A few general functions with multiple use cases.

Includes a few general python functions,
a lot of numpy transformations and also some tools
to handle the datetime of python and numpy. Also some
tests of these functions are included.
"""
import os
import random
from datetime import datetime
from functools import wraps
from typing import Union, List, Tuple, Any, Sequence

import numpy as np

# Determine platform, assuming we are on Euler if it is not a windows platform.
EULER: bool = os.name != 'nt'

SEED: int = 42  #: Default seed value.


def fix_seed(seed: int = SEED) -> None:
    """Fixes the random seed."""
    np.random.seed(seed)
    random.seed(seed)


#######################################################################################################
# Typing

# Type for general number
Num = Union[int, float]

# Type for general array, including 0-D ones, i.e. single numbers
Arr = Union[Num, np.ndarray]


#######################################################################################################
# Python stuff

def make_param_ext(l: List[Tuple[str, Any]]) -> str:
    """Converts a list of parameters to a string.

    Can be used as an extension of a file name to differentiate
    files generated with different parameters.

    Args:
        l: List of (Name, parameter) tuples.

    Returns:
        String combining all parameters.
    """
    s = ""
    for t in l:
        pre_str, el = t
        if el is None:
            continue
        elif type(el) is bool:
            if el:
                s += "_" + pre_str
        elif type(el) is int:
            s += "_" + pre_str + str(el)
        elif type(el) is float:
            s += "_" + pre_str + "{:.4g}".format(el)
        elif type(el) is list or type(el) is tuple:
            s += "_" + pre_str + '-'.join(map(str, el))
        else:
            raise ValueError("Type not supported!")
    return s


def tot_size(t: Tuple[int, ...]) -> int:
    """Computes the product of all numbers in `t`.

    Returns 0 for empty `t`.
    """
    res = 1 if len(t) > 0 else 0
    for k in t:
        res *= k
    return res


def scale_to_range(x: Num, tot_len: Num, ran: Sequence[Num]) -> float:
    """Interval transformation.

    Assumes `x` is in [0, `tot_len`] and scales it affine linearly
    to the interval `ran`.

    Args:
        x: Point to transform.
        tot_len: Length of initial interval.
        ran: Interval to transform into.

    Returns:
        New point in requested interval.
    """
    # Check input
    assert tot_len >= np.nanmax(x) and np.nanmin(x) >= 0.0, "Invalid values!"
    assert len(ran) == 2, "Range must have length 2!"
    assert ran[1] > ran[0], "Interval must have positive length!"

    # Compute output
    d_ran = ran[1] - ran[0]
    return ran[0] + x / tot_len * d_ran


def check_and_scale(action: Num, tot_n_actions: int, interval: Sequence[Num]):
    """Checks if `action` is in the right range and scales it.

    Works for an array of actions. Ignores nans.

    Args:
        action: The action to scale.
        tot_n_actions: The total number of possible actions.
        interval: The range to scale `action` into.

    Returns:
        The scaled action.
    """
    if not 0 <= np.nanmin(action) or not np.nanmax(action) <= tot_n_actions:
        raise ValueError(f"Action: {action} not in correct range!")
    cont_action = scale_to_range(action, tot_n_actions - 1, interval)
    return cont_action


def linear_oob_penalty(x: Num, bounds: Sequence[Num]) -> float:
    """Computes the linear penalty for `x` not lying within `bounds`.

    Args:
        x: Value that should be within bounds.
        bounds: The specified bounds.

    Returns:
        Penalty value.
    """
    assert bounds[0] <= bounds[1], "Invalid bounds!"
    if x < bounds[0]:
        return bounds[0] - x
    elif x > bounds[1]:
        return x - bounds[1]
    return 0


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

    TODO: Make it work for non-member functions!!
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
        # print("Init decorator!!!")

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
    TODO: Find out how or decide to remove!
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


def train_decorator(verbose: bool = True):
    """Decorator factory for fit method of ML model.

    Assumes the model has a keras model `m` as member
    variable and a name `name`. Then it tries loading
    the model, if that fails the actual fitting is done.
    """

    def decorator(fit):

        @wraps(fit)
        def decorated(self):

            loaded = self.load_if_exists(self.m, self.name)
            if not loaded:
                # Set seed for reproducibility
                np.random.seed(SEED)

                # Fit and save
                if verbose:
                    print("Fitting Model...")
                fit(self)
                self.save_model(self.m, self.name)
            elif verbose:
                print("Restored trained model")

        return decorated

    return decorator


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