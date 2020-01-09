"""Controller interface for opcua client.

Defines controllers that can be used to
do control on the real system using the opcua client.
"""
import datetime
from abc import ABC, abstractmethod

import numpy as np

from util.util import Num, get_min_diff


class Controller(ABC):

    @abstractmethod
    def __call__(self, values):
        pass

    def terminate(self):
        return False


class FixTimeConstController(Controller):

    """Const Controller

    Sets the value to be controlled to constant `val`.
    """

    val: Num  #: The numerical value to be set.
    max_n_minutes: int  #: The maximum allowed runtime in minutes.

    def __init__(self, val: Num = 20, max_n_minutes: int = None):
        self.val = val
        self.max_n_minutes = max_n_minutes
        self._start_time = datetime.datetime.now()

    def __call__(self, values=None) -> Num:
        return self.val

    def terminate(self) -> bool:
        """Checks if the maximum time is reached.

        Returns:
            True if the max. runtime is reached, else False.
        """
        if self.max_n_minutes is None:
            return False
        time_now = datetime.datetime.now()
        h_diff = get_min_diff(self._start_time, time_now)
        return h_diff > self.max_n_minutes


class ToggleController(FixTimeConstController):

    def __init__(self, val_low: Num = 20, val_high: Num = 22, n_mins: int = 2,
                 start_low: bool = True, max_n_minutes: int = None):
        """Controller that toggles every `n_mins` between two values.

        Args:
            val_low: The lower value.
            val_high: The higher value.
            n_mins: The number of minutes in an interval.
            start_low: Whether to start with `val_low`.
            max_n_minutes: The maximum number of minutes the controller should run.
        """
        super().__init__(val_low, max_n_minutes)
        self.v_low = val_low
        self.v_high = val_high
        self.dt = n_mins
        self.start_low = start_low

    def __call__(self, values=None) -> Num:
        """Computes the current value according to the current time."""
        time_now = datetime.datetime.now()
        min_diff = get_min_diff(self._start_time, time_now)
        is_start_state = int(min_diff) % (2 * self.dt) < self.dt
        is_low = is_start_state if self.start_low else not is_start_state
        return self.v_low if is_low else self.v_high


class StatefulController(Controller, ABC):

    state: np.ndarray

    def set_state(self, curr_state: np.ndarray):
        self.state = curr_state
