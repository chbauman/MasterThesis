"""Controller interface for opcua client.

Defines controllers that can be used to
do control on the real system using the opcua client.
"""
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from util.util import Num, get_min_diff

MAX_TEMP: int = 28  #: Them maximum temperature to set.
MIN_TEMP: int = 10  #: Them minimum temperature to set.


class Controller(ABC):
    """Base controller interface.

    A controller needs to implement the __call__ function
    and optionally a termination criterion: `terminate()`.
    """

    @abstractmethod
    def __call__(self, values):
        """Returns the current control input."""
        pass

    def terminate(self):
        return False


ControlT = List[Tuple[int, Controller]]  #: Room number to controller map type


class FixTimeConstController(Controller):
    """Const Controller.

    Runs for a fixed amount of time if `max_n_minutes` is specified.
    Sets the value to be controlled to constant `val`.
    """

    val: Num  #: The numerical value to be set.
    max_n_minutes: int  #: The maximum allowed runtime in minutes.

    _start_time: datetime  #: The starting time.

    def __init__(self, val: Num = MIN_TEMP, max_n_minutes: int = None):
        self.val = val
        self.max_n_minutes = max_n_minutes
        self._start_time = datetime.now()

    def __call__(self, values=None) -> Num:
        return self.val

    def terminate(self) -> bool:
        """Checks if the maximum time is reached.

        Returns:
            True if the max. runtime is reached, else False.
        """
        if self.max_n_minutes is None:
            return False
        h_diff = get_min_diff(self._start_time, t2=None)
        return h_diff > self.max_n_minutes


class ToggleController(FixTimeConstController):
    """Toggle controller.

    Toggles every `n_mins` between two values.
    """

    def __init__(self, val_low: Num = MIN_TEMP, val_high: Num = MAX_TEMP, n_mins: int = 2,
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
        min_diff = get_min_diff(self._start_time, t2=None)
        is_start_state = int(min_diff) % (2 * self.dt) < self.dt
        is_low = is_start_state if self.start_low else not is_start_state
        return self.v_low if is_low else self.v_high


class StatefulController(Controller, ABC):
    """Interface of a stateful controller.

    Contains a state, the return value of `self.call`
    should depend on this state.
    """
    state: np.ndarray

    def set_state(self, curr_state: np.ndarray):
        self.state = curr_state


class ValveToggler(StatefulController):
    """Controller that toggles as soon as the valves have toggled."""

    n_delay: int  #: How many steps to wait with toggling back.
    TOL: float = 0.05

    _step_count: int = 0
    _curr_valve_state: bool = False

    def __init__(self, n_steps_delay: int = 10):
        self.n_delay = n_steps_delay
        pass

    def __call__(self, values=None):

        v = self.state[4]  # Extract valve state
        if v > 1.0 - self.TOL:
            if not self._curr_valve_state:
                # Valves just opened
                self._step_count = 0
                self._curr_valve_state = True
        elif v < self.TOL:
            if self._curr_valve_state:
                # Valves just closed
                self._step_count = 0
                self._curr_valve_state = False

        ret = MIN_TEMP if self._curr_valve_state else MAX_TEMP
        # If valves just switched, ignore change
        if self._step_count < self.n_delay:
            ret = not ret

        # Increment and return
        self._step_count += 1
        return ret
