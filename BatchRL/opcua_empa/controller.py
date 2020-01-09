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

    state: np.ndarray = None

    @abstractmethod
    def __call__(self, values):
        """Returns the current control input."""
        pass

    def terminate(self):
        return False

    def set_state(self, curr_state: np.ndarray):
        self.state = curr_state


class FixTimeController(Controller, ABC):
    """Fixed-time controller.

    Runs for a fixed number of timesteps.
    """
    max_n_minutes: int  #: The maximum allowed runtime in minutes.

    _start_time: datetime  #: The starting time.

    def __init__(self, max_n_minutes: int = None):
        self.max_n_minutes = max_n_minutes
        self._start_time = datetime.now()

    def terminate(self) -> bool:
        """Checks if the maximum time is reached.

        Returns:
            True if the max. runtime is reached, else False.
        """
        if self.max_n_minutes is None:
            return False
        h_diff = get_min_diff(self._start_time, t2=None)
        return h_diff > self.max_n_minutes


ControlT = List[Tuple[int, Controller]]  #: Room number to controller map type


class FixTimeConstController(FixTimeController):
    """Const Controller.

    Runs for a fixed amount of time if `max_n_minutes` is specified.
    Sets the value to be controlled to constant `val`.
    Control inputs do not depend on current time or on state!
    """

    val: Num  #: The numerical value to be set.

    def __init__(self, val: Num = MIN_TEMP, max_n_minutes: int = None):
        super().__init__(max_n_minutes)
        self.val = val

    def __call__(self, values=None) -> Num:
        return self.val


class ToggleController(FixTimeController):
    """Toggle controller.

    Toggles every `n_mins` between two values.
    Control inputs only depend on current time and not on state!
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
        super().__init__(max_n_minutes)
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


class ValveToggler(FixTimeController):
    """Controller that toggles as soon as the valves have toggled."""

    n_delay: int  #: How many steps to wait with toggling back.
    TOL: float = 0.05

    _step_count: int = 0
    _curr_valve_state: bool = False

    def __init__(self, n_steps_delay: int = 10, n_steps_max: int = 60 * 60):
        super().__init__(n_steps_max)
        self.n_delay = n_steps_delay

    def __call__(self, values=None):
        print(self.state[4])

        v = self.state[4]  # Extract valve state
        if v > 1.0 - self.TOL:
            if not self._curr_valve_state:
                # Valves just opened
                self._step_count = 0
                print("Valves opened!!!")
                self._curr_valve_state = True
        elif v < self.TOL:
            if self._curr_valve_state:
                # Valves just closed
                print("Valves closed!!!")
                self._step_count = 0
                self._curr_valve_state = False

        ret_min = self._curr_valve_state

        # If valves just switched, ignore change
        if self._step_count < self.n_delay:
            ret_min = not ret_min

        # Convert bool to temperature
        ret = MIN_TEMP if ret_min else MAX_TEMP

        # Increment and return
        self._step_count += 1
        return ret
