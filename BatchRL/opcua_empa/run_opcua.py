"""Module for running the opcua client.

May be removed later and moved to BatchRL.py if
it is high-level enough.
"""
import logging
from typing import List

from opcua_empa.controller import ToggleController
from opcua_empa.opcua_util import analyze_experiment, check_room_list
from opcua_empa.room_control_client import ControlClient

print_fun = logging.warning  # Maybe use print instead of logging?


def try_opcua(verbose: int = 2, room_list: List[int] = None, debug: bool = True):
    """Runs the opcua client."""

    # Analyze previous experiment
    analyze_experiment("../Data/Experiments/2020_01_09T11_29_28_Toggle_6min.pkl")

    # Choose experiment name
    exp_name = "Test"

    # Check list with room numbers
    if room_list is not None:
        check_room_list(room_list)

    # Define room and control
    tc = ToggleController(val_low=10, val_high=28, n_mins=60 * 100, start_low=True, max_n_minutes=60 * 16)
    room_list = [475] if room_list is None else room_list
    used_control = [(i, tc) for i in room_list]
    if debug:
        room_list = [472]
        used_control = [(r, ToggleController(val_low=10, val_high=28,
                                             n_mins=6, start_low=False,
                                             max_n_minutes=60))
                        for r in room_list]
        exp_name = "Debug"

    with ControlClient(used_control=used_control,
                       exp_name=exp_name,
                       user='ChristianBaumannETH2020',
                       password='Christian4_ever',
                       verbose=verbose) as client:

        # Run controller
        cont = True
        while cont:
            cont = client.read_publish_wait_check()
