"""Module for running the opcua client.

May be removed later and moved to BatchRL.py if
it is high-level enough.
"""
import logging
from typing import List

from opcua_empa.controller import ToggleController, ValveToggler
from opcua_empa.opcua_util import analyze_experiment, check_room_list
from opcua_empa.opcuaclient_subscription import OpcuaClient
from opcua_empa.room_control_client import run_control
from tests.test_opcua import OfflineClient

print_fun = logging.warning  # Maybe use print instead of logging?


def try_opcua(verbose: int = 2, room_list: List[int] = None, debug: bool = True):
    """Runs the opcua client."""

    if verbose:
        if debug:
            print_fun("Running in debug mode!")

    # Analyze previous experiment
    analyze_experiment("../Data/Experiments/2020_01_09T14_16_08_R475_Test.pkl")

    # Choose experiment name
    exp_name = "Test"

    # Check list with room numbers
    if room_list is not None:
        check_room_list(room_list)

    # Define room and control
    # tc = ToggleController(n_mins=60 * 100, start_low=True, max_n_minutes=60 * 16)
    tc = ValveToggler(n_steps_delay=30)
    room_list = [475] if room_list is None else room_list
    used_control = [(i, tc) for i in room_list]
    if debug:
        room_list = [472]
        used_control = [(r, ToggleController(n_mins=6, start_low=False,
                                             max_n_minutes=60))
                        for r in room_list]
        used_control = [(r, ValveToggler(n_steps_delay=30))
                        for r in room_list]
        exp_name = "DebugValveToggle"

    # Use offline client in debug mode
    cl_class = OfflineClient if debug else OpcuaClient
    run_control(used_control=used_control,
                exp_name=exp_name,
                user='ChristianBaumannETH2020',
                password='Christian4_ever',
                verbose=verbose,
                _client_class=cl_class)
