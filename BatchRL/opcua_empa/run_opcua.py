"""Module for running the opcua client.

May be removed later and moved to BatchRL.py if
it is high-level enough.
"""
from typing import List

from opcua_empa.controller import ValveToggler, ValveTest2Controller, FixTimeConstController
from opcua_empa.opcua_util import analyze_experiment, check_room_list
from opcua_empa.opcuaclient_subscription import OpcuaClient
from opcua_empa.room_control_client import run_control
from tests.test_opcua import OfflineClient
from util.util import prog_verb, ProgWrap, DEFAULT_ROOM_NR


def try_opcua(verbose: int = 1, room_list: List[int] = None, debug: bool = True):
    """Runs the opcua client."""

    if verbose:
        if debug:
            print("Running in debug mode!")

    # Analyze previous experiment
    exp_name = "2020_01_15T21_14_51_R475_Experiment_15min_PT_0"
    with ProgWrap(f"Analyzing experiment {exp_name}...", verbose > 0):
        analyze_experiment(exp_name,
                           compute_valve_delay=True,
                           verbose=prog_verb(verbose))

    # Choose experiment name
    exp_name = "Test"

    # Check list with room numbers
    check_room_list(room_list)

    # Define room and control
    # tc = ToggleController(n_mins=60 * 100, start_low=True, max_n_minutes=60 * 16)
    # tc = ValveToggler(n_steps_delay=30, n_steps_max=2 * 60)
    tc = ValveTest2Controller()
    room_list = [43] if room_list is None else room_list
    used_control = [(i, tc) for i in room_list]
    if debug:
        room_list = [41]
        used_control = [(r, ValveToggler(n_steps_delay=30))
                        for r in room_list]
        exp_name = "Offline_DebugValveToggle"

    # Use offline client in debug mode
    cl_class = OfflineClient if debug else OpcuaClient
    run_control(used_control=used_control,
                exp_name=exp_name,
                user='ChristianBaumannETH2020',
                password='Christian4_ever',
                verbose=verbose,
                _client_class=cl_class)


def run_rl_control(room_nr: int = DEFAULT_ROOM_NR,
                   notify_failure: bool = False,
                   debug: bool = False,
                   verbose: int = 1):

    assert room_nr in [41, 43], f"Invalid room number: {room_nr}"

    used_control = [(room_nr, FixTimeConstController(val=21.0, max_n_minutes=12 * 60))]
    exp_name = "DefaultExperimentName"

    cl_class = OfflineClient if debug else OpcuaClient
    run_control(used_control=used_control,
                exp_name=exp_name,
                user='ChristianBaumannETH2020',
                password='Christian4_ever',
                verbose=verbose,
                _client_class=cl_class,
                notify_failures=notify_failure)
    pass
