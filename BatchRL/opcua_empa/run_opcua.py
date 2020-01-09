import datetime
import logging
import time
from typing import List

import numpy as np
import pandas as pd

from opcua_empa.opcua_util import NodeAndValues, ALL_ROOM_NRS, \
    analyze_experiment, check_room_list
from opcua_empa.controller import FixTimeConstController, ToggleController
from opcua_empa.opcuaclient_subscription import OpcuaClient
from util.numerics import check_in_range

TEMP_MIN_MAX = (18.0, 25.0)

print_fun = logging.warning  # Maybe use print instead of logging?


def try_opcua(verbose: int = 2, room_list: List[int] = None, debug: bool = True):
    """User credentials"""

    # Analyze previous experiment
    analyze_experiment("../Data/Experiments/2020_01_08T13_10_34.pkl")

    # Choose experiment name
    exp_name = None

    # Check list with room numbers
    if room_list is not None:
        check_room_list(room_list)

    # Define room and control
    tc = ToggleController(val_low=10, val_high=28, n_mins=60 * 100, start_low=True, max_n_minutes=60 * 16)
    room_list = [475, 571] if room_list is None else room_list
    used_control = [(i, tc) for i in room_list]
    if debug:
        room_list = [475]
        used_control = [(r, FixTimeConstController(val=25, max_n_minutes=5)) for r in room_list]
        used_control = [(r, ToggleController(val_low=10, val_high=28,
                                             n_mins=5, start_low=False, max_n_minutes=40))
                        for r in room_list]
        exp_name = "Debug"

    # Define value and node generator
    node_value_gen = NodeAndValues(used_control, exp_name=exp_name)
    write_nodes = node_value_gen.get_nodes()
    read_nodes = node_value_gen.get_read_nodes()

    # Initialize data frame
    curr_vals = node_value_gen.compute_current_values()
    df_write = pd.DataFrame({'node': write_nodes, 'value': curr_vals})
    df_read = pd.DataFrame({'node': read_nodes})

    with OpcuaClient() as opcua_client:

        # Subscribe, need to wait a bit before reading for the first time
        opcua_client.subscribe(df_read)
        time.sleep(1.0)

        cont = True
        iter_ind = 0
        while cont:
            # Read values
            read_vals = opcua_client.read_values()

            # Compute the current values
            df_write["value"] = node_value_gen.compute_current_values()
            if verbose:
                print_fun(f"Temperature setpoint: {df_write['value'][0]}")

            # Write (publish) values and wait
            t0 = datetime.datetime.now()
            opcua_client.publish(df_write)
            dt = datetime.datetime.now() - t0
            print_fun(f"Publishing took: {dt}")
            time.sleep(1.0)

            # Check termination criterion
            ext_values = node_value_gen.extract_values(read_vals)
            print(node_value_gen.get_valve_values()[0])

            # Check that the research acknowledgement is true.
            # Wait for at least 20s before requiring to be true, takes some time.
            res_ack_true = np.all(ext_values[0]) or iter_ind < 20

            # Check measured temperatures, stop if too low or high.
            temps_in_bound = check_in_range(np.array(ext_values[1]), *TEMP_MIN_MAX)

            # Stop if (first) controller gives termination signal.
            terminate_now = used_control[0][1].terminate()
            cont = res_ack_true and temps_in_bound and not terminate_now

            # Print the reason of termination.
            if verbose > 0:
                print_fun(f"Extracted: {ext_values}")
                if not temps_in_bound:
                    print_fun("Temperature bounds reached, aborting experiment.")
                if not res_ack_true:
                    print_fun("Research mode confirmation lost :(")
                if terminate_now:
                    print_fun("Experiment time over!")

            # Increment counter and wait.
            iter_ind += 1

        node_value_gen.save_cached_data()
