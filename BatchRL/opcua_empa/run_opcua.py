import datetime
import logging
import time
from typing import List

import numpy as np
import pandas as pd

from opcua_empa.opcua_util import NodeAndValues, ToggleController, FixTimeConstController, ALL_ROOM_NRS
from opcua_empa.opcuaclient_subscription import OpcuaClient
from util.numerics import check_in_range
from util.visualize import plot_valve_opening

TEMP_MIN_MAX = (18.0, 25.0)


def try_opcua(verbose: int = 2, room_list: List[int] = None, debug: bool = True):
    """User credentials"""
    print_fun = logging.warning  # Maybe use print instead of logging?

    # Check list with room numbers
    if room_list is not None:
        assert isinstance(room_list, list), f"Room list: {room_list} needs to be a list!"
        for k in room_list:
            assert k in ALL_ROOM_NRS, f"Invalid room number: {k}"

    # Define room and control
    tc = ToggleController(val_low=10, val_high=35, n_mins=60 * 100, start_low=True, max_n_minutes=60 * 16)
    room_list = [475, 571] if room_list is None else room_list
    used_control = [(i, tc) for i in room_list]
    if debug:
        room_list = [475]
        used_control = [(r, FixTimeConstController(val=25, max_n_minutes=5)) for r in room_list]
        used_control = [(r, ToggleController(val_low=10, val_high=35,
                                             n_mins=5, start_low=False, max_n_minutes=1))
                        for r in room_list]

    # Define value and node generator
    node_value_gen = NodeAndValues(used_control)
    write_nodes = node_value_gen.get_nodes()
    read_nodes = node_value_gen.get_read_nodes()

    # Initialize data frame
    curr_vals = node_value_gen.compute_current_values()
    df_write = pd.DataFrame({'node': write_nodes, 'value': curr_vals})
    df_read = pd.DataFrame({'node': read_nodes})

    with OpcuaClient(user='ChristianBaumannETH2020',
                     password='Christian4_ever') as opcua_client:

        # Subscribe
        opcua_client.subscribe(json_read=df_read.to_json())

        cont = True
        iter_ind = 0
        while cont:
            # Compute the current values
            df_write["value"] = node_value_gen.compute_current_values()
            if verbose:
                print_fun(f"Temperature setpoint: {df_write['value'][0]}")

            # Write (publish) values and wait
            t0 = datetime.datetime.now()
            opcua_client.publish(json_write=df_write.to_json())
            dt = datetime.datetime.now() - t0
            print_fun(f"Publishing took: {dt}")
            time.sleep(1.0)

            # Read values
            read_vals = opcua_client.read_values()

            # Check termination criterion
            ext_values = node_value_gen.extract_values(read_vals)
            print(node_value_gen.get_valve_values())

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
                print_fun(f"Extracted : {ext_values}")
                if not temps_in_bound:
                    print_fun("Temperature bounds reached, aborting experiment.")
                if not res_ack_true:
                    print_fun("Research mode confirmation lost :(")
                if terminate_now:
                    print_fun("Experiment time over!")

            # Increment counter and wait.
            iter_ind += 1

        all_valves = node_value_gen.get_valve_values(all_prev=True)[0]
        all_timesteps = node_value_gen.read_timestamps
        pub_ts = node_value_gen.write_timestamps
        temp_sps = node_value_gen.write_values[:, ]
        plot_valve_opening(all_timesteps, all_valves, "test", pub_ts, temp_sps)
