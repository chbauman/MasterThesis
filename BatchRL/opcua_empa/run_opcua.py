import datetime
import logging
import time
from typing import List

import pandas as pd
import numpy as np

from opcua_empa.opcua_util import NodeAndValues, ToggleController, FixTimeConstController, ALL_ROOM_NRS
from opcua_empa.opcuaclient_subscription import OpcuaClient
from util.numerics import check_in_range

TEMP_MIN_MAX = (18.0, 25.0)


def try_opcua(verbose: int = 0, room_list: List[int] = None, debug: bool = False):
    """User credentials"""
    opcua_client = OpcuaClient(user='ChristianBaumannETH2020', password='Christian4_ever')

    publish_only = False

    print_fun = logging.warning  # Maybe use print instead of logging?

    # TODO: Check room_list!

    # Connect client
    if not opcua_client.connect():
        return

    # Define room and control
    tc = ToggleController(val_low=10, val_high=35, n_mins=15, start_low=True, max_n_minutes=60 * 16)
    room_list = [475, 571] if room_list is None else room_list
    used_control = [(i, tc) for i in room_list]
    if debug:
        used_control = [(575, FixTimeConstController(val=50, max_n_minutes=1))]

    # Define value and node generator
    node_value_gen = NodeAndValues(used_control)
    write_nodes = node_value_gen.get_nodes()
    read_nodes = node_value_gen.get_read_nodes()

    # Subscribe
    if not publish_only:
        df_read = pd.DataFrame({'node': read_nodes})
        opcua_client.subscribe(json_read=df_read.to_json())

    # Initialize data frame
    curr_vals = node_value_gen.compute_current_values()
    df_write = pd.DataFrame({'node': write_nodes, 'value': curr_vals})

    cont = True
    iter_ind = 0
    while cont:
        # Compute the current values
        df_write["value"] = node_value_gen.compute_current_values()

        # Write (publish) values and wait
        t0 = datetime.datetime.now()
        opcua_client.publish(json_write=df_write.to_json())
        dt = datetime.datetime.now() - t0
        print_fun(f"{dt}")
        time.sleep(0.4)

        res_ack_true, temps_in_bound = True, True
        ext_values = None
        if not publish_only:
            opcua_client.handler.df_Read.set_index('node', drop=True)

            # Check termination criterion
            ext_values = node_value_gen.extract_values(opcua_client.handler.df_Read)

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
            if not publish_only:
                print_fun(f"Extracted : {ext_values}")
            else:
                print_fun(f"Step: {iter_ind}")
            if not temps_in_bound:
                print_fun("Temperature bounds reached, aborting experiment.")
            if not res_ack_true:
                print_fun("Research mode confirmation lost :(")
            if terminate_now:
                print_fun("Experiment time over!")

        # Increment counter and wait.
        iter_ind += 1

    # Disconnect the client.
    opcua_client.disconnect()
