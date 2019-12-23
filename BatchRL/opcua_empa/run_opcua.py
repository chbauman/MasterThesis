import time

import pandas as pd
import numpy as np

from opcua_empa.opcua_util import NodeAndValues, ToggleController, FixTimeConstController
from opcua_empa.opcuaclient_subscription import OpcuaClient
from util.numerics import check_in_range

TEMP_MIN_MAX = (18.0, 26.0)

# Set pandas printing options
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.options.display.max_colwidth = 200


def try_opcua(verbose: int = 0):
    """User credentials"""
    opcua_client = OpcuaClient(user='ChristianBaumannETH2020', password='Christian4_ever')

    # Connect client
    connect_success = opcua_client.connect()
    if not connect_success:
        return

    # Define room and control
    curr_control = [
        (475, ToggleController(val_low=10, val_high=35, n_mins=60 * 4, start_low=False)),
        (571, ToggleController(val_low=10, val_high=35, n_mins=60 * 4, start_low=False)),
    ]
    curr_control = [(575, FixTimeConstController(val=30, max_n_minutes=1))]

    # Define value and node generator
    value_gen = NodeAndValues(curr_control)
    w_nodes = value_gen.get_nodes()
    read_nodes = value_gen.get_read_nodes()
    if verbose:
        print(read_nodes)
    print(value_gen.room_inds)

    # Subscribe
    df_Read = pd.DataFrame({'node': read_nodes})
    opcua_client.subscribe(json_read=df_Read.to_json())

    cont = True
    while cont:
        # Compute the current values
        v = value_gen.get_values()
        df_Write = pd.DataFrame({'node': w_nodes, 'value': v})

        opcua_client.publish(json_write=df_Write.to_json())
        time.sleep(1)
        opcua_client.handler.df_Read.set_index('node', drop=True)
        read_values = opcua_client.handler.df_Read

        # Check termination criterion
        ext_values = value_gen.extract_values(read_values)
        res_ack_true = np.all(ext_values[0])
        temps_in_bound = check_in_range(np.array(ext_values[1]), *TEMP_MIN_MAX)
        terminate_now = curr_control[0][1].terminate()
        cont = res_ack_true and temps_in_bound and not terminate_now

        if verbose > 0:
            print(f"Extracted : {ext_values}")
            if not temps_in_bound:
                print("Temperature bounds reached, aborting experiment.")
            if not res_ack_true:
                print("Research mode confirmation lost :(")
            if terminate_now:
                print("Experiment time over!")

    opcua_client.disconnect()
