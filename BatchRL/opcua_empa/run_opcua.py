import time

import pandas as pd

from opcua_empa.opcua_util import NodeAndValues, ToggleController
from opcua_empa.opcuaclient_subscription import OpcuaClient

# Set pandas printing options
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.options.display.max_colwidth = 200


def terminate(_=None) -> bool:
    return False


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
    curr_control = [(575, 25)]

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
        if verbose > 0:
            print(read_values)
            print(f"Extracted : {value_gen.extract_values(read_values)}")

        # Check termination criterion
        cont = not terminate(read_values)

    opcua_client.disconnect()
