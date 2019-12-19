import logging
import time

import pandas as pd

from opcua_empa.opcua_util import NodeAndValues, ToggleController
from opcua_empa.opcuaclient_subscription import OpcuaClient

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)
logger = logging.getLogger('opc ua client')

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
    # curr_control = [(575, 25)]
    curr_control = [(575, ToggleController(val_low=23, val_high=25, n_mins=1))]

    # Define value and node generator
    value_gen = NodeAndValues(curr_control)
    w_nodes = value_gen.get_nodes()
    read_nodes = value_gen.get_read_nodes()
    print(read_nodes)

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

        if verbose > 0:
            print(opcua_client.handler.df_Read)

        # Check termination criterion
        cont = not terminate()

    opcua_client.disconnect()
