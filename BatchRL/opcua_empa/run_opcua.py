import logging
import time

import pandas as pd

from opcua_empa.opcua_util import NodeAndValues, ToggleController
from opcua_empa.opcuaclient_subscription import OpcuaClient

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)
logger = logging.getLogger('opc ua client')

read_node_names = [
    # Weather:
    'ns=2;s=Gateway.PLC1.65NT-03032-D001.PLC1.MET51.strMET51Read.strWetterstation.strStation1.lrLufttemperatur',
    'ns=2;s=Gateway.PLC1.65NT-06421-D001.PLC1.Units.str2T5.strRead.strSensoren.strW1.strB870.rValue5',

    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strM1.strP890.rValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR1.strB870.bAckResearch',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR1.strB870.rValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR1.strB870.rValue2',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ1.strY700.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ1.strY701.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ1.strY702.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ1.strY703.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ1.strY704.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ1.strY705.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ1.strY706.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB870.bAckResearch',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB870.rValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB870.rValue2',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ2.strY700.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ2.strY701.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ2.strY706.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB871.bAckResearch',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB871.rValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB871.rValue2',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ2.strY702.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB872.bAckResearch',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB872.rValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR2.strB872.rValue2',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ2.strY703.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ2.strY704.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ2.strY705.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR3.strB870.bAckResearch',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR3.strB870.rValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR3.strB870.rValue2',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ3.strY700.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ3.strY705.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ3.strY706.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR3.strB871.bAckResearch',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR3.strB871.rValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR3.strB871.rValue2',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ3.strY704.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR3.strB872.bAckResearch',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR3.strB872.rValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strR3.strB872.rValue2',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ3.strY701.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ3.strY702.bValue1',
    'ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strAktoren.strZ3.strY703.bValue1',
]

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

    df_Read = pd.DataFrame({'node': read_node_names})
    opcua_client.subscribe(json_read=df_Read.to_json())

    # Define room and control
    # curr_control = [(575, 25)]
    curr_control = [(575, ToggleController(val_low=19, val_high=15, n_mins=1))]

    # Define value and node generator
    value_gen = NodeAndValues(curr_control)
    w_nodes = value_gen.get_nodes()

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
