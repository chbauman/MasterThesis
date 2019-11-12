from .opcuaclient_subscription import OpcuaClient
from .opcuaclient_subscription import toggle
import pandas as pd
import logging
import time

# Configuration logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('opc ua client')


def try_opcua():
    opcua_client = OpcuaClient(user='Aurelio', password='Aurelio4ever')
    if opcua_client.connect():
        opcua_client.subscribe(json_read=pd.DataFrame(
            {'node': ["ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bAnlageEin",
                      "ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bQuittierung",
                      "ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bReqResearch",
                      "ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bWdResearch",
                      "ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.rSollWirkleistung"
                      ]}).to_json())

        while True:
            opcua_client.publish(json_write=pd.DataFrame(
                {'node': ["ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bAnlageEin",
                          "ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bQuittierung",
                          "ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bReqResearch",
                          "ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bWdResearch",
                          "ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.rSollWirkleistung"
                          ],

                 'value': [True,
                           False,
                           True,
                           toggle(),
                           2
                           ]}).to_json())
            time.sleep(0.100)
            print(opcua_client.handler.json_Read)

    opcua_client.disconnect()
