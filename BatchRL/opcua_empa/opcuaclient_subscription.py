#########################################################################################################################
# Name: opc ua client
# Version: 0.1

# Activities:                                           Author:                         Date:
# Initial comment                                       RK                              20190409
# Add toggle module                                     RK                              20190411
# Add computer name to the client                       RK                              20190411
# Changed order to subscribe/ publish                   RK                              20190411
# Moved the try statement inside the for loop           RK                              20190425
# Changed from time to datetime to show milliseconds    RK                              20190508

########################################################################################################################
from opcua import Client
from opcua import ua
from opcua.common import ua_utils
import pandas as pd
import datetime
import logging
import socket

"""initialize logger"""
logger = logging.getLogger('opc ua client')


# toggle function
def toggle(tonf=5000):
    if datetime.datetime.now().second % 10 < 5:
        is_toggled = False
    else:
        is_toggled = True
    return is_toggled


# Definition action if you receive read variable
class SubHandler(object):
    """
    Subscription Handler. To receive events from server for a subscription
    data_change and event methods are called directly from receiving thread.
    Do not do expensive, slow or network operation there. Create another
    thread if you need to do such a thing. You have to define here
    what to do with the received date.
    """

    def __init__(self):
        self.df_Read = pd.DataFrame(data={'node': [], 'value': []}).astype('object')
        self.json_Read = self.df_Read.to_json()

    def datachange_notification(self, node, val, data):
        try:
            df_New = pd.DataFrame(data={'node': [], 'value': []}).astype('object')
            df_New.at[0, 'node'] = str(node)
            df_New.at[0, 'value'] = str(val)
            self.df_Read = self.df_Read.merge(df_New, on=list(self.df_Read), how='outer')
            self.df_Read.drop_duplicates(subset=['node'], inplace=True, keep='last')
            self.json_Read = self.df_Read.to_json()
            logger.info('read %s %s' % (node, val))
        except Exception as e:
            logger.error(e)

    def event_notification(self, event):
        logger.info("Python: New event", event)


# Definition of opcua_client client
class OpcuaClient(object):
    """Initialized the opc ua client."""

    df_write = None

    _node_objects = None
    _data_types = None
    _ua_values = None

    def __init__(self, url='opc.tcp://ehub.nestcollaboration.ch:49320',
                 application_uri='Researchclient',
                 product_uri='Researchclient',
                 user='JustforTest',
                 password='JustforTest'):

        # You have to enter the url of the opc ua server e.g "opc.tcp://ehub.nestcollaboration.ch:49320"
        self.client = Client(url=url, timeout=4)
        self.client.set_user(user)  # You have to enter your User name*
        self.client.set_password(password)  # You have to enter your password*
        self.client.application_uri = application_uri + ":" + socket.gethostname() + ":" + user  # You have to enter the uri according to the name or path of your certificate and key*
        # You have to enter the uri according to the name or path of your certificate and key*
        self.client.product_uri = product_uri + ":" + socket.gethostname() + ":" + user
        self.handler = SubHandler()
        self.bInitPublish = False

    def connect(self):
        try:
            self.client.connect()
            self.client.load_type_definitions()  # load definition of server specific structures/extension objects
            print('%s OPC UA Connection to server established' % (datetime.datetime.now()))
            return True
        except Exception as e:
            print(e)
            return False

    def disconnect(self):
        self.client.disconnect()
        print('%sOPC UA Server disconnected' % (datetime.datetime.now()))

    """All values you want to read"""

    def subscribe(self, json_read="""{"node":{	"0":"ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bAnlageEin",
                                    "1":"ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bQuittierung",
                                    "2":"ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bReqResearch",
                                    "3":"ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bWdResearch",
                                    "4":"ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.rSollWirkleistung"}}"""
                  ):

        df_read = pd.read_json(json_read)
        nodelist_read = []
        for index, row in df_read.iterrows():
            nodelist_read.append(self.client.get_node(row['node']))

        try:
            handler = self.handler
            sub = self.client.create_subscription(period=0, handler=handler)
            sub.subscribe_data_change(nodelist_read)
            print('%s OPC UA Subscription requested' % (datetime.datetime.now()))
        except Exception as e:
            logging.warning(e)

    """All values wou want to write"""

    def publish(self, json_write="""{"node":{	"0":"ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bAnlageEin",
                                                "1":"ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bQuittierung",
                                                "2":"ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bReqResearch",
                                                "3":"ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.bWdResearch",
                                                "4":"ns=2;s=Gateway.PLC1.65LK-06420-D001.PLC1.Batan.strWrite_L.strBatterienanlage.rSollWirkleistung"},
                                        
                                        "value":{	"0":true,
                                                "1":false,
                                                "2":true,
                                                "3":false,
                                                "4":2}}"""
                ):

        self.df_write = pd.read_json(json_write)
        if not self.bInitPublish:
            self._node_objects = [self.client.get_node(node) for node in self.df_write['node'].tolist()]
            try:
                self._data_types = [nodeObject.get_data_type_as_variant_type() for nodeObject in self._node_objects]
                self.bInitPublish = True
                print('%s OPC UA Publishing initialized' % (datetime.datetime.now()))
            except Exception as e:
                print(e)
                logging.warning('The node you want to write does not exist')

        try:
            self._ua_values = [ua.DataValue(ua.Variant(ua_utils.string_to_val(str(value), datatype), datatype)) for
                               value, datatype in zip(self.df_write['value'].tolist(), self._data_types)]
            self.client.set_values(nodes=self._node_objects, values=self._ua_values)
            [logger.info('write %s %s' % (nodeobject, value)) for nodeobject, value in
             zip(self._node_objects, self._ua_values)]
        except Exception as e:
            logging.warning(e)
