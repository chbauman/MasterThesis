########################################################################################################################
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
import warnings
from typing import List

import opcua
from opcua import Client
from opcua import ua
from opcua.common import ua_utils
import pandas as pd
import datetime
import logging
import socket

# Initialize and configure logger
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)
logger = logging.getLogger('opc ua client')


def toggle():
    """Toggles every 5 seconds.

    The watchdog has to toggle every 5 seconds
    otherwise the connection will be refused.
    """
    if datetime.datetime.now().second % 10 < 5:
        toggle_state = False
    else:
        toggle_state = True
    return toggle_state


class SubHandler(object):
    """Subscription Handler.

    To receive events from server for a subscription
    data_change and event methods are called directly from receiving thread.
    Do not do expensive, slow or network operation there. Create another
    thread if you need to do such a thing. You have to define here
    what to do with the received date.
    """

    def __init__(self):
        self.df_Read = pd.DataFrame(data={'node': [], 'value': []}).astype('object')
        self.json_Read = self.df_Read.to_json()

    def datachange_notification(self, node, val, _):
        try:
            df_new = pd.DataFrame(data={'node': [], 'value': []}).astype('object')
            df_new.at[0, 'node'] = str(node)
            df_new.at[0, 'value'] = str(val)
            self.df_Read = self.df_Read.merge(df_new, on=list(self.df_Read), how='outer')
            self.df_Read.drop_duplicates(subset=['node'], inplace=True, keep='last')
            self.json_Read = self.df_Read.to_json()
            logger.info('read %s %s' % (node, val))
        except Exception as e:
            logger.error(e)

    @staticmethod
    def event_notification(event):
        logger.info("Python: New event", event)


# Definition of opcua client
class OpcuaClient(object):
    """Wrapper class for Opcua Client."""

    # Read and write data frames.
    df_Read: pd.DataFrame = None
    df_Write: pd.DataFrame = None

    # Private members
    _node_objects: List
    _data_types: List
    _ua_values: List

    def __init__(self, url='opc.tcp://ehub.nestcollaboration.ch:49320',
                 application_uri='Researchclient',
                 product_uri='Researchclient',
                 user='JustForTest',
                 password='JustForTest'):
        """Initialize the opc ua client."""

        self.client = Client(url=url, timeout=4)
        self.client.set_user(user)
        self.client.set_password(password)
        self.client.application_uri = application_uri + ":" + socket.gethostname() + ":" + user
        self.client.product_uri = product_uri + ":" + socket.gethostname() + ":" + user
        self.handler = SubHandler()
        self.bInitPublish = False

    def connect(self):
        """Connect the client to the server"""
        try:
            self.client.connect()
            self.client.load_type_definitions()
            print('%s OPC UA Connection to server established' % (datetime.datetime.now()))
            return True
        except Exception as e:
            print(f"Exception: {e} happened while connecting!")
            print(f"Check your internet connection!")
            return False

    def disconnect(self):
        """Disconnect the client"""
        self.client.disconnect()
        print('%sOPC UA Server disconnected' % (datetime.datetime.now()))

    def subscribe(self, json_read: str):
        """Subscribe all values you want to read"""
        self.df_Read = pd.read_json(json_read)
        nodelist_read = [self.client.get_node(row['node'])
                         for i, row in self.df_Read.iterrows()]

        try:
            sub = self.client.create_subscription(period=0, handler=self.handler)
            sub_res = sub.subscribe_data_change(nodelist_read)

            # Check if subscription was successful
            for ct, s in enumerate(sub_res):
                if not type(s) is int:
                    print(s)
                    warnings.warn(f"Node: {nodelist_read[ct]} not found!")
            print('%s OPC UA Subscription requested' % (datetime.datetime.now()))
        except Exception as e:
            print(f"Exception: {e} happened while subscribing!")
            logging.warning(e)

    def publish(self, json_write: str):
        self.df_Write = pd.read_json(json_write)
        if not self.bInitPublish:
            self._node_objects = [self.client.get_node(node)
                                  for node in self.df_Write['node'].tolist()]
            try:
                self._data_types = [nodeObject.get_data_type_as_variant_type()
                                    for nodeObject in self._node_objects]
                self.bInitPublish = True
                print('%s OPC UA Publishing initialized' % (datetime.datetime.now()))
            except opcua.ua.uaerrors._auto.BadNodeIdUnknown as e:
                print(f"Exception: {e} happened while initializing publishing!")
                logging.warning('The node you want to write does not exist')
                raise e

        try:
            self._ua_values = [ua.DataValue(ua.Variant(ua_utils.string_to_val(str(value), d_t), d_t)) for
                               value, d_t in zip(self.df_Write['value'].tolist(), self._data_types)]

            for n, val in zip(self._node_objects, self._ua_values):
                n.set_value(val)
                logger.info('write %s %s' % (n, val))

        except Exception as e:
            print(f"Exception: {e} happened while publishing!")
            logging.warning(e)
