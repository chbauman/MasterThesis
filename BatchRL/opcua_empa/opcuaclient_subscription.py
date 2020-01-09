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

import datetime
import logging
import socket
import warnings
from typing import List

import pandas as pd
from opcua import Client, Subscription
from opcua import ua
from opcua.common import ua_utils
from opcua.ua import UaStatusCodeError

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

    # The subscription object
    _sub: Subscription = None
    _sub_init: bool = False

    # Bool specifying successful connection
    _connected: bool = False

    def __init__(self, url='opc.tcp://ehub.nestcollaboration.ch:49320',
                 application_uri='Researchclient',
                 product_uri='Researchclient',
                 user='ChristianBaumannETH2020',
                 password='Christian4_ever'):
        """Initialize the opcua client."""

        self.client = Client(url=url, timeout=4)
        self.client.set_user(user)
        self.client.set_password(password)
        self.client.application_uri = application_uri + ":" + socket.gethostname() + ":" + user
        self.client.product_uri = product_uri + ":" + socket.gethostname() + ":" + user
        self.handler = SubHandler()
        self.bInitPublish = False

    def __enter__(self):
        """Enter method for use as context manager."""
        suc_connect = self.connect()
        if suc_connect:
            return self
        self.disconnect()
        raise UaStatusCodeError("Connection failed!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit method for use as context manager."""
        self.disconnect()

    def connect(self) -> bool:
        """Connect the client to the server.

        If connection fails, prints a possible solution.

        Returns:
            True if connection successful, else False.
        """
        try:
            self.client.connect()
            self._connected = True
            self.client.load_type_definitions()
            self._sub = self.client.create_subscription(period=0, handler=self.handler)
            logging.warning("OPC UA Connection to server established.")
            return True
        except UaStatusCodeError as e:
            logging.warning(f"Exception: {e} happened while connecting!")
            print(f"Try waiting a bit more and rerun!")
            return False
        except Exception as e:
            logging.warning(f"Exception: {e} happened while connecting!")
            print(f"Check your internet connection!")
            return False

    def read_values(self) -> pd.DataFrame:
        """Returns the read values in the dataframe."""
        if not self._sub_init:
            logging.warning("You need to subscribe first!")

        try:
            self.handler.df_Read.set_index('node', drop=True)
            return self.handler.df_Read

        except ValueError as e:
            logging.warning(f"Exception: {e} while reading values")
        return self.handler.df_Read

    def disconnect(self) -> None:
        """Disconnect the client.

        Deletes the subscription first to avoid error.
        """
        # If it wasn't connected, do nothing
        if not self._connected:
            return
        try:
            # Need to delete the subscription first before disconnecting
            self._sub.delete()
            self.client.disconnect()
            logging.warning("OPC UA Server disconnected.")
        except UaStatusCodeError as e:
            # This does not catch the error :(
            logging.warning(f"OPC UA Server disconnected with error: {e}")

    def subscribe(self, df_read: pd.DataFrame) -> None:
        """Subscribe all values you want to read.

        TODO: Why take a json string as input????

        If it fails, a warning is printed and some values might
        not be read correctly.
        """
        if self._sub_init:
            logging.warning("You already subscribed!")

        self.df_Read = df_read
        nodelist_read = [self.client.get_node(row['node'])
                         for i, row in self.df_Read.iterrows()]

        # Try subscribing to the nodes in the list.
        try:
            sub_res = self._sub.subscribe_data_change(nodelist_read)
            self._sub_init = True

            # Check if subscription was successful
            for ct, s in enumerate(sub_res):
                if not type(s) is int:
                    warnings.warn(f"Node: {nodelist_read[ct]} not found!")
            logging.warning("OPC UA Subscription requested.")
        except Exception as e:
            # TODO: Remove or catch more specific error!
            logging.warning(f"Exception: {e} happened while subscribing!")
            raise e

    def publish(self, json_write: str) -> None:
        """Publish (write) values to server.

        Initializes publishing if called for first time. If the actual
        publishing fails, a warning message is printed.

        Args:
            json_write: The string with the json to write.

        Raises:
            UaStatusCodeError: If initialization fails.
        """
        self.df_Write = pd.read_json(json_write)

        # Initialize publishing
        if not self.bInitPublish:
            self._node_objects = [self.client.get_node(node)
                                  for node in self.df_Write['node'].tolist()]
            try:
                self._data_types = [nodeObject.get_data_type_as_variant_type()
                                    for nodeObject in self._node_objects]
                self.bInitPublish = True
                logging.warning("OPC UA Publishing initialized.")
            except UaStatusCodeError as e:
                print(f"UaStatusCodeError while initializing publishing!: {e}")
                raise e

        # Publish values, failures to publish will not raise an exception.
        try:
            self._ua_values = [ua.DataValue(ua.Variant(ua_utils.string_to_val(str(value), d_t), d_t)) for
                               value, d_t in zip(self.df_Write['value'].tolist(), self._data_types)]
            self.client.set_values(nodes=self._node_objects, values=self._ua_values)
            for n, val in zip(self._node_objects, self._ua_values):
                logger.info('write %s %s' % (n, val))
        except UaStatusCodeError as e:
            logging.warning(f"UaStatusCodeError: {e} happened while publishing!")
