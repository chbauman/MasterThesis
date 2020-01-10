import datetime
from typing import List
from unittest import TestCase

import pandas as pd

import opcua_empa.opcua_util
from opcua_empa.controller import FixTimeConstController
from opcua_empa.opcua_util import NodeAndValues
from opcua_empa.opcuaclient_subscription import OpcuaClient
from opcua_empa.room_control_client import ControlClient
from util.util import get_min_diff


class OfflineClient(OpcuaClient):
    """Test client that works offline and returns arbitrary values.

    One room only!
    """

    connected: bool = False
    subscribed: bool = False

    node_strs: List[str]
    n_read_vals: int

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.assert_connected()
        pass

    def read_values(self) -> pd.DataFrame:
        assert self.subscribed, "No subscription!"
        # TODO: Make this useful!
        vals = ["1" for _ in range(self.n_read_vals)]
        return pd.DataFrame({'node': self.node_strs,
                             'value': vals})

    def publish(self, df_write: pd.DataFrame,
                log_time: bool = False,
                sleep_after: float = None) -> None:
        self.assert_connected()

    def subscribe(self, df_read: pd.DataFrame, sleep_after: float = None) -> None:
        self.assert_connected()
        self.subscribed = True
        pd_sub_df: pd.DataFrame = df_read.sort_index()
        self.node_strs = [opcua_empa.opcua_util._trf_node(i) for i in pd_sub_df['node']]
        self.n_read_vals = len(self.node_strs)

    def assert_connected(self):
        assert self.connected, "Not connected!"

    pass


class TestOpcua(TestCase):
    """Tests the opcua client and related stuff.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_val = 10.0
        self.cont = [(575, FixTimeConstController(val=self.c_val, max_n_minutes=1))]

    def test_string_manipulation(self):
        inp = "Hoi_Du"
        exp = "strHoi.strDu"
        res = opcua_empa.opcua_util._th_string_to_node_name(inp)
        self.assertEqual(res[-len(exp):], exp)

    def test_min_diff(self):
        d1 = datetime.datetime(2005, 7, 14, 13, 30)
        d2 = datetime.datetime(2005, 7, 14, 12, 30)
        min_diff = get_min_diff(d2, d1)
        self.assertAlmostEqual(min_diff, 60.0)

    def test_node_and_values(self):
        nav = NodeAndValues(self.cont)
        nodes = nav.get_nodes()
        self.assertEqual(len(self.cont) * 3, len(nodes))
        vals = nav.compute_current_values()
        self.assertEqual(vals[0], self.c_val)
        self.assertEqual(vals[1], True)

    def test_offline_client(self):
        nav = NodeAndValues(self.cont)
        read_nodes = nav.get_read_nodes()
        df_read = pd.DataFrame({'node': read_nodes})
        with OfflineClient() as client:
            client.subscribe(df_read)
            client.publish(df_read)
            r_vals = client.read_values()
            nav.extract_values(r_vals)

    def test_control_client(self):
        with ControlClient(self.cont,
                           exp_name="OfflineTest",
                           client_class=OfflineClient) as cc:
            cc.read_publish_wait_check()
        pass
