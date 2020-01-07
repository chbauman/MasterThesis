import datetime
from unittest import TestCase

import pandas as pd

import opcua_empa.opcua_util
from opcua_empa.opcua_util import FixTimeConstController, get_min_diff, NodeAndValues
from opcua_empa.opcuaclient_subscription import OpcuaClient


class OfflineClient(OpcuaClient):
    """Test client that works offline and returns arbitrary values."""

    connected: bool = False

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        pass

    def read_values(self) -> pd.DataFrame:
        return pd.DataFrame({'node': ["test1", "test2"]})

    def publish(self, json_write: str) -> None:
        pass

    def subscribe(self, json_read: str) -> None:
        pass

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
