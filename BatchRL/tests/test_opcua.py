import datetime
from typing import List
from unittest import TestCase

import pandas as pd

import opcua_empa.opcua_util
from opcua_empa.controller import FixTimeConstController, ValveToggler, MIN_TEMP, MAX_TEMP
from opcua_empa.opcua_util import NodeAndValues
from opcua_empa.opcuaclient_subscription import OpcuaClient
from opcua_empa.room_control_client import ControlClient, run_control
from util.util import get_min_diff


class OfflineClient(OpcuaClient):
    """Test client that works offline and returns arbitrary values.

    One room only, with three valves!
    Will run until `read_values` is called `N_STEPS_MAX` times, then,
    the read temperature will be set to out of bounds and the
    experiment will terminate.
    """

    N_STEPS_MAX = 10

    _step_ind: int = 0

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
        self._step_ind += 1
        r_temp = "22.0" if self._step_ind < self.N_STEPS_MAX else "35.0"
        r_vals = ["1", r_temp, "28.0", "1"]
        valves = ["1", "1", "1"]
        exo = ["5.0", "0.0", "26.0", "26.0"]
        vals = r_vals + valves + exo
        return pd.DataFrame({'node': self.node_strs,
                             'value': vals})

    def publish(self, df_write: pd.DataFrame,
                log_time: bool = False,
                sleep_after: float = None) -> None:
        assert len(df_write) == 3, f"Only one room supported! (df_write = {df_write})"
        self.assert_connected()

    def subscribe(self, df_read: pd.DataFrame,
                  sleep_after: float = None) -> None:
        self.assert_connected()
        self.subscribed = True
        pd_sub_df: pd.DataFrame = df_read.sort_index()
        self.node_strs = [opcua_empa.opcua_util._trf_node(i) for i in pd_sub_df['node']]
        self.n_read_vals = len(self.node_strs)
        assert self.n_read_vals == 11, f"Wrong number of read nodes: {self.n_read_vals}"

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
        write_nodes = nav.get_nodes()
        df_read = pd.DataFrame({'node': read_nodes})
        df_write = pd.DataFrame({'node': write_nodes, 'value': None})
        with OfflineClient() as client:
            client.subscribe(df_read)
            client.publish(df_write)
            r_vals = client.read_values()
            nav.extract_values(r_vals)

    def test_valve_toggler(self):
        class OCToggle(OfflineClient):
            t_state = False
            op = ["1" for _ in range(3)]
            cl = ["0" for _ in range(3)]

            def read_values(self):
                self.t_state = not self.t_state
                vals = super().read_values()
                vals['value'][4:7] = self.op if self.t_state else self.cl
                return vals

            def publish(self, df_write: pd.DataFrame,
                        log_time: bool = False,
                        sleep_after: float = None) -> None:
                super().publish(df_write, log_time, sleep_after)
                temp_set = df_write['value'][0]

                assert (self.t_state and temp_set == MIN_TEMP) or\
                       (not self.t_state and temp_set == MAX_TEMP)

        vt = [(575, ValveToggler(n_steps_delay=0))]
        run_control(vt,
                    exp_name="OfflineTest",
                    verbose=0,
                    _client_class=OCToggle)

    def test_control_client(self):
        # Experiment will terminate after first iteration since
        # temperature is out of bound.
        with ControlClient(self.cont,
                           exp_name="OfflineTest",
                           verbose=0,
                           _client_class=OfflineClient) as cc:
            cc.read_publish_wait_check()
        pass

    def test_run_control(self):
        run_control(self.cont,
                    exp_name="OfflineTest",
                    verbose=0,
                    _client_class=OfflineClient)
