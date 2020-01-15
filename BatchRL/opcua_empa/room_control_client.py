"""Client that combines the node definitions and the client.

Mainly about the class :class:`ControlClient` which
uses composition to combine the classes :class:`opcua_empa.opcua_util.NodeAndValues`
and :class:`opcua_empa.opcuaclient_subscription.OpcuaClient`.

.. moduleauthor:: Christian Baumann
"""
import logging
from typing import List, Callable

import pandas as pd
import numpy as np

from opcua_empa.opcua_util import NodeAndValues
from opcua_empa.controller import ControlT
from opcua_empa.opcuaclient_subscription import OpcuaClient
from util.numerics import check_in_range


print_fun = logging.warning


def run_control(*args, **kwargs):
    """Runs the controller until termination.

    Takes the same arguments as :func:`ControlClient.__init__`.
    """
    with ControlClient(*args, **kwargs) as client:

        cont = True
        while cont:
            cont = client.read_publish_wait_check()


class ControlClient:
    """Client combining the node definition and the opcua client.

    Use it as a context manager!
    """

    TEMP_MIN_MAX = (20.0, 25.0)  #: Temperature bounds, experiment will be aborted if temperature leaves these bounds.

    write_nodes: List[str]  #: List with the read nodes as strings.
    read_nodes: List[str]  #: List with the write nodes as strings.

    _n_pub: int = 0

    def __init__(self,
                 used_control: ControlT,
                 exp_name: str = None,
                 user: str = 'ChristianBaumannETH2020',
                 password: str = 'Christian4_ever',
                 verbose: int = 3,
                 no_data_saving: bool = False,
                 _client_class: Callable = OpcuaClient):
        """Initializer.

        A non-default `_client_class` should be used for testing / debugging only.
        E.g. use :class:`tests.test_opcua.OfflineClient` if you are working offline and
        want to test something.
        """

        assert len(used_control) == 1, "Only one room supported!"

        self.verbose = verbose
        self.client = _client_class(user=user, password=password)
        self.node_gen = NodeAndValues(used_control, exp_name=exp_name)

        if no_data_saving:
            self.node_gen.save_cached_data = self._no_save

    def _no_save(self, verbose: bool = False):
        """Used to overwrite the save function of `self.node_gen`."""
        if self.verbose or verbose:
            print("Saving data...")

    def __enter__(self):

        # Get node strings
        self.write_nodes = self.node_gen.get_nodes()
        self.read_nodes = self.node_gen.get_read_nodes()

        # Initialize dataframes
        self.df_write = pd.DataFrame({'node': self.write_nodes, 'value': None})
        self.df_read = pd.DataFrame({'node': self.read_nodes})

        # Connect client and subscribe
        self.client.__enter__()
        self.client.subscribe(self.df_read, sleep_after=1.0)

        return self

    def __exit__(self, *args, **kwargs):
        # Save data and exit client
        self.node_gen.save_cached_data(self.verbose)
        self.client.__exit__(*args, **kwargs)

    def read_publish_wait_check(self) -> bool:
        """Read and publish values, wait, and check if termination is reached.

        If `self.verbose` is True, some information is logged.

        Returns:
            Whether termination is reached.
        """
        # Read values
        read_vals = self.client.read_values()
        ext_values = self.node_gen.extract_values(read_vals)

        # Compute and publish current control input
        self.df_write["value"] = self.node_gen.compute_current_values()
        self.client.publish(self.df_write, log_time=True, sleep_after=1.0)

        # Extract values
        if self.verbose:
            print_fun(f"Temperature setpoint: {self.df_write['value'][0]}")
            print_fun(f"Valves: {self.node_gen.get_valve_values()[0]}")

        # Check that the research acknowledgement is true.
        # Wait for at least 20s before requiring to be true, takes some time.
        res_ack_true = np.all(ext_values[0]) or self._n_pub < 20

        # Check measured temperatures, stop if too low or high.
        temps_in_bound = check_in_range(np.array(ext_values[1]), *self.TEMP_MIN_MAX)

        # Stop if (first) controller gives termination signal.
        terminate_now = self.node_gen.control[0][1].terminate()
        cont = res_ack_true and temps_in_bound and not terminate_now

        # Print the reason of termination.
        if self.verbose > 0:
            print_fun(f"Extracted: {ext_values}")
            if not temps_in_bound:
                print_fun("Temperature bounds reached, aborting experiment.")
            if not res_ack_true:
                print_fun("Research mode confirmation lost :(")
            if terminate_now:
                print_fun("Experiment time over!")

        # Increment publishing counter and return termination criterion.
        self._n_pub += 1
        return cont
