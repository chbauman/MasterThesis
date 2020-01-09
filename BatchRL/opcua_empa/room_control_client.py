"""Client that combines the node definitions and the client.

"""
import logging
from typing import List

import pandas as pd
import numpy as np

from opcua_empa.opcua_util import NodeAndValues, ControlT
from opcua_empa.opcuaclient_subscription import OpcuaClient
from util.numerics import check_in_range


print_fun = logging.warning


class ControlClient:

    TEMP_MIN_MAX = (18.0, 25.0)

    write_nodes: List[str]
    read_nodes: List[str]

    n_pub: int = 0

    def __init__(self,
                 used_control: ControlT,
                 exp_name: str = None,
                 user: str = 'ChristianBaumannETH2020',
                 password: str = 'Christian4_ever',
                 verbose: int = 3):

        self.verbose = verbose
        self.client = OpcuaClient(user=user, password=password)
        self.node_gen = NodeAndValues(used_control, exp_name=exp_name)

    def __enter__(self):

        # Get node strings
        self.write_nodes = self.node_gen.get_nodes()
        self.read_nodes = self.node_gen.get_read_nodes()

        curr_vals = self.node_gen.compute_current_values()
        self.df_write = pd.DataFrame({'node': self.write_nodes, 'value': curr_vals})
        self.df_read = pd.DataFrame({'node': self.read_nodes})

        # Connect client and subscribe
        self.client.__enter__()
        self.client.subscribe(self.df_read, sleep_after=1.0)

        return self

    def __exit__(self, *args, **kwargs):
        # Save data and exit client
        self.node_gen.save_cached_data()
        self.client.__exit__(*args, **kwargs)

    def read_publish_wait_check(self) -> bool:
        # Read values
        read_vals = self.client.read_values()

        # Compute and publish current control input
        self.df_write["value"] = self.node_gen.compute_current_values()
        self.client.publish(self.df_write, log_time=True, sleep_after=1.0)

        # Extract values
        ext_values = self.node_gen.extract_values(read_vals)
        if self.verbose:
            print_fun(f"Temperature setpoint: {self.df_write['value'][0]}")
            print_fun(f"Valves: {self.node_gen.get_valve_values()[0]}")

        # Check that the research acknowledgement is true.
        # Wait for at least 20s before requiring to be true, takes some time.
        res_ack_true = np.all(ext_values[0]) or self.n_pub < 20

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
        self.n_pub += 1
        return cont
