"""Client that combines the node definitions and the client.

Mainly about the class :class:`ControlClient` which
uses composition to combine the classes :class:`opcua_empa.opcua_util.NodeAndValues`
and :class:`opcua_empa.opcuaclient_subscription.OpcuaClient`.

.. moduleauthor:: Christian Baumann
"""
import logging
import time
from datetime import datetime
from threading import Lock
from typing import List, Callable, Tuple

import numpy as np
import pandas as pd

from opcua_empa.controller import ControlT
from opcua_empa.opcua_util import NodeAndValues
from opcua_empa.opcuaclient_subscription import OpcuaClient
from util.notify import send_mail, set_exit_handler, read_pw_from_file
from util.numerics import check_in_range
from util.util import ProgWrap

print_fun = logging.warning


def run_control(used_control: ControlT,
                exp_name: str = None,
                *args, verbose: int = 0,
                debug: bool = False,
                **kwargs):
    """Runs the controller until termination.

    Takes the same arguments as :func:`ControlClient.__init__`, except
    for an additional one, `debug` which decides where to send the mail to.
    """

    with ControlClient(used_control, exp_name, *args,
                       verbose=1 if verbose > 0 else 0,
                       debug_mail=debug, **kwargs) as client:
        cont = True
        while cont:
            if not client._exited:
                cont = client.read_publish_wait_check()
            else:
                time.sleep(0.5)


class ControlClient:
    """Client combining the node definition and the opcua client.

    Use it as a context manager!
    """

    TEMP_MIN_MAX = (20.0, 25.0)  #: Temperature bounds, experiment will be aborted if temperature leaves these bounds.

    write_nodes: List[str]  #: List with the read nodes as strings.
    read_nodes: List[str]  #: List with the write nodes as strings.

    termination_reason: str = None

    _n_pub: int = 0

    _curr_ex_vals: Tuple = None
    _curr_temp_sp: float = None
    _curr_valves: Tuple = None
    _start_time: datetime = None

    _started_exiting: bool = False
    _exited: bool = False
    exit_lock = Lock()
    _add_msg: str = None

    def __init__(self,
                 used_control: ControlT,
                 exp_name: str = None,
                 user: str = 'ChristianBaumannETH2020',
                 password: str = 'Christian4_ever', *,
                 verbose: int = 1,
                 no_data_saving: bool = False,
                 notify_failures: bool = False,
                 debug_mail: bool = True,
                 _client_class: Callable = OpcuaClient):
        """Initializer.

        A non-default `_client_class` should be used for testing / debugging only.
        E.g. use :class:`tests.test_opcua.OfflineClient` if you are working offline and
        want to test something.
        """
        assert len(used_control) == 1, "Only one room supported!"

        self.notify_failures = notify_failures
        self.verbose = verbose
        self._start_time = datetime.now()
        self.client = _client_class(user=user, password=password)
        self.node_gen = NodeAndValues(used_control, exp_name=exp_name)

        self._pw = read_pw_from_file()
        self.deb_mail = debug_mail

        if no_data_saving:
            self.node_gen.save_cached_data = self._no_save

    def _no_save(self, verbose: bool = False):
        """Used to overwrite the save function of `self.node_gen`."""
        if self.verbose or verbose:
            print("Not saving data...")

    def __enter__(self):
        """Setup the ControlClient.

        Define nodes, initialize dataframes and enter
        and subscribe with client."""

        # Get node strings
        self.write_nodes = self.node_gen.get_nodes()
        self.read_nodes = self.node_gen.get_read_nodes()

        # Initialize dataframes
        self.df_write = pd.DataFrame({'node': self.write_nodes, 'value': None})
        self.df_read = pd.DataFrame({'node': self.read_nodes})

        # Connect client and subscribe
        self.client.__enter__()
        self.client.subscribe(self.df_read, sleep_after=1.0)

        # Set exit handler
        def on_exit(sig, func=None):
            add_msg = f"Program was mysteriously killed by somebody or something. "
            self._add_msg = add_msg
            self.__exit__(None, None, None)

        set_exit_handler(on_exit)

        return self

    def __exit__(self, *args, **kwargs):
        """Save data and exit client."""
        if self.verbose:
            print("Exiting...")

        self.exit_lock.acquire()
        if not self._exited:
            self._exited = True
            self.exit_lock.release()
            print("Actually exiting :)")

            self.client.__exit__(*args, **kwargs)
            self.node_gen.save_cached_data(self.verbose)

            # Notify reason of termination
            time.sleep(0.5)
            with ProgWrap(f"Sending notification...", self.verbose > 0):
                self.notify_me()

        else:
            self.exit_lock.release()

    def _print_set_on_change(self, attr_name: str, val, msg: str) -> None:
        curr_val = getattr(self, attr_name)
        if curr_val != val:
            setattr(self, attr_name, val)
            if self.verbose > 0:
                print_fun(f"{msg}: {val}")
        elif self.verbose > 1:
            print_fun(f"{msg}: {val}")

    def notify_me(self) -> None:
        """Sends a notification mail with the reason of termination.

        Does nothing if `self.notify_failures` is False.
        """
        # Set subject
        sub = "Experiment Termination Notification"

        # Set message
        msg = self.termination_reason
        if msg is None:
            msg = "Unknown termination reason :("

        if self._add_msg is not None:
            msg += f"\n\n{self._add_msg}"

        # Add some more information
        msg += f"\n\nExperiment name: {self.node_gen.experiment_name}"
        msg += f"\n\nStarting date and time: {self._start_time}"

        # Send mail
        print("Hoi")
        send_mail(subject=sub, msg=msg, debug=self.deb_mail,
                  password=self._pw)

        print("Hoi2")

    def read_publish_wait_check(self) -> bool:
        """Read and publish values, wait, and check if termination is reached.

        If `self.verbose` is True, some information is logged.

        Returns:
            Whether termination is reached.
        """
        # Read and extract values
        read_vals = self.client.read_values()
        ext_values = self.node_gen.extract_values(read_vals)
        ex_vals = (ext_values[0][0], ext_values[1][0])
        self._print_set_on_change("_curr_ex_vals", ex_vals,
                                  msg="Extracted")
        valve_tuple = tuple(self.node_gen.get_valve_values()[0])
        self._print_set_on_change("_curr_valves", valve_tuple,
                                  msg="Valves")

        # Compute and publish current control input
        self.df_write["value"] = self.node_gen.compute_current_values()
        self.client.publish(self.df_write, log_time=self.verbose > 1, sleep_after=1.0)

        self._print_set_on_change("_curr_temp_sp", self.df_write['value'][0],
                                  msg="Temperature setpoint")

        # Check that the research acknowledgement is true.
        # Wait for at least 20s before requiring to be true, takes some time.
        res_ack_true = np.all(ext_values[0]) or self._n_pub < 20

        # Check measured temperatures, stop if too low or high.
        temps_in_bound = check_in_range(np.array(ext_values[1]), *self.TEMP_MIN_MAX)

        # Stop if (first) controller gives termination signal.
        terminate_now = self.node_gen.control[0][1].terminate()
        cont = res_ack_true and temps_in_bound and not terminate_now

        # Print the reason of termination.
        if not temps_in_bound:
            self.termination_reason = "Temperature bounds reached, aborting experiment."
        if not res_ack_true:
            self.termination_reason = "Research mode confirmation lost :("
        if terminate_now:
            self.termination_reason = "Experiment time over!"

        if not cont and self.verbose > 0:
            print_fun(self.termination_reason)

        raise Exception("Fuck")

        # Increment publishing counter and return termination criterion.
        self._n_pub += 1
        return cont
