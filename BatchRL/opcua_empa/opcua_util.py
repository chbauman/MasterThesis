"""Utility module for the opcua client.

Defines node strings for the nodes needed to control rooms at DFAB.
Includes the read and the write nodes.
"""

import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from opcua_empa.controller import StatefulController, ControlT
from opcua_empa.opcuaclient_subscription import toggle
from rest.client import save_dir
from util.numerics import has_duplicates, nan_avg_between
from util.util import str2bool, create_dir, now_str
from util.visualize import plot_valve_opening, PLOT_DIR

experiment_data_path = os.path.join(save_dir, "Experiments")  #: Experiment data directory
experiment_plot_path = os.path.join(PLOT_DIR, "Experiments")  #: Experiment plot directory
create_dir(experiment_data_path)
create_dir(experiment_plot_path)

# The dictionary mapping room numbers to thermostat strings
ROOM_DICT: Dict[int, str] = {
    371: "R1_B870",
    472: "R2_B870",
    473: "R2_B871",
    475: "R2_B872",
    571: "R3_B870",
    573: "R3_B871",
    575: "R3_B872",
}

ALL_ROOM_NRS = [k for k in ROOM_DICT]

# The inverse dictionary of the above one
INV_ROOM_DICT = {v: k for k, v in ROOM_DICT.items()}

# Valves of each room
ROOM_VALVE_DICT: Dict[int, List[str]] = {
    371: ["Y700", "Y701", "Y702", "Y703", "Y704", "Y705", "Y706"],
    472: ["Y700", "Y701", "Y706"],
    473: ["Y702"],
    475: ["Y703", "Y704", "Y705"],
    571: ["Y700", "Y705", "Y706"],
    573: ["Y704"],
    575: ["Y701", "Y702", "Y703"],
}

# Read nodes that are the same for each room
read_node_names = [
    # Weather
    'ns=2;s=Gateway.PLC1.65NT-03032-D001.PLC1.MET51.strMET51Read.strWetterstation.strStation1.lrLufttemperatur',
    'ns=2;s=Gateway.PLC1.65NT-06421-D001.PLC1.Units.str2T5.strRead.strSensoren.strW1.strB870.rValue5',
    # Heating water temperatures
    "ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strM1.strB810.rValue1",
    "ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strRead.strSensoren.strM1.strB814.rValue1",
]
read_node_descs_and_types = ([
                                 "Outside Temp.",
                                 "Irradiance",
                                 "Water Temp. In",
                                 "Water Temp. Out",
                             ], [
                                 float,
                                 float,
                                 float,
                                 float,
                             ])

TH_SUFFIXES: List[str] = [
    "rValue1",
    "bReqResearch",
    "bWdResearch",
]

READ_SUF_NAME_TYPES: List[Tuple[str, str, type]] = [
    ("bAckResearch", "Research Acknowledged", bool),
    ("rValue1", "Measured Temp.", float),
    ("rValue2", "Temp. Set-point Feedback", float),
    ("bValue1", "", bool),
]

BASE_NODE_STR = f"ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3."


def check_room_list(room_list: List[int]) -> None:
    """Checks if IDs of rooms in list are valid.

    Raises:
        AssertionError: If any ID is invalid or the argument is
            not a list.
    """
    # Check list with room numbers
    assert isinstance(room_list, list), f"Room list: {room_list} needs to be a list!"
    for k in room_list:
        assert k in ALL_ROOM_NRS, f"Invalid room number: {k}"


def _th_string_to_node_name(th_str: str, ext: str = "", read: bool = False) -> str:
    """Turns a thermostat string into a node name string.

    `th_str` should be contained as a value in `ROOM_DICT`.
    """
    n1, n2 = th_str.split("_")
    rw_part = "strRead" if read else "strWrite_L"
    pre = BASE_NODE_STR + rw_part + f".strSensoren.str{n1}.str{n2}"
    return pre + ext


def _trf_node(node_str: str) -> str:
    return f"Node(StringNodeId({node_str}))"


def _get_values(control: ControlT) -> List:
    val_list = []
    for c in control:
        r_nr, val_fun = c
        val_list += [
            val_fun(),
            True,
            toggle(),
        ]
    return val_list


def _get_nodes(control: ControlT) -> List:
    node_list = []
    for c in control:
        r_nr, val_fun = c
        n_str = _th_string_to_node_name(ROOM_DICT[r_nr])
        node_list += [n_str + "." + s for s in TH_SUFFIXES]
    return node_list


def _get_read_nodes(control: ControlT) -> Tuple[List[str], List[str], List[int], List[type]]:
    # Initialize lists
    node_list, node_descs, room_inds, types = [], [], [], []

    # Iterate over all rooms to be controlled.
    for c in control:
        r_nr, _ = c
        valves = ROOM_VALVE_DICT[r_nr]
        room_str = ROOM_DICT[r_nr]
        s1, s2 = room_str.split("_")

        # Add temperature feedback
        b_s = _th_string_to_node_name(room_str, read=True)
        room_inds += [len(node_descs)]
        for s, d, t in READ_SUF_NAME_TYPES:
            node_list += [b_s + "." + s]
            node_descs += [f"{r_nr}: {d}"]
            types += [t]

        # Add valves
        for v in valves:
            n = s1[1]
            v_s = BASE_NODE_STR + f"strRead.strAktoren.strZ{n}.str{v}.bValue1"
            node_list += [v_s]
            node_descs += [f"{r_nr}: Valve {v}"]
            types += [bool]
    return node_list, node_descs, room_inds, types


def str_to_dt(s: str, dt: type):
    """Converts a string to type `dt`."""
    if dt is bool:
        return str2bool(s)
    elif dt in [int, float, str]:
        return dt(s)
    else:
        raise NotImplementedError(f"Dtype: {dt} not supported!")


def analyze_experiment(exp_file_name: str):
    """Analyzes the data generated in an experiment.

    Assumes one room only, with three valves."""
    exp_name = os.path.splitext(os.path.basename(exp_file_name))[0]

    # Load data
    with open(exp_file_name, "rb") as f:
        data = pickle.load(f)
    read_vals, read_ts, write_vals, write_ts = data

    # Extract relevant data parts
    valve_data = read_vals[:, 4:7]
    temp_set_p = write_vals[:, 0]
    temp_set_p_meas = read_vals[:, 2]
    res_req = read_vals[:, 1]

    # TODO: Check res_req values!
    assert np.any(res_req)

    # Plot valve opening and closing
    valve_plt_path = os.path.join(experiment_plot_path, exp_name)
    plot_valve_opening(read_ts, valve_data, valve_plt_path, write_ts, temp_set_p, temp_set_p_meas)

    pass


class NodeAndValues:
    """Class that defines nodes and values to be used with the opcua client.

    Designed for room temperature control at DFAB.
    Number of rooms to be controlled can vary.

    TODO: Integrate into Opcua Client???
    """
    n_rooms: int  #: The number of rooms.
    control: ControlT  #: The controller list.
    nodes: List[str]  #: The nodes to be written.
    read_nodes: List[str]  #: The nodes to be read.
    read_desc: List[str]  #: The descriptions of the read nodes.
    room_inds: List[int]  #: The indices of the rooms.

    # Read data arrays
    read_timestamps: np.ndarray = None
    read_values: np.ndarray = None

    # Write data arrays
    write_timestamps: np.ndarray = None
    write_values: np.ndarray = None

    experiment_name: str

    _extract_node_strs: List[List]
    _curr_read_n: int = 0
    _curr_write_n: int = 0

    def __init__(self, control: ControlT, exp_name: str = None):

        self.n_rooms = len(control)
        assert self.n_rooms > 0, "No rooms to be controlled!"

        self.control = control
        self.nodes = _get_nodes(control)
        n, d, i, t = _get_read_nodes(control)
        add_read_nt = read_node_descs_and_types
        self.read_nodes, self.read_desc = n + read_node_names, d + add_read_nt[0]
        self.room_inds, self.read_types = i, t + add_read_nt[1]

        # Check for duplicate room numbers in control
        room_inds = np.array([c[0] for c in control])
        assert not has_duplicates(room_inds), f"Multiply controlled rooms: {room_inds}!"

        # Strings used for value extraction
        inds = [0, 1]
        self._extract_node_strs = [
            [_trf_node(self.read_nodes[r_ind + i]) for i in inds]
            for r_ind in self.room_inds
        ]
        self.read_dict = self._get_read_dict()

        # Initialize data arrays
        self.n_max = 3600
        dtypes = np.dtype([(s, t)
                           for s, t in zip(self.read_desc, self.read_types)])

        self.read_df = np.empty((self.n_max,), dtype=dtypes)
        self.read_values = np.empty((self.n_max, len(self.read_desc)), dtype=np.float32)
        self.read_timestamps = np.empty((self.n_max,), dtype='datetime64[s]')
        self.write_values = np.empty((self.n_max, 3 * self.n_rooms), dtype=np.float32)
        self.write_timestamps = np.empty((self.n_max,), dtype='datetime64[s]')

        # Fill arrays with nans
        self.read_values.fill(np.nan)
        self.write_values.fill(np.nan)
        self.read_timestamps.fill(np.nan)
        self.write_timestamps.fill(np.nan)

        self.n_valve_list = [len(ROOM_VALVE_DICT[r]) for r, _ in control]

        # Define experiment name
        self.experiment_name = now_str() + ("" if exp_name is None else "_" + exp_name)

    def get_filename(self, ext: str = ".pkl"):
        return os.path.join(experiment_data_path, self.experiment_name + ext)

    def save_cached_data(self) -> None:
        """Save the current data in cache to file."""
        all_data = [self.read_values, self.read_timestamps,
                    self.write_values, self.write_timestamps]
        logging.warning("Saving Experiment Data")
        f_name = self.get_filename()
        with open(f_name, "wb") as f:
            pickle.dump(all_data, f)

    def _inc(self, att_str: str):
        """Increments or resets counter."""
        curr_ct = getattr(self, att_str)
        curr_ct += 1
        if curr_ct == self.n_max:
            # TODO: Save to disk or do something with the data
            curr_ct = 0
        setattr(self, att_str, curr_ct)

    def get_nodes(self) -> List[str]:
        return self.nodes

    def get_read_nodes(self) -> List[str]:
        return self.read_nodes

    def get_read_node_descs(self) -> List[str]:
        return self.read_desc

    def _get_read_dict(self) -> Dict:
        """Creates a dict that maps node strings to indices."""
        inds = range(len(self.read_nodes))
        return {_trf_node(s): ind for s, ind in zip(self.read_nodes, inds)}

    def compute_current_values(self) -> List:
        """Computes current control inputs."""

        # Get values
        values = _get_values(self.control)

        # TODO: Make control depending on state!
        if isinstance(self.control, StatefulController):
            # Compute state
            state = np.empty((6,), dtype=np.float32)
            curr_read_vals = self.read_values[self._curr_read_n]
            state[:4] = curr_read_vals[-4:]  # Assign weather and water
            state[4] = np.mean(curr_read_vals[4:7])  # Assign valve value
            state[5] = curr_read_vals[1]  # Assign Room temperature

            # Set state
            self.control.set_state(state)

        # Save values in memory
        self.write_timestamps[self._curr_write_n] = np.datetime64('now')
        for ct, v in enumerate(values):
            self.write_values[self._curr_write_n, ct * 3: (ct + 1) * 3] = v

        # Increment counter and return values
        self._inc("_curr_write_n")
        return values

    def get_avg_last_vals(self, n_mins: int = 15) -> np.ndarray:
        return nan_avg_between(self.read_timestamps, self.read_values, n_mins)

    def get_valve_values(self, all_prev: bool = False) -> List[np.ndarray]:
        """Returns the state of the valves of all rooms.

        Need to call `extract_values(...)` first.
        If `all_prev` is true, returns also all previous values in memory.
        """
        last_n = max(self._curr_read_n - 1, 0)
        ind = slice(None) if all_prev else last_n
        val_vals = [self.read_values[ind,
                                     self.room_inds[i] + 4: (self.room_inds[i] + self.n_valve_list[i] + 4)]
                    for i in range(self.n_rooms)]
        return val_vals

    def extract_values(self, read_df: pd.DataFrame) -> Tuple[List, List]:

        # Save current time and set values to nan
        self.read_timestamps[self._curr_read_n] = np.datetime64('now')
        self.read_values[self._curr_read_n] = np.nan

        # Order may vary, so we iterate and find index
        for k, row in read_df.iterrows():
            s, val = row["node"], row["value"]
            ind = self.read_dict.get(s)
            if ind is None:
                logging.warning(f"String: {s} not found!")
                continue
            val = str_to_dt(val, self.read_types[ind])
            self.read_values[self._curr_read_n, ind] = val

        # Extract research acknowledgement and current room temp
        inds = [0, 1]
        res_ack = [self.read_values[self._curr_read_n, i + inds[0]]
                   for i in self.room_inds]
        temps = [self.read_values[self._curr_read_n, i + inds[1]]
                 for i in self.room_inds]

        # Increment counter and return values
        self._inc("_curr_read_n")
        return res_ack, temps
