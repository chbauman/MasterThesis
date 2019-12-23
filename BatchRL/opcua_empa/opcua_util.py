import datetime
import time
from typing import Dict, List, Tuple, Any, Union, Callable

import numpy as np
import pandas as pd

from opcua_empa.opcuaclient_subscription import toggle
from util.numerics import has_duplicates
from util.util import Num, str2bool

# The dictionary mapping room numbers to thermostat strings
ROOM_DICT: Dict[int, str] = {
    472: "R2_B870",
    473: "R2_B871",
    475: "R2_B872",
    571: "R3_B870",
    573: "R3_B871",
    575: "R3_B872",
}

# The inverse dictionary of the above one
INV_ROOM_DICT = {v: k for k, v in ROOM_DICT.items()}

# Values: [write code,
EXT_ROOM_VALVE_DICT: Dict[int, Any] = {
    472: ["Y700", "Y701", "Y706"],
    473: ["Y702"],
    475: ["Y703", "Y704", "Y705"],
    571: ["Y700", "Y705", "Y706"],
    573: ["Y704"],
    575: ["Y701", "Y702", "Y703"],
}

read_node_names = [
    # Weather:
    'ns=2;s=Gateway.PLC1.65NT-03032-D001.PLC1.MET51.strMET51Read.strWetterstation.strStation1.lrLufttemperatur',
    'ns=2;s=Gateway.PLC1.65NT-06421-D001.PLC1.Units.str2T5.strRead.strSensoren.strW1.strB870.rValue5',
]
read_node_descs = [
    "Outside Temp.",
    "Irradiance",
]

TH_SUFFIXES: List[str] = [
    "rValue1",
    "bReqResearch",
    "bWdResearch",
]

READ_SUFFIXES: List[Tuple[str, str, type]] = [
    ("bAckResearch", "Research Acknowledged", bool),
    ("rValue1", "Measured Temp.", float),
    ("rValue2", "", float),
    ("bValue1", "Temp. Set-point Feedback", bool),
]

base_s = f"ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3."


def _th_string_to_node_name(th_str: str, ext: str = "", read: bool = False) -> str:
    n1, n2 = th_str.split("_")
    rw_part = "strRead" if read else "strWrite_L"
    pre = base_s + rw_part + f".strSensoren.str{n1}.str{n2}"
    return pre + ext


def get_min_diff(t1, t2):
    d1_ts = time.mktime(t1.timetuple())
    d2_ts = time.mktime(t2.timetuple())
    return (d2_ts - d1_ts) / 60


# Type definitions for control
ControllerT = Union[Callable[[], Num], Num]
ControlT = List[Tuple[int, ControllerT]]


class FixTimeConstController:

    def __init__(self, val: Num = 20, max_n_minutes: int = None):
        self.val = val
        self.max_n_minutes = max_n_minutes
        self.start_time = datetime.datetime.now()

    def __call__(self) -> Num:
        return self.val

    def terminate(self) -> bool:
        """Checks if the maximum time is reached.

        Returns:
            True if the max. runtime is reached, else False.
        """
        if self.max_n_minutes is None:
            return False
        time_now = datetime.datetime.now()
        h_diff = get_min_diff(self.start_time, time_now)
        return h_diff > self.max_n_minutes


class ToggleController(FixTimeConstController):

    def __init__(self, val_low: Num = 20, val_high: Num = 22, n_mins: int = 2,
                 start_low: bool = True, max_n_minutes: int = None):
        """Controller that toggles every `n_mins` between two values.

        If you need a constant controller, set `val_low` == `val_high`.

        Args:
            val_low: The lower value.
            val_high: The higher value.
            n_mins: The number of minutes in an interval.
            start_low: Whether to start with `val_low`.
            max_n_minutes: The maximum number of minutes the controller should run.
        """
        super().__init__(val_low, max_n_minutes)
        self.v_low = val_low
        self.v_high = val_high
        self.dt = n_mins
        self.start_low = start_low
        self.start_time = datetime.datetime.now()
        # self.max_n_minutes = max_n_minutes

    def __call__(self) -> Num:
        """Computes the current value according to the current time."""
        time_now = datetime.datetime.now()
        min_diff = get_min_diff(self.start_time, time_now)
        is_start_state = int(min_diff) % (2 * self.dt) < self.dt
        is_low = is_start_state if self.start_low else not is_start_state
        return self.v_low if is_low else self.v_high


def comp_val(v: ControllerT) -> Num:
    if type(v) in [float, int]:
        return v
    elif callable(v):
        return v()
    else:
        raise NotImplementedError("Only numerical values or functions "
                                  "taking no arguments allowed!")


def _trf_node(node_str: str) -> str:
    return f"Node(StringNodeId({node_str}))"


def _get_values(control: ControlT) -> List:
    val_list = []
    for c in control:
        r_nr, val_fun = c
        val_list += [
            comp_val(val_fun),
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
    node_list, node_descs, room_inds, types = [], [], [], []
    for c in control:
        r_nr, _ = c
        valves = EXT_ROOM_VALVE_DICT[r_nr]
        room_str = ROOM_DICT[r_nr]
        s1, s2 = room_str.split("_")

        # Add temperature feedback
        b_s = _th_string_to_node_name(room_str, read=True)
        room_inds += [len(node_descs)]
        for s, d, t in READ_SUFFIXES:
            node_list += [b_s + "." + s]
            node_descs += [f"{r_nr}: {d}"]
            types += [t]

        # Add valves
        for v in valves:
            n = s1[1]
            v_s = base_s + f"strRead.strAktoren.strZ{n}.str{v}.bValue1"
            node_list += [v_s]
            node_descs += [f"{r_nr}: Valve {v}"]
            types += [bool]
    return node_list, node_descs, room_inds, types


def str_to_dt(s: str, dt: type):
    if dt is bool:
        return str2bool(s)
    elif dt is int:
        return int(s)
    elif dt is float:
        return float(s)
    elif dt is str:
        return s
    else:
        raise NotImplementedError(f"Dtype: {dt} not supported!")


class NodeAndValues:
    n_rooms: int
    control: ControlT
    nodes: List[str]
    read_nodes: List[str]
    read_desc: List[str]
    room_inds: List[int]

    read_df: np.ndarray = None
    write_df: pd.DataFrame = None

    _extract_node_strs: List[List]
    _curr_read_n: int = 0

    def __init__(self, control: ControlT):

        self.n_rooms = len(control)
        assert self.n_rooms > 0, "No rooms to be controlled!"

        self.control = control
        self.nodes = _get_nodes(control)
        n, d, i, t = _get_read_nodes(control)
        self.read_nodes, self.read_desc = n, d
        self.room_inds, self.read_types = i, t

        # Check for duplicate room numbers in control
        room_inds = np.array([c[0] for c in control])
        assert not has_duplicates(room_inds), "Multiply controlled rooms!"

        # Strings used for value extraction
        inds = [0, 1]
        self._extract_node_strs = [
            [_trf_node(self.read_nodes[r_ind + i]) for i in inds]
            for r_ind in self.room_inds
        ]
        self.read_dict = self._get_read_dict()

        self.n_max = 3600
        dtypes = np.dtype([(s, t)
                           for s, t in zip(self.read_desc, self.read_types)])
        self.read_df = np.empty((self.n_max,), dtype=dtypes)

    def get_nodes(self) -> List[str]:
        return self.nodes

    def get_read_nodes(self) -> List[str]:
        return self.read_nodes + read_node_names

    def get_read_node_descs(self) -> List[str]:
        return self.read_desc + read_node_descs

    def _get_read_dict(self) -> Dict:
        """Creates a dict that maps node strings to indices."""
        inds = range(len(self.read_nodes))
        return {_trf_node(s): ind for s, ind in zip(self.read_nodes, inds)}

    def compute_current_values(self) -> List:
        return _get_values(self.control)

    def extract_values(self, read_df: pd.DataFrame) -> Tuple[List, List]:

        for k, row in read_df.iterrows():
            s, val = row["node"], row["value"]
            ind = self.read_dict.get(s)
            if ind is None:
                print(f"String: {s} not found!")
                continue
            col_name = self.read_desc[ind]
            val = str_to_dt(val, self.read_types[ind])
            self.read_df[self._curr_read_n][col_name] = val

        # Initialize empty
        res_ack, temps = [], []

        # Iterate over rooms
        for node_strs in self._extract_node_strs:
            # Find research acknowledgement and room temperature values.
            for k, row in read_df.iterrows():
                s, val = row["node"], row["value"]
                if s == node_strs[0]:
                    res_ack += [str2bool(val)]
                elif s == node_strs[1]:
                    temps += [float(val)]

        inds = [0, 1]

        res_ack = [self.read_df[self._curr_read_n][i + inds[0]]
                   for i in self.room_inds]
        temps = [self.read_df[self._curr_read_n][i + inds[1]]
                 for i in self.room_inds]

        print(self.read_desc)
        print(self.read_df[self._curr_read_n])

        return res_ack, temps
