import datetime
import time
from typing import Dict, List, Tuple, Any, Union, Callable

from opcua_empa.opcuaclient_subscription import toggle
from util.util import Num

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


TH_SUFFIXES: List[str] = [
    "rValue1",
    "bReqResearch",
    "bWdResearch",
]

READ_SUFFIXES: List[str] = [
    "bAckResearch",
    "rValue1",
    "rValue2",
    "bValue1",
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


class ToggleController:

    def __init__(self, val_low: Num = 20, val_high: Num = 22, n_mins: int = 2):
        self.v_low = val_low
        self.v_high = val_high
        self.dt = n_mins
        self.start_time = datetime.datetime.now()

    def __call__(self):
        time_now = datetime.datetime.now()
        min_diff = get_min_diff(self.start_time, time_now)
        is_low = int(min_diff) % (2 * self.dt) < self.dt
        return self.v_low if is_low else self.v_high


def comp_val(v: Union[Callable, Num]) -> Num:
    if type(v) in [float, int]:
        return v
    elif callable(v):
        return v()
    else:
        raise NotImplementedError("Only numerical values allowed!")


ControlT = List[Tuple[int, Any]]


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


def _get_read_nodes(control: ControlT) -> List[str]:
    node_list = []
    for c in control:
        r_nr, _ = c
        valves = EXT_ROOM_VALVE_DICT[r_nr]
        room_str = ROOM_DICT[r_nr]
        s1, s2 = room_str.split("_")

        # Add temperature feedback
        b_s = _th_string_to_node_name(room_str, read=True)
        for s in READ_SUFFIXES:
            node_list += [b_s + "." + s]

        # Add valves
        for v in valves:
            n = s1[1]
            v_s = base_s + f"strRead.strAktoren.strZ{n}.str{v}.bValue1"
            node_list += [v_s]
    return node_list


class NodeAndValues:
    control: ControlT
    nodes: List[str]
    read_nodes: List[str]

    def __init__(self, control: ControlT):
        self.control = control
        self.nodes = _get_nodes(control)
        self.read_nodes = _get_read_nodes(control)

    def get_nodes(self) -> List[str]:
        return self.nodes

    def get_read_nodes(self) -> List[str]:
        return self.read_nodes + read_node_names

    def get_values(self) -> List:
        return _get_values(self.control)
