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

TH_SUFFIXES: List[str] = [
    "rValue1",
    "bReqResearch",
    "bWdResearch",
]


def th_string_to_node_name(th_str: str, ext: str = "") -> str:
    n1, n2 = th_str.split("_")
    pre = f"ns=2;s=Gateway.PLC1.65NT-71331-D001.PLC1.Units.str3T3.strWrite_L.strSensoren.str{n1}.str{n2}"
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
        n_str = th_string_to_node_name(ROOM_DICT[r_nr])
        node_list += [n_str + "." + s for s in TH_SUFFIXES]
    return node_list


class NodeAndValues:
    control: ControlT
    nodes: List[str]

    def __init__(self, control: ControlT):
        self.control = control
        self.nodes = _get_nodes(control)

    def get_nodes(self) -> List[str]:
        return self.nodes

    def get_values(self) -> List:
        return _get_values(self.control)
