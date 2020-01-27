"""Module for running the opcua client.

May be removed later and moved to BatchRL.py if
it is high-level enough.
"""
from typing import List

from agents.keras_agents import default_ddpg_agent
from dynamics.load_models import load_room_env
from envs.dynamics_envs import RangeT
from opcua_empa.controller import ValveToggler, ValveTest2Controller, FixTimeConstController, BaseRLController
from opcua_empa.opcua_util import analyze_experiment, check_room_list
from opcua_empa.opcuaclient_subscription import OpcuaClient
from opcua_empa.room_control_client import run_control
from tests.test_opcua import OfflineClient
from util.util import prog_verb, ProgWrap, DEFAULT_ROOM_NR, DEFAULT_EVAL_SET, DEFAULT_TRAIN_SET, DEFAULT_END_DATE


def try_opcua(verbose: int = 1, room_list: List[int] = None, debug: bool = True):
    """Runs the opcua client."""

    if verbose:
        if debug:
            print("Running in debug mode!")

    # Analyze previous experiment
    exp_name = "2020_01_15T21_14_51_R475_Experiment_15min_PT_0"
    with ProgWrap(f"Analyzing experiment {exp_name}...", verbose > 0):
        analyze_experiment(exp_name,
                           compute_valve_delay=True,
                           verbose=prog_verb(verbose))

    # Choose experiment name
    exp_name = "Test"

    # Check list with room numbers
    check_room_list(room_list)

    # Define room and control
    # tc = ToggleController(n_mins=60 * 100, start_low=True, max_n_minutes=60 * 16)
    # tc = ValveToggler(n_steps_delay=30, n_steps_max=2 * 60)
    tc = ValveTest2Controller()
    room_list = [43] if room_list is None else room_list
    used_control = [(i, tc) for i in room_list]
    if debug:
        room_list = [41]
        used_control = [(r, ValveToggler(n_steps_delay=30))
                        for r in room_list]
        exp_name = "Offline_DebugValveToggle"

    # Use offline client in debug mode
    cl_class = OfflineClient if debug else OpcuaClient
    run_control(used_control=used_control,
                exp_name=exp_name,
                user='ChristianBaumannETH2020',
                password='Christian4_ever',
                verbose=verbose,
                _client_class=cl_class)


def run_rl_control(room_nr: int = DEFAULT_ROOM_NR,
                   notify_failure: bool = False,
                   debug: bool = False,
                   verbose: int = 5,
                   alpha: float = 50.0,
                   n_steps: int = None,
                   date_str: str = DEFAULT_END_DATE,
                   temp_bds: RangeT = None,
                   train_data: str = DEFAULT_TRAIN_SET,
                   hop_eval_set: str = DEFAULT_EVAL_SET,
                   include_battery: bool = False,
                   notify_debug: bool = None,
                   dummy_env_mode: bool = True,
                   ):

    assert room_nr in [41, 43], f"Invalid room number: {room_nr}"

    if dummy_env_mode and verbose:
        print("Using dummy environment, will raise an error if there "
              "is no fitted agent available!")

    if notify_debug is None:
        notify_debug = debug
        msg = f"Using {'debug' if debug else 'original'} " \
              f"mail address for notifications."
        if verbose:
            print(msg)

    next_verbose = prog_verb(verbose)
    m_name = "FullState_Comp_ReducedTempConstWaterWeather"

    rl_cont = None
    if not debug:
        # Load the model and init env
        with ProgWrap(f"Loading environment...", verbose > 0):
            env = load_room_env(m_name,
                                verbose=next_verbose,
                                alpha=alpha,
                                include_battery=include_battery,
                                date_str=date_str,
                                temp_bds=temp_bds,
                                train_data=train_data,
                                room_nr=room_nr,
                                hop_eval_set=hop_eval_set,
                                dummy_use=dummy_env_mode)

        # Define default agents and compare
        with ProgWrap(f"Initializing agents...", verbose > 0):
            agent = default_ddpg_agent(env, n_steps, fitted=True,
                                       verbose=next_verbose,
                                       hop_eval_set=hop_eval_set)
            if verbose:
                print(agent)

        # Choose controller
        rl_cont = BaseRLController(agent, dt=env.m.data.dt, n_steps_max=3600,
                                   verbose=next_verbose)

    f_cont = FixTimeConstController(val=21.0, max_n_minutes=12 * 60)
    cont = f_cont if debug else rl_cont
    used_control = [(room_nr, cont)]

    exp_name = "DefaultExperimentName"
    if debug:
        exp_name += "Debug"

    # Run fucking control
    run_control(used_control=used_control,
                exp_name=exp_name,
                user='ChristianBaumannETH2020',
                password='Christian4_ever',
                debug=notify_debug,
                verbose=verbose,
                _client_class=OpcuaClient,
                notify_failures=notify_failure)
    pass
