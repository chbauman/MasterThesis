"""The main script of this project.

The `main` function runs all the necessary high-level
functions. The complicated stuff is hidden in the other
modules / packages.
"""
import argparse
import os
import warnings
from functools import reduce
from typing import List, Tuple, Sequence, Type

import numpy as np

from agents.agents_heuristic import RuleBasedAgent, get_const_agents
from agents.base_agent import upload_trained_agents, download_trained_agents
from agents.keras_agents import DDPGBaseAgent, default_ddpg_agent
from data_processing.data import get_battery_data, \
    choose_dataset_and_constraints, update_data, unique_room_nr
from data_processing.dataset import Dataset, check_dataset_part
from dynamics.base_hyperopt import HyperOptimizableModel, optimize_model, check_eval_data, upload_hop_pars, \
    download_hop_pars
from dynamics.base_model import compare_models, check_train_str
from dynamics.battery_model import BatteryModel
from dynamics.load_models import base_rnn_models, full_models, full_models_short_names, get_model, load_room_models, \
    load_room_env
from dynamics.recurrent import test_rnn_models
from envs.dynamics_envs import BatteryEnv, heat_marker, RangeT
from opcua_empa.run_opcua import try_opcua, run_rl_control
from rest.client import check_date_str
from tests.test_util import cleanup_test_data, TEST_DIR
from util.numerics import MSE, MAE, MaxAbsEer, ErrMetric
from util.share_data import test_folder_zip
from util.util import EULER, get_rl_steps, ProgWrap, prog_verb, w_temp_str, str2bool, extract_args, DEFAULT_TRAIN_SET, \
    DEFAULT_ROOM_NR, DEFAULT_EVAL_SET, DEFAULT_END_DATE, data_ext, BASE_DIR, execute_powershell
from util.visualize import plot_performance_table, plot_performance_graph, OVERLEAF_IMG_DIR, plot_dataset, \
    plot_heat_cool_rew_det

# Model performance evaluation
N_PERFORMANCE_STEPS = (1, 4, 12, 24, 48)
METRICS: Tuple[Type[ErrMetric], ...] = (MSE, MAE, MaxAbsEer)
PARTS = ["Val", "Train"]


def run_integration_tests(verbose: int = 1) -> None:
    """Runs a few rather time consuming tests.

    Raises:
        AssertionError: If a test fails.
    """
    # Do all the tests.
    with ProgWrap(f"Running a few tests...", verbose > 0):
        test_rnn_models()


def test_cleanup(verbose: int = 0):
    """Cleans the data that was generated for the tests."""
    # Do some cleanup.
    with ProgWrap("Cleanup...", verbose=verbose > 0):
        cleanup_test_data(verbose=prog_verb(verbose))


def run_battery(do_rl: bool = True, overwrite: bool = False,
                verbose: int = 0, steps: Sequence = (24,),
                put_on_ol: bool = False) -> None:
    """Runs all battery related stuff.

    Loads and prepares the battery data, fits the
    battery model and evaluates some agents.
    """
    # Print info to console
    if verbose:
        print("Running battery modeling...")
        if put_on_ol:
            print("Putting images into Overleaf directory.")
        if overwrite:
            print("Overwriting existing images.")

    # Load and prepare battery data.
    with ProgWrap(f"Loading battery data...", verbose > 0):
        bat_name = "Battery"
        get_battery_data()
        bat_ds = Dataset.loadDataset(bat_name)
        bat_ds.standardize()
        bat_ds.split_data()

    # Initialize and fit battery model.
    with ProgWrap(f"Fitting and analyzing battery...", verbose > 0):
        bat_mod = BatteryModel(bat_ds)
        bat_mod.fit(verbose=prog_verb(verbose))
        bat_mod.analyze_bat_model(put_on_ol=put_on_ol, overwrite=overwrite)
        bat_mod.analyze_visually(save_to_ol=put_on_ol, base_name="Bat",
                                 overwrite=overwrite, n_steps=steps, verbose=verbose > 0)

    if not do_rl:
        if verbose:
            print("No RL this time.")
        return
    if verbose:
        print("Running battery RL.")

    with ProgWrap(f"Defining environment...", verbose > 0):
        # Get numbers of steps
        n_steps = get_rl_steps(True)
        n_eval_steps = 10000 if EULER else 100

        # Define the environment
        bat_env = BatteryEnv(bat_mod,
                             disturb_fac=0.3,
                             cont_actions=True,
                             n_cont_actions=1)

    # Define the agents
    const_ag_1, const_ag_2 = get_const_agents(bat_env)
    dqn_agent = DDPGBaseAgent(bat_env,
                              action_range=bat_env.action_range,
                              n_steps=n_steps,
                              gamma=0.99)

    # Fit agents and evaluate.
    ag_list = [const_ag_1, const_ag_2, dqn_agent]
    bat_env.analyze_agents_visually(ag_list, start_ind=0, fitted=False)
    bat_env.analyze_agents_visually(ag_list, start_ind=1267, fitted=False)
    bat_env.analyze_agents_visually(ag_list, start_ind=100, fitted=False)
    bat_env.detailed_eval_agents(ag_list,
                                 use_noise=False,
                                 n_steps=n_eval_steps)


def run_dynamic_model_hyperopt(use_bat_data: bool = True,
                               verbose: int = 1,
                               enforce_optimize: bool = False,
                               n_fit_calls: int = None,
                               hop_eval_set: str = DEFAULT_EVAL_SET,
                               date_str: str = DEFAULT_END_DATE,
                               room_nr: int = DEFAULT_ROOM_NR,
                               model_indices: List[int] = None) -> None:
    """Runs the hyperparameter optimization for all base RNN models.

    Does not much if not on Euler, except if `enforce_optimize`
    is True, then it will optimize anyways.

    Args:
        use_bat_data: Whether to include the battery data.
        verbose: Verbosity level.
        enforce_optimize: Whether to enforce the optimization.
        n_fit_calls: Number of fit evaluations, default if None.
        hop_eval_set: Evaluation set for the optimization.
        date_str: End date string specifying data.
        room_nr: Integer specifying the room number.
        model_indices: Indices of models for hyperparameter tuning referring
            to the list `base_rnn_models`.
    """
    assert hop_eval_set in ["val", "test"], f"Fuck: {hop_eval_set}"

    next_verb = prog_verb(verbose)
    if verbose:
        print(f"Doing hyperparameter optimization using "
              f"evaluation on {hop_eval_set} set. Using room {room_nr} "
              f"with data up to {date_str}.")

    # Check model indices and set model list
    if model_indices is not None:
        if len(model_indices) == 0:
            model_indices = None
        else:
            msg = f"Invalid indices: {model_indices}"
            assert max(model_indices) < len(base_rnn_models), msg
            assert min(model_indices) >= 0, msg
            if verbose:
                print("Not optimizing all models.")
    model_list = base_rnn_models
    if model_indices is not None:
        model_list = [model_list[i] for i in model_indices]

    # Load models
    model_dict = load_room_models(model_list,
                                  use_bat_data,
                                  from_hop=False,
                                  fit=False,
                                  date_str=date_str,
                                  room_nr=room_nr,
                                  hop_eval_set=hop_eval_set,
                                  verbose=verbose)
    # Hyper-optimize model(s)
    with ProgWrap(f"Hyperoptimizing models...", verbose > 0):
        for name, mod in model_dict.items():

            # Optimize model
            if isinstance(mod, HyperOptimizableModel):
                # Create extension based on room number and data end date
                full_ext = data_ext(date_str, room_nr, hop_eval_set)

                if EULER or enforce_optimize:
                    with ProgWrap(f"Optimizing model: {name}...", next_verb > 0):
                        optimize_model(mod, verbose=next_verb > 0,
                                       n_restarts=n_fit_calls,
                                       eval_data=hop_eval_set,
                                       data_ext=full_ext)
                else:
                    print("Not optimizing!")
            else:
                warnings.warn(f"Model {name} not hyperparameter-optimizable!")
                # raise ValueError(f"Model {name} not hyperparameter-optimizable!")


def run_dynamic_model_fit_from_hop(use_bat_data: bool = False,
                                   verbose: int = 1,
                                   visual_analyze: bool = True,
                                   perf_analyze: bool = False,
                                   include_composite: bool = False,
                                   date_str: str = DEFAULT_END_DATE,
                                   train_data: str = DEFAULT_TRAIN_SET,
                                   room_nr: int = DEFAULT_ROOM_NR,
                                   hop_eval_set: str = DEFAULT_EVAL_SET,
                                   ) -> None:
    """Runs the hyperparameter optimization for all base RNN models.

    Does not much if not on Euler.

    Args:
        use_bat_data: Whether to include the battery data.
        verbose: Verbosity level.
        visual_analyze: Whether to do the visual analysis.
        perf_analyze: Whether to do the performance analysis.
        include_composite: Whether to also do all the stuff for the composite models.
        date_str: End date string specifying data.
        train_data: String specifying the part of the data to train the model on.
        room_nr: Integer specifying the room number.
        hop_eval_set: Evaluation set for the hyperparameter optimization.
    """
    if verbose:
        print(f"Fitting dynamics ML models based on parameters "
              f"optimized by hyperparameter tuning. Using room {room_nr} "
              f"with data up to {date_str}. The models are fitted "
              f"using the {train_data} portion of the data. "
              f"Hyperparameter tuning used evaluation on {hop_eval_set} "
              f"set.")

    check_train_str(train_data)
    next_verb = prog_verb(verbose)

    # Create model list
    lst = base_rnn_models[:]
    if include_composite:
        lst += full_models

    # Load and fit all models
    with ProgWrap(f"Loading models...", verbose > 0):
        # Load models
        all_mods = load_room_models(lst,
                                    use_bat_data,
                                    from_hop=True,
                                    fit=True,
                                    date_str=date_str,
                                    room_nr=room_nr,
                                    hop_eval_set=hop_eval_set,
                                    train_data=train_data,
                                    verbose=next_verb)

    # Fit or load all initialized models
    with ProgWrap(f"Analyzing models...", verbose > 0):
        for name, m_to_use in all_mods.items():
            if verbose:
                print(f"Model: {name}")
            # Visual analysis
            if visual_analyze:
                with ProgWrap(f"Analyzing model visually...", verbose > 0):
                    m_to_use.analyze_visually(overwrite=False, verbose=next_verb > 0)

            # Do the performance analysis
            if perf_analyze:
                with ProgWrap(f"Analyzing model performance...", verbose > 0):
                    m_to_use.analyze_performance(N_PERFORMANCE_STEPS, verbose=next_verb,
                                                 overwrite=False,
                                                 metrics=METRICS)

    # Create the performance table
    if include_composite:
        with ProgWrap("Creating performance table and plots...", verbose > 0):

            orig_mask = np.array([0, 1, 2, 3, 5])
            full_mods = [all_mods[n] for n in full_models]
            metric_names = [m.name for m in METRICS]

            # Plot the performance
            name = "EvalTable"
            if use_bat_data:
                name += "WithBat"
            plot_performance_table(full_mods, PARTS, metric_names, name,
                                   short_mod_names=full_models_short_names,
                                   series_mask=orig_mask)
            plot_name = "EvalPlot"
            plot_performance_graph(full_mods, PARTS, METRICS, plot_name + "_RTempOnly",
                                   short_mod_names=full_models_short_names,
                                   series_mask=np.array([5]), scale_back=True, remove_units=False)
            plot_performance_graph(full_mods, PARTS, METRICS, plot_name,
                                   short_mod_names=full_models_short_names,
                                   series_mask=orig_mask)


def run_room_models(verbose: int = 1,
                    put_on_ol: bool = False,
                    eval_list: List[int] = None,
                    perf_eval: bool = False,
                    alpha: float = 50.0,
                    n_steps: int = None,
                    overwrite: bool = False,
                    include_battery: bool = False,
                    physically_consistent: bool = False,
                    date_str: str = DEFAULT_END_DATE,
                    temp_bds: RangeT = None,
                    train_data: str = DEFAULT_TRAIN_SET,
                    room_nr: int = DEFAULT_ROOM_NR,
                    hop_eval_set: str = DEFAULT_EVAL_SET,
                    ) -> None:
    # Print what the code does
    if verbose:
        print("Running RL agents on learned room model.")
        if include_battery:
            print("Model includes battery.")
    next_verbose = prog_verb(verbose)

    # Select model
    m_name = "FullState_Comp_ReducedTempConstWaterWeather"
    if physically_consistent:
        m_name = "FullState_Comp_Phys"

    if eval_list is None:
        eval_list = [2595, 8221, 0, 2042, 12067, None]

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
                            hop_eval_set=hop_eval_set)

    # Define default agents and compare
    with ProgWrap(f"Initializing agents...", verbose > 0):
        closed_agent, open_agent = get_const_agents(env)
        ch_rate = 10.0 if include_battery else None
        rule_based_agent = RuleBasedAgent(env, env.temp_bounds,
                                          const_charge_rate=ch_rate)

        agent = default_ddpg_agent(env, n_steps, fitted=True,
                                   verbose=next_verbose,
                                   hop_eval_set=hop_eval_set)

        agent_list = [open_agent, closed_agent, rule_based_agent, agent]

    with ProgWrap(f"Analyzing agents...", verbose > 0):

        b_ind = -2 if include_battery else -1
        bounds = [(b_ind, (22.0, 26.0))]

        for s in eval_list:
            if s is None:
                s = np.random.randint(0, env.n_start_data)

            # Find the current heating water temperatures
            heat_inds = np.array([2, 3])
            h_in_and_out = env.get_scaled_init_state(s, heat_inds)
            title_ext = w_temp_str(h_in_and_out)

            # Plot
            env.analyze_agents_visually(agent_list, state_mask=None, start_ind=s,
                                        plot_constrain_actions=False,
                                        show_rewards=True, series_merging_list=None,
                                        bounds=bounds, title_ext=title_ext,
                                        put_on_ol=put_on_ol, plot_rewards=True,
                                        overwrite=overwrite)

    # Do performance evaluation
    if perf_eval:
        with ProgWrap(f"Evaluating agents...", verbose > 0):
            n_eval_steps = 10000  # n_steps // 100
            env.detailed_eval_agents(agent_list, use_noise=False, n_steps=n_eval_steps,
                                     put_on_ol=put_on_ol, overwrite=overwrite,
                                     verbose=prog_verb(verbose),
                                     plt_fun=plot_heat_cool_rew_det,
                                     episode_marker=heat_marker)
    elif verbose > 0:
        print("No performance evaluation!")


def update_overleaf_plots(verbose: int = 2, overwrite: bool = False,
                          debug: bool = False):
    # If debug is true, the plots are not saved to Overleaf.
    if verbose > 0 and debug:
        print("Running in debug mode!")

    # Battery model plots
    with ProgWrap(f"Running battery...", verbose > 0):
        run_battery(do_rl=False, overwrite=overwrite, verbose=prog_verb(verbose), put_on_ol=not debug)

    # Get data and constraints
    with ProgWrap(f"Loading data...", verbose > 0):
        ds, rnn_consts = choose_dataset_and_constraints(seq_len=20,
                                                        add_battery_data=False)

        ds_bat, rnn_consts_bat = choose_dataset_and_constraints(seq_len=20,
                                                                add_battery_data=True)

    # Weather model
    with ProgWrap(f"Analyzing weather model visually...", verbose > 0):
        # Define model indices and names
        mod_inds = [0, 4]
        model_names = ["RNN", "Linear"]
        n_steps = (0, 24)

        # Load weather models
        mod_list = [get_model(base_rnn_models[k], ds, rnn_consts,
                              from_hop=True, fit=True, verbose=prog_verb(verbose)) for k in mod_inds]

        # Compare the models for one week continuous and 6h predictions
        dir_to_use = OVERLEAF_IMG_DIR if not debug else TEST_DIR
        ol_file_name = os.path.join(dir_to_use, "WeatherComparison")

        compare_models(mod_list, ol_file_name,
                       n_steps=n_steps,
                       model_names=model_names,
                       overwrite=overwrite)

        # Plot prediction performance
        try:
            # for m in mod_list:
            #     m.analyze_performance(metrics=METRICS)
            plot_performance_graph(mod_list, PARTS, METRICS, "WeatherPerformance",
                                   short_mod_names=model_names,
                                   series_mask=None, scale_back=True,
                                   remove_units=False, put_on_ol=not debug,
                                   compare_models=True, overwrite=overwrite,
                                   scale_over_series=True)
        except OSError as e:
            if verbose:
                print(f"{e}")
                print(f"Need to analyze performance of model first!")

    # Heating water constant
    with ProgWrap(f"Plotting heating water...", verbose > 0):
        s_name = os.path.join(OVERLEAF_IMG_DIR, "WaterTemp")
        if overwrite or not os.path.isfile(s_name + ".pdf"):
            ds_heat = ds[2:4]
            n_tot = ds_heat.data.shape[0]
            ds_heat_rel = ds_heat.slice_time(int(n_tot * 0.6), int(n_tot * 0.66))
            plot_dataset(ds_heat_rel, show=False,
                         title_and_ylab=["Heating Water Temperatures", "Temperature [°C]"],
                         save_name=s_name)

    # Room temperature model
    with ProgWrap(f"Analyzing room temperature model visually...", verbose > 0):
        r_mod_name = base_rnn_models[2]
        r_mod = get_model(r_mod_name, ds_bat, rnn_consts_bat, from_hop=True,
                          fit=True, verbose=prog_verb(verbose))
        r_mod.analyze_visually(n_steps=[24], overwrite=overwrite,
                               verbose=prog_verb(verbose) > 0, one_file=True,
                               save_to_ol=not debug, base_name="Room1W_E",
                               add_errors=True)
        r_mod.analyze_visually(n_steps=[24], overwrite=overwrite,
                               verbose=prog_verb(verbose) > 0, one_file=True,
                               save_to_ol=not debug, base_name="Room1W",
                               add_errors=False)

    # # Combined model evaluation
    # with ProgWrap(f"Analyzing full model performance...", verbose > 0):
    #     full_mod_name = full_models[0]
    #     full_mod = get_model(full_mod_name, ds_bat, rnn_consts_bat, from_hop=True, fit=True, verbose=False)
    #
    #     metrics: Tuple[ErrMetric] = (MSE, MAE, MaxAbsEer)
    #
    #     full_mods = [full_mod]
    #
    #
    #     plot_name = "EvalPlot"
    #     plot_performance_graph(full_mods, parts, metrics, plot_name + "_RTempOnly",
    #                            short_mod_names=["TempPredCombModel"],
    #                            series_mask=np.array([5]), scale_back=True, remove_units=False,
    #                            put_on_ol=True)

    # # DDPG Performance Evaluation
    # with ProgWrap(f"Analyzing DDPG performance...", verbose > 0):
    #     eval_list = [11889]
    #     run_room_models(verbose=prog_verb(verbose), put_on_ol=not debug,
    #                     eval_list=eval_list, alpha=2.5)
    pass


def curr_tests() -> None:
    """The code that I am currently experimenting with."""
    try_opcua()
    return


arg_def_list = [
    # The following arguments can be provided.
    ("verbose", "use verbose mode"),
    ("file_transfer", "transfer data"),
    ("mod_eval", "fit and evaluate the room ML models"),
    ("optimize", "optimize hyperparameters of ML models"),
    ("data", "update the data from the nest database"),
    ("battery", "run the battery model"),
    ("room", "run the room simulation model to train and evaluate a rl agent"),
    ("test", "run a few integration tests, not running unit tests"),
    ("cleanup", "cleanup all test files, including ones from unit tests"),
    ("plot", "run overleaf plot creation"),
    ("ua", "run opcua control"),
]
opt_param_l = [
    ("int", int, "additional integer parameter(s)"),
    ("float", float, "additional floating point parameter(s)"),
    ("str", str, "additional string parameter(s)"),
    ("bool", str2bool, "additional boolean parameter(s)"),
]
common_params = [
    # ("arg_name", type, "help string", default_value),
    ("train_data", str, "Data used for training the models, can be one of "
                        "'train', 'train_val' or 'all'.", DEFAULT_TRAIN_SET),
    ("eval_data", str, "Data used for evaluation of the models, can be one of "
                       "'train', 'val', 'train_val', 'test' or 'all'.",
     DEFAULT_EVAL_SET),
    ("hop_eval_data", str, "Data used for evaluation of the models in "
                           "hyperparameter optimization, can be one either "
                           "'val' or 'test'.", DEFAULT_EVAL_SET),
    ("data_end_date", str, "String specifying the date when the data was "
                           "loaded from NEST database, e.g. 2020-01-21",
     "2020-01-21"),
    ("room_nr", int, "Integer specifying the room number.", 43),
]


def def_parser() -> argparse.ArgumentParser:
    """The argument parser factory.

    Returns:
        An argument parser.
    """
    # Define argument parser
    parser = argparse.ArgumentParser()

    # Add boolean args
    for kw, h in arg_def_list:
        short_kw = f"-{kw[0]}"
        parser.add_argument(short_kw, f"--{kw}", action="store_true", help=h)

    # Add parameters used for many tasks
    for kw, t, h, d in common_params:
        parser.add_argument(f"--{kw}", type=t, help=h, default=d)

    # Add general optional parameters
    for kw, t, h in opt_param_l:
        short_kw = f"-{kw[0:2]}"
        parser.add_argument(short_kw, f"--{kw}", nargs='+', type=t, help=h)

    return parser


def transfer_data(gd_upload: bool, gd_download: bool, data_to_euler: bool,
                  models_from_euler: bool, verbose: int = 5):
    next_verb = prog_verb(verbose)

    # Upload to / download from Google Drive
    if gd_upload:
        with ProgWrap("Uploading data to Google Drive", verbose > 0):
            upload_trained_agents()
            upload_hop_pars()
    if gd_download:
        with ProgWrap("Downloading data from Google Drive", verbose > 0):
            download_trained_agents()
            download_hop_pars()

    # Upload to / download from Euler
    auto_script_path = os.path.join(BASE_DIR, "automate.ps1")
    if data_to_euler or models_from_euler:
        assert not EULER, "Cannot be executed on Euler"
        print("Make sure you have an active VPN connection to ETH.")

    if data_to_euler:
        execute_powershell(auto_script_path, "-cp_data")

    if models_from_euler:
        execute_powershell(auto_script_path, "-cp_hop")
        execute_powershell(auto_script_path, "-cp_rl")


def main() -> None:
    """The main function, here all the important, high-level stuff happens.

    Defines command line arguments that can be specified to run certain
    portions of the code. If no such flag is specified, the current
    experiments (defined in the function `curr_tests`) are run, especially
    this is the default in PyCharm.
    """

    # Parse arguments
    parser = def_parser()
    args = parser.parse_args()
    verbose = 5 if args.verbose else 0
    if args.verbose:
        print("Verbosity turned on.")

    # Extract arguments
    train_data, eval_data = args.train_data, args.eval_data
    hop_eval_data = args.hop_eval_data
    room_nr, date_str = args.room_nr, args.data_end_date

    # Check arguments
    check_date_str(date_str)
    check_train_str(train_data)
    check_dataset_part(eval_data)
    check_eval_data(hop_eval_data)
    room_nr = unique_room_nr(room_nr)
    if room_nr in [41, 51] and date_str == DEFAULT_END_DATE:
        raise ValueError(f"Room number and data end date combination "
                         f"not supported because of backwards compatibility "
                         f"reasons :(")

    # Run integration tests and optionally the cleanup after.
    if args.test:
        run_integration_tests(verbose=verbose)
    if args.cleanup:
        test_cleanup(verbose=verbose)

    # Update stored data
    if args.data:
        update_data(date_str=date_str)

    # Transfer data
    if args.file_transfer:
        gd_upload, gd_download, data_to_euler, models_from_euler = \
            extract_args(args.bool, False, False, False, False)
        transfer_data(gd_upload=gd_upload, gd_download=gd_download,
                      data_to_euler=data_to_euler,
                      models_from_euler=models_from_euler,
                      verbose=verbose)

    # Run hyperparameter optimization
    if args.optimize:
        n_steps = extract_args(args.int, None, raise_too_many_error=False)[0]
        ind_list = []
        if n_steps is not None:
            _, *ind_list = args.int
        use_bat_data, enf_opt = extract_args(args.bool, True, False)
        run_dynamic_model_hyperopt(use_bat_data=use_bat_data,
                                   verbose=verbose,
                                   enforce_optimize=enf_opt,
                                   n_fit_calls=n_steps,
                                   hop_eval_set=hop_eval_data,
                                   date_str=date_str,
                                   room_nr=room_nr,
                                   model_indices=ind_list)

    # Fit and analyze all models
    if args.mod_eval:
        perf_analyze, visual_analyze, include_composite = extract_args(args.bool, True, False, False)
        run_dynamic_model_fit_from_hop(verbose=verbose, perf_analyze=perf_analyze,
                                       visual_analyze=visual_analyze,
                                       include_composite=include_composite,
                                       date_str=date_str, train_data=train_data,
                                       room_nr=room_nr, hop_eval_set=hop_eval_data)

    # Train and analyze the battery model
    if args.battery:
        ext_args = extract_args(args.bool, False, False, False)
        do_rl, put_on_ol, overwrite = ext_args
        run_battery(verbose=verbose, do_rl=do_rl, put_on_ol=put_on_ol,
                    overwrite=overwrite)

    # Evaluate room model
    if args.room:
        alpha, tb_low, tb_high = extract_args(args.float, 50.0, None, None)
        n_steps = extract_args(args.int, None)[0]
        ext_args = extract_args(args.bool, False, False, False, False)
        add_bat, perf_eval, phys_cons, overwrite = ext_args
        temp_bds = None if tb_high is None else (tb_low, tb_high)
        run_room_models(verbose=verbose, alpha=alpha, n_steps=n_steps,
                        include_battery=add_bat, perf_eval=perf_eval,
                        physically_consistent=phys_cons, overwrite=overwrite,
                        date_str=date_str, temp_bds=temp_bds,
                        train_data=train_data, room_nr=room_nr,
                        hop_eval_set=hop_eval_data)

    # Overleaf plots
    if args.plot:
        debug, overwrite = extract_args(args.bool, False, False)
        update_overleaf_plots(verbose, overwrite=overwrite, debug=debug)

    # Opcua
    if args.ua:
        debug, notify_failure, phys_cons, notify_debug, dummy_env_mode = \
            extract_args(args.bool, False, False, False, None, True)
        n_steps = extract_args(args.int, None)[0]
        alpha, tb_low, tb_high = extract_args(args.float, 50.0, None, None)
        temp_bds = None if tb_high is None else (tb_low, tb_high)
        run_rl_control(room_nr=room_nr, notify_failure=notify_failure,
                       debug=debug, alpha=alpha, n_steps=n_steps,
                       date_str=date_str, temp_bds=temp_bds,
                       train_data=train_data,
                       hop_eval_set=hop_eval_data,
                       notify_debug=notify_debug,
                       dummy_env_mode=dummy_env_mode,
                       )

    # Check if any flag is set, if not, do current experiments.
    var_dict = vars(args)
    var_dict = {k: val for k, val in var_dict.items() if k != "verbose"}
    any_flag_set = reduce(lambda x, k: x or var_dict[k] is True, var_dict, 0)
    if not any_flag_set:
        print("No flags set")
        curr_tests()


if __name__ == '__main__':
    main()
