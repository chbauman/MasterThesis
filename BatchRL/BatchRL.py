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
from sklearn.linear_model import MultiTaskLassoCV

from agents.agents_heuristic import ConstActionAgent, RuleBasedAgent, get_const_agents
from agents.keras_agents import DDPGBaseAgent
from data_processing.data import get_battery_data, \
    choose_dataset_and_constraints, update_data
from data_processing.dataset import DatasetConstraints, Dataset
from dynamics.base_hyperopt import HyperOptimizableModel, optimize_model
from dynamics.base_model import BaseDynamicsModel, compare_models
from dynamics.battery_model import BatteryModel
from dynamics.classical import SKLearnModel
from dynamics.composite import CompositeModel
from dynamics.const import ConstModel
from dynamics.recurrent import RNNDynamicModel, test_rnn_models, RNNDynamicOvershootModel, PhysicallyConsistentRNN
from dynamics.sin_cos_time import SCTimeModel
from envs.dynamics_envs import FullRoomEnv, BatteryEnv, RoomBatteryEnv, LowHighProfile, heat_marker
from opcua_empa.run_opcua import try_opcua
from rest.client import test_rest_client, DEFAULT_END_DATE
from tests.test_util import cleanup_test_data, TEST_DIR
from util.numerics import MSE, MAE, MaxAbsEer, ErrMetric
from util.util import EULER, get_rl_steps, ProgWrap, prog_verb, w_temp_str, str2bool, extract_args
from util.visualize import plot_performance_table, plot_performance_graph, OVERLEAF_IMG_DIR, plot_dataset, \
    plot_heat_cool_rew_det

# Define the models by name
base_rnn_models = [
    "WeatherFromWeatherTime_RNN",
    "Apartment_RNN",
    "RoomTempFromReduced_RNN",
    "RoomTemp_RNN",
    "WeatherFromWeatherTime_Linear",
    "PhysConsModel",
    # "Full_RNN",
]
full_models = [
    "FullState_Comp_ReducedTempConstWaterWeather",
    "FullState_Comp_TempConstWaterWeather",
    "FullState_Comp_WeatherAptTime",
    "FullState_Naive",
    "FullState_Comp_Phys",
]
full_models_short_names = [
    "Weather, Constant Water, Reduced Room Temp",
    "Weather, Constant Water, Room Temp.",
    "Weather, joint Room and Water",
    "Naive",
    "Weather, Constant Water, Consistent Room Temp",
]

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
        test_rest_client()


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
        # bat_mod_naive = ConstModel(bat_ds)
        # bat_mod_naive.analyze_visually()

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
                               n_fit_calls: int = None) -> None:
    """Runs the hyperparameter optimization for all base RNN models.

    Does not much if not on Euler, except if `enforce_optimize`
    is True, then it will optimize anyways.

    Args:
        use_bat_data: Whether to include the battery data.
        verbose: Verbosity level.
        enforce_optimize: Whether to enforce the optimization.
        n_fit_calls: Number of fit evaluations, default if None.
    """
    next_verb = prog_verb(verbose)

    # Get data and constraints
    with ProgWrap(f"Loading data...", verbose > 0):
        ds, rnn_consts = choose_dataset_and_constraints('Model_Room43', seq_len=20,
                                                        add_battery_data=use_bat_data)

    # Hyper-optimize model(s)
    with ProgWrap(f"Hyperoptimizing models...", verbose > 0):
        for name in base_rnn_models:
            # Load model
            with ProgWrap(f"Loading model: {name}...", next_verb > 0):
                mod = get_model(name, ds, rnn_consts, from_hop=False, fit=False)

            # Optimize model
            if isinstance(mod, HyperOptimizableModel):
                if EULER or enforce_optimize:
                    with ProgWrap(f"Optimizing model: {name}...", next_verb > 0):
                        optimize_model(mod, verbose=next_verb > 0, n_restarts=n_fit_calls)
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
                                   date_str: str = DEFAULT_END_DATE) -> None:
    """Runs the hyperparameter optimization for all base RNN models.

    Does not much if not on Euler.

    Args:
        use_bat_data: Whether to include the battery data.
        verbose: Verbosity level.
        visual_analyze: Whether to do the visual analysis.
        perf_analyze: Whether to do the performance analysis.
        include_composite: Whether to also do all the stuff for the composite models.
        date_str: End date string specifying data.
    """
    next_verb = prog_verb(verbose)

    # Get data and constraints
    with ProgWrap(f"Loading data...", verbose > 0):
        ds, rnn_consts = choose_dataset_and_constraints('Model_Room43',
                                                        seq_len=20,
                                                        add_battery_data=use_bat_data,
                                                        date_str=date_str)

    # Load and fit all models
    with ProgWrap(f"Loading models...", verbose > 0):
        lst = base_rnn_models[:]
        if include_composite:
            lst += full_models
        all_mods = {nm: get_model(nm, ds, rnn_consts,
                                  from_hop=True, fit=True, verbose=next_verb) for nm in lst}

    # Fit or load all initialized models
    with ProgWrap(f"Analyzing models...", verbose > 0):
        for name, m_to_use in all_mods.items():
            if verbose:
                print(f"Model: {name}")
            # Visual analysis
            if visual_analyze:
                with ProgWrap(f"Analyzing model visually...", verbose > 0):
                    m_to_use.analyze_visually(overwrite=False, verbose=next_verb)

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


def run_room_models(verbose: int = 1, put_on_ol: bool = False,
                    eval_list: List[int] = None,
                    perf_eval: bool = False,
                    alpha: float = 5.0,
                    n_steps: int = None,
                    overwrite: bool = False,
                    include_battery: bool = False,
                    physically_consistent: bool = False,
                    date_str: str = DEFAULT_END_DATE) -> None:

    # Print what the code does
    if verbose:
        print("Running RL agents on learned room model.")
        if include_battery:
            print("Model includes battery.")

    # Select model
    m_name = "FullState_Comp_ReducedTempConstWaterWeather"
    if physically_consistent:
        m_name = "FullState_Comp_Phys"

    if eval_list is None:
        eval_list = [2595, 8221, 0, 2042, 12067, None]

    # Get dataset and constraints
    with ProgWrap(f"Loading dataset...", verbose > 0):
        ds, rnn_consts = choose_dataset_and_constraints('Model_Room43', seq_len=20,
                                                        add_battery_data=include_battery,
                                                        date_str=date_str)

    # Load the model and init env
    with ProgWrap(f"Preparing environment...", verbose > 0):
        m = get_model(m_name, ds, rnn_consts, from_hop=True, fit=True, verbose=prog_verb(verbose))
        # m.analyze_visually(overwrite=False, plot_acf=False, verbose=prog_verb(verbose) > 0)
        if include_battery:
            c_prof = LowHighProfile(ds.dt)
            assert isinstance(m, CompositeModel), f"Invalid model: {m}, needs to be composite!"
            env = RoomBatteryEnv(m, p=c_prof,
                                 cont_actions=True,
                                 disturb_fac=0.3, alpha=alpha)
        else:
            env = FullRoomEnv(m, cont_actions=True, n_cont_actions=1,
                              disturb_fac=0.3, alpha=alpha)

    # Define default agents and compare
    with ProgWrap(f"Initializing agents...", verbose > 0):
        closed_agent, open_agent = get_const_agents(env)
        ch_rate = 10.0 if include_battery else None
        rule_based_agent = RuleBasedAgent(env, env.temp_bounds, const_charge_rate=ch_rate)

        # Choose agent and fit to env.
        if n_steps is None:
            n_steps = get_rl_steps(eul=True)
        agent = DDPGBaseAgent(env,
                              action_range=env.action_range,
                              n_steps=n_steps,
                              gamma=0.99, lr=0.00001)
        name_ext = "_BAT" if include_battery else ""
        name_ext += "_PHYS" if physically_consistent else ""
        agent.name = f"DDPG_FS_RT_CW_NEP{n_steps}_Al_{alpha}{name_ext}"

        agent_list = [open_agent, closed_agent, rule_based_agent, agent]

    with ProgWrap(f"Fitting DDPG agent...", verbose > 0):
        agent.fit(verbose=prog_verb(verbose))

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
        ds, rnn_consts = choose_dataset_and_constraints('Model_Room43',
                                                        seq_len=20,
                                                        add_battery_data=False)

        ds_bat, rnn_consts_bat = choose_dataset_and_constraints('Model_Room43',
                                                                seq_len=20,
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


def get_model(name: str, ds: Dataset,
              rnn_consts: DatasetConstraints = None,
              from_hop: bool = False,
              fit: bool = False,
              verbose: int = 0) -> BaseDynamicsModel:
    """Loads and optionally fits a model.

    Args:
        name: The name specifying the model.
        ds: The dataset to initialize the model with.
        rnn_consts: The constraints for the recurrent models.
        from_hop: Whether to initialize the model from optimal hyperparameters.
        fit: Whether to fit the model before returning it.
        verbose: Verbosity.

    Returns:
        The requested model.
    """
    # Check input
    assert (ds.d == 8 and ds.n_c == 1) or (ds.d == 10 and ds.n_c == 2)
    battery_used = ds.d == 10

    # Fit if required using one step recursion
    if fit:
        mod = get_model(name, ds, rnn_consts, from_hop, fit=False, verbose=prog_verb(verbose))
        mod.fit(verbose=prog_verb(verbose))
        return mod

    if verbose and not fit:
        print(f"Loading model {name}.")

    # Load battery model if required.
    battery_mod = None
    if battery_used:
        if verbose > 0:
            print("Dataset contains battery data.")
        battery_mod = BatteryModel(dataset=ds, base_ind=8)

    # Helper function to build composite models including the battery model.
    def _build_composite(model_list: List[BaseDynamicsModel], comp_name: str):

        # Load battery model.
        if battery_used:
            assert battery_mod is not None, "Need to rethink this!"
            if fit:
                battery_mod.fit()
            model_list = model_list + [battery_mod]

        # Print model names
        if verbose > 1:
            for ct, m in enumerate(model_list):
                print(f"Model {ct}: {m.name}")

        # Adjust the name for full models.
        if comp_name.startswith("FullState_"):
            if battery_used:
                comp_name = comp_name + "_Battery"
            if rnn_consts is not None:
                comp_name += "_CON"
        return CompositeModel(ds, model_list, new_name=comp_name)

    # Basic parameter set
    hop_pars = {
        'n_iter_max': 10,
        'hidden_sizes': (50, 50),
        'input_noise_std': 0.001,
        'lr': 0.01,
        'gru': False,
    }
    nec_pars = {
        'name': name,
        'data': ds,
    }
    fix_pars = {
        'residual_learning': True,
        'constraint_list': rnn_consts,
        'weight_vec': None,
    }
    fix_pars = dict(fix_pars, **nec_pars)
    all_out = {'out_inds': np.array([0, 1, 2, 3, 5], dtype=np.int32)}
    all_in = {'in_inds': np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)}
    all_inds = dict(all_out, **all_in)
    base_params = dict(hop_pars, **fix_pars, **all_inds)
    base_params_no_inds = dict(hop_pars, **fix_pars)

    # Choose the model
    if name == "Time_Exact":
        # Time model: Predicts the deterministic time variable exactly.
        return SCTimeModel(ds, 6)
    elif name == "FullState_Naive":
        # The naive model that predicts all series as the last seen input.
        bat_inds = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int32)
        no_bat_inds = np.array([0, 1, 2, 3, 5, 6, 7], dtype=np.int32)
        inds = bat_inds if battery_used else no_bat_inds
        return ConstModel(ds, in_inds=inds, pred_inds=inds)
    elif name == "Battery":
        # Battery model.
        assert battery_mod is not None, "I fucked up somewhere!"
        return battery_mod
    elif name == "Full_RNN":
        # Full model: Predicting all series except for the controllable and the time
        # series. Weather predictions might depend on apartment data.
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **all_inds)
        return RNNDynamicModel(**base_params)
    elif name == "WeatherFromWeatherTime_RNN":
        # The weather model, predicting only the weather and the time, i.e. outside temperature and
        # irradiance from the past values and the time variable.
        inds = {
            'out_inds': np.array([0, 1], dtype=np.int32),
            'in_inds': np.array([0, 1, 6, 7], dtype=np.int32),
        }
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **inds)
        return RNNDynamicModel(**inds, **base_params_no_inds)
    elif name == "WeatherFromWeatherTime_Linear":
        # The weather model, predicting only the weather and the time, i.e. outside temperature and
        # irradiance from the past values and the time variable using a linear model.
        inds = {
            'out_inds': np.array([0, 1], dtype=np.int32),
            'in_inds': np.array([0, 1, 6, 7], dtype=np.int32),
        }
        skl_base_mod = MultiTaskLassoCV(max_iter=1000, cv=5)
        return SKLearnModel(skl_model=skl_base_mod, **nec_pars, **inds)
    elif name == "Apartment_RNN":
        # The apartment model, predicting only the apartment variables, i.e. water
        # temperatures and room temperature based on all input variables including the weather.
        out_inds = {'out_inds': np.array([2, 3, 5], dtype=np.int32)}
        out_inds = dict(out_inds, **all_in)
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **out_inds)
        return RNNDynamicModel(**out_inds, **base_params_no_inds)
    elif name == "RoomTemp_RNN":
        # The temperature only model, predicting only the room temperature from
        # all the variables in the dataset. Can e.g. be used with a constant water
        # temperature model.
        out_inds = {'out_inds': np.array([5], dtype=np.int32)}
        out_inds = dict(out_inds, **all_in)
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **out_inds)
        return RNNDynamicModel(**out_inds, **base_params_no_inds)
    elif name == "RoomTempFromReduced_RNN":
        # The temperature only model, predicting only the room temperature from
        # a reduced number of variables. Can e.g. be used with a constant water
        # temperature model.
        inds = {
            'out_inds': np.array([5], dtype=np.int32),
            'in_inds': np.array([0, 2, 4, 5, 6, 7], dtype=np.int32),
        }
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **inds)
        return RNNDynamicModel(**inds, **base_params_no_inds)
    elif name == "PhysConsModel":
        # The physically consistent temperature only model.
        inds = {
            'out_inds': np.array([5], dtype=np.int32),
            'in_inds': np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
        }
        if from_hop:
            return PhysicallyConsistentRNN.from_best_hp(**fix_pars, **inds)
        return PhysicallyConsistentRNN(**inds, **base_params_no_inds)
    elif name == "WaterTemp_Const":
        # Constant model for water temperatures
        return ConstModel(ds, pred_inds=np.array([2, 3], dtype=np.int32))
    elif name == "Full_RNNOvershootDecay":
        # Similar to the model "FullModel", but trained with overshoot.
        return RNNDynamicOvershootModel(n_overshoot=5,
                                        decay_rate=0.8,
                                        **base_params)
    elif name == "Full_Comp_WeatherApt":
        # The full model combining the weather only model and the
        # apartment only model to predict all
        # variables except for the control and the time variables.
        mod_weather = get_model("WeatherFromWeather_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        mod_apt = get_model("Apartment_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        mod_list = [mod_weather, mod_apt]
        return _build_composite(mod_list, name)

    elif name == "FullState_Comp_WeatherAptTime":
        # The full state model combining the weather only model, the
        # apartment only model and the exact time model to predict all
        # variables except for the control variable.
        mod_weather = get_model("WeatherFromWeatherTime_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        mod_apt = get_model("Apartment_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_time_exact = get_model("Time_Exact", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        mod_list = [mod_weather, mod_apt, model_time_exact]
        return _build_composite(mod_list, name)

    elif name == "FullState_Comp_FullTime":
        # The full state model combining the combined weather and apartment model
        # and the exact time model to predict all
        # variables except for the control variable.
        mod_full = get_model("Full_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_time_exact = get_model("Time_Exact", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        mod_list = [mod_full, model_time_exact]
        return _build_composite(mod_list, name)

    elif name == "FullState_Comp_ReducedTempConstWaterWeather":
        # The full state model combining the weather, the constant water temperature,
        # the reduced room temperature and the exact time model to predict all
        # variables except for the control variable.
        mod_wt = get_model("WaterTemp_Const", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_reduced_temp = get_model("RoomTempFromReduced_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_time_exact = get_model("Time_Exact", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_weather = get_model("WeatherFromWeatherTime_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        mod_list = [model_weather, mod_wt, model_reduced_temp, model_time_exact]
        return _build_composite(mod_list, name)

    elif name == "FullState_Comp_Phys":
        # The full state model combining the weather, the constant water temperature,
        # the physically consistent room temperature and the exact time model to predict all
        # variables except for the control variable.
        mod_wt = get_model("WaterTemp_Const", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_reduced_temp = get_model("PhysConsModel", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_time_exact = get_model("Time_Exact", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_weather = get_model("WeatherFromWeatherTime_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        mod_list = [model_weather, mod_wt, model_reduced_temp, model_time_exact]
        return _build_composite(mod_list, name)

    elif name == "FullState_Comp_TempConstWaterWeather":
        # The full state model combining the weather, the constant water temperature,
        # the reduced room temperature and the exact time model to predict all
        # variables except for the control variable.
        mod_wt = get_model("WaterTemp_Const", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_reduced_temp = get_model("RoomTemp_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_time_exact = get_model("Time_Exact", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_weather = get_model("WeatherFromWeatherTime_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        mod_list = [model_weather, mod_wt, model_reduced_temp, model_time_exact]
        return _build_composite(mod_list, name)
    else:
        raise ValueError("No such model defined!")


def curr_tests() -> None:
    """The code that I am currently experimenting with."""

    # Load the dataset and setup the model
    ds_full, rnn_consts_full = choose_dataset_and_constraints('Model_Room43', seq_len=20, add_battery_data=True)
    # mod = get_model("FullState_Comp_ReducedTempConstWaterWeather", ds_full,
    #                 rnn_consts=rnn_consts_full, fit=True, from_hop=True)
    mod = get_model("FullState_Comp_Phys", ds_full,
                    rnn_consts=rnn_consts_full, fit=True, from_hop=True)

    # Setup env
    assert isinstance(mod, CompositeModel), "Model not suited"
    full_env = RoomBatteryEnv(mod, max_eps=24)

    # Define agent and fit
    ac_range_list = full_env.action_range
    n_steps = get_rl_steps()
    ddpg_ag = DDPGBaseAgent(full_env,
                            n_steps=n_steps,
                            layers=(50, 50),
                            action_range=ac_range_list)
    ddpg_ag.fit(verbose=1)

    # Define agents to compare with.
    ag1 = ConstActionAgent(full_env, np.array([0.0, 0.0], dtype=np.float32))

    # Evaluate
    full_env.detailed_eval_agents([ddpg_ag, ag1], n_steps=3, use_noise=False)

    return


arg_def_list = [
    # The following arguments can be provided.
    ("verbose", "output verbosity"),
    ("mod_eval", "fit and evaluate the room models"),
    ("optimize", "optimize hyperparameters of models"),
    ("data", "update the data from the nest database"),
    ("battery", "run the battery model."),
    ("room", "run the room model"),
    ("test", "run tests"),
    ("cleanup", "cleanup test files"),
    ("plot", "run overleaf plot creation"),
    ("ua", "run opcua"),
]
opt_param_l = [
    ("int", int, "additional integer parameter(s)"),
    ("float", float, "additional floating point parameter(s)"),
    ("str", str, "additional string parameter(s)"),
    ("bool", str2bool, "additional boolean parameter(s)"),
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

    # Add more optional parameters
    for kw, t, h in opt_param_l:
        short_kw = f"-{kw[0:2]}"
        parser.add_argument(short_kw, f"--{kw}", nargs='+', type=t, help=h)
    return parser


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

    # Common arguments
    date_str = extract_args(args.str, DEFAULT_END_DATE)[0]

    # Run integration tests and optionally the cleanup after.
    if args.test:
        run_integration_tests(verbose=verbose)
    if args.cleanup:
        test_cleanup(verbose=verbose)

    # Update stored data
    if args.data:
        update_data(date_str=date_str)

    # Run hyperparameter optimization
    if args.optimize:
        n_steps = extract_args(args.int, None)[0]
        use_bat_data, enf_opt = extract_args(args.bool, True, False)
        run_dynamic_model_hyperopt(use_bat_data=use_bat_data,
                                   verbose=verbose,
                                   enforce_optimize=enf_opt,
                                   n_fit_calls=n_steps)

    # Fit and analyze all models
    if args.mod_eval:
        date_str = extract_args(args.str, DEFAULT_END_DATE)[0]
        perf_analyze, visual_analyze, include_composite = extract_args(args.bool, True, False, False)
        run_dynamic_model_fit_from_hop(verbose=verbose, perf_analyze=perf_analyze,
                                       visual_analyze=visual_analyze,
                                       include_composite=include_composite,
                                       date_str=date_str)

    # Train and analyze the battery model
    if args.battery:
        ext_args = extract_args(args.bool, False, False, False)
        do_rl, put_on_ol, overwrite = ext_args
        run_battery(verbose=verbose, do_rl=do_rl, put_on_ol=put_on_ol,
                    overwrite=overwrite)

    # Evaluate room model
    if args.room:
        alpha = args.float[0] if args.float is not None else None
        n_steps = extract_args(args.int, None)[0]
        ext_args = extract_args(args.bool, False, False, False, False)
        add_bat, perf_eval, phys_cons, overwrite = ext_args
        run_room_models(verbose=verbose, alpha=alpha, n_steps=n_steps,
                        include_battery=add_bat, perf_eval=perf_eval,
                        physically_consistent=phys_cons, overwrite=overwrite,
                        date_str=date_str)

    # Overleaf plots
    if args.plot:
        debug, overwrite = extract_args(args.bool, False, False)
        update_overleaf_plots(verbose, overwrite=overwrite, debug=debug)

    # Opcua
    if args.ua:
        debug = args.bool[0] if args.bool is not None else False
        try_opcua(verbose + 1, room_list=args.int, debug=debug)

    # Check if any flag is set, if not, do current experiments.
    var_dict = vars(args)
    any_flag_set = reduce(lambda x, k: x or var_dict[k] is True, var_dict, 0)
    if not any_flag_set:
        print("No flags set")
        curr_tests()


if __name__ == '__main__':
    main()
