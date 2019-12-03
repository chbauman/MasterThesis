"""The main script of this project.

The `main` function runs all the necessary high-level
functions. The complicated stuff is hidden in the other
modules / packages.
"""
import argparse
import os
import time
from functools import reduce
from typing import List, Tuple

import numpy as np

from agents.agents_heuristic import ConstActionAgent, RuleBasedHeating
from agents.keras_agents import DDPGBaseAgent
from data_processing.data import get_battery_data, get_data_test, \
    choose_dataset_and_constraints
from data_processing.dataset import DatasetConstraints, Dataset
from dynamics.base_hyperopt import HyperOptimizableModel, optimize_model
from dynamics.base_model import test_dyn_model, BaseDynamicsModel
from opcua_empa.run_opcua import try_opcua
from tests.test_util import cleanup_test_data
from dynamics.battery_model import BatteryModel
from dynamics.composite import CompositeModel
from dynamics.const import ConstModel
from dynamics.recurrent import RNNDynamicModel, test_rnn_models, RNNDynamicOvershootModel
from dynamics.sin_cos_time import SCTimeModel
from envs.dynamics_envs import FullRoomEnv, BatteryEnv, RoomBatteryEnv
from rest.client import test_rest_client
from util.numerics import max_abs_err, mae, mse, MSE, MAE, MaxAbsEer, ErrMetric
from util.util import EULER, get_rl_steps, print_if_verb, ProgWrap

# Define the models by name
from util.visualize import plot_performance_table, plot_performance_graph, plot_dataset, OVERLEAF_IMG_DIR

base_rnn_models = [
    "WeatherFromWeatherTime_RNN",
    "Apartment_RNN",
    "RoomTempFromReduced_RNN",
    "RoomTemp_RNN",
    # "Full_RNN",
]
full_models = [
    "FullState_Comp_ReducedTempConstWaterWeather",
    "FullState_Comp_TempConstWaterWeather",
    "FullState_Comp_WeatherAptTime",
    "FullState_Naive",
]
full_models_short_names = [
    "Weather, Constant Water, Reduced Room Temp",
    "Weather, Constant Water, Room Temp.",
    "Weather, joint Room and Water",
    "Naive",
]


def run_integration_tests() -> None:
    """Runs a few rather time consuming tests.

    Raises:
        AssertionError: If a test fails.
    """
    # Do all the tests.
    test_rnn_models()
    test_dyn_model()
    test_rest_client()
    get_data_test()


def test_cleanup():
    # Do some cleanup.
    cleanup_test_data(verbose=1)


def run_battery(do_rl: bool = True, overwrite: bool = False, verbose: int = 0) -> None:
    """Runs all battery related stuff.

    Loads and prepares the battery data, fits the
    battery model and evaluates some agents.
    """
    # Load and prepare battery data.
    bat_name = "Battery"
    get_battery_data()
    bat_ds = Dataset.loadDataset(bat_name)
    bat_ds.standardize()
    bat_ds.split_data()

    # Initialize and fit battery model.
    bat_mod = BatteryModel(bat_ds)
    bat_mod.analyze_bat_model(put_on_ol=True)
    bat_mod.analyze_visually(one_week_to_ol=True, base_name="Bat", overwrite=overwrite)
    # bat_mod_naive = ConstModel(bat_ds)
    # bat_mod_naive.analyze_visually()

    if not do_rl:
        return

    # Get numbers of steps
    n_steps = get_rl_steps(True)
    n_eval_steps = 10000 if EULER else 100

    # Define the environment
    bat_env = BatteryEnv(bat_mod,
                         disturb_fac=0.3,
                         cont_actions=True,
                         n_cont_actions=1)

    # Define the agents
    const_ag_1 = ConstActionAgent(bat_env, 6.0)  # Charge
    const_ag_2 = ConstActionAgent(bat_env, -3.0)  # Discharge
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
                               verbose: int = 1) -> None:
    """Runs the hyperparameter optimization for all base RNN models.

    Does not much if not on Euler.

    Args:
        use_bat_data: Whether to include the battery data.
        verbose: Verbosity level.
    """

    # Get data and constraints
    ds, rnn_consts = choose_dataset_and_constraints('Model_Room43', seq_len=20, add_battery_data=use_bat_data)

    # Hyper-optimize model(s)
    for name in base_rnn_models:
        mod = get_model(name, ds, rnn_consts, from_hop=False, fit=False)
        if isinstance(mod, HyperOptimizableModel):
            if verbose:
                print(f"Optimizing: {name}")
            if EULER:
                optimize_model(mod, verbose=verbose > 0)
            else:
                print("Not optimizing!")
        else:
            raise ValueError(f"Model {name} not hyperparameter-optimizable!")
    pass


def run_dynamic_model_fit_from_hop(use_bat_data: bool = True,
                                   verbose: int = 1,
                                   visual_analyze: bool = True,
                                   perf_analyze: bool = False,
                                   include_composite: bool = False) -> None:
    """Runs the hyperparameter optimization for all base RNN models.

    Does not much if not on Euler.

    Args:
        use_bat_data: Whether to include the battery data.
        verbose: Verbosity level.
        visual_analyze: Whether to do the visual analysis.
        perf_analyze: Whether to do the performance analysis.
        include_composite: Whether to also do all the stuff for the composite models.
    """
    # Data for performance analysis
    n_steps = (1, 4, 12, 24, 48)
    metrics: Tuple[ErrMetric] = (MSE, MAE, MaxAbsEer)  # What is this shit warning?

    # Get data and constraints
    ds, rnn_consts = choose_dataset_and_constraints('Model_Room43',
                                                    seq_len=20,
                                                    add_battery_data=use_bat_data)

    # Load and fit all models
    lst = base_rnn_models[:]
    if include_composite:
        lst += full_models
    all_mods = {nm: get_model(nm, ds, rnn_consts, from_hop=True, fit=True) for nm in lst}

    # Fit or load all initialized models
    for name, m_to_use in all_mods.items():
        if verbose:
            print(f"Model: {name}")
        # Visual analysis
        if visual_analyze:
            with ProgWrap(f"Analyzing model visually...", verbose > 0):
                m_to_use.analyze_visually(overwrite=False, verbose=False)

        # Do the performance analysis
        if perf_analyze:
            if verbose:
                pass
                # print(f"Model: {name}, performance: {m_to_use.hyper_obj()}")
            with ProgWrap(f"Analyzing model performance...", verbose > 0):
                m_to_use.analyze_performance(n_steps, verbose=False,
                                             overwrite=False,
                                             metrics=metrics)

        # m_to_use.analyze_disturbed("Valid", 'val', 10)
        # m_to_use.analyze_disturbed("Train", 'train', 10)

    # Create the performance table
    with ProgWrap("Creating performance table and plots...", verbose > 0):

        orig_mask = np.array([0, 1, 2, 3, 5])

        full_mods = [all_mods[n] for n in full_models]
        parts = ["Val", "Train"]
        metric_names = [m.name for m in metrics]
        name = "EvalTable"

        if use_bat_data:
            name += "WithBat"

        plot_performance_table(full_mods, parts, metric_names, name,
                               short_mod_names=full_models_short_names,
                               series_mask=orig_mask)
        plot_name = "EvalPlot"
        plot_performance_graph(full_mods, parts, metrics, plot_name + "_RTempOnly",
                               short_mod_names=full_models_short_names,
                               series_mask=np.array([5]), scale_back=True, remove_units=False)
        plot_performance_graph(full_mods, parts, metrics, plot_name,
                               short_mod_names=full_models_short_names,
                               series_mask=orig_mask)


def run_room_models(verbose: int = 1) -> None:
    # Get dataset and constraints
    ds, rnn_consts = choose_dataset_and_constraints('Model_Room43', seq_len=20)

    # Test all models
    for m_name in full_models[0:1]:
        # Load the model and init env
        m = get_model(m_name, ds, rnn_consts, from_hop=True, fit=True)
        m.analyze_visually(overwrite=False, plot_acf=False, verbose=False)
        alpha = 10.0
        env = FullRoomEnv(m, cont_actions=True, n_cont_actions=1, disturb_fac=0.3, alpha=alpha)

        # Define default agents and compare
        open_agent = ConstActionAgent(env, 1.0)
        closed_agent = ConstActionAgent(env, 0.0)
        rule_based_agent = RuleBasedHeating(env, env.temp_bounds)

        # Choose agent and fit to env.
        n_steps = get_rl_steps() * 30
        n_eval_steps = 2000  # n_steps // 100
        if m_name == "FullState_Comp_ReducedTempConstWaterWeather":
            agent = DDPGBaseAgent(env,
                                  action_range=env.action_range,
                                  n_steps=n_steps,
                                  gamma=0.99, lr=0.00001)
            agent.name = f"DDPG_FS_RT_CW_NEP{n_steps}_Al_{alpha}"
            print_if_verb(verbose, "Fitting agent...")
            agent.fit()
            print_if_verb(verbose, "Analyzing agents...")
            agent_list = [open_agent, closed_agent, rule_based_agent, agent]
            mask = np.array([0, 1, 4])
            for s in [0, None]:
                env.analyze_agents_visually(agent_list, state_mask=mask, start_ind=s,
                                            plot_constrain_actions=False,
                                            show_rewards=True)

            # env.detailed_eval_agents(agent_list, use_noise=False, n_steps=n_eval_steps)


def update_overleaf_plots(verbose: int = 1):
    # # Battery model plots
    # with ProgWrap(f"Running battery...", verbose > 0):
    #     run_battery(do_rl=False, overwrite=True, verbose=0)
    #
    # Get data and constraints
    with ProgWrap(f"Loading data...", verbose > 0):
        ds, rnn_consts = choose_dataset_and_constraints('Model_Room43',
                                                        seq_len=20,
                                                        add_battery_data=False)

        ds_bat, rnn_consts_bat = choose_dataset_and_constraints('Model_Room43',
                                                                seq_len=20,
                                                                add_battery_data=True)

    # # Weather model
    # w_mod_name = base_rnn_models[0]
    # w_mod = get_model(w_mod_name, ds, rnn_consts, from_hop=True, fit=True, verbose=False)
    # with ProgWrap(f"Analyzing weather model visually...", verbose > 0):
    #     w_mod.analyze_visually(overwrite=True, verbose=False, one_file=True,
    #                            one_week_to_ol=True, base_name="Weather1W")

    # # Heating water constant
    # with ProgWrap(f"Plotting heating water...", verbose > 0):
    #     ds_heat = ds[2:4]
    #     n_tot = ds_heat.data.shape[0]
    #     ds_heat_rel = ds_heat.slice_time(int(n_tot * 0.6), int(n_tot * 0.66))
    #     plot_dataset(ds_heat_rel, show=False,
    #                  title_and_ylab=["Heating Water Temperatures", "Temperature [°C]"],
    #                  save_name=os.path.join(OVERLEAF_IMG_DIR, "WaterTemp"))
    #
    # # Room temperature model
    # r_mod_name = base_rnn_models[2]
    # with ProgWrap(f"Analyzing room temperature model visually...", verbose > 0):
    #     r_mod = get_model(r_mod_name, ds_bat, rnn_consts_bat, from_hop=True, fit=True, verbose=False)
    #     r_mod.analyze_visually(overwrite=True, verbose=False, one_file=True,
    #                            one_week_to_ol=True, base_name="Room1W")

    # Combined model evaluation
    with ProgWrap(f"Analyzing full model performance...", verbose > 0):
        full_mod_name = full_models[0]
        full_mod = get_model(full_mod_name, ds_bat, rnn_consts_bat, from_hop=True, fit=True, verbose=False)

        metrics: Tuple[ErrMetric] = (MSE, MAE, MaxAbsEer)

        full_mods = [full_mod]
        parts = ["Val", "Train"]

        plot_name = "EvalPlot"
        plot_performance_graph(full_mods, parts, metrics, plot_name + "_RTempOnly",
                               short_mod_names=full_models_short_names[0:1],
                               series_mask=np.array([5]), scale_back=True, remove_units=False,
                               put_on_ol=True)
    pass


def get_model(name: str, ds: Dataset,
              rnn_consts: DatasetConstraints = None,
              from_hop: bool = False,
              fit: bool = False,
              verbose: int = 1) -> BaseDynamicsModel:
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

        # Adjust the name for full models.
        if comp_name.startswith("FullState_"):
            if battery_used:
                comp_name = comp_name + "_Battery"
            if rnn_consts is not None:
                comp_name += "_CON"
        return CompositeModel(ds, model_list, new_name=comp_name)

    # Fit if required.
    if fit:
        mod = get_model(name, ds, rnn_consts, from_hop, fit=False)
        mod.fit()
        return mod

    # Basic parameter set
    hop_pars = {
        'n_iter_max': 10,
        'hidden_sizes': (50, 50),
        'input_noise_std': 0.001,
        'lr': 0.01,
        'gru': False,
    }
    fix_pars = {
        'name': name,
        'data': ds,
        'residual_learning': True,
        'constraint_list': rnn_consts,
        'weight_vec': None,
    }
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
        inds = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int32)
        return ConstModel(ds, in_indices=inds, pred_inds=inds)
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

    # try_opcua()
    # return

    with ProgWrap("Testing..."):
        time.sleep(5)

    # Load the dataset and setup the model
    ds_full, rnn_consts_full = choose_dataset_and_constraints('Model_Room43', seq_len=20, add_battery_data=True)
    mod = get_model("FullState_Comp_ReducedTempConstWaterWeather", ds_full,
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


def def_parser() -> argparse.ArgumentParser:
    # Define argument parser
    parser = argparse.ArgumentParser()
    arg_def_list = [
        # The following arguments can be provided.
        ("verbose", "Increase output verbosity."),
        ("mod_eval", "Fit and evaluate the room models."),
        ("optimize", "Execute the hyperparameter optimization."),
        ("battery", "Run the battery model."),
        ("room", "Run the room model."),
        ("test", "Run tests."),
        ("cleanup", "Run test cleanup."),
        ("plot", "Run overleaf plot creation."),

    ]
    for kw, h in arg_def_list:
        short_kw = "-" + kw[0]
        parser.add_argument(short_kw, "--" + kw, action="store_true", help=h)
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
    verbose = args.verbose
    if verbose:
        print("Verbosity turned on.")

    # Run integration tests and optionally the cleanup after.
    if args.test:
        run_integration_tests()
    if args.cleanup:
        test_cleanup()

    # Run hyperparameter optimization
    if args.optimize:
        run_dynamic_model_hyperopt(use_bat_data=True)

    # Fit and analyze all models
    if args.mod_eval:
        run_dynamic_model_fit_from_hop(verbose=verbose, perf_analyze=True,
                                       visual_analyze=False,
                                       include_composite=True)

    if args.battery:
        # Train and analyze the battery model
        run_battery()

    if args.room:
        # Room model
        run_room_models()

    # Overleaf plots
    if args.plot:
        update_overleaf_plots(verbose)

    # Check if any flag is set, if not, do current experiments.
    var_dict = vars(args)
    any_flag_set = reduce(lambda x, k: x or var_dict[k], var_dict, 0)
    if not any_flag_set:
        print("No flags set")
        curr_tests()


if __name__ == '__main__':
    main()
