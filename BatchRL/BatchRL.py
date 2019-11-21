"""The main script of this project.

The `main` function runs all the necessary high-level
functions. The complicated stuff is hidden in the other
modules / packages.
"""
from typing import Tuple, List

import numpy as np

from agents.agents_heuristic import ConstHeating, RuleBasedHeating
from agents.keras_agents import DDPGBaseAgent
from data_processing.data import get_battery_data, get_data_test, \
    get_constraints, choose_dataset
from data_processing.dataset import DatasetConstraints, Dataset
from dynamics.base_hyperopt import HyperOptimizableModel, optimize_model
from dynamics.base_model import test_dyn_model, BaseDynamicsModel, cleanup_test_data
from dynamics.battery_model import BatteryModel
from dynamics.composite import CompositeModel
from dynamics.const import ConstModel
from dynamics.recurrent import RNNDynamicModel, test_rnn_models, RNNDynamicOvershootModel
from dynamics.sin_cos_time import SCTimeModel
from envs.dynamics_envs import FullRoomEnv, BatteryEnv
from rest.client import test_rest_client
from util.util import EULER


def run_tests() -> None:
    """Runs a few tests.

    Raises:
        AssertionError: If a test fails.
    """
    # Do all the tests.
    test_rnn_models()
    test_dyn_model()
    test_rest_client()
    get_data_test()

    # Do some cleanup.
    cleanup_test_data()


def run_battery() -> None:
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
    bat_mod.analyze_bat_model()
    bat_mod.analyze()
    # bat_mod_naive = ConstModel(bat_ds)
    # bat_mod_naive.analyze()

    # Define the environment and agents.
    bat_env = BatteryEnv(bat_mod,
                         disturb_fac=0.3,
                         cont_actions=True,
                         n_cont_actions=1)
    const_ag_1 = ConstHeating(bat_env, 6.0)  # Charge
    const_ag_2 = ConstHeating(bat_env, -3.0)  # Discharge
    dqn_agent = DDPGBaseAgent(bat_env, action_range=bat_env.action_range,
                              n_steps=100000, gamma=0.99)

    bat_env.detailed_eval_agents([const_ag_1, const_ag_2, dqn_agent], use_noise=False, n_steps=1000)

    # Fit agent and evaluate.
    bat_env.analyze_agents_visually([const_ag_1, const_ag_2, dqn_agent],
                                    start_ind=0, fitted=False)
    bat_env.analyze_agents_visually([const_ag_1, const_ag_2, dqn_agent],
                                    start_ind=1267, fitted=False)
    bat_env.analyze_agents_visually([const_ag_1, const_ag_2, dqn_agent],
                                    start_ind=100, fitted=False)


def choose_dataset_and_constraints(base_ds_name: str = "Model_Room43",
                                   seq_len: int = 20,
                                   add_battery_data: bool = False,
                                   ) -> Tuple[Dataset, DatasetConstraints]:
    """Let's you choose a dataset.

    Reads a room dataset, if it is not found, it is generated.
    Then the sequence length is set, the time variable is added and
    it is standardized and split into parts for training, validation
    and testing. Finally it is returned with the corresponding constraints.

    Args:
        base_ds_name: The name of the base dataset, must be of the form "Model_Room<nr>",
            with nr = 43 or 53.
        seq_len: The sequence length to use for the RNN training.
        add_battery_data: Whether to add the battery data.

    Returns:
        The prepared dataset and the corresponding list of constraints.
    """

    ds = choose_dataset(base_ds_name, seq_len, add_battery_data)
    rnn_consts = get_constraints(ds, add_battery_data)

    # Return
    return ds, rnn_consts


def analyze_control_influence(m: BaseDynamicsModel):
    n_actions = 2
    env = FullRoomEnv(m, n_disc_actions=n_actions)
    const_ag_1 = ConstHeating(env, 0)
    const_ag_2 = ConstHeating(env, n_actions - 1)
    env.analyze_agents_visually([const_ag_1, const_ag_2])


def get_model(name: str, ds: Dataset,
              rnn_consts: DatasetConstraints = None,
              from_hop: bool = False,
              fit: bool = False) -> BaseDynamicsModel:
    """Loads and optionally fits a model.

    Args:
        name: The name specifying the model.
        ds: The dataset to initialize the model with.
        rnn_consts: The constraints for the recurrent models.
        from_hop: Whether to initialize the model from optimal hyperparameters.
        fit: Whether to fit the model before returning it.

    Returns:
        The requested model.
    """
    # Check input
    assert (ds.d == 8 and ds.n_c == 1) or (ds.d == 10 and ds.n_c == 2)
    battery_used = ds.d == 10

    # Load battery model if required.
    battery_mod = None
    if battery_used:
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
    # base_params_no_inds = {k: base_params[k] for k in base_params if k != 'out_inds'}
    base_params_no_inds = dict(hop_pars, **fix_pars)

    # Choose the model
    if name == "Time_Exact":
        # Time model: Predicts the deterministic time variable exactly.
        return SCTimeModel(ds, 6)
    elif name == "FullState_Naive":
        # The naive model that predicts all series as the last seen input.
        return ConstModel(ds)
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
    ds_full, rnn_consts_full = choose_dataset_and_constraints('Model_Room43', seq_len=20, add_battery_data=True)
    bat_mod = get_model("FullState_Comp_ReducedTempConstWaterWeather", ds_full,
                        rnn_consts=rnn_consts_full, fit=True, from_hop=True)
    # bat_mod.analyze_bat_model()
    bat_mod.analyze(overwrite=True)
    return

    # Get dataset and constraints
    ds, rnn_consts = choose_dataset_and_constraints('Model_Room43', seq_len=20)

    fit_custom = False
    if fit_custom:
        # Basic parameter set
        name = "RoomTempFromReduced_RNN"
        hop_pars = {
            'n_iter_max': 25,
            'hidden_sizes': (50, 50, 50),
            'input_noise_std': 0.0001,
            'lr': 0.0004,
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
        base_params = dict(hop_pars, **fix_pars, **all_out)
        base_params_no_inds = {k: base_params[k] for k in base_params if k != 'out_inds'}
        inds = {
            'out_inds': np.array([5], dtype=np.int32),
            'in_inds': np.array([0, 2, 4, 5, 6, 7], dtype=np.int32),
        }
        rnn = RNNDynamicModel(**inds, **base_params_no_inds)
        rnn.fit()
        rnn.analyze()
        return

    # try_opcua()
    # return

    # Choose a model
    full_mod_names = [
        "FullState_Comp_ReducedTempConstWaterWeather",
        "FullState_Comp_TempConstWaterWeather",
        "FullState_Comp_WeatherAptTime"
    ]

    # Test all models
    for m_name in full_mod_names[0:1]:
        # Load the model and init env
        m = get_model(m_name, ds, rnn_consts, from_hop=True, fit=True)
        # m.analyze()
        env = FullRoomEnv(m, cont_actions=True, n_cont_actions=1, disturb_fac=0.3)

        # Define default agents and compare
        open_agent = ConstHeating(env, 1.0)
        closed_agent = ConstHeating(env, 0.0)
        rule_based_agent = RuleBasedHeating(env, env.temp_bounds)
        # ag_list = [open_agent, closed_agent, rule_based_agent]
        # env.analyze_agents_visually(ag_list,
        #                             # start_ind=0,
        #                             use_noise=False,
        #                             max_steps=None)
        # env.detailed_eval_agents(ag_list, use_noise=False, n_steps=1000)

        # Choose agent and fit to env.
        if m_name == "FullState_Comp_ReducedTempConstWaterWeather":
            agent = DDPGBaseAgent(env, action_range=env.action_range,
                                  n_steps=50000, gamma=0.99, lr=0.00001)
            agent.fit()
            agent_list = [open_agent, closed_agent, rule_based_agent, agent]
            env.analyze_agents_visually(agent_list)
            env.detailed_eval_agents(agent_list, use_noise=False, n_steps=2000)


def main() -> None:
    """The main function, here all the important, high-level stuff happens.

    Changes a lot, so I won't put a more accurate description here ;)
    """
    # Run tests.
    # run_tests()

    # Full test model
    curr_tests()
    # return

    # Train and analyze the battery model
    # run_battery()
    return

    # Get dataset and constraints
    ds, rnn_consts = choose_dataset_and_constraints('Model_Room43', seq_len=20)

    # Get the needed models
    needed = [
        # "Time_Exact",
        # "WaterTemp_Const",
        # "Full_RNN",
        "WeatherFromWeatherTime_RNN",
        "Apartment_RNN",
        "RoomTempFromReduced_RNN",
        "RoomTemp_RNN",
        # "FullState_Comp_WeatherAptTime",
        # "FullState_Comp_FullTime",
        # "FullState_Comp_ReducedTempConstWaterWeather",
        # "FullState_Comp_TempConstWaterWeather",
    ]

    # Hyper-optimize model(s)
    for name in needed:
        mod = get_model(name, ds, rnn_consts, from_hop=False)
        if isinstance(mod, HyperOptimizableModel):
            print(f"Optimizing: {name}")
            if EULER:
                optimize_model(mod)
            else:
                print("Not optimizing!")
        else:
            raise ValueError(f"Model {name} not hyperparameter-optimizable!")

    all_mods = {nm: get_model(nm, ds, rnn_consts, from_hop=True) for nm in needed}

    # Fit or load all initialized models
    for name, m_to_use in all_mods.items():
        m_to_use.fit()
        print(f"Model: {name}, performance: {m_to_use.hyper_obj()}")
        m_to_use.analyze(overwrite=True)
        # analyze_control_influence(m_to_use)
        # m_to_use.analyze_disturbed("Valid", 'val', 10)
        # m_to_use.analyze_disturbed("Train", 'train', 10)
        pass


if __name__ == '__main__':
    main()
