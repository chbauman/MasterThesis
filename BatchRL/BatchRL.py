"""The main script of this project.

The `main` function runs all the necessary high-level
functions. The complicated stuff is hidden in the other
modules.
"""
from agents_heuristic import ConstHeating, RuleBasedHeating
from base_dynamics_env import test_test_env
from base_dynamics_model import test_dyn_model, BaseDynamicsModel
from base_hyperopt import HyperOptimizableModel, test_hyperopt
from battery_model import BatteryModel
from data import get_battery_data, Dataset, test_dataset_artificially, SeriesConstraint, \
    generate_room_datasets, get_DFAB_heating_data, DatasetConstraints
from dm_Composite import CompositeModel
from dm_Const import ConstModel
from dm_LSTM import RNNDynamicModel, test_rnn_models, RNNDynamicOvershootModel
from dm_Time import SCTimeModel
from dynamics_envs import FullRoomEnv, BatteryEnv, PWProfile
from keras_agents import DDPGBaseAgent, NAFBaseAgent
from keras_layers import test_layers
from util import *


def run_tests() -> None:
    """Runs a few tests.

    Raises:
        AssertionError: If a test fails.
    """
    test_hyperopt()
    test_layers()
    test_rnn_models()
    test_dyn_model()
    test_test_env()
    test_time_stuff()
    test_numpy_functions()
    # test_rest_client()
    test_python_stuff()
    # get_data_test()
    # test_align()
    test_dataset_artificially()


def run_battery() -> None:
    """Runs all battery related stuff.

    Loads and prepares the battery data, fits the
    battery model and evaluates some agents.
    """
    # Load and prepare battery data.
    bat_name = "Battery"
    get_battery_data()
    bat_ds = Dataset.loadDataset(bat_name)
    bat_ds.split_data()

    # Initialize and fit battery model.
    bat_mod = BatteryModel(bat_ds)
    bat_mod.analyze_bat_model()
    bat_mod.analyze()
    bat_mod_naive = ConstModel(bat_ds)
    bat_mod_naive.analyze()

    # Define the environment and agents.
    bat_env = BatteryEnv(bat_mod,
                         disturb_fac=0.3, cont_actions=True, n_cont_actions=1)
    const_ag_1 = ConstHeating(bat_env, 6.0)  # Charge
    const_ag_2 = ConstHeating(bat_env, -3.0)  # Discharge
    dqn_agent = DDPGBaseAgent(bat_env)

    # Fit agent and evaluate.
    dqn_agent.fit()
    bat_env.analyze_agent([const_ag_1, const_ag_2, dqn_agent])


def choose_dataset(base_ds_name: str = "Model_Room43",
                   seq_len: int = 20) -> Tuple[Dataset, DatasetConstraints]:
    """Let's you choose a dataset.

    Reads a room dataset, if it is not found, it is generated.
    Then the sequence length is set, the time variable is added and
    it is standardized and split into parts for training, validation
    and testing. Finally it is returned with the corresponding constraints.

    Args:
        base_ds_name: The name of the base dataset, must be of the form "Model_Room<nr>",
            with nr = 43 or 53.
        seq_len: The sequence length to use for the RNN training.

    Returns:
        The prepared dataset and the corresponding list of constraints.
    """
    # Check `base_ds_name`.
    if base_ds_name[:10] != "Model_Room" or base_ds_name[-2:] not in ["43", "53"]:
        raise ValueError(f"Dataset: {base_ds_name} does not exist!")

    # Load dataset, generate if not found.
    try:
        ds = Dataset.loadDataset(base_ds_name)
    except FileNotFoundError:
        get_DFAB_heating_data()
        generate_room_datasets()
        ds = Dataset.loadDataset(base_ds_name)

    # Set sequence length
    ds.seq_len = seq_len
    ds.name = base_ds_name[-6:] + "_" + str(ds.seq_len)

    # Add time variables, standardize and prepare different parts of dataset.
    ds = ds.add_time()
    ds.standardize()
    ds.split_data()

    # Room temperature model
    rnn_consts = [
        SeriesConstraint('interval', [-15.0, 40.0]),
        SeriesConstraint('interval', [0.0, 1300.0]),
        SeriesConstraint('interval', [-10.0, 100.0]),
        SeriesConstraint('interval', [-10.0, 100.0]),
        SeriesConstraint('interval', [0.0, 1.0]),
        SeriesConstraint('interval', [0.0, 40.0]),
        SeriesConstraint('exact'),
        SeriesConstraint('exact'),
    ]
    ds.transform_c_list(rnn_consts)

    # Return
    return ds, rnn_consts


def optimize_model(mod: HyperOptimizableModel) -> None:
    """Executes the hyperparameter optimization of a model.

    Uses reduced number of model trainings if not on Euler.

    Args:
        mod: Model whose hyperparameters are to be optimized.
    """
    n_opt = 60 if EULER else 3
    opt_params = mod.optimize(n_opt)
    # print("All tried parameter combinations: {}.".format(mod.param_list))
    print("Optimal parameters: {}.".format(opt_params))


def analyze_control_influence(m: BaseDynamicsModel):
    n_actions = 2
    env = FullRoomEnv(m, n_disc_actions=n_actions)
    const_ag_1 = ConstHeating(env, 0)
    const_ag_2 = ConstHeating(env, n_actions - 1)
    env.analyze_agent([const_ag_1, const_ag_2])


def get_model(name: str, ds: Dataset, rnn_consts: DatasetConstraints = None, from_hop: bool = False):
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
    base_params = dict(hop_pars, **fix_pars, **all_out)
    base_params_no_inds = {k: base_params[k] for k in base_params if k != 'out_inds'}
    if name == "Time_Exact":
        # Time model: Predicts the deterministic time variable exactly.
        return SCTimeModel(ds, 6)
    elif name == "FullState_Naive":
        # The naive model that predicts all series as the last seen input.
        return ConstModel(ds)
    elif name == "Full_RNN":
        # Full model: Predicting all series except for the controllable and the time
        # series. Weather predictions might depend on apartment data.
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **all_out)
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
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **out_inds)
        return RNNDynamicModel(**out_inds, **base_params_no_inds)
    elif name == "RoomTemp_RNN":
        # The temperature only model, predicting only the room temperature from
        # all the variables in the dataset. Can e.g. be used with a constant water
        # temperature model.
        out_inds = {'out_inds': np.array([5], dtype=np.int32)}
        if from_hop:
            return RNNDynamicModel.from_best_hp(**fix_pars, **out_inds)
        return RNNDynamicModel(**out_inds, **base_params_no_inds)
    elif name == "RoomTempFromReduced_RNN":
        # The temperature only model, predicting only the room temperature from
        # a reduced number of variables. Can e.g. be used with a constant water
        # temperature model.
        inds = {
            'out_inds': np.array([5], dtype=np.int32),
            'in_inds': np.array([0, 2, 5, 6, 7], dtype=np.int32),
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
        return CompositeModel(ds, [mod_weather, mod_apt],
                              new_name="Full_Comp_WeatherApt")
    elif name == "FullState_Comp_WeatherAptTime":
        # The full state model combining the weather only model, the
        # apartment only model and the exact time model to predict all
        # variables except for the control variable.
        mod_weather = get_model("WeatherFromWeatherTime_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        mod_apt = get_model("Apartment_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_time_exact = get_model("Time_Exact", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        return CompositeModel(ds, [mod_weather, mod_apt, model_time_exact],
                              new_name="FullState_Comp_WeatherAptTime")
    elif name == "FullState_Comp_FullTime":
        # The full state model combining the combined weather and apartment model
        # and the exact time model to predict all
        # variables except for the control variable.
        mod_full = get_model("Full_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_time_exact = get_model("Time_Exact", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        return CompositeModel(ds, [mod_full, model_time_exact],
                              new_name="FullState_Comp_FullTime")
    elif name == "FullState_Comp_ReducedTempConstWaterWeather":
        # The full state model combining the weather, the constant water temperature,
        # the reduced room temperature and the exact time model to predict all
        # variables except for the control variable.
        mod_wt = get_model("WaterTemp_Const", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_reduced_temp = get_model("RoomTempFromReduced_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_time_exact = get_model("Time_Exact", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_weather = get_model("WeatherFromWeatherTime_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        return CompositeModel(ds, [model_reduced_temp, mod_wt, model_weather, model_time_exact],
                              new_name="FullState_Comp_ReducedTempConstWaterWeather")
    elif name == "FullState_Comp_TempConstWaterWeather":
        # The full state model combining the weather, the constant water temperature,
        # the reduced room temperature and the exact time model to predict all
        # variables except for the control variable.
        mod_wt = get_model("WaterTemp_Const", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_reduced_temp = get_model("RoomTemp_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_time_exact = get_model("Time_Exact", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        model_weather = get_model("WeatherFromWeatherTime_RNN", ds, rnn_consts=rnn_consts, from_hop=from_hop)
        return CompositeModel(ds, [model_reduced_temp, mod_wt, model_weather, model_time_exact],
                              new_name="FullState_Comp_TempConstWaterWeather")
    else:
        raise ValueError("No such model defined!")


def curr_tests(ds: Dataset = None) -> None:
    """The code that I am currently experimenting with."""

    # Get dataset and constraints
    ds, rnn_consts = choose_dataset('Model_Room43', seq_len=20)

    # Choose a model
    # m = get_model("FullState_Comp_ReducedTempConstWaterWeather", ds, rnn_consts, from_hop=True)
    # m = get_model("FullState_Comp_TempConstWaterWeather", ds, rnn_consts, from_hop=True)
    m = get_model("FullState_Comp_WeatherAptTime", ds, rnn_consts, from_hop=True)
    # m.analyze()

    # And an environment
    env = FullRoomEnv(m, cont_actions=True, n_cont_actions=1)

    # Choose agent and fit to env.
    agent = DDPGBaseAgent(env)
    agent.fit()

    # Analyze comparing to other agents.
    open_agent = ConstHeating(env, 1.0)
    closed_agent = ConstHeating(env, 0.0)
    rule_based_agent = RuleBasedHeating(env, env.temp_bounds)
    env.analyze_agent([open_agent, closed_agent, rule_based_agent, agent])
    env.analyze_agent([open_agent, closed_agent, rule_based_agent])

    pass


def main() -> None:
    """The main function, here all the important, high-level stuff happens.

    Changes a lot, so I won't put a more accurate description here ;)
    """
    # Run tests.
    # run_tests()

    # Full test model
    curr_tests()
    return

    # Train and analyze the battery model
    # run_battery()

    # Get dataset and constraints
    ds, rnn_consts = choose_dataset('Model_Room43', seq_len=20)

    # Get the needed models
    needed = [
        # "Time_Exact",
        # "WaterTemp_Const",
        # "Full_RNN",
        "WeatherFromWeatherTime_RNN",
        "Apartment_RNN",
        "RoomTempFromReduced_RNN",
        "RoomTemp_RNN",
        # "Full_Comp_WeatherApt",
        # "FullState_Comp_WeatherAptTime",
        # "FullState_Comp_FullTime",
        # "FullState_Comp_ReducedTempConstWaterWeather",
        # "FullState_Comp_TempConstWaterWeather",
    ]
    all_mods = {nm: get_model(nm, ds, rnn_consts, from_hop=True) for nm in needed}

    # Hyper-optimize model(s)
    # optimize_model(get_model("WeatherFromWeatherTime_RNN", ds, rnn_consts))
    # optimize_model(get_model("RoomTempFromReduced_RNN", ds, rnn_consts))
    # optimize_model(get_model("Apartment_RNN", ds, rnn_consts))
    # optimize_model(get_model("RoomTemp_RNN", ds, rnn_consts))

    # Fit or load all initialized models
    for name, m_to_use in all_mods.items():
        m_to_use.fit()
        print(f"Model: {name}, performance: {m_to_use.hyper_obj()}")
        m_to_use.analyze()
        # analyze_control_influence(m_to_use)
        # m_to_use.analyze_disturbed("Valid", 'val', 10)
        # m_to_use.analyze_disturbed("Train", 'train', 10)
        pass


if __name__ == '__main__':
    main()
