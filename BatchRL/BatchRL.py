"""The main script of this project.

The `main` function runs all the necessary high-level
functions. The complicated stuff is hidden in the other
modules.
"""
from agents_heuristic import ConstHeating
from base_dynamics_env import test_test_env
from base_dynamics_model import test_dyn_model
from battery_model import BatteryModel
from data import get_battery_data, Dataset, test_dataset_artificially, SeriesConstraint, \
    generate_room_datasets, get_DFAB_heating_data
from dm_Composite import CompositeModel
from dm_Const import ConstModel
from dm_LSTM import RNNDynamicModel, test_rnn_models
from dm_Time import SCTimeModel
from dm_TimePeriodic import Periodic1DayModel
from dynamics_envs import FullRoomEnv
from keras_layers import test_layers
from keras_rl_wrap import DQNRoomHeatingAgent
from util import *


def run_tests() -> None:
    """Runs a few tests.

    Raises:
        AssertionError: If a test fails.
    """
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


def choose_dataset(base_ds_name: str = "Model_Room43",
                   seq_len: int = 20) -> Tuple[Dataset, List[SeriesConstraint]]:
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
        The prepared dataset and the corresponding constraints list.
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


def main() -> None:
    """The main function, here all the important, high-level stuff happens.

    Changes a lot, so I won't put a more accurate description here ;)
    """
    # Run tests.
    run_tests()

    # Get dataset
    ds, rnn_consts = choose_dataset('Model_Room43', seq_len=20)

    # Time model: Predicts the deterministic time variable
    model_time_exact = SCTimeModel(ds, 6)

    # Constant model for water temperatures
    mod_const_water = ConstModel(ds, pred_inds=np.array([2, 3], dtype=np.int32))

    # Basic parameter set
    base_params = {'input_noise_std': 0.001,
                   'lr': 0.01,
                   'residual_learning': True,
                   'weight_vec': None,
                   'out_inds': np.array([0, 1, 2, 3, 5], dtype=np.int32),
                   }
    base_params_no_inds = {k: base_params[k] for k in base_params if k != 'out_inds'}

    # Different models

    # Full model: Predicting all series except for the controllable and the time
    # series. Weather predictions might depend on apartment data.
    mod_full = RNNDynamicModel(ds,
                               name="FullModel",
                               hidden_sizes=(50, 50),
                               n_iter_max=10,
                               constraint_list=rnn_consts,
                               **base_params)

    # The weather model, predicting only the weather, i.e. outside temperature and
    # irradiance from the past values and the time variable.
    mod_weather = RNNDynamicModel(ds,
                                  name="WeatherFromWeatherOnly",
                                  hidden_sizes=(10, 10),
                                  n_iter_max=10,
                                  constraint_list=rnn_consts,
                                  out_inds=np.array([0, 1], dtype=np.int32),
                                  in_inds=np.array([0, 1], dtype=np.int32),
                                  **base_params_no_inds)

    # The apartment model, predicting only the apartment variables, i.e. water
    # temperatures and room temperature based on all input variables including the weather.
    mod_apt = RNNDynamicModel(ds,
                              name="ApartmentOnly",
                              hidden_sizes=(10, 10),
                              n_iter_max=10,
                              constraint_list=rnn_consts,
                              out_inds=np.array([2, 3, 5], dtype=np.int32),
                              **base_params_no_inds)
    # The full model combining weather and apartment model.
    mod_weather_and_apt = CompositeModel(ds, [mod_weather, mod_apt, model_time_exact], new_name="FullWeatherAndApt")

    # mod_const_wt = RNNDynamicModel(ds,
    #                                name="RoomTempOnly",
    #                                hidden_sizes=(50, 50),
    #                                n_iter_max=10,
    #                                out_inds=np.array([5], dtype=np.int32),
    #                                **base_params_no_inds)
    # mod_overshoot = RNNDynamicOvershootModel(n_overshoot=5,
    #                                          data=ds,
    #                                          name="Overshoot",
    #                                          hidden_sizes=(50, 50),
    #                                          n_iter_max=10,
    #                                          **base_params)
    # mod_overshoot_dec = RNNDynamicOvershootModel(n_overshoot=5,
    #                                              decay_rate=0.8,
    #                                              data=ds,
    #                                              name="Overshoot_Decay0.8",
    #                                              hidden_sizes=(50, 50),
    #                                              n_iter_max=10,
    #                                              **base_params)
    optimize = False
    if optimize:
        opt_params = mod_full.optimize(5)
        # print("All tried parameter combinations: {}.".format(mod.param_list))
        print("Optimal parameters: {}.".format(opt_params))

    mods = []  # mod_overshoot_dec, mod_overshoot, mod, mod_test]  # , mod_const_wt, mod_overshoot, mod_test, mod_no_consts]
    for m_to_use in mods:
        continue
        m_to_use.fit()
        print("16 Timestep performance: {}".format(m_to_use.hyper_objective()))
        # m_to_use.analyze()
        # m_to_use.analyze_disturbed("Valid", 'val', 10)
        # m_to_use.analyze_disturbed("Train", 'train', 10)

    # Full test model
    comp_model = CompositeModel(ds, [mod_test, time_model_ds], new_name="CompositeTimeRNNFull")
    comp_model.fit()
    env = FullRoomEnv(comp_model, disturb_fac=0.3)
    const_ag_1 = ConstHeating(env, 0.0)
    const_ag_2 = ConstHeating(env, env.nb_actions - 1)
    const_ag_3 = ConstHeating(env, env.nb_actions - 1)
    env.analyze_agent([const_ag_1, const_ag_2, const_ag_3])
    return

    dqn_agent = DQNRoomHeatingAgent(env)
    dqn_agent.fit()
    return

    # Exogenous variable model
    exo_inds = np.array([0, 1, 2], dtype=np.int32)
    pre_mod = Periodic1DayModel(ds, exo_inds, alpha=0.1)
    pre_mod.analyze_6_days()

    comp_model.analyze_6_days()

    # mod.fit()
    # mod.model_disturbance()
    # mod.disturb()
    # mod.analyze()
    # #
    mod_naive = ConstModel(ds)
    mod_naive.analyze()

    # Battery data
    bat_name = "Battery"
    get_battery_data()
    bat_ds = Dataset.loadDataset(bat_name)
    bat_ds.split_data()
    bat_mod = BatteryModel(bat_ds)
    bat_mod.analyze_bat_model()
    bat_mod.analyze()
    bat_mod_naive = ConstModel(bat_ds)
    bat_mod_naive.analyze()
    return


if __name__ == '__main__':
    main()
