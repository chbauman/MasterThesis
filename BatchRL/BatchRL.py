from FQI import NFQI
from LSPI import LSPI
from batchDDPG import bDDPG
from dm_Composite import CompositeModel

from dm_LSTM import RNNDynamicModel
from dm_GPR import GPR_DM
from dm_Const import ConstModel
from battery_model import BatteryModel
from dm_Time import SCTimeModel
from dm_TimePeriodic import Periodic1DayModel
from keras_layers import test_layers
from simple_battery_test import SimpleBatteryTest
# Environments for debugging
from cart_pole import CartPole
from mount_car_cont import MountCarCont
from pendulum import Pendulum
from util import *
from dynamics_env import DynEnv
from data import WeatherData, TestData, \
    get_UMAR_heating_data, get_data_test, \
    cut_data_into_sequences, extract_streak, get_battery_data, cut_and_split, \
    get_DFAB_heating_data, \
    compute_DFAB_energy_usage, get_weather_data, generate_room_datasets, \
    analyze_room_energy_consumption, Dataset, test_align, test_dataset_artificially, no_inds, \
    generate_sin_cos_time_ds, test_rest_client, SeriesConstraint


def simple_battery_FQI():
    sbt = SimpleBatteryTest(bidirectional=True)
    sbt = CartPole()
    sbt = MountCarCont()
    sbt = Pendulum()

    state_dim = sbt.state_dim
    nb_actions = sbt.nb_actions

    [s_t, a_t, r_t, s_tp1] = sbt.get_transition_tuples(n_tuples=100000)

    print((np.c_[s_t, a_t, r_t, s_tp1])[:50])

    # fqi = NFQI(state_dim, nb_actions, stoch_policy_imp = False, use_diff_target_net=False,
    # param_updata_fac=0.5, max_iters = 20, lr = 0.001)
    fqi = bDDPG(state_dim, nb_actions)
    # fqi = LSPI(state_dim, nb_actions, stoch_policy_imp=True)

    fqi.fit(s_t, a_t, r_t, s_tp1)

    sbt.eval_policy(fqi.get_policy())


def main():
    # get_DFAB_heating_data()
    # generate_room_datasets()

    # w_dat = get_weather_data()
    # w_dat.split_train_test()
    # pre_mod = Periodic1DayModel(w_dat, None, alpha=0.1)
    # pre_mod.analyze_6_days()

    # Do tests
    test_time_stuff()
    test_layers()
    test_numpy_functions()
    test_rest_client()
    test_python_stuff()
    # get_data_test()
    # test_align()
    test_dataset_artificially()
    return

    # Dataset
    name_ds = 'Model_Room43'
    ds = Dataset.loadDataset(name_ds)
    ds = ds.add_time()
    ds.standardize()
    ds.split_data()

    # Time variable prediction
    time_model_ds = SCTimeModel(ds, 6)
    # time_model_ds.analyze()
    # time_model_ds.analyze_disturbed()

    # Constant model for water temperatures
    mod_naive = ConstModel(ds, pred_inds=np.array([2, 3], dtype=np.int32))
    # mod_naive.analyze()
    # mod_naive.analyze_disturbed()

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
    mod_test = RNNDynamicModel(ds,
                               name="Test",
                               hidden_sizes=(10, 10),
                               n_iter_max=5,
                               input_noise_std=0.001,
                               lr=0.01,
                               residual_learning=True,
                               weight_vec=None,
                               out_inds=np.array([0, 1, 2, 3, 5], dtype=np.int32),
                               constraint_list=rnn_consts)
    mod = RNNDynamicModel(ds,
                          hidden_sizes=(100, 100),
                          n_iter_max=100,
                          input_noise_std=0.001,
                          lr=0.001,
                          residual_learning=True,
                          weight_vec=None,
                          out_inds=np.array([0, 1, 2, 3, 5], dtype=np.int32),
                          constraint_list=rnn_consts
                          )
    mod_no_consts = RNNDynamicModel(ds,
                                    name="RNN_No_Consts",
                                    hidden_sizes=(100, 100),
                                    n_iter_max=100,
                                    input_noise_std=0.001,
                                    lr=0.001,
                                    residual_learning=True,
                                    weight_vec=None,
                                    out_inds=np.array([0, 1, 2, 3, 5], dtype=np.int32),
                                    constraint_list=None
                                    )
    mod_no_wt = RNNDynamicModel(ds,
                                name="RNN_NoWT",
                                hidden_sizes=(100, 100),
                                n_iter_max=100,
                                input_noise_std=0.001,
                                lr=0.001,
                                residual_learning=True,
                                weight_vec=None,
                                out_inds=np.array([0, 1, 5], dtype=np.int32),
                                constraint_list=rnn_consts)

    mods = [mod_test]  # mod, mod_no_consts, mod_no_wt]
    for m_to_use in mods:
        m_to_use.fit()
        m_to_use.analyze()
        m_to_use.analyze_disturbed("Valid", 'val', 10)
        m_to_use.analyze_disturbed("Train", 'train', 10)
    return
    # mod.optimize(2)

    # Exogenous variable model
    exo_inds = np.array([0, 1, 2], dtype=np.int32)
    pre_mod = Periodic1DayModel(ds, exo_inds, alpha=0.1)
    pre_mod.analyze_6_days()

    # Full model
    comp_model = CompositeModel(ds, [mod, time_model_ds], new_name="CompositeTimeRNNFull")
    comp_model.fit()
    comp_model.analyze_6_days()

    # mod.fit()
    # mod.model_disturbance()
    # mod.disturb()
    # mod.analyze()
    # #
    mod_naive = ConstModel(ds)
    mod_naive.analyze()

    # compute_DFAB_energy_usage()

    # Battery data
    bat_name = "Battery"
    get_battery_data()
    bat_ds = Dataset.loadDataset(bat_name)
    bat_ds.split_train_test(7)
    bat_ds.get_prepared_data()
    bat_mod = BatteryModel(bat_ds)
    bat_mod.analyze_bat_model()
    bat_mod.analyze()
    bat_mod_naive = ConstModel(bat_ds)
    bat_mod_naive.analyze()
    return

    # simple_battery_FQI()


if __name__ == '__main__':
    main()
