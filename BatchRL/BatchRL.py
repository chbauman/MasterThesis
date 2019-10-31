"""The main script of this project.

The `main` function runs all the necessary high-level
functions. The complicated stuff is hidden in the other
modules.
"""
from agents_heuristic import ConstHeating
from base_dynamics_env import test_test_env
from batchDDPG import bDDPG
from battery_model import BatteryModel
from cart_pole import CartPole
from data import get_battery_data, \
    Dataset, test_dataset_artificially, SeriesConstraint
from dm_Composite import CompositeModel
from dm_Const import ConstModel
from dm_LSTM import RNNDynamicModel, test_rnn_models, RNNDynamicOvershootModel
from dm_Time import SCTimeModel
from dm_TimePeriodic import Periodic1DayModel
from dynamics_envs import FullRoomEnv
from keras_layers import test_layers
from keras_rl_wrap import DQNRoomHeatingAgent
from mount_car_cont import MountCarCont
from pendulum import Pendulum
from simple_battery_test import SimpleBatteryTest
from util import *


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

    # Run tests
    test_layers()
    test_rnn_models()
    # test_dyn_model()
    test_test_env()
    test_time_stuff()
    test_numpy_functions()
    # test_rest_client()
    test_python_stuff()
    # # get_data_test()
    # # test_align()
    test_dataset_artificially()

    # Dataset
    name_ds = 'Model_Room43'
    ds = Dataset.loadDataset(name_ds)

    # Change seq_len
    # ds.seq_len = 7 * 96
    # ds.name = "Room_" + str(ds.seq_len)

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
    base_params = {'input_noise_std': 0.001,
                   'lr': 0.01,
                   'residual_learning': True,
                   'weight_vec': None,
                   'out_inds': np.array([0, 1, 2, 3, 5], dtype=np.int32),
                   }
    base_params_no_inds = {k: base_params[k] for k in base_params if k != 'out_inds'}
    mod_test = RNNDynamicModel(ds,
                               name="Test",
                               hidden_sizes=(10, 10),
                               n_iter_max=5,
                               constraint_list=rnn_consts,
                               **base_params)
    mod = RNNDynamicModel(ds,
                          hidden_sizes=(50, 50),
                          n_iter_max=10,
                          constraint_list=rnn_consts,
                          **base_params)
    mod_no_consts = RNNDynamicModel(ds,
                                    name="RNN_No_Consts",
                                    hidden_sizes=(50, 50),
                                    n_iter_max=10,
                                    **base_params)
    mod_const_wt = RNNDynamicModel(ds,
                                   name="RNN_ConstWT",
                                   hidden_sizes=(50, 50),
                                   n_iter_max=10,
                                   out_inds=np.array([0, 1, 5], dtype=np.int32),
                                   **base_params_no_inds)
    mod_overshoot = RNNDynamicOvershootModel(n_overshoot=5,
                                             data=ds,
                                             name="Overshoot",
                                             hidden_sizes=(50, 50),
                                             n_iter_max=10,
                                             **base_params)
    mod_overshoot_dec = RNNDynamicOvershootModel(n_overshoot=5,
                                                 decay_rate=0.8,
                                                 data=ds,
                                                 name="Overshoot_Decay0.8",
                                                 hidden_sizes=(50, 50),
                                                 n_iter_max=10,
                                                 **base_params)
    optimize = False
    if optimize:
        opt_params = mod.optimize(5)
        # print("All tried parameter combinations: {}.".format(mod.param_list))
        print("Optimal parameters: {}.".format(opt_params))

    mods = [mod_overshoot_dec, mod_overshoot, mod, mod_test]  # , mod_const_wt, mod_overshoot, mod_test, mod_no_consts]
    for m_to_use in mods:
        # m_to_use.fit()
        print("16 Timestep performance: {}".format(m_to_use.hyper_objective()))
        # m_to_use.analyze()
        # m_to_use.analyze_disturbed("Valid", 'val', 10)
        # m_to_use.analyze_disturbed("Train", 'train', 10)

    # Full test model
    comp_model = CompositeModel(ds, [mod_test, time_model_ds], new_name="CompositeTimeRNNFull")
    comp_model.fit()
    env = FullRoomEnv(comp_model, disturb_fac=0.3)
    const_ag_1 = ConstHeating(env, 0.0)
    const_ag_2 = ConstHeating(env, 1.0)
    env.analyze_agent([const_ag_1, const_ag_2])
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

    # compute_DFAB_energy_usage()

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

    # simple_battery_FQI()


if __name__ == '__main__':
    main()
