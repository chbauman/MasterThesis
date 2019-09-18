
import numpy as np

from FQI import NFQI
from LSPI import LSPI
from batchDDPG import bDDPG

from dm_LSTM import BaseRNN_DM
from dm_GPR import GPR_DM

from battery_model import BatteryModel

# Environments for debugging
from simple_battery_test import SimpleBatteryTest
from cart_pole import CartPole
from mount_car_cont import MountCarCont
from pendulum import Pendulum

from data import Room274Data, Room272Data, WeatherData, TestData, \
    analyze_data, get_heating_data, get_data_test, \
    cut_data_into_sequences, extract_streak, get_battery_data, cut_and_split, \
    process_DFAB_heating_data, test_plotting_withDFAB_data, test_dataset_with_DFAB
from visualize import plot_time_series, plot_ip_time_series

def simple_battery_FQI():

    sbt = SimpleBatteryTest(bidirectional = True)
    sbt = CartPole()
    sbt = MountCarCont()
    sbt = Pendulum()

    state_dim = sbt.state_dim
    nb_actions = sbt.nb_actions

    [s_t, a_t, r_t, s_tp1] = sbt.get_transition_tuples(n_tuples = 100000)

    print((np.c_[s_t, a_t, r_t, s_tp1])[:50])

    #fqi = NFQI(state_dim, nb_actions, stoch_policy_imp = False, use_diff_target_net=False, param_updata_fac=0.5, max_iters = 20, lr = 0.001)
    fqi = bDDPG(state_dim, nb_actions)
    #fqi = LSPI(state_dim, nb_actions, stoch_policy_imp=True)

    fqi.fit(s_t, a_t, r_t, s_tp1)

    sbt.eval_policy(fqi.get_policy())

def main():

    #test_dataset_with_DFAB()
    #return
    #TestData.getData()
    ##get_data_test()
    #return 0

    process_DFAB_heating_data(show_plots = True)
    #test_plotting_withDFAB_data()

    # Battery data
    dat_bat, m_bat, name_bat = get_battery_data(show_plot = True, show = False)
    train_bat, test_bat = cut_and_split(dat_bat, 2, 96 * 7)
    bat_mod = BatteryModel()
    bat_mod.fit(train_bat, m_bat)
    return

    # This crashes:
    #mod.analyze(test_bat)
    #return

    # Parameters
    seq_len = 20

    # Heating data
    dat_heat, m_heat, name_heat = get_heating_data(2.0)
    dat_s = dat_heat.shape
    if False:
        for k in range(dat_s[1]):
            plot_ip_time_series(dat_heat[:,k], m = m_heat[k], show = k == 5)
    train_heat, test_heat = cut_and_split(dat_heat, seq_len, 96 * 7)
    train_shape = train_heat.shape
    n_feats = train_shape[-1]
    print("Train data shape:", train_shape)
    print("Analysis data shape:", test_heat.shape)

    # Train Heating model
    mod = BaseRNN_DM(seq_len - 1, n_feats, hidden_sizes=[50, 50], n_iter_max=50, input_noise_std = 0.01, name = "Train50_50-50_" + name_heat)
    mod.fit(train_heat)
    mod.analyze(test_heat)
    
    # GP Model
    seq_len_gp = 4
    train_heat, test_heat = cut_and_split(dat_heat, seq_len_gp, 96 * 7)
    mod = GPR_DM(alpha = 5.0)
    mod.fit(train_heat)
    mod.analyze(test_heat)

    return

    #simple_battery_FQI()

    return 0


main()




