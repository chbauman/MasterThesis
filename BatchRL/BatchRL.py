
import numpy as np

from FQI import NFQI
from LSPI import LSPI
from batchDDPG import bDDPG

from dm_LSTM import BaseRNN_DM
from dm_GPR import GPR_DM

# Environments for debugging
from simple_battery_test import SimpleBatteryTest
from cart_pole import CartPole
from mount_car_cont import MountCarCont
from pendulum import Pendulum

from data import Room274Data, Room272Data, WeatherData, TestData, \
    analyze_data, get_all_relevant_data, get_data_test, \
    cut_data_into_sequences, extract_streak
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
    
    #get_data_test()
    #return 0

    # Parameters
    seq_len = 20

    # Prepare data
    dat, m, name = get_all_relevant_data(2.0)
    dat_s = dat.shape
    for k in range(dat_s[1]):
        plot_ip_time_series(dat[:,k], m = m[k], show = k == 5)
    dat_train, dat_test = extract_streak(dat, 96 * 7, seq_len - 1)
    cut_train_dat = cut_data_into_sequences(dat_train, seq_len, interleave = True)
    cut_test_dat = cut_data_into_sequences(dat_test, seq_len, interleave = True)

    train_shape = cut_train_dat.shape
    n_feats = train_shape[-1]

    print("Train data shape:", train_shape)
    print("Analysis data shape:", cut_test_dat.shape)

    mod = BaseRNN_DM(seq_len - 1, n_feats, n_iter_max=50, input_noise_std = 0.05, name = "Train50_" + name)
    #mod = GPR_DM()
    mod.fit(cut_train_dat)
    mod.analyze(cut_test_dat)
    return

    # GP model
    cut_dat_5 = cut_data_into_sequences(dat, 5, interleave = True)
    m_gp = GPR_DM()
    m_gp.fit(cut_dat_5)
    m_gp.analyze()

    #plot_ip_time_series(dat[:,0], m[0], show = False)
    #plot_ip_time_series(dat[:,1], m[1], show = False)
    #plot_ip_time_series(dat[:,2], m[2], show = False)
    #plot_ip_time_series(dat[:,3], m[3], show = False)
    #plot_ip_time_series(dat[:,4], m[4], show = False)
    #plot_ip_time_series(dat[:,5], m[5], show = True)
    return 0

    #simple_battery_FQI()

    return 0


main()




