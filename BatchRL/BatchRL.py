
import numpy as np

from FQI import NFQI
from LSPI import LSPI
from batchDDPG import bDDPG

# Environments for debugging
from simple_battery_test import SimpleBatteryTest
from cart_pole import CartPole
from mount_car_cont import MountCarCont
from pendulum import Pendulum

from data import Room274Data, Room272Data, WeatherData, TestData, analyze_data, get_all_relevant_data
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
    dat, m = WeatherData.getData()
    dat = get_all_relevant_data()
    plot_ip_time_series(dat[:,0], m[0], show = False)
    plot_ip_time_series(dat[:,1], m[2], show = True)
    return 0

    dat, m = Room274Data.getData()
    dat, m = Room272Data.getData()
    dat, m = WeatherData.getData()
    analyze_data(dat[0])

    [y, dt] = interpolate_time_series(dat[0], 15)
    plot_ip_time_series(y, m[0], show = False)

    for ct, el in enumerate(dat):
        plot_time_series(el[1], el[0], m[ct])
    dat, m = TestData.getData()
    
    #print(len(dat))
    #print(m)

    #simple_battery_FQI()

    return 0


main()




