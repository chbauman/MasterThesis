
import numpy as np

from FQI import NFQI
from LSPI import LSPI
from batchDDPG import bDDPG
from simple_battery_test import SimpleBatteryTest
from cart_pole import CartPole

from data import TestData, ElData, HeatingData, ValveData
from visualize import plot_time_series




def simple_battery_FQI():

    sbt = SimpleBatteryTest(bidirectional = True)
    sbt = CartPole()


    state_dim = sbt.state_dim
    nb_actions = sbt.nb_actions

    [s_t, a_t, r_t, s_tp1] = sbt.get_transition_tuples(n_tuples = 30000)

    print((np.c_[s_t, a_t, r_t, s_tp1])[:50])

    fqi = NFQI(state_dim, nb_actions, stoch_policy_imp = False, use_diff_target_net=False, param_updata_fac=0.5, max_iters = 20, lr = 0.001)
    #fqi = LSPI(state_dim, nb_actions, stoch_policy_imp=True)

    fqi.fit(s_t, a_t, r_t, s_tp1)

    sbt.eval_policy(fqi.get_policy())

def main():


    #dat, m = ValveData.getData()
    #dat, m = HeatingData.getData()
    #dat, m = ElData.getData()
    #plot_time_series(dat[1][1], dat[1][0])
    #dat, m = TestData.getData()
    
    #print(len(dat))
    #print(m)

    simple_battery_FQI()

    return 0


main()




