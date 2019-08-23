
import numpy as np

from FQI import NFQI
from LSPI import LSPI
from simple_battery_test import SimpleBatteryTest

from data import TestData, ElData, HeatingData, ValveData
from visualize import plot_time_series




def simple_battery_FQI():

    sbt = SimpleBatteryTest(bidirectional = True)
    state_dim = sbt.state_dim
    nb_actions = sbt.nb_actions

    [s_t, a_t, r_t, s_tp1] = sbt.get_transition_tuples(n_tuples = 30000)

    #print((np.c_[s_t, a_t, r_t, s_tp1])[:15])

    fqi = NFQI(state_dim, nb_actions, stoch_policy_imp = True)
    #fqi = LSPI(state_dim, nb_actions, stoch_policy_imp=True)

    fqi.fit(s_t, a_t, r_t, s_tp1)

    sbt.eval_policy(fqi.get_policy())

def main():

    from keral_custom_optim import DDPGOpt

    ddpgOpt = DDPGOpt()

    print("Success!!")
    return 0

    dat, m = ValveData.getData()
    dat, m = HeatingData.getData()
    dat, m = ElData.getData()
    plot_time_series(dat[1][1], dat[1][0])
    dat, m = TestData.getData()
    
    print(len(dat))
    print(m)

    #simple_battery_FQI()

    return 0


main()




