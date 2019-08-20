
import numpy as np

from FQI import NFQI
from LSPI import LSPI
from simple_battery_test import SimpleBatteryTest

from data import ElData, HeatingData, TestData





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

    dat = ElData.getData()
    dat = HeatingData.getData()
    dat = TestData.getData()
    print(len(dat))
    print(dat)

    #simple_battery_FQI()

    return 0


main()




