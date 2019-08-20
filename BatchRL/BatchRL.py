
import numpy as np

from FQI import NFQI
from LSPI import LSPI
from simple_battery_test import SimpleBatteryTest

import restclient
import pandas as pd


def getData():
    """
    
    """
    REST = restclient.client(username='bach', password='Welc0me$2019!') # User of visualizer

    df_read = REST.read(df_data=['42190140', '42190141'],
                                             startDate='2019-08-03',
                                             endDate='2019-12-04')
    print(df_read)
    return df_read





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

    getData()

    #simple_battery_FQI()

    return 0


main()




