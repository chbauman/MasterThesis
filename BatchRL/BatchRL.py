
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

    # Electricity Stuff
    data_desc = [
        "alarm active",
        "el. active power total",
        "el. active power consumpted",
        "el. active power L1",
        "el. reactive power L1",
        "el. cos Phi L1",
        "el. current L1",
        "el. voltage L1",
        "el. active power L2",
        "el. reactive power L2",
        "el. cos Phi L2",
        "el. current L2",
        "el. voltage L2",
        "el. active power L3",
        "el. reactive power L3",
        "el. cos Phi L3",
        "el. current L3",
        "el. voltage L3",
        ]
    num_cols = len(data_desc)
    data_ids = [str(42190138 + i) for i in range(num_cols)]

    df_read = REST.read(df_data=data_ids,
                        startDate='2019-01-01',
                        endDate='2019-12-31')
    print(df_read)

    REST.write_np("Test")

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




