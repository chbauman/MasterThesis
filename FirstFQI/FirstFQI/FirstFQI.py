


from MyFQI import NFQI
from simple_battery_test import SimpleBatteryTest

def simple_battery_FQI():

    sbt = SimpleBatteryTest()
    state_dim = sbt.state_dim
    nb_actions = sbt.nb_actions

    [s_t, a_t, r_t, s_tp1] = sbt.get_transition_tuples()

    fqi = NFQI(state_dim, nb_actions)
    fqi.fit(s_t, a_t, r_t, s_tp1)

    sbt.eval_policy(fqi.get_policy())

def main():

    simple_battery_FQI()

    return 0



main()



