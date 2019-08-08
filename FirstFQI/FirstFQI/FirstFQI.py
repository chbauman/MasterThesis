


from rcnfq import NFQ
from MyFQI import NFQI

def simple_battery_FQI():

    from simple_battery_test import state_dim, nb_actions, get_transition_tuples

    [s_t, a_t, r_t, s_tp1] = get_transition_tuples()

    fqi = NFQI(state_dim, nb_actions)
    fqi.fit(s_t, a_t, r_t, s_tp1)

def main():

    simple_battery_FQI()

    return 0



main()



