


from rcnfq import NFQ

def simple_battery_FQI():

    from simple_battery_test import state_dim, nb_actions, get_transition_tuples

    [s_t, a_t, r_t, s_tp1] = get_transition_tuples()

    fqi = NFQ(state_dim, nb_actions, None)
    fqi.fit_vectorized(s_t, a_t, r_t, s_tp1)


def main():

    simple_battery_FQI()

    return 0



main()



