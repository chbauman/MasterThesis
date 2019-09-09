
import datetime

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

clr_map = ['blue', 'green', 'c', 'k']
n_cols = len(clr_map)


def plot_time_series(x, y, m, show = True):
    """
    Plots a time-series where x are the dates and
    y are the values.
    """

    # Format the date
    formatter = DateFormatter('%d/%m/%y')

    # Define plot
    fig, ax = plt.subplots()
    plt.plot_date(x, y, linestyle=':', marker='^', color='red', markersize=5, mfc = 'blue', mec = 'blue')
    plt.title(m['description'])
    plt.ylabel(m['unit'])
    plt.xlabel('Time')
    #ax.xaxis.set_major_formatter(formatter)
    #ax.xaxis.set_tick_params(rotation=30, labelsize=10)

    # Show plot
    if show:
        plt.show()
    return

def plot_ip_time_series(y, lab = None, m = None, show = True):
    """
    Plots an interpolated time series
    where x is assumed to be uniform.
    """

    # Define plot
    fig, ax = plt.subplots()
    if isinstance(y, list):
        n = y[0].shape[0]
        x = [15 * i for i in range(n)]
        for ct, ts in enumerate(y):
            clr = clr_map[ct % n_cols]
            curr_lab = None if lab is None else lab[ct]
            plt.plot(x, ts, linestyle=':', marker='^', color='red', label = curr_lab, markersize=5, mfc = clr, mec = clr)
    else:
        plt.plot(y, linestyle=':', marker='^', color='red', label = lab, markersize=5, mfc = 'blue', mec = 'blue')

    if m is not None:
        plt.title(m['description'])
        plt.ylabel(m['unit'])

    plt.xlabel('Time [15 min.]')
    plt.legend()

    # Show plot
    if show:
        plt.show()
    return



