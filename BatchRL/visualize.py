
import datetime

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

clr_map = ['blue', 'green', 'c']
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


def plot_helper(x, y, m_col = 'blue', label = None, dates = False):

    """
    Basic plot style for all plots.
    """
    ls = ':'
    color = 'red'
    marker = '^'
    ms = 5
    kwargs = {'marker': marker, 'c':color, 'linestyle': ls, 'label': label, 'markersize': ms, 'mfc': m_col, 'mec': m_col}

    if dates:
        plt.plot_date(x, y, **kwargs)
    else:
        plt.plot(x, y, **kwargs)


def plot_ip_time_series(y, lab = None, m = None, show = True, init = None, mean_and_stds = None):
    """
    Plots an interpolated time series
    where x is assumed to be uniform.
    """

    # Define plot
    fig, ax = plt.subplots()
    if isinstance(y, list):
        n = y[0].shape[0]
        n_init = 0 if init is None else init.shape[0]
        if init is not None:
            x_init = [15 * i for i in range(n_init)]
            plot_helper(x_init, init, m_col = 'k')
            #plt.plot(x_init, init, linestyle=':', marker='^', color='red', markersize=5, mfc = 'k', mec = 'k')

        x = [15 * i for i in range(n_init, n_init + n)]
        for ct, ts in enumerate(y):
            if mean_and_stds is not None:
                ts = mean_and_stds[ct][1] * ts + mean_and_stds[ct][0]
            clr = clr_map[ct % n_cols]
            curr_lab = None if lab is None else lab[ct]
            plot_helper(x, ts, m_col = clr, label = curr_lab)
    else:
        y_curr = y
        if mean_and_stds is not None:
            y_curr = mean_and_stds[1] * y + mean_and_stds[0]
        x = range(len(y_curr))
        plt.plot(y_curr, linestyle=':', marker='^', color='red', label = lab, markersize=5, mfc = 'blue', mec = 'blue')

    if m is not None:
        plt.title(m['description'])
        plt.ylabel(m['unit'])

    plt.xlabel('Time [min.]')
    plt.legend()

    # Show plot
    if show:
        plt.show()
    return

def scatter_plot(x, y, *, show = True, lab_dict = None, m_and_std_x = None, m_and_std_y = None):
    """
    Scatter Plot. 
    """

    # Transform data back to original mean and std.
    x_curr = x
    if m_and_std_x is not None:
        x_curr = m_and_std_x[1] * x + m_and_std_x[0]
    y_curr = y
    if m_and_std_y is not None:
        y_curr = m_and_std_y[1] * y + m_and_std_y[0]

    # Plot
    plt.scatter(x_curr, y_curr,  marker='^', c='red')
    
    # Add Labels
    if lab_dict is not None:
        plt.title(lab_dict['title'])
        plt.ylabel(lab_dict['ylab'])
        plt.xlabel(lab_dict['xlab'])

    if show:
        plt.show()