

import datetime

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



def plot_time_series(x, y, m):
    """
    Plots a time-series where x are the dates and
    y are the values.
    """

    # Format the date
    formatter = DateFormatter('%d/%m/%y')

    # Define plot
    fig, ax = plt.subplots()
    plt.plot_date(x, y, ls = '-', ms = 0.1)
    plt.title(m['description'])
    plt.ylabel(m['unit'])
    plt.xlabel('Time')
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(rotation=30, labelsize=10)

    # Show plot
    plt.show()
