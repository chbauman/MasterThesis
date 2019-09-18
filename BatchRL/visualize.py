
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters

from util import *

#mpl.use("pgf")
register_matplotlib_converters()

plt.rc('font', family='serif')
#plt.rc('text', usetex=True) # Makes Problems with the Celsius sign

clr_map = ['blue', 'green', 'c']
n_cols = len(clr_map)


def plot_time_series(x, y, m, show = True):
    """
    Plots a time-series where x are the dates and
    y are the values.
    """

    # Define plot
    fig, ax = plt.subplots()
    plot_helper(x, y, 'blue', dates = True)
    title = cleas_desc(m['description'])
    plt.title(title)
    plt.ylabel(m['unit'])
    plt.xlabel('Time')

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

def plot_ip_time_series(y, lab = None, m = None, show = True, init = None, mean_and_stds = None, use_time = False):
    """
    Plots an interpolated time series
    where x is assumed to be uniform.
    DEPRECATED!!!!!
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

        if use_time:
            mins = m[0]['dt']
            interv = np.timedelta64(mins, 'm')
            dt_init = datetime_to_npdatetime(string_to_dt(m[0]['t_init']))
            x = [dt_init + i * interv for i in range(n)]
        else:
            x = [15 * i for i in range(n_init, n_init + n)]

        for ct, ts in enumerate(y):
            if mean_and_stds is not None:
                ts = mean_and_stds[ct][1] * ts + mean_and_stds[ct][0]
            clr = clr_map[ct % n_cols]
            curr_lab = None if lab is None else lab[ct]
            plot_helper(x, ts, m_col = clr, label = curr_lab, dates = use_time)
    else:
        y_curr = y
        if mean_and_stds is not None:
            y_curr = mean_and_stds[1] * y + mean_and_stds[0]
        x = range(len(y_curr))
        plot_helper(x, y_curr, m_col = 'blue', label = lab, dates = use_time)

        if m is not None:
            plt.title(m['description'])
            plt.ylabel(m['unit'])

    plt.xlabel('Time [min.]')
    plt.legend()

    # Show plot
    if show:
        plt.show()
    return

def plot_single_ip_ts(y,
                       lab = None,
                       show = True,
                       *,
                       mean_and_std = None,
                       use_time = False,
                       title_and_ylab = None, 
                       dt_mins = 15, 
                       dt_init_str = None):
    """
    Wrapper function with fewer arguments for single
    time series plotting.
    """
    plot_ip_ts(y,
               lab = lab,
               show = show,
               mean_and_std = mean_and_std,
               title_and_ylab = title_and_ylab, 
               dt_mins = dt_mins, 
               dt_init_str = dt_init_str, 
               use_time = use_time,
               last_series = True, 
               series_index = 0,
               timestep_offset = 0)

def plot_ip_ts(y,
               lab = None,
               show = True,
               mean_and_std = None,
               use_time = False,
               series_index = 0,
               last_series = True, 
               title_and_ylab = None, 
               dt_mins = 15, 
               dt_init_str = None, 
               timestep_offset = 0):
    """
    Plots an interpolated time series
    where x is assumed to be uniform.
    """

    if series_index == 0:
        # Define plot
        fig, ax = plt.subplots()

    n = len(y)
    y_curr = np.copy(y)    

    # Add std and mean back
    if mean_and_std is not None:
        y_curr = mean_and_std[1] * y + mean_and_std[0]

    # Use datetimes for x values
    if use_time:
        if dt_init_str is None:
            raise ValueError("Need to know the initial time of the time series when plotting with dates!")

        mins = dt_mins
        interv = np.timedelta64(mins, 'm')
        dt_init = datetime_to_npdatetime(string_to_dt(dt_init_str))
        x = [dt_init + (timestep_offset + i) * interv for i in range(n)]
    else:
        x = range(n)

    plot_helper(x, y_curr, m_col = clr_map[series_index], label = cleas_desc(lab), dates = use_time)

    if last_series:
        if title_and_ylab is not None:
            plt.title(title_and_ylab[0])
            plt.ylabel(title_and_ylab[1])

        x_lab = 'Time' if use_time else 'Time [' + str(dt_mins) + ' min.]'
        plt.xlabel(x_lab)
        plt.legend()

        # Show plot
        if show:
            plt.show()
    return

def plot_multiple_ip_ts(y_list, lab_list = None, mean_and_std_list = None, use_time = False, timestep_offset_list = None, dt_init_str_list = None, show_last = True, title_and_ylab = None):
    """
    Plotting function for multiple time series.
    """
    n = len(y_list)
    for k, y in enumerate(y_list):
        ts_offset = get_if_nnone(timestep_offset_list, k, 0)
        lab = get_if_nnone(lab_list, k)
        m_a_s = get_if_nnone(mean_and_std_list, k)
        dt_init_str = get_if_nnone(dt_init_str_list, k)

        last_series = k == n - 1
        plot_ip_ts(y,
               lab = lab,
               show = last_series and show_last,
               mean_and_std = m_a_s,
               use_time = use_time,
               series_index = k,
               last_series = last_series, 
               title_and_ylab = title_and_ylab, 
               dt_mins = 15, 
               dt_init_str = dt_init_str, 
               timestep_offset = ts_offset)

def plot_single(time_series, m, use_time = True, show = True, title_and_ylab = None, scale_back = True):
    """
    Higher level plot function for single time series.
    """
    m_a_s = m.get('mean_and_std') if scale_back else None
    plot_single_ip_ts(time_series,
                       lab = m.get('description'),
                       show = show,
                       mean_and_std = m_a_s,
                       use_time = use_time,
                       title_and_ylab = title_and_ylab, 
                       dt_mins = m.get('dt'), 
                       dt_init_str = m.get('t_init'))

def plot_all(all_data, m, use_time = True, show = True, title_and_ylab = None, scale_back = True):
    """
    Higher level plot funciton for multiple time series
    stored in the matrix 'all_data'
    as they are e.g. saved in processed form.
    """

    n_series = all_data.shape[1]
    all_series = [all_data[:, i] for i in range(n_series)]

    mean_and_std_list = [m[i].get('mean_and_std') for i in range(n_series)] if scale_back else None

    plot_multiple_ip_ts(all_series,
                        lab_list = [m[i].get('description') for i in range(n_series)],
                        mean_and_std_list = mean_and_std_list,
                        use_time = use_time,
                        timestep_offset_list = [0 for i in range(n_series)],
                        dt_init_str_list = [m[i].get('t_init') for i in range(n_series)],
                        show_last = show, 
                        title_and_ylab = title_and_ylab)

def scatter_plot(x, y, *, show = True, lab_dict = None, lab = 'Measurements', m_and_std_x = None, m_and_std_y = None, add_line = False, custom_line = None, custom_label = None):
    """
    Scatter Plot. 
    """

    fig, ax = plt.subplots()
        
    # Transform data back to original mean and std.
    x_curr = x
    if m_and_std_x is not None:
        x_curr = m_and_std_x[1] * x + m_and_std_x[0]
    y_curr = y
    if m_and_std_y is not None:
        y_curr = m_and_std_y[1] * y + m_and_std_y[0]

    if add_line:
        # Fit a line with Least Squares
        n = x_curr.shape[0]
        ls_mat = np.empty((n, 2), dtype = np.float32)
        ls_mat[:, 0] = 1
        ls_mat[:, 1] = x_curr
        m, c = np.linalg.lstsq(ls_mat, y_curr, rcond=None)[0]

        max_x = np.max(x_curr)
        min_x = np.min(x_curr)
        x = np.linspace(min_x, max_x, 5)
        y = m + c * x
        plt.plot(x, y, label = 'Linear Fit')

    if custom_line:
        # Add a custom line to plot
        x, y = custom_line
        if m_and_std_y is not None:
            y = m_and_std_y[1] * y + m_and_std_y[0]
        if m_and_std_x is not None:
            x = m_and_std_x[1] * x + m_and_std_x[0]
        plt.plot(x, y, label = custom_label)

    # Plot
    plt.scatter(x_curr, y_curr,  marker='^', c='red', label = lab)
    
    # Add Labels
    if lab_dict is not None:
        plt.title(lab_dict['title'])
        plt.ylabel(lab_dict['ylab'])
        plt.xlabel(lab_dict['xlab'])
    plt.legend()

    if show:
        plt.show()