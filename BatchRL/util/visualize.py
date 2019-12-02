import os
import warnings
from typing import Dict, Sequence, Tuple, List

import matplotlib as mpl
import numpy as np
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from util.numerics import fit_linear_1d, load_performance, check_shape, ErrMetric
from util.util import EULER, datetime_to_np_datetime, string_to_dt, get_if_not_none, clean_desc, split_desc_units, \
    create_dir, Num, yeet, tot_size, mins_to_str

if EULER:
    # Do not use GUI based backend.
    mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors

"""This file contains all plotting stuff.

Plotting is based on matplotlib. 
If we are not on a Windows platform, the backend 'Agg' is
used, since this works also on Euler.

Most functions plot some kind of time series, but there
are also more specialized plot functions. E.g. plotting 
the training of a keras neural network.
"""

register_matplotlib_converters()

font = {'family': 'serif',
        # 'weight': 'bold',
        'size': 18}

plt.rc('font', **font)
# plt.rc('text', usetex=True)  # Makes Problems with the Celsius sign :(

# Plotting colors
colors = mpl_colors.TABLEAU_COLORS
names = list(colors)
clr_map = [colors[name] for name in names]
clr_map[0], clr_map[1] = clr_map[1], clr_map[0]  # For Bratislav
n_cols: int = len(clr_map)  #: Number of colors in colormap.

# Plotting styles
styles = [
    ("--", 'o'),
    ("-", 's'),
    ("-.", 'v'),
    (":", '*'),
]
joint_styles = [j + i for i, j in styles]

# Saving
PLOT_DIR = '../Plots'  #: Base plot folder.
preprocess_plot_path = os.path.join(PLOT_DIR, "Preprocessing")  #: Data processing plot folder.
model_plot_path = os.path.join(PLOT_DIR, "Models")  #: Dynamics modeling plot folder.
rl_plot_path = os.path.join(PLOT_DIR, "RL")
EVAL_MODEL_PLOT_DIR = os.path.join(model_plot_path, "EvalTables")

# Create folders if they do not exist
create_dir(preprocess_plot_path)
create_dir(model_plot_path)
create_dir(rl_plot_path)
create_dir(EVAL_MODEL_PLOT_DIR)


def save_figure(save_name, show: bool = False,
                vector_format: bool = True,
                size: Tuple[Num, Num] = None) -> None:
    """Saves the current figure.

    Args:
        size: The size in inches, if None, (16, 9) is used.
        save_name: Path where to save the plot.
        show: If true, does nothing.
        vector_format: Whether to save image in vector format.
    """
    if save_name is not None and not show:
        # Set figure size
        fig = plt.gcf()
        sz = size if size is not None else (16, 9)
        fig.set_size_inches(*sz)

        # Save and clear
        save_format = '.pdf' if vector_format else '.png'
        # save_kwargs = {'bbox_inches': 'tight', 'dpi': 500}
        save_kwargs = {'bbox_inches': 'tight'}
        plt.savefig(save_name + save_format, **save_kwargs)
        plt.close()


def _plot_helper(x, y, m_col='blue', label: str = None,
                 dates: bool = False, steps: bool = False, ax=plt) -> None:
    """Defining basic plot style for all plots.

    TODO: Make x optional. (Except for `dates` == True case!)

    Args:
        x: X values
        y: Y values
        m_col: Marker and line color.
        label: The label of the current series.
        dates: Whether to use datetimes in x-axis.
        steps: Whether to plot piecewise constant series.
        ax: The axis to plot the series on.
    """
    # Determine style
    ls = ':'
    marker = '^'
    ms = 2
    kwargs = {'marker': marker, 'c': m_col, 'linestyle': ls, 'label': label, 'markersize': ms, 'mfc': m_col,
              'mec': m_col}

    # Choose plotting method
    plot_method = ax.plot
    if dates:
        plot_method = ax.plot_date
    elif steps:
        plot_method = ax.step

    # Finally plot
    plot_method(x, y, **kwargs)


# Plotting raw data series
def plot_time_series(x, y, m: Dict, show: bool = True,
                     series_index: int = 0,
                     title: str = None,
                     save_name: str = None):
    """Plots a raw time-series where x are the dates and y are the values.
    """

    # Define plot
    lab = clean_desc(m['description'])
    if series_index == 0:
        # Init new plot
        plt.subplots()
    _plot_helper(x, y, clr_map[series_index], label=lab, dates=True)
    if title:
        plt.title(title)
    plt.ylabel(m['unit'])
    plt.xlabel('Time')
    plt.legend()

    # Show plot
    if show:
        plt.show()

    # Sate to raster image since vector image would be too large
    save_figure(save_name, show, vector_format=False)


def plot_multiple_time_series(x_list, y_list, m_list, *,
                              show: bool = True,
                              title_and_ylab: Sequence = None,
                              save_name: str = None):
    """Plots multiple raw time series.
    """
    n = len(x_list)
    for ct, x in enumerate(x_list):
        plot_time_series(x, y_list[ct], m_list[ct], show=show and ct == n - 1, series_index=ct)

    # Set title
    if title_and_ylab is not None:
        plt.title(title_and_ylab[0])
        plt.ylabel(title_and_ylab[1])
    plt.legend()

    # Sate to raster image since vector image would be too large
    save_figure(save_name, show, vector_format=False)


def plot_ip_time_series(y, lab=None, m=None, show=True, init=None, mean_and_stds=None, use_time=False):
    """
    Plots an interpolated time series
    where x is assumed to be uniform.
    DEPRECATED: DO NOT USE!!!!!
    """

    # Define plot
    plt.subplots()
    if isinstance(y, list):
        n = y[0].shape[0]
        n_init = 0 if init is None else init.shape[0]
        if init is not None:
            x_init = [15 * i for i in range(n_init)]
            _plot_helper(x_init, init, m_col='k')
            # plt.plot(x_init, init, linestyle=':', marker='^', color='red', markersize=5, mfc = 'k', mec = 'k')

        if use_time:
            mins = m[0]['dt']
            interval = np.timedelta64(mins, 'm')
            dt_init = datetime_to_np_datetime(string_to_dt(m[0]['t_init']))
            x = [dt_init + i * interval for i in range(n)]
        else:
            x = [15 * i for i in range(n_init, n_init + n)]

        for ct, ts in enumerate(y):
            if mean_and_stds is not None:
                ts = mean_and_stds[ct][1] * ts + mean_and_stds[ct][0]
            clr = clr_map[ct % n_cols]
            curr_lab = None if lab is None else lab[ct]
            _plot_helper(x, ts, m_col=clr, label=curr_lab, dates=use_time)
    else:
        y_curr = y
        if mean_and_stds is not None:
            y_curr = mean_and_stds[1] * y + mean_and_stds[0]
        x = range(len(y_curr))
        _plot_helper(x, y_curr, m_col='blue', label=lab, dates=use_time)

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
                      lab=None,
                      show=True,
                      *,
                      mean_and_std=None,
                      use_time=False,
                      title_and_ylab=None,
                      dt_mins=15,
                      dt_init_str=None):
    """
    Wrapper function with fewer arguments for single
    time series plotting.
    """
    plot_ip_ts(y,
               lab=lab,
               show=show,
               mean_and_std=mean_and_std,
               title_and_ylab=title_and_ylab,
               dt_mins=dt_mins,
               dt_init_str=dt_init_str,
               use_time=use_time,
               last_series=True,
               series_index=0,
               timestep_offset=0)


def plot_ip_ts(y,
               lab=None,
               show=True,
               mean_and_std=None,
               use_time=False,
               series_index=0,
               last_series=True,
               title_and_ylab=None,
               dt_mins=15,
               dt_init_str=None,
               timestep_offset=0):
    """
    Plots an interpolated time series
    where x is assumed to be uniform.
    """

    if series_index == 0:
        # Define new plot
        plt.subplots()

    n = len(y)
    y_curr = np.copy(y)

    # Add std and mean back
    if mean_and_std is not None:
        y_curr = mean_and_std[1] * y + mean_and_std[0]

    # Use datetime for x values
    if use_time:
        if dt_init_str is None:
            raise ValueError("Need to know the initial time of the time series when plotting with dates!")

        mins = dt_mins
        interval = np.timedelta64(mins, 'm')
        dt_init = datetime_to_np_datetime(string_to_dt(dt_init_str))
        x = [dt_init + (timestep_offset + i) * interval for i in range(n)]
    else:
        x = range(n)

    _plot_helper(x, y_curr, m_col=clr_map[series_index], label=clean_desc(lab), dates=use_time)

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


def plot_multiple_ip_ts(y_list,
                        lab_list=None,
                        mean_and_std_list=None,
                        use_time=False,
                        timestep_offset_list=None,
                        dt_init_str_list=None,
                        show_last=True,
                        title_and_ylab=None,
                        dt_mins=15):
    """
    Plotting function for multiple time series.
    """
    n = len(y_list)
    for k, y in enumerate(y_list):
        ts_offset = get_if_not_none(timestep_offset_list, k, 0)
        lab = get_if_not_none(lab_list, k)
        m_a_s = get_if_not_none(mean_and_std_list, k)
        dt_init_str = get_if_not_none(dt_init_str_list, k)

        last_series = k == n - 1
        plot_ip_ts(y,
                   lab=lab,
                   show=last_series and show_last,
                   mean_and_std=m_a_s,
                   use_time=use_time,
                   series_index=k,
                   last_series=last_series,
                   title_and_ylab=title_and_ylab,
                   dt_mins=dt_mins,
                   dt_init_str=dt_init_str,
                   timestep_offset=ts_offset)


def plot_single(time_series, m, use_time=True, show=True, title_and_ylab=None, scale_back=True, save_name=None):
    """
    Higher level plot function for single time series.
    """
    m_a_s = m.get('mean_and_std') if scale_back else None
    plot_single_ip_ts(time_series,
                      lab=m.get('description'),
                      show=show,
                      mean_and_std=m_a_s,
                      use_time=use_time,
                      title_and_ylab=title_and_ylab,
                      dt_mins=m.get('dt'),
                      dt_init_str=m.get('t_init'))
    save_figure(save_name, show)


def plot_all(all_data, m, use_time=True, show=True, title_and_ylab=None, scale_back=True, save_name=None):
    """
    Higher level plot function for multiple time series
    stored in the matrix 'all_data'
    as they are e.g. saved in processed form.
    """

    n_series = all_data.shape[1]
    all_series = [all_data[:, i] for i in range(n_series)]

    mean_and_std_list = [m[i].get('mean_and_std') for i in range(n_series)] if scale_back else None

    plot_multiple_ip_ts(all_series,
                        lab_list=[m[i].get('description') for i in range(n_series)],
                        mean_and_std_list=mean_and_std_list,
                        use_time=use_time,
                        timestep_offset_list=[0 for _ in range(n_series)],
                        dt_init_str_list=[m[i].get('t_init') for i in range(n_series)],
                        show_last=show,
                        title_and_ylab=title_and_ylab,
                        dt_mins=m[0].get('dt'))
    save_figure(save_name, show)


def plot_dataset(dataset, show: bool = True, title_and_ylab=None, save_name: str = None) -> None:
    """Plots the unscaled series in a dataset.

    Args:
        dataset: The dataset to plot.
        show: Whether to show the plot.
        title_and_ylab: List with title and y-label.
        save_name: The file path for saving the plot.
    """
    all_data = dataset.get_unscaled_data()
    n_series = all_data.shape[1]
    all_series = [np.copy(all_data[:, i]) for i in range(n_series)]
    labs = [d for d in dataset.descriptions]
    t_init = dataset.t_init

    plot_multiple_ip_ts(all_series,
                        lab_list=labs,
                        mean_and_std_list=None,
                        use_time=True,
                        timestep_offset_list=[0 for _ in range(n_series)],
                        dt_init_str_list=[t_init for _ in range(n_series)],
                        show_last=show,
                        title_and_ylab=title_and_ylab,
                        dt_mins=dataset.dt)
    save_figure(save_name, show)


def scatter_plot(x, y, *,
                 show=True,
                 lab_dict=None,
                 lab='Measurements',
                 m_and_std_x=None,
                 m_and_std_y=None,
                 add_line=False,
                 custom_line=None,
                 custom_label=None,
                 save_name=None):
    """
    Scatter Plot. 
    """

    plt.subplots()
    plt.grid(True)

    # Transform data back to original mean and std.
    x_curr = x
    if m_and_std_x is not None:
        x_curr = m_and_std_x[1] * x + m_and_std_x[0]
    y_curr = y
    if m_and_std_y is not None:
        y_curr = m_and_std_y[1] * y + m_and_std_y[0]

    if add_line:
        # Fit a line with Least Squares
        max_x = np.max(x_curr)
        min_x = np.min(x_curr)
        x = np.linspace(min_x, max_x, 5)
        y = fit_linear_1d(x_curr, y_curr, x)
        plt.plot(x, y, label='Linear Fit')

    if custom_line:
        # Add a custom line to plot
        x, y = custom_line
        if m_and_std_y is not None:
            y = m_and_std_y[1] * y + m_and_std_y[0]
        if m_and_std_x is not None:
            x = m_and_std_x[1] * x + m_and_std_x[0]
        plt.plot(x, y, label=custom_label)

    # Plot
    plt.scatter(x_curr, y_curr, marker='^', c='red', label=lab)

    # Add Labels
    if lab_dict is not None:
        plt.title(lab_dict['title'])
        plt.ylabel(lab_dict['ylab'])
        plt.xlabel(lab_dict['xlab'])
    plt.legend()

    # Show or save
    if show:
        plt.show()
    save_figure(save_name, show)


def plot_train_history(hist, name: str = None, val: bool = True) -> None:
    """Visualizes the training of a keras model.

    Plot training & validation loss values of 
    a history object returned by keras.Model.fit().

    Args:
        hist: The history object returned by the fit method.
        name: The path of the plot file if saving it.
        val: Whether to include the validation curve.
    """
    plt.subplots()
    plt.plot(hist.history['loss'])
    if val:
        plt.plot(hist.history['val_loss'])
    plt.yscale('log')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    leg = ['Training']
    if val:
        leg += ['Validation']
    plt.legend(leg, loc='upper right')
    if name is not None:
        save_figure(name, False)
    else:
        plt.show()


def plot_rewards(hist, name: str = None) -> None:
    """Plots the rewards from RL training.

    Args:
        hist: The history object.
        name: The path where to save the figure.
    """
    plt.subplots()
    plt.plot(hist.history['episode_reward'])
    plt.title('Model reward')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    leg = ['Training']
    plt.legend(leg, loc='upper left')
    if name is not None:
        save_figure(name, False)
    else:
        plt.show()


def plot_simple_ts(y_list, title=None, name=None):
    """
    Plots the given aligned time series.
    """
    n = len(y_list[0])
    x = range(n)

    plt.subplots()
    for y in y_list:
        plt.plot(x, y)

    if title is not None:
        plt.title(title)
    if name is not None:
        save_figure(name, False)
    else:
        plt.show()


def stack_compare_plot(stack_y, y_compare, title=None, name=None):
    """
    Plots the given aligned time series.
    """
    n = len(y_compare[0])
    x = range(n)
    ys = [stack_y[:, i] for i in range(stack_y.shape[1])]

    fig, ax = plt.subplots()
    ax.stackplot(x, *ys)
    for y in y_compare:
        ax.plot(x, y)

    if title is not None:
        plt.title(title)
    if name is not None:
        save_figure(name, False)
    else:
        plt.show()


def plot_residuals_acf(residuals: np.ndarray,
                       name: str = None,
                       lags: int = 50,
                       partial: bool = False) -> None:
    """Plots the ACF of the residuals given.

    If `name` is None, the plot will be shown, else
    it will be saved to the path specified by `name`.

    Args:
        residuals: The array with the residuals for one series.
        name: The filename for saving the plot.
        lags: The number of lags to consider.
        partial: Whether to use the partial ACF.

    Returns:
        None

    Raises:
        ValueError: If residuals do not have the right shape.
    """
    if len(residuals.shape) != 1:
        yeet("Residuals needs to be a vector!")

    # Initialize and plot data
    plt.subplots()
    plot_fun = plot_pacf if partial else plot_acf
    plot_fun(residuals, lags=lags)

    # Set title and labels
    title = "Autocorrelation of Residuals"
    if partial:
        title = "Partial " + title
    plt.title(title)
    plt.ylabel("Correlation")
    plt.xlabel("Lag")

    # Save or show
    if name is not None:
        save_figure(name, False)
    else:
        plt.show()


def _setup_axis(ax, base_title: str, desc: str, title: bool = True):
    """Helper function for `plot_env_evaluation`.

    Adds label and title to axis."""
    t, u = split_desc_units(desc)
    ax.set_ylabel(u)
    if title:
        ax.set_title(base_title + ": " + t)


def plot_env_evaluation(actions: np.ndarray,
                        states: np.ndarray,
                        rewards: np.ndarray,
                        ds,
                        agent_names: Sequence[str],
                        save_path: str = None,
                        extra_actions: np.ndarray = None,
                        series_mask: np.ndarray = None,
                        title_ext: str = None,
                        show_rewards: bool = True) -> None:
    """Plots the evaluation of multiple agents on an environment.

    Only for one specific initial condition.
    """
    assert len(agent_names) == actions.shape[0], "Incorrect number of names!"
    assert rewards.shape[0] == actions.shape[0], "Incorrect shapes!"
    if series_mask is not None:
        check_shape(series_mask, (-1,))
        assert len(series_mask) < states.shape[2]

    # Check fallback actions
    plot_extra = extra_actions is not None

    # Extract shapes
    n_agents, episode_len, n_feats = states.shape
    if series_mask is not None:
        n_feats = len(series_mask)
    n_actions = actions.shape[-1]
    tot_n_plots = n_actions + n_feats + show_rewards + plot_extra * n_actions

    # We'll use a separate GridSpecs for controls, states and rewards
    fig = plt.figure()
    h_s = 0.6
    t = 0.95
    gs_con = plt.GridSpec(tot_n_plots, 1, hspace=h_s, top=t, bottom=0.0, figure=fig)
    gs_state = plt.GridSpec(tot_n_plots, 1, hspace=h_s, top=t, bottom=0.0, figure=fig)
    gs_rew = plt.GridSpec(tot_n_plots, 1, hspace=h_s, top=t, bottom=0.0, figure=fig)
    gs_con_fb = plt.GridSpec(tot_n_plots, 1, hspace=h_s, top=t, bottom=0.0, figure=fig)

    # Define axes
    n_act_plots = n_actions * (1 + plot_extra)
    rew_ax = fig.add_subplot(gs_rew[-1, :])
    con_axs = [fig.add_subplot(gs_con[i, :], sharex=rew_ax) for i in range(n_actions)]
    state_axs = [fig.add_subplot(gs_state[i, :], sharex=rew_ax) for i in range(n_act_plots, tot_n_plots - 1)]
    con_fb_axs = [fig.add_subplot(gs_con_fb[i, :], sharex=rew_ax) for i in range(n_actions, n_act_plots)]
    assert plot_extra or con_fb_axs == [], "Something is wrong!"

    # Find legends
    n_tot_vars = ds.d
    c_inds = ds.c_inds
    control_descs = [ds.descriptions[c] for c in c_inds]
    state_descs = [ds.descriptions[c] for c in range(n_tot_vars) if c not in c_inds]

    # Reduce series
    if series_mask is not None:
        state_descs = [state_descs[i] for i in series_mask]
        states = states[:, :, series_mask]

    # Set titles
    if show_rewards:
        rew_ax.set_title("Rewards")
    for k in range(n_actions):
        _setup_axis(con_axs[k], "Original Control Inputs", control_descs[k])
        if plot_extra:
            _setup_axis(con_fb_axs[k], "Constrained Control Inputs", control_descs[k])
    for k in range(n_feats):
        _setup_axis(state_axs[k], "States", state_descs[k])

    # Plot all the things!
    x = range(len(rewards[0]))
    for k in range(n_agents):
        # Plot actions
        for i in range(n_actions):
            _plot_helper(x, actions[k, :, i], m_col=clr_map[k],
                         label=agent_names[k], ax=con_axs[i], steps=True)
            if plot_extra:
                _plot_helper(x, extra_actions[k, :, i], m_col=clr_map[k],
                             label=agent_names[k], ax=con_fb_axs[i], steps=True)
        # Plot states
        for i in range(n_feats):
            _plot_helper(x, states[k, :, i], m_col=clr_map[k],
                         label=agent_names[k], ax=state_axs[i])

        # Plot reward
        if show_rewards:
            _plot_helper(x, rewards[k, :], m_col=clr_map[k],
                         label=agent_names[k], ax=rew_ax)

    # Set time
    last_c_ax = con_fb_axs[-1] if plot_extra else con_axs[-1]
    last_c_ax.set_xlabel(f"Timestep [{ds.dt}min]")
    state_axs[-1].set_xlabel(f"Timestep [{ds.dt}min]")

    # Add legends
    con_axs[0].legend()
    state_axs[0].legend()

    if show_rewards:
        rew_ax.set_xlabel(f"Timestep [{ds.dt}min]")
        rew_ax.legend()

    # Super title
    sup_t = 'Visual Analysis. '
    if title_ext is not None:
        sup_t += title_ext
    con_axs[0].annotate('Visual Analysis. ', (0.5, 0.975),
                        xycoords='figure fraction', ha='center',
                        fontsize=24)
    # plt.suptitle('Main title')
    # plt.subplots_adjust(top=1.85)

    # Save
    s = (16, tot_n_plots * 1.8)
    if save_path is not None:
        save_figure(save_path, size=s)


def plot_reward_details(a_list: List,
                        rewards: np.ndarray,
                        path_name: str,
                        rew_descs: List[str],
                        dt: int = 15,
                        n_eval_steps: int = 2000,
                        title_ext: str = None,
                        scale_tot_rew: bool = True) -> None:
    """Creates a bar plot with the different rewards of the different agents.

    Args:
        a_list: List with agents.
        rewards: Array with all rewards for all agents.
        path_name: Path of the plot to save.
        rew_descs: Descriptions of the parts of the reward.
        dt: Number of minutes in a timestep.
        n_eval_steps: Number of evaluation steps.
        title_ext: Extension to add to the title.
        scale_tot_rew: Whether to scale the total reward to a nice range.
    """
    n_rewards = rewards.shape[-1]
    assert n_rewards == len(rew_descs) + 1, "No correct number of descriptions!"
    labels = [a.get_short_name() for a in a_list]
    mean_rewards = np.mean(rewards, axis=1)
    all_descs = ["Total Reward"] + rew_descs

    title = f"Mean rewards per hour for {n_eval_steps} steps. "
    if title_ext is not None:
        title += title_ext
    fac = 60 / dt
    mean_rewards *= fac

    if scale_tot_rew:
        # Scale the maximum reward to the same magnitude of any of the other parts
        max_tot = np.max(np.abs(mean_rewards[:, 0]))
        max_not_tot = np.max(np.abs(mean_rewards[:, 1:]))
        mean_rewards[:, 0] *= max_not_tot / max_tot

    fig, ax = plt.subplots()

    # Define the bars
    x = np.arange(len(labels))  # the label locations
    width = 0.8 / n_rewards  # the width of the bars
    offs = (n_rewards - 1) * width / 2
    rects = [ax.bar(x - offs + i * width,
                    mean_rewards[:, i],
                    width,
                    label=all_descs[i])
             for i in range(n_rewards)]

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Label the rectangles with the values.
    def auto_label(rect_list):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rect_list:
            height = rect.get_height()
            ax.annotate('{:.4g}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    for r in rects:
        auto_label(r)

    # Set layout and save
    fig.tight_layout()
    save_figure(save_name=path_name)


def _load_all_model_data(model_list: List,
                         parts: List[str],
                         metric_list: List[str],
                         series_mask=None) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function for `plot_performance_table` to load the performance data of all models.

    Args:
        model_list:
        parts:
        metric_list:

    Returns:
        The loaded data.
    """
    # Get sizes and check them
    n_models, n_parts, n_metrics = len(model_list), len(parts), len(metric_list)
    assert n_models > 0 and n_parts > 0 and n_metrics > 0, "Nothing to do here!"
    dt = model_list[0].data.dt

    # Load data of all models
    data_list = []
    inds = None
    for ct, m in enumerate(model_list):

        # # Define function generating path.
        # def gen_fun(name):
        #     return m.get_plt_path(name)

        # Load data of current model and check shape
        data, inds_curr = load_performance(m.get_plt_path, parts, dt, n_metrics)
        if ct == 0:
            inds = inds_curr
        else:
            assert np.array_equal(inds, inds_curr)
            check_shape(data_list[0], data.shape)
        data_list += [data]

    # Check shape and stack
    check_shape(data_list[0], (n_parts, -1, n_metrics, len(inds)))
    data_array = np.stack(data_list)

    # Reduce data
    if series_mask is not None:
        d = model_list[0].data
        prep_mask = d.to_prepared(series_mask)
        data_array = data_array[:, :, prep_mask]

    return data_array, inds


def _get_descs(model_list: List, remove_units: bool = True,
               series_mask=None, short_mod_names: List = None):
    # Edit series descriptions
    d = model_list[0].data
    all_desc = np.ones((d.d,), dtype=np.bool)
    all_desc[d.c_inds] = False
    if series_mask is None:
        series_descs = d.descriptions[all_desc]
        rel_scaling = d.scaling[all_desc]
    else:
        series_descs = d.descriptions[series_mask]
        rel_scaling = d.scaling[series_mask]
    if remove_units:
        series_descs = [sd.split("[")[0] for sd in series_descs]

    # Edit model names
    mod_names = [m.name for m in model_list]
    if short_mod_names is not None:
        assert len(short_mod_names) == len(mod_names), "Incorrect number of model names!"
        mod_names = short_mod_names

    return series_descs, mod_names, rel_scaling


def plot_performance_table(model_list: List, parts: List[str], metric_list: List[str],
                           name: str = "Test", short_mod_names: List = None,
                           remove_units: bool = True,
                           series_mask=None) -> None:
    # Define the ordering of the rows.
    order = (0, 1, 4, 2, 3)

    # Prepare the labels
    series_descs, mod_names, _ = _get_descs(model_list, remove_units, series_mask, short_mod_names)

    # Construct the path of the plot
    plot_path = os.path.join(EVAL_MODEL_PLOT_DIR, name)

    # Load data
    data_array, inds = _load_all_model_data(model_list, parts, metric_list, series_mask)

    # Handle indices and shapes
    sec_order = np.argsort(order)
    last_ind = order[-1]
    dat_shape = data_array.shape
    n_dim = len(dat_shape)
    n_models, n_parts, n_series, n_metrics, n_steps = dat_shape
    tot_s = tot_size(dat_shape)
    n_last = dat_shape[last_ind]
    tot_n_rows = tot_s // n_last

    # Compute indices to convert 5d array to 2d
    n_sub = []
    curr_sz = tot_n_rows
    for k in range(n_dim - 1):
        curr_sh = dat_shape[order[k]]
        curr_sz //= curr_sh
        n_sub += [(curr_sz, curr_sh)]

    # Initialize empty string array
    table_array = np.empty((tot_n_rows, n_dim - 1 + n_last), dtype="<U50")

    for k in range(tot_n_rows):

        all_inds = [(k // n) % m for n, m in n_sub]

        # Fill header cols
        table_array[k, 0] = mod_names[all_inds[0]]
        table_array[k, 1] = parts[all_inds[1]]
        table_array[k, 2] = "{}".format(int(inds[all_inds[2]]))
        table_array[k, 3] = series_descs[all_inds[3]]

        for i in range(n_last):
            ind_list = np.array(all_inds + [i])[sec_order]
            table_array[k, 4 + i] = "{:.4g}".format(data_array[tuple(ind_list)])

    # Define column labels
    col_labels = ["Model", "Set", "Steps", "Series"]
    for k in range(n_last):
        col_labels += [metric_list[k]]

    base_w = 0.07
    col_w = [base_w for _ in col_labels]
    col_w[0] = col_w[3] = base_w * 4

    fig, ax = plt.subplots()

    # hide axes
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=table_array, colLabels=col_labels,
             loc='center', colWidths=col_w)

    fig.tight_layout()

    save_figure(plot_path)


def _trf_desc_units(curr_desc: str, m: ErrMetric) -> str:
    split_d = curr_desc.split("[")
    assert len(split_d) == 2, "Invalid description!"
    base, rest = split_d
    unit = rest.split("]")[0]
    trf_unit = m.unit_trf(unit)
    return f"{base}[{trf_unit}]"


def plot_performance_graph(model_list: List, parts: List[str],
                           metric_list: Sequence[ErrMetric],
                           name: str = "Test",
                           short_mod_names: List = None,
                           remove_units: bool = True,
                           series_mask=None,
                           scale_back: bool = False) -> None:
    metric_names = [m.name for m in metric_list]

    # Prepare the labels
    series_descs, mod_names, scaling = _get_descs(model_list, remove_units, series_mask, short_mod_names)

    # Load data
    data_array, inds = _load_all_model_data(model_list, parts, metric_names, series_mask)
    n_models, n_parts, n_series, n_metrics, n_steps = data_array.shape

    for model_ind, m_name in enumerate(mod_names):

        share_y = False
        dt = model_list[0].data.dt

        if n_parts > len(joint_styles):
            warnings.warn("Duplicate plot styles!")

        ax1 = None
        for ct_m, m in enumerate(metric_list):
            subplot_ind = 311 + ct_m
            ax1 = plt.subplot(subplot_ind, sharex=ax1, sharey=ax1 if share_y else None)

            # Set ticks
            tick_label = [mins_to_str(dt * int(i)) + f"\n{int(i)} Steps" for i in inds]
            plt.xticks(inds, tick_label)
            plt.ylabel(m.name)

            # Plot all series
            for set_id in range(n_parts):
                for series_id in range(n_series):
                    # Get labels and errors
                    s_desc = series_descs[series_id]
                    if scale_back:
                        s_desc = _trf_desc_units(s_desc, m)
                    lab = parts[set_id] + ": " + s_desc
                    si = data_array[model_ind, set_id, series_id, ct_m]

                    # Scale the errors
                    m_and_sd = scaling[series_id]
                    if scale_back:
                        si = m.scaling_fun(si, m_and_sd[1])

                    # Plot
                    plt.plot(inds, si, joint_styles[set_id],
                             c=clr_map[series_id], label=lab)

            # Add title, legend and x-label
            if ct_m == 0 or scale_back:
                if ct_m == 0:
                    plt.title(mod_names[model_ind])
                plt.legend()
            # if ct_m == len(metric_names) - 1:
            #     plt.xlabel(f"Steps [{mins_to_str(dt)}]")

        # Construct the path of the plot
        plot_path = os.path.join(EVAL_MODEL_PLOT_DIR, f"{name}_{m_name}")
        save_figure(plot_path)
