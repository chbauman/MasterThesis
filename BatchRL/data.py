import os
from typing import Dict

import scipy
import pickle
import warnings

import numpy as np
import pandas as pd

from ast import literal_eval
from datetime import datetime

from visualize import plot_time_series, plot_ip_time_series, \
    plot_single_ip_ts, plot_multiple_ip_ts, \
    plot_all, plot_single, preprocess_plot_path, \
    plot_multiple_time_series, plot_dataset, plot_dir, \
    plot_simple_ts, stack_compare_plot
from restclient import DataStruct, save_dir
from util import *

# Data directories
dataset_data_path = os.path.join(save_dir, "Datasets")

#######################################################################################################
# NEST Data

# UMAR Room Data
Room272Data = DataStruct(id_list=[42150280,
                                  42150288,
                                  42150289,
                                  42150290,
                                  42150291,
                                  42150292,
                                  42150293,
                                  42150294,
                                  42150295,
                                  42150483,
                                  42150484,
                                  42150284,
                                  42150270],
                         name="UMAR_Room272",
                         start_date='2017-01-01',
                         end_date='2019-12-31')

Room274Data = DataStruct(id_list=[42150281,
                                  42150312,
                                  42150313,
                                  42150314,
                                  42150315,
                                  42150316,
                                  42150317,
                                  42150318,
                                  42150319,
                                  42150491,
                                  42150492,
                                  42150287,
                                  42150274],
                         name="UMAR_Room274",
                         start_date='2017-01-01',
                         end_date='2019-12-31')

# DFAB Data
Room4BlueData = DataStruct(id_list=[421110054,  # Temp
                                    421110023,  # Valves
                                    421110024,
                                    421110029,
                                    421110209  # Blinds
                                    ],
                           name="DFAB_Room41",
                           start_date='2017-01-01',
                           end_date='2019-12-31')

Room5BlueData = DataStruct(id_list=[421110072,  # Temp
                                    421110038,  # Valves
                                    421110043,
                                    421110044,
                                    421110219  # Blinds
                                    ],
                           name="DFAB_Room51",
                           start_date='2017-01-01',
                           end_date='2019-12-31')

Room4RedData = DataStruct(id_list=[421110066,  # Temp
                                   421110026,  # Valves
                                   421110027,
                                   421110028,
                                   ],
                          name="DFAB_Room43",
                          start_date='2017-01-01',
                          end_date='2019-12-31')

Room5RedData = DataStruct(id_list=[421110084,  # Temp
                                   421110039,  # Valves
                                   421110040,
                                   421110041,
                                   ],
                          name="DFAB_Room53",
                          start_date='2017-01-01',
                          end_date='2019-12-31')

DFAB_AddData = DataStruct(id_list=[421100168,  # Vorlauf Temp
                                   421100170,  # Rücklauf Temp
                                   421100174,  # Tot volume flow
                                   421100163,  # Pump running
                                   421110169,  # Speed of other pump
                                   421100070,  # Volume flow through other part
                                   ],
                          name="DFAB_Extra",
                          start_date='2017-01-01',
                          end_date='2019-12-31')

DFAB_AllValves = DataStruct(id_list=[421110008,  # First Floor
                                     421110009,
                                     421110010,
                                     421110011,
                                     421110012,
                                     421110013,
                                     421110014,
                                     421110023,  # Second Floor,
                                     421110024,
                                     421110025,
                                     421110026,
                                     421110027,
                                     421110028,
                                     421110029,
                                     421110038,  # Third Floor
                                     421110039,
                                     421110040,
                                     421110041,
                                     421110042,
                                     421110043,
                                     421110044,
                                     ],
                            name="DFAB_Valves",
                            start_date='2017-01-01',
                            end_date='2019-12-31')

rooms = [Room4BlueData, Room5BlueData, Room4RedData, Room5RedData]

# Weather Data
WeatherData = DataStruct(id_list=[3200000,
                                  3200002,
                                  3200008],
                         name="Weather",
                         start_date='2019-01-01',
                         end_date='2019-12-31')

# Battery Data
BatteryData = DataStruct(id_list=[40200000,
                                  40200001,
                                  40200002,
                                  40200003,
                                  40200004,
                                  40200005,
                                  40200006,
                                  40200007,
                                  40200008,
                                  40200009,
                                  40200010,
                                  40200011,
                                  40200012,
                                  40200013,
                                  40200014,
                                  40200015,
                                  40200016,
                                  40200017,
                                  40200018,
                                  40200019,
                                  40200087,
                                  40200088,
                                  40200089,
                                  40200090,
                                  40200098,
                                  40200099,
                                  40200102,
                                  40200103,
                                  40200104,
                                  40200105,
                                  40200106,
                                  40200107,
                                  40200108],
                         name="Battery",
                         start_date='2018-01-01',
                         end_date='2019-12-31')


#######################################################################################################
# Time Series Processing

def extract_date_interval(dates, values, d1, d2):
    """
    Extracts all data that lies within the dates d1 and d2.
    Returns None if there are no such points.
    """
    if d1 >= d2:
        raise ValueError("Invalid date interval passed.")
    mask = np.logical_and(dates < d2, dates > d1)
    if np.sum(mask) == 0:
        print("No values in provided date interval.")
        return None
    dates_new = dates[mask]
    values_new = values[mask]
    dt_init_new = dates_new[0]
    return dates_new, values_new, dt_init_new


def analyze_data(dat):
    """
    Analyzes the provided data.
    DEPRECATED!
    """

    values, dates = dat
    n_data_p = len(values)

    tot_time_span = dates[-1] - dates[0]
    print("Data ranges from ", dates[0], "to", dates[-1])

    t_diffs = dates[1:] - dates[:-1]
    max_t_diff = np.max(t_diffs)
    mean_t_diff = np.mean(t_diffs)
    print("Largest gap:", np.timedelta64(max_t_diff, 'D'), "or", np.timedelta64(max_t_diff, 'h'))
    print("Mean gap:", np.timedelta64(mean_t_diff, 'm'), "or", np.timedelta64(mean_t_diff, 's'))

    print("Positive differences:", np.all(t_diffs > np.timedelta64(0, 'ns')))
    return


def clean_data(dat, rem_values=(), n_cons_least: int = 60, const_excepts=()):
    """
    Removes all values with a specified value 'rem_val'
    and removes all sequences where there are at 
    least 'n_cons_least' consecutive
    values having the exact same value. If the value 
    occurring multiple times is in 'const_excepts' then
    it is not removed.
    """

    values, dates = dat
    tot_dat = values.shape[0]

    # Make copy
    new_values = np.copy(values)
    new_dates = np.copy(dates)

    # Initialize
    prev_val = np.nan
    count = 0
    num_occ = 1
    con_streak = False

    # Add cleaned values and dates
    for (v, d) in zip(values, dates):

        if v not in rem_values:

            # Monitor how many times the same value occurred
            if v == prev_val and v not in const_excepts:

                num_occ += 1
                if num_occ == n_cons_least:
                    con_streak = True
                    count -= n_cons_least - 1
            else:
                con_streak = False
                num_occ = 1

            # Add value if it has not occurred too many times
            if not con_streak:
                new_values[count] = v
                new_dates[count] = d
                count += 1
                prev_val = v

        else:
            # Reset streak
            con_streak = False
            num_occ = 1

    # Return clean data
    print(tot_dat - count, "data points removed.")
    return [new_values[:count], new_dates[:count]]


def remove_out_interval(dat: Tuple, interval: Tuple[Num, Num] = (0.0, 100.0)) -> None:
    """
    Removes values that do not lie within the interval.

    :param dat: Raw time series tuple (values, dates)
    :param interval: Interval where the values have to lie within.
    :return: None
    """
    values, dates = dat
    values[values > interval[1]] = np.nan
    values[values < interval[0]] = np.nan


def clip_to_interval(dat: Tuple, interval: Sequence = (0.0, 100.0)) -> None:
    """
    Clips the values of the time_series that are
    out of the interval to lie within.

    :param dat: Raw time series tuple (values, dates)
    :param interval: Interval where the values will lie within.
    :return: None
    """
    values, dates = dat
    values[values > interval[1]] = interval[1]
    values[values < interval[0]] = interval[0]


def interpolate_time_series(dat, dt_mins, lin_ip=False):
    """
    Interpolates the given time series
    to produce another one with equidistant timesteps
    and NaNs if values are missing.
    """

    # Unpack
    values, dates = dat

    # Datetime of first and last data point
    start_dt = floor_datetime_to_min(dates[0], dt_mins)
    end_dt = floor_datetime_to_min(dates[-1], dt_mins)
    interval = np.timedelta64(dt_mins, 'm')
    n_ts = int((end_dt - start_dt) / interval + 1)
    print(n_ts, "Timesteps")

    # Initialize
    new_values = np.empty((n_ts,), dtype=np.float32)
    new_values.fill(np.nan)
    count = 0
    last_dt = dates[0]
    last_val = values[0]
    curr_val = (last_dt - start_dt) / interval * last_val
    curr_dt = dates[0]
    v = 0.0

    # Loop over data points
    for ct, v in enumerate(values[1:]):
        curr_dt = dates[ct + 1]
        curr_upper_lim = start_dt + (count + 1) * interval
        if curr_dt >= curr_upper_lim:
            if curr_dt <= curr_upper_lim + interval:
                # Next datetime in next interval
                curr_val += (curr_upper_lim - last_dt) / interval * v
                if not lin_ip:
                    new_values[count] = curr_val
                else:
                    new_values[count] = last_val + (v - last_val) * (curr_upper_lim - last_dt) / (curr_dt - last_dt)
                count += 1
                curr_val = (curr_dt - curr_upper_lim) / interval * v
            else:
                # Data missing!                
                curr_val += (curr_upper_lim - last_dt) / interval * last_val
                if not lin_ip:
                    new_values[count] = curr_val
                else:
                    new_values[count] = last_val
                count += 1
                n_data_missing = int((curr_dt - curr_upper_lim) / interval)
                print("Missing", n_data_missing, "data points :(")
                for k in range(n_data_missing):
                    new_values[count] = np.nan
                    count += 1
                dt_start_new_iv = curr_dt - curr_upper_lim - n_data_missing * interval
                curr_val = dt_start_new_iv / interval * v

        else:
            # Next datetime still in same interval
            curr_val += (curr_dt - last_dt) / interval * v

        # Update
        last_dt = curr_dt
        last_val = v

    # Add last one
    curr_val += (end_dt + interval - curr_dt) / interval * v
    new_values[count] = curr_val

    # Return
    return [new_values, start_dt]


def add_col(full_dat_array, data, dt_init, dt_init_new, col_ind, dt_mins=15):
    """
    Add time series as column to data array at the right index.
    If the second time series exceeds the datetime range of the
    first one it is cut to fit the first one. If it is too short
    the missing values are filled with NaNs.
    """

    n_data = full_dat_array.shape[0]
    n_data_new = data.shape[0]

    # Compute indices
    interval = np.timedelta64(dt_mins, 'm')
    offset_before = int(np.round((dt_init_new - dt_init) / interval))
    offset_after = n_data_new - n_data + offset_before
    dat_inds = [np.maximum(0, offset_before), n_data + np.minimum(0, offset_after)]
    new_inds = [np.maximum(0, -offset_before), n_data_new + np.minimum(0, -offset_after)]

    # Add to column
    full_dat_array[dat_inds[0]:dat_inds[1], col_ind] = data[new_inds[0]:new_inds[1]]
    return


def add_time(all_data, dt_init1, col_ind=0, dt_mins=15):
    """
    Adds the time as indices to the data,
    periodic with period one day.
    """

    n_data = all_data.shape[0]
    interval = np.timedelta64(dt_mins, 'm')
    n_ts_per_day = 24 * 60 / dt_mins
    t_temp_round = np.datetime64(dt_init1, 'D')
    start_t = (dt_init1 - t_temp_round) / interval
    for k in range(n_data):
        all_data[k, col_ind] = (start_t + k) % n_ts_per_day
    return


def fill_holes_linear_interpolate(time_series: np.ndarray, max_width: int = 1) -> None:
    """
    Fills the holes of a equi-spaced time series
    with a width up to 'max_width'
    by linearly interpolating between the previous and
    next data point.

    :param time_series: The time series that is processed.
    :param max_width: Sequences of at most that many nans are removed by interpolation.
    :return: None
    """

    n = time_series.shape[0]
    nan_bool = np.isnan(time_series)

    # Return if there are no NaNs
    if np.sum(nan_bool) == 0:
        return

    # Neglect NaNs at beginning and end
    non_nans = np.where(nan_bool == False)[0]
    nan_bool[:non_nans[0]] = False
    nan_bool[non_nans[-1]:] = False

    # Find all indices with NaNs
    all_nans = np.argwhere(nan_bool)

    # Initialize iterators
    ind_ind = 0
    s_ind = all_nans[ind_ind][0]

    while ind_ind < all_nans.shape[0]:
        s_ind = all_nans[ind_ind][0]
        streak_len = np.where(nan_bool[s_ind:] == False)[0][0]
        if streak_len <= max_width:

            # Interpolate values
            low_val = time_series[s_ind - 1]
            high_val = time_series[s_ind + streak_len]
            for k in range(streak_len):
                curr_val = low_val * (k + 1) + high_val * (streak_len - k)
                curr_val /= streak_len + 1
                time_series[s_ind + k] = curr_val

        ind_ind += streak_len
    return


def remove_outliers(time_series: np.ndarray, grad_clip: Num = 100.0,
                    clip_int: Optional[Sequence] = None) -> np.ndarray:
    """
    Removes data points that lie outside
    the specified interval 'clip_int' and ones
    with a gradient larger than grad_clip.

    :param time_series: The time series to process.
    :param grad_clip: The maximum gradient magnitude.
    :param clip_int: The interval where the data has to lie within.
    :return: A copy of the series without the outlliers.
    """

    # Helper functions
    def grad_fd(x1, x2):
        if x2 is None or x1 is None:
            return np.nan
        if np.isnan(x1) or np.isnan(x2):
            return np.nan
        return x2 - x1

    def is_outlier(x, x_tm1, x_tp1=None):
        g1 = grad_fd(x_tm1, x)
        g2 = grad_fd(x, x_tp1)
        if np.isnan(g1):
            return True if np.absolute(g2) > 1.5 * grad_clip else False
        if np.isnan(g2):
            return True if np.absolute(g1) > 1.5 * grad_clip else False
        rej = np.absolute(g1) > grad_clip and np.absolute(g2) > grad_clip
        rej = rej and g1 * g2 < 0
        return rej

    def reject_outliers(x, x_tm1, x_tp1=None):
        if is_outlier(x, x_tm1, x_tp1):
            return np.nan
        return x

    # First and last values
    time_series[0] = reject_outliers(time_series[0], time_series[1])
    time_series[-1] = reject_outliers(time_series[-1], time_series[-2])

    # Iterate
    for ct, el in enumerate(time_series[1:-1]):
        if el != np.nan:
            # Remove large gradient outliers
            time_series[ct + 1] = reject_outliers(el,
                                                  time_series[ct + 2],
                                                  time_series[ct])

            # Clip to interval
            if clip_int is not None:
                if el < clip_int[0] or el > clip_int[1]:
                    time_series[ct + 1] = np.nan
    return


def gaussian_filter_ignoring_nans(time_series: np.ndarray, sigma=2.0) -> np.ndarray:
    """
    Applies 1-dimensional Gaussian Filtering ignoring
    occurrences of NaNs. From:
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python

    :param time_series: The time series to process.
    :param sigma: Gaussian filter standard deviation.
    :return: Filtered time series.
    """

    v = time_series.copy()
    v[np.isnan(time_series)] = 0
    vv = scipy.ndimage.filters.gaussian_filter1d(v, sigma=sigma)

    w = 0 * time_series.copy() + 1
    w[np.isnan(time_series)] = 0
    ww = scipy.ndimage.filters.gaussian_filter1d(w, sigma=sigma)

    z = vv / ww
    z[np.isnan(time_series)] = np.nan
    return z


def pipeline_preps(orig_dat,
                   dt_mins,
                   all_data=None,
                   *,
                   dt_init=None,
                   row_ind=None,
                   clean_args=None,
                   clip_to_int_args=None,
                   remove_out_int_args=None,
                   rem_out_args=None,
                   hole_fill_args=None,
                   n_tot_cols=None,
                   gauss_sigma=None,
                   lin_ip=False):
    """
    Applies all the specified pre-processing to the
    given data.
    """
    modified_data = orig_dat

    # Clean Data
    if clean_args is not None:
        for k in clean_args:
            modified_data = clean_data(orig_dat, *k)

            # Clip to interval
    if remove_out_int_args is not None:
        remove_out_interval(modified_data, remove_out_int_args)

    # Clip to interval
    if clip_to_int_args is not None:
        clip_to_interval(modified_data, clip_to_int_args)

    # Interpolate / Subsample
    [modified_data, dt_init_new] = interpolate_time_series(modified_data, dt_mins, lin_ip=lin_ip)

    # Remove Outliers
    if rem_out_args is not None:
        remove_outliers(modified_data, *rem_out_args)

    # Fill holes
    if hole_fill_args is not None:
        fill_holes_linear_interpolate(modified_data, hole_fill_args)

    # Gaussian Filtering
    if gauss_sigma is not None:
        modified_data = gaussian_filter_ignoring_nans(modified_data, gauss_sigma)

    if all_data is not None:
        if dt_init is None or row_ind is None:
            raise ValueError("Need to provide the initial time of the first series and the column index!")

        # Add to rest of data
        add_col(all_data, modified_data, dt_init, dt_init_new, row_ind, dt_mins)
    else:
        if n_tot_cols is None:
            print("Need to know the total number of columns!")
            raise ValueError("Need to know the total number of columns!")

        # Initialize np array for compact storage
        n_data = modified_data.shape[0]
        all_data = np.empty((n_data, n_tot_cols), dtype=np.float32)
        all_data.fill(np.nan)
        all_data[:, 0] = modified_data

    return all_data, dt_init_new


def add_and_save_plot_series(data, m, curr_all_dat, ind, dt_mins, dt_init, plot_name, base_plot_dir,
                             title="",
                             pipeline_kwargs={},
                             n_cols=None,
                             col_ind=None):
    """
    Adds the series with index 'ind' to curr_all_dat
    and plots the series before and after processing
    with the pipeline.
    ind: index of series in raw data
    col_ind: column index of series in processed data 
    """

    dt_init_new = np.copy(dt_init)
    all_dat = curr_all_dat
    if col_ind is None:
        col_ind = ind
    elif curr_all_dat is None and col_ind != 0:
        raise NotImplementedError("col_ind cannot be chosen if curr_all_dat is None!")

    # Process data
    if curr_all_dat is None:
        col_ind = 0
        if n_cols is None:
            raise ValueError("Need to specify n_cols if data is None.")
        all_dat, dt_init_new = pipeline_preps(copy_arr_list(data[ind]),
                                              dt_mins,
                                              n_tot_cols=n_cols,
                                              **pipeline_kwargs)
        add_dt_and_tinit(m, dt_mins, dt_init_new)
    else:
        if dt_init is None:
            raise ValueError("dt_init cannot be None!")
        all_dat, _ = pipeline_preps(copy_arr_list(data[ind]),
                                    dt_mins,
                                    all_data=curr_all_dat,
                                    dt_init=dt_init,
                                    row_ind=col_ind,
                                    **pipeline_kwargs)

    if np.isnan(np.nanmax(all_dat[:, col_ind])):
        raise ValueError("Something went very fucking wrong!!")

    # Plot before data
    plot_file_name = base_plot_dir + "_" + plot_name + "_Raw"
    plot_time_series(data[ind][1], data[ind][0], m=m[ind], show=False, save_name=plot_file_name)

    # Plot after data
    plot_file_name2 = base_plot_dir + "_" + plot_name
    plot_single(np.copy(all_dat[:, col_ind]),
                m[ind],
                use_time=True,
                show=False,
                title_and_ylab=[title, m[ind]['unit']],
                save_name=plot_file_name2)

    return all_dat, dt_init_new


def save_ds_from_raw(all_data: np.ndarray, m_out: List[Dict], name: str,
                     c_inds: np.ndarray = None,
                     p_inds: np.ndarray = None,
                     standardize_data: bool = False):
    """
    Creates a dataset from the raw input data.

    :param all_data: 2D numpy data array
    :param m_out: List with metadata dictionaries.
    :param name: Name of the dataset.
    :param c_inds: Control indices.
    :param p_inds: Prediction indices.
    :param standardize_data: Whether to standardize the data.
    :return: Dataset
    """
    if c_inds is None:
        c_inds = no_inds
    if p_inds is None:
        p_inds = no_inds
    if standardize_data:
        all_data, m_out = standardize(all_data, m_out)
    dataset = Dataset.fromRaw(all_data, m_out, name, c_inds=c_inds, p_inds=p_inds)
    dataset.save()
    return dataset


def get_from_data_struct(dat_struct, base_plot_dir, dt_mins, new_name, ind_list, prep_arg_list,
                         desc_list=None,
                         c_inds=None,
                         p_inds=None,
                         standardize_data=False):
    """
    Extracts the specified series and applies
    pre-processing steps to all of them and puts them into a
    Dataset.
    """

    # Get name
    name = dat_struct.name
    if new_name is None:
        new_name = name

    # Try loading data
    try:
        loaded = Dataset.loadDataset(name)
        return loaded
    except FileNotFoundError:
        data, m = dat_struct.getData()
        n_cols = len(data)

        # Check arguments
        n_inds = len(ind_list)
        n_preps = len(prep_arg_list)
        if n_inds > n_cols or n_preps > n_inds:
            raise ValueError("Too long lists!")
        if desc_list is not None:
            if len(desc_list) != n_inds:
                raise ValueError("List of description does not have correct length!!")

        all_data = None
        dt_init = None
        m_out = []

        # Sets
        for ct, i in enumerate(ind_list):
            n_cs = n_inds if ct == 0 else None
            title = clean_desc(m[i]['description'])
            title = title if desc_list is None else desc_list[ct]
            added_cols = m[i]['additionalColumns']
            plot_name = ""
            plot_file_name = os.path.join(base_plot_dir, added_cols['AKS Code'])
            all_data, dt_init = add_and_save_plot_series(data, m, all_data, i, dt_mins, dt_init, plot_name,
                                                         plot_file_name, title,
                                                         n_cols=n_cs,
                                                         pipeline_kwargs=prep_arg_list[ct],
                                                         col_ind=ct)
            m_out += [m[i]]

        # Save
        return save_ds_from_raw(all_data, m_out, new_name, c_inds, p_inds, standardize_data)
    pass


def convert_data_struct(dat_struct, base_plot_dir, dt_mins, pl_kwargs,
                        c_inds=None,
                        p_inds=None,
                        standardize_data=False):
    """
    Converts a DataStruct to a Dataset.
    Using the same pre-processing steps for each series
    int the DataStruct.
    """

    # Get name
    name = dat_struct.name

    # Try loading data
    try:
        loaded = Dataset.loadDataset(name)
        return loaded
    except FileNotFoundError:
        data, m = dat_struct.getData()
        n_cols = len(data)
        pl_kwargs = b_cast(pl_kwargs, n_cols)
        all_data = None
        dt_init = None

        # Sets
        for i in range(n_cols):
            n_cs = n_cols if i == 0 else None
            title = clean_desc(m[i]['description'])
            added_cols = m[i]['additionalColumns']
            plot_name = added_cols['AKS Code']
            plot_file_name = os.path.join(base_plot_dir, name)
            all_data, dt_init = add_and_save_plot_series(data, m, all_data, i, dt_mins, dt_init, plot_name,
                                                         plot_file_name, title,
                                                         n_cols=n_cs,
                                                         pipeline_kwargs=pl_kwargs[i])

        # Save
        return save_ds_from_raw(all_data, m, name, c_inds, p_inds, standardize_data)


#######################################################################################################
# Preparing Data for model fitting

def standardize(data, m):
    """
    Removes mean and scales std to 1.0.
    Stores the parameters in the meta information.
    """
    s = data.shape
    n_feat = s[1]

    # Compute Mean and StD ignoring NaNs
    f_mean = np.nanmean(data, axis=0).reshape((1, n_feat))
    f_std = np.nanstd(data, axis=0).reshape((1, n_feat))

    # Process and store info 
    proc_data = (data - f_mean) / f_std
    for k in range(n_feat):
        m[k]['mean_and_std'] = [f_mean[0, k], f_std[0, k]]

    return proc_data, m


def find_rows_with_nans(all_data):
    """
    Returns a boolean vector indicating which
    rows of 'all_dat' contain NaNs.
    """

    n = all_data.shape[0]
    m = all_data.shape[1]
    col_has_nan = np.empty((n,), dtype=np.bool)
    col_has_nan.fill(False)

    for k in range(m):
        col_has_nan = np.logical_or(col_has_nan, np.isnan(all_data[:, k]))

    return col_has_nan


def cut_into_fixed_len(col_has_nan, seq_len: int = 20, interleave: bool = True):
    """
    Cuts the time series into pieces of length 'seq_len'
    for training of RNN.
    """

    n = col_has_nan.shape[0]
    indices = np.arange(0, n)

    # Initialize and find first non-NaN
    max_n_seq = n if interleave else n // seq_len
    seqs = np.empty((seq_len, max_n_seq), dtype=np.int32)
    ct = np.where(col_has_nan == False)[0][0]
    seq_count = 0

    while True:
        # Find next NaN
        zeros = np.where(col_has_nan[ct:])[0]
        curr_seq_len = n - ct if zeros.shape[0] == 0 else zeros[0]

        # Add sequences
        if interleave:
            n_seq_curr = curr_seq_len - seq_len + 1
            for k in range(n_seq_curr):
                seqs[:, seq_count] = indices[(ct + k):(ct + k + seq_len)]
                seq_count += 1
        else:
            n_seq_curr = curr_seq_len // seq_len
            for k in range(n_seq_curr):
                seqs[:, seq_count] = indices[(ct + k * seq_len):(ct + (k + 1) * seq_len)]
                seq_count += 1

        # Find next non-NaN
        ct += curr_seq_len
        non_zs = np.where(col_has_nan[ct:] == False)[0]

        # Break if none found
        if non_zs.shape[0] == 0:
            break
        ct += non_zs[0]

    # Return all found sequences
    return seqs[:, :seq_count]


def cut_data_into_sequences(all_data, seq_len: int, interleave: bool = True):
    """
    Use the two functions above to cut the data into
    sequences for RNN training.
    """

    # Use helper functions
    nans = find_rows_with_nans(all_data)
    seq_inds = cut_into_fixed_len(nans, seq_len, interleave)

    # Get shapes
    n = all_data.shape[0]
    n_feat = all_data.shape[1]
    n_seqs = seq_inds.shape[1]

    # Initialize empty data
    out_dat = np.empty((n_seqs, seq_len, n_feat), dtype=np.float32)

    # Fill and return
    for k in range(n_seqs):
        out_dat[k, :, :] = all_data[seq_inds[:, k], :]
    return out_dat


def extract_streak(all_data, s_len, lag):
    """
    Finds the last sequence where all data is available
    for at least s_len + lag timesteps.
    """

    tot_s_len = s_len + lag
    not_nans = np.logical_not(find_rows_with_nans(all_data))
    rwn = np.int32(not_nans)
    true_seq = np.empty((tot_s_len,), dtype=np.int32)
    true_seq.fill(1)

    # Find sequences of length tot_s_len
    tmp = np.convolve(rwn, true_seq, 'valid')
    inds = np.where(tmp == tot_s_len)[0]
    if len(inds) < 1:
        raise IndexError("No fucking streak of length {} found!!!".format(tot_s_len))
    last_seq_start = inds[-1]

    # Extract
    first_dat = all_data[:last_seq_start, :]
    streak_dat = all_data[last_seq_start:(last_seq_start + tot_s_len), :]
    return first_dat, streak_dat


def cut_and_split(dat, seq_len, streak_len, ret_orig=False):
    """
    Finds the latest series of 'streak_len' timesteps
    where all data is valid and splits the data there.
    The it cuts both parts it into sequences of length 'seq_len'.
    """
    dat_train, dat_test = extract_streak(dat, streak_len, seq_len - 1)
    if ret_orig:
        return dat_train, dat_test
    cut_train_dat = cut_data_into_sequences(dat_train, seq_len, interleave=True)
    cut_test_dat = cut_data_into_sequences(dat_test, seq_len, interleave=True)
    return cut_train_dat, cut_test_dat


#######################################################################################################
# Full Data Retrieval and Pre-processing

def get_battery_data():
    """
    Loads the battery dataset if existing else
    creates it from the raw data and creates a few plots.
    Then returns the dataset.
    """
    # Constants
    dt_mins = 15
    name = "Battery"
    bat_plot_path = os.path.join(preprocess_plot_path, name)
    create_dir(bat_plot_path)
    inds = [19, 17]
    n_feats = len(inds)

    # Define arguments
    p_kwargs_soc = {'clean_args': [([0.0], 24 * 60, [])],
                    'rem_out_args': (100, [0.0, 100.0]),
                    'lin_ip': True}
    p_kwargs_ap = {'clean_args': [([], 6 * 60, [])]}
    kws = [p_kwargs_soc, p_kwargs_ap]
    c_inds = np.array([1], dtype=np.int32)
    p_inds = np.array([0], dtype=np.int32)

    # Get the data
    ds = get_from_data_struct(BatteryData, bat_plot_path, dt_mins, name, inds, kws,
                              c_inds=c_inds,
                              p_inds=p_inds,
                              standardize_data=True)

    # Plot files
    plot_name_roi = os.path.join(bat_plot_path, "Strange")
    plot_name_after = os.path.join(bat_plot_path, "Processed")

    # Plot all data
    y_lab = '% / kW'
    plot_dataset(ds, False, ['Processed Battery Data', y_lab], plot_name_after)

    # Get data
    dat, m = BatteryData.getData()
    x = [dat[i][1] for i in inds]
    y = [dat[i][0] for i in inds]
    m_used = [m[i] for i in inds]

    # Extract and plot ROI of data where it behaves strangely
    d1 = np.datetime64('2019-05-24T12:00')
    d2 = np.datetime64('2019-05-25T12:00')
    x_ext, y_ext = [[extract_date_interval(x[i], y[i], d1, d2)[k] for i in range(n_feats)] for k in range(2)]
    plot_multiple_time_series(x_ext, y_ext, m_used,
                              show=False,
                              title_and_ylab=["Strange Battery Behavior", y_lab],
                              save_name=plot_name_roi)
    return ds


def get_weather_data(save_plots=True):
    """
    Load and interpolate the weather data.
    """

    # Constants
    dt_mins = 15
    filter_sigma = 2.0
    fill_by_ip_max = 2
    name = "Weather"
    name += "" if filter_sigma is None else str(filter_sigma)

    # Plot files
    prep_plot_dir = os.path.join(preprocess_plot_path, name)
    create_dir(prep_plot_dir)
    plot_name_temp = os.path.join(prep_plot_dir, "Outside_Temp")
    plot_name_irr = os.path.join(prep_plot_dir, "Irradiance")
    plot_name_temp_raw = os.path.join(prep_plot_dir, "Raw Outside_Temp")
    plot_name_irr_raw = os.path.join(prep_plot_dir, "Raw Irradiance")

    # Try loading data
    try:
        loaded = Dataset.loadDataset(name)
        return loaded
    except FileNotFoundError:
        pass

    # Initialize meta data dict list
    m_out = []

    # Weather data
    dat, m = WeatherData.getData()

    # Add Temperature
    all_data, dt_init = pipeline_preps(dat[0],
                                       dt_mins,
                                       clean_args=[([], 30, [])],
                                       rem_out_args=None,
                                       hole_fill_args=fill_by_ip_max,
                                       n_tot_cols=2,
                                       gauss_sigma=filter_sigma)
    add_dt_and_tinit(m, dt_mins, dt_init)
    m_out += [m[0]]

    if save_plots:
        plot_time_series(dat[0][1], dat[0][0], m=m[0], show=False, save_name=plot_name_temp_raw)
        plot_single(all_data[:, 0],
                    m[0],
                    use_time=True,
                    show=False,
                    title_and_ylab=['Outside Temperature Processed', m[0]['unit']],
                    save_name=plot_name_temp)

    # Add Irradiance Data
    all_data, _ = pipeline_preps(dat[2],
                                 dt_mins,
                                 all_data=all_data,
                                 dt_init=dt_init,
                                 row_ind=1,
                                 clean_args=[([], 60, [1300.0, 0.0]), ([], 60 * 20)],
                                 rem_out_args=None,
                                 hole_fill_args=fill_by_ip_max,
                                 gauss_sigma=filter_sigma)
    m_out += [m[2]]

    if save_plots:
        plot_time_series(dat[2][1], dat[2][0], m=m[2], show=False, save_name=plot_name_irr_raw)
        plot_single(all_data[:, 1],
                    m[2],
                    use_time=True,
                    show=False,
                    title_and_ylab=['Irradiance Processed', m[2]['unit']],
                    save_name=plot_name_irr)

    # Save and return
    w_dataset = Dataset.fromRaw(all_data, m_out, name, c_inds=no_inds, p_inds=no_inds)
    w_dataset.save()
    return w_dataset


def get_UMAR_heating_data():
    """
    Load and interpolate all the necessary data.
    """

    dat_structs = [Room272Data, Room274Data]
    dt_mins = 15
    fill_by_ip_max = 2
    filter_sigma = 2.0

    name = "UMAR"
    umar_rooms_plot_path = os.path.join(preprocess_plot_path, name)
    create_dir(umar_rooms_plot_path)
    all_ds = []

    for dat_struct in dat_structs:
        p_kwargs_room_temp = {'clean_args': [([0.0], 10 * 60)],
                              'rem_out_args': (4.5, [10, 100]),
                              'hole_fill_args': fill_by_ip_max,
                              'gauss_sigma': filter_sigma}
        p_kwargs_win = {'hole_fill_args': fill_by_ip_max,
                        'gauss_sigma': filter_sigma}
        p_kwargs_valve = {'hole_fill_args': 3,
                          'gauss_sigma': filter_sigma}
        kws = [p_kwargs_room_temp, p_kwargs_win, p_kwargs_valve]
        inds = [1, 11, 12]
        all_ds += [get_from_data_struct(dat_struct, umar_rooms_plot_path, dt_mins, dat_struct.name, inds, kws)]

    return all_ds


def get_DFAB_heating_data():
    """
    Loads or creates all data from DFAB then returns
    a list of all datasets. 4 for the rooms, one for the heating water
    and one with all the valves.

    :return: list(Datasets)
    """
    data_list = []
    dt_mins = 15
    dfab_rooms_plot_path = os.path.join(preprocess_plot_path, "DFAB")
    create_dir(dfab_rooms_plot_path)

    # Single Rooms
    for e in rooms:
        data, m = e.getData()
        n_cols = len(data)

        # Single Room Heating Data  
        temp_kwargs = {'clean_args': [([0.0], 24 * 60, [])], 'gauss_sigma': 5.0, 'rem_out_args': (1.5, None)}
        valve_kwargs = {'clean_args': [([], 30 * 24 * 60, [])]}
        blinds_kwargs = {'clip_to_int_args': [0.0, 100.0], 'clean_args': [([], 7 * 24 * 60, [])]}
        prep_kwargs = [temp_kwargs, valve_kwargs, valve_kwargs, valve_kwargs]
        if n_cols == 5:
            prep_kwargs += [blinds_kwargs]
        data_list += [convert_data_struct(e, dfab_rooms_plot_path, dt_mins, prep_kwargs)]

    # General Heating Data
    temp_kwargs = {'remove_out_int_args': [10, 50], 'gauss_sigma': 5.0}
    prep_kwargs = [temp_kwargs, temp_kwargs, {}, {}, {}, {}]
    data_list += [convert_data_struct(DFAB_AddData, dfab_rooms_plot_path, dt_mins, prep_kwargs)]

    # All Valves Together
    prep_kwargs = {'clean_args': [([], 30 * 24 * 60, [])]}
    data_list += [convert_data_struct(DFAB_AllValves, dfab_rooms_plot_path, dt_mins, prep_kwargs)]
    return data_list


def compute_DFAB_energy_usage(show_plots=True):
    """
    Computes the energy usage for every room at DFAB
    using the valves data and the inlet and outlet water
    temperature difference.
    """

    # Load data from Dataset
    w_name = "DFAB_Extra"
    w_dataset = Dataset.loadDataset(w_name)
    w_dat = w_dataset.getUnscaledData()
    t_init_w = w_dataset.t_init
    dt = w_dataset.dt

    v_name = "DFAB_Valves"
    dfab_rooms_plot_path = os.path.join(preprocess_plot_path, "DFAB")
    v_dataset = Dataset.loadDataset(v_name)
    v_dat = v_dataset.getUnscaledData()
    t_init_v = v_dataset.t_init

    # Align data
    aligned_data, t_init_new = align_ts(v_dat, w_dat, t_init_v, t_init_w, dt)
    aligned_len = aligned_data.shape[0]
    w_dat = aligned_data[:, 21:]
    v_dat = aligned_data[:, :21]

    # Find nans
    not_nans = np.logical_not(find_rows_with_nans(aligned_data))
    aligned_not_nan = np.copy(aligned_data[not_nans])
    n_not_nans = aligned_not_nan.shape[0]
    w_dat_not_nan = aligned_not_nan[:, 21:]
    v_dat_not_nan = aligned_not_nan[:, :21]
    thresh = 0.05
    usable = np.logical_and(w_dat_not_nan[:, 3] > 1 - thresh,
                            w_dat_not_nan[:, 4] < thresh)
    usable = np.logical_and(usable, w_dat_not_nan[:, 5] > 0.1)
    usable = np.logical_and(usable, w_dat_not_nan[:, 5] < 0.2)
    n_usable = np.sum(usable)
    first_n_del = n_usable // 3

    # Room info
    room_dict = {0: "31", 1: "41", 2: "42", 3: "43", 4: "51", 5: "52", 6: "53"}
    valve_room_allocation = np.array(["31", "31", "31", "31", "31", "31", "31",  # 3rd Floor
                                      "41", "41", "42", "43", "43", "43", "41",  # 4th Floor
                                      "51", "51", "53", "53", "53", "52", "51",  # 5th Floor
                                      ])
    n_rooms = len(room_dict)
    n_valves = len(valve_room_allocation)

    # Loop over rooms and compute flow per room
    A = np.empty((n_not_nans, n_rooms), dtype=np.float32)
    for i, room_nr in room_dict.items():
        room_valves = v_dat_not_nan[:, valve_room_allocation == room_nr]
        A[:, i] = np.mean(room_valves, axis=1)
    b = w_dat_not_nan[:, 2]
    x = solve_ls(A[usable][first_n_del:], b[usable][first_n_del:], offset=True)
    print("Flow", x)

    # Loop over rooms and compute flow per room
    A = np.empty((n_not_nans, n_valves), dtype=np.float32)
    for i in range(n_valves):
        A[:, i] = v_dat_not_nan[:, i]
    b = w_dat_not_nan[:, 2]
    x, fitted = solve_ls(A[usable][first_n_del:], b[usable][first_n_del:], offset=False, ret_fit=True)
    print("Flow per valve", x)

    x = solve_ls(A[usable][first_n_del:], b[usable][first_n_del:], non_neg=True)
    print("Non Negative Flow per valve", x)
    x = solve_ls(A[usable][first_n_del:], b[usable][first_n_del:], non_neg=True, offset=True)
    print("Non Negative Flow per valve with offset", x)

    stack_compare_plot(A[usable][first_n_del:], [21 * b[usable][first_n_del:], 21 * fitted], title="Valve model")
    # PO
    if show_plots:
        tot_room_valves_plot_path = os.path.join(dfab_rooms_plot_path, "DFAB_All_Valves")
        m_all = {'description': 'All valves summed',
                 'unit': 'TBD',
                 'dt': dt,
                 't_init': t_init_new}
        plot_single(np.sum(v_dat, axis=1),
                    m_all,
                    use_time=True,
                    show=False,
                    title_and_ylab=['Sum All Valves', '0/1'],
                    save_name=tot_room_valves_plot_path)

    raise NotImplementedError("Hahaha")

    flow_rates_f3 = np.array([134, 123, 129, 94, 145, 129, 81], dtype=np.float32)
    print(np.sum(flow_rates_f3), "Flow rates sum, 3. OG")
    del_temp = 13
    powers_f45 = np.array([137, 80, 130, 118, 131, 136, 207,
                           200, 192, 147, 209, 190, 258, 258], dtype=np.float32)
    c_p = 4.186
    d_w = 997
    h_to_s = 3600

    flow_rates_f45 = h_to_s / (c_p * d_w * del_temp) * powers_f45
    print(flow_rates_f45)

    tot_n_vals_open = np.sum(v_dat, axis=1)
    dTemp = w_dat[:, 0] - w_dat[:, 1]

    # Prepare output
    out_dat = np.empty((aligned_len, n_rooms), dtype=np.float32)
    m_list = []
    m_room = {'description': 'Energy consumption room',
              'unit': 'TBD',
              'dt': dt,
              't_init': t_init_new}
    if show_plots:
        w_plot_path = os.path.join(dfab_rooms_plot_path, w_name + "_WaterTemps")
        dw_plot_path = os.path.join(dfab_rooms_plot_path, w_name + "_WaterTempDiff")
        plot_dataset(w_dataset, show=False,
                     title_and_ylab=["Water Temps", "Temperature"],
                     save_name=w_plot_path)
        plot_single(dTemp, m_room, use_time=True, show=False,
                    title_and_ylab=["Temperature Difference", "DT"],
                    scale_back=False,
                    save_name=dw_plot_path)

    # Loop over rooms and compute energy
    for i, room_nr in room_dict.items():
        room_valves = v_dat[:, valve_room_allocation == room_nr]
        room_sum_valves = np.sum(room_valves, axis=1)

        # Divide ignoring division by zero
        room_energy = dTemp * np.divide(room_sum_valves,
                                        tot_n_vals_open,
                                        out=np.zeros_like(room_sum_valves),
                                        where=tot_n_vals_open != 0)

        m_room['description'] = 'Room ' + room_nr
        if show_plots:
            tot_room_valves_plot_path = os.path.join(dfab_rooms_plot_path,
                                                     "DFAB_" + m_room['description'] + "_Tot_Valves")
            room_energy_plot_path = os.path.join(dfab_rooms_plot_path, "DFAB_" + m_room['description'] + "_Tot_Energy")
            plot_single(room_energy,
                        m_room,
                        use_time=True,
                        show=False,
                        title_and_ylab=['Energy Consumption', 'Energy'],
                        save_name=room_energy_plot_path)
            plot_single(room_sum_valves,
                        m_room,
                        use_time=True,
                        show=False,
                        title_and_ylab=['Sum All Valves', '0/1'],
                        save_name=tot_room_valves_plot_path)

        # Add data to output
        out_dat[:, i] = room_energy
        m_list += [m_room.copy()]

    # Save dataset
    ds_out = Dataset.fromRaw(out_dat,
                             m_list,
                             "DFAB_Room_Energy_Consumption")
    ds_out.save()


def analyze_room_energy_consumption():
    """
    Compares the energy consumption of the different rooms
    summed over whole days.
    """

    ds = Dataset.loadDataset("DFAB_Room_Energy_Consumption")
    relevant_rooms = np.array([False, True, False, True, True, False, True], dtype=np.bool)
    dat = ds.data[:, relevant_rooms]
    n = dat.shape[0]
    d = dat.shape[1]

    # Set rows with nans to nan
    row_with_nans = find_rows_with_nans(dat)
    dat[row_with_nans, :] = np.nan

    # Sum Energy consumption over days
    n_ts = 4 * 24  # 1 Day
    n_mins = ds.dt * n_ts
    offset = n % n_ts
    dat = dat[offset:, :]
    dat = np.sum(dat.reshape((-1, n_ts, d)), axis=1)

    # Plot
    m = [{'description': ds.descriptions[relevant_rooms][k], 'dt': n_mins} for k in range(d)]
    f_name = os.path.join(preprocess_plot_path, "DFAB")
    f_name = os.path.join(f_name, "energy_comparison")
    plot_all(dat, m, use_time=False, show=False, title_and_ylab=["Room Energy Consumption", "Energy over one day"],
             save_name=f_name)


#######################################################################################################
# Dataset definition and generation

no_inds = np.array([], dtype=np.int32)


class Dataset:
    """
    This class contains all infos about a given dataset.
    """

    def __init__(self, all_data, dt, t_init, scaling, is_scaled, descs, c_inds=no_inds, p_inds=no_inds, name: str = ""):
        """
        Base constructor.
        """

        # Constants
        self.val_percent = 0.1
        self.seq_len = 20

        # Dimensions
        self.n = all_data.shape[0]
        self.d = get_shape1(all_data)
        self.n_c = c_inds.shape[0]
        self.n_p = p_inds.shape[0]

        # Check that indices are in range
        if not check_in_range(c_inds, 0, self.d):
            raise ValueError("Control indices out of bound.")
        if not check_in_range(p_inds, 0, self.d):
            raise ValueError("Prediction indices out of bound.")

        # Meta data
        self.name = name
        self.dt = dt
        self.t_init = t_init
        self.is_scaled = is_scaled
        self.scaling = scaling
        self.descriptions = descs
        self.c_inds = c_inds
        self.p_inds = p_inds

        # Actual data
        self.data = all_data
        if self.d == 1:
            self.data = np.reshape(self.data, (-1, 1))

        # Variables for later use
        self.streak_len = None
        self.orig_train_val = None
        self.orig_test = None
        self.orig_train = None
        self.orig_val = None
        self.train_streak = None
        self.test_data = None
        self.train_val_data = None
        self.train_data = None
        self.val_data = None
        self.train_streak_data = None
        self.val_streak = None
        self.val_streak_data = None

        self.c_inds_prep = None
        self.p_inds_prep = None
        return

    def split_train_test(self, streak_len: int = 7):
        """
        Split dataset into train, validation and test set.
        """

        # split data        
        s_len = int(60 / self.dt * 24 * streak_len)
        self.streak_len = s_len
        self.orig_train_val, self.orig_test = cut_and_split(np.copy(self.data), self.seq_len, s_len, ret_orig=True)
        self.orig_train, self.orig_val = split_arr(np.copy(self.orig_train_val), self.val_percent)
        _, self.train_streak = extract_streak(np.copy(self.orig_train), s_len, self.seq_len - 1)
        _, self.val_streak = extract_streak(np.copy(self.orig_val), s_len, self.seq_len - 1)

        # Cut into sequences and save to self.
        self.test_data = cut_data_into_sequences(np.copy(self.orig_test), self.seq_len, interleave=True)
        self.train_val_data = cut_data_into_sequences(np.copy(self.orig_train_val), self.seq_len, interleave=True)
        self.train_data = cut_data_into_sequences(np.copy(self.orig_train), self.seq_len, interleave=True)
        self.val_data = cut_data_into_sequences(np.copy(self.orig_val), self.seq_len, interleave=True)
        self.train_streak_data = cut_data_into_sequences(np.copy(self.train_streak), self.seq_len, interleave=True)
        self.val_streak_data = cut_data_into_sequences(np.copy(self.val_streak), self.seq_len, interleave=True)

    def get_prepared_data(self, what_data: str = 'train', *, get_all_preds: bool = False):
        """
        Prepares the data for supervised learning.
        """

        # Get the right data
        if what_data == 'train':
            data_to_use = self.train_data
        elif what_data == 'val':
            data_to_use = self.val_data
        elif what_data == 'test':
            data_to_use = self.test_data
        elif what_data == 'train_val':
            data_to_use = self.train_val_data
        elif what_data == 'train_streak':
            data_to_use = self.train_streak_data
        elif what_data == 'val_streak':
            data_to_use = self.val_streak_data
        else:
            raise ValueError("No such data available: " + what_data)
        data_to_use = np.copy(data_to_use)

        # Get dimensions
        s = data_to_use.shape
        n, s_len, d = s
        n_c = self.n_c

        # Get control and other column indices
        cont_inds = np.empty([d], dtype=np.bool)
        cont_inds.fill(False)
        cont_inds[self.c_inds] = True
        other_inds = np.logical_not(cont_inds)

        # Fill the data
        input_data = np.empty((n, s_len - 1, d), dtype=np.float32)
        input_data[:, :, :self.d - n_c] = data_to_use[:, :-1, other_inds]
        input_data[:, :, self.d - n_c:] = data_to_use[:, 1:, cont_inds]
        if not get_all_preds:
            output_data = data_to_use[:, -1, other_inds]
        else:
            output_data = data_to_use[:, 1:, other_inds]

        # Store more parameters, assuming only one control variable
        self.c_inds_prep = self.d - 1
        self.p_inds_prep = np.copy(self.p_inds)
        for c_ind in self.c_inds:
            self.p_inds_prep[self.p_inds_prep > c_ind] -= 1

        return input_data, output_data

    @classmethod
    def fromRaw(cls, all_data: np.ndarray, m: List, name: str,
                c_inds: np.ndarray = no_inds,
                p_inds: np.ndarray = no_inds) -> 'Dataset':
        """
        Constructor from data and dict m:
        Extracts the important metadata from the
        dict m.

        :param all_data: Numpy array with all the time series.
        :param m: List of metadata dictionaries.
        :param name: Name of the data collection.
        :param c_inds: Control indices.
        :param p_inds: Prediction indices.
        :return: Generated Dataset.
        """
        d = all_data.shape[1]

        # Extract data from m
        dt = m[0]['dt']
        t_init = m[0]['t_init']
        is_scaled = np.empty((d,), dtype=np.bool)
        is_scaled.fill(True)
        scaling = np.empty((d, 2), dtype=np.float32)
        descriptions = np.empty((d,), dtype="U100")
        for ct, el in enumerate(m):
            desc = el['description']
            descriptions[ct] = desc
            m_a_s = el.get('mean_and_std')
            if m_a_s is not None:
                scaling[ct, 0] = m_a_s[0]
                scaling[ct, 1] = m_a_s[1]
            else:
                is_scaled[ct] = False

        ret_val = cls(np.copy(all_data), dt, t_init, scaling, is_scaled, descriptions, c_inds, p_inds, name)
        return ret_val

    def __len__(self) -> int:
        """
        Returns the number of features per sample per timestep.

        :return: Number of features.
        """
        return self.d

    def __str__(self) -> str:
        """
        Creates a string containing the most important
        information about this dataset.

        :return: Dataset description string.
        """
        out_str = "Dataset(" + repr(self.data) + ", \ndt = " + repr(self.dt)
        out_str += ", t_init = " + repr(self.t_init) + ", \nis_scaled = " + repr(self.is_scaled)
        out_str += ", \ndescriptions = " + repr(self.descriptions) + ", \nc_inds = " + repr(self.c_inds)
        out_str += ", \np_inds = " + repr(self.p_inds) + ", name = " + str(self.name) + ")"
        return out_str

    @classmethod
    def copy(cls, dataset: 'Dataset') -> 'Dataset':
        """
        Returns a deep copy of the passed Dataset.

        :param dataset: The dataset to copy.
        :return: The new dataset.
        """
        return cls(np.copy(dataset.data),
                   dataset.dt,
                   dataset.t_init,
                   np.copy(dataset.scaling),
                   np.copy(dataset.is_scaled),
                   np.copy(dataset.descriptions),
                   np.copy(dataset.c_inds),
                   np.copy(dataset.p_inds),
                   dataset.name)

    def __add__(self, other: 'Dataset') -> 'Dataset':
        """
        Merges dataset other into self.
        Does not commute!

        :param other: The dataset to merge self with.
        :return: A new dataset with the combined data.
        """
        # Check compatibility
        if self.dt != other.dt:
            raise ValueError("Datasets not compatible!")

        # Merge data
        ti1 = self.t_init
        ti2 = other.t_init
        data, t_init = align_ts(self.data, other.data, ti1, ti2, self.dt)

        # Merge metadata
        d = self.d
        scaling = np.concatenate([self.scaling, other.scaling], axis=0)
        is_scaled = np.concatenate([self.is_scaled, other.is_scaled], axis=0)
        descs = np.concatenate([self.descriptions, other.descriptions], axis=0)
        c_inds = np.concatenate([self.c_inds, other.c_inds + d], axis=0)
        p_inds = np.concatenate([self.p_inds, other.p_inds + d], axis=0)
        name = self.name + other.name

        return Dataset(data, self.dt, t_init, scaling, is_scaled, descs, c_inds, p_inds, name)

    def add_time(self, sine_cos: bool = True) -> 'Dataset':
        """
        Adds time to current dataset.

        :param sine_cos: Whether to use sin(t) and cos(t) instead of t directly.
        :return: self + the time dataset.
        """
        dt = self.dt
        t_init = datetime_to_npdatetime(string_to_dt(self.t_init))
        one_day = np.timedelta64(1, 'D')
        dt_td64 = np.timedelta64(dt, 'm')
        n_tint_per_day = int(one_day / dt_td64)
        floor_day = np.array([t_init], dtype='datetime64[D]')[0]
        begin_ind = int((t_init - floor_day) / dt_td64)
        dat = np.empty((self.n,), dtype=np.float32)
        for k in range(self.n):
            dat[k] = (begin_ind + k) % n_tint_per_day

        if not sine_cos:
            return self + Dataset(dat,
                                  self.dt,
                                  self.t_init,
                                  np.array([0.0, 1.0]),
                                  np.array([False]),
                                  np.array(["Time of day [" + dt + " mins.]"]),
                                  no_inds,
                                  no_inds,
                                  "Time")
        else:
            all_dat = np.empty((self.n, 2), dtype=np.float32)
            all_dat[:, 0] = np.sin(2 * np.pi * dat / n_tint_per_day)
            all_dat[:, 1] = np.cos(2 * np.pi * dat / n_tint_per_day)
            return self + Dataset(all_dat,
                                  self.dt,
                                  self.t_init,
                                  np.array([[0.0, 1.0], [0.0, 1.0]]),
                                  np.array([False, False]),
                                  np.array(["sin(Time of day)", "cos(Time of day)"]),
                                  no_inds,
                                  no_inds,
                                  "Time")
        pass

    def getSlice(self, ind_low: int, ind_high: int) -> 'Dataset':
        """
        Returns a new dataset with the columns
        'ind_low' through 'ind_high'.

        :param ind_low: Lower range index.
        :param ind_high: Upper range index.
        :return: Dataset containing series [ind_low: ind_high) of current
        dataset.
        """

        warnings.warn("Prediction and control indices are lost when slicing.")
        low = ind_low

        if ind_low < 0 or ind_high > self.d or ind_low >= ind_high:
            raise ValueError("Slice indices are invalid.")
        if ind_low + 1 != ind_high:
            return Dataset(np.copy(self.data[:, ind_low: ind_high]),
                           self.dt,
                           self.t_init,
                           np.copy(self.scaling[ind_low: ind_high]),
                           np.copy(self.is_scaled[ind_low: ind_high]),
                           np.copy(self.descriptions[ind_low: ind_high]),
                           no_inds,
                           no_inds,
                           self.name + "[" + str(ind_low) + ":" + str(ind_high) + "]")
        else:
            return Dataset(np.copy(self.data[:, low:low + 1]),
                           self.dt,
                           self.t_init,
                           np.copy(self.scaling[low:low + 1]),
                           np.copy(self.is_scaled[low:low + 1]),
                           np.copy(self.descriptions[low:low + 1]),
                           no_inds,
                           no_inds,
                           self.name + "[" + str(low) + "]")

    def __getitem__(self, key):
        """
        Allows for slicing. Returns a copy not a reference.
        """

        if isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise NotImplementedError("Only implemented for contiguous ranges!")
            return self.getSlice(key.start, key.stop)
        return self.getSlice(key, key + 1)

    def save(self) -> None:
        """
        Save the class object to a file.

        :return: None
        """
        create_dir(dataset_data_path)

        file_name = self.get_filename(self.name)
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def getUnscaledData(self):
        """
        Adds the mean and std back to every column and returns
        the data.
        """
        data_out = np.copy(self.data)
        for k in range(self.d):
            if self.is_scaled[k]:
                data_out[:, k] = add_mean_and_std(data_out[:, k], self.scaling[k, :])

        return data_out

    @staticmethod
    def get_filename(name: str) -> str:
        return os.path.join(dataset_data_path, name) + '.pkl'

    @staticmethod
    def loadDataset(name: str) -> 'Dataset':
        """
        Load a saved Dataset object.

        :param name: Name of dataset.
        :return: Loaded dataset.
        """
        f_name = Dataset.get_filename(name)
        if not os.path.isfile(f_name):
            raise FileNotFoundError("Dataset {} does not exist.".format(f_name))
        with open(f_name, 'rb') as f:
            ds = pickle.load(f)
            return ds

    def standardize_col(self, col_ind: int) -> None:
        """
        Standardizes a certain column of the data.
        Nans are ignored.

        :param col_ind: Index of the column to be standardized.
        :return: None
        """
        if col_ind >= self.d:
            raise ValueError("Column index too big!")
        if self.is_scaled[col_ind]:
            return
        m = np.nanmean(self.data[:, col_ind])
        std = np.nanstd(self.data[:, col_ind])
        self.data[:, col_ind] = (self.data[:, col_ind] - m) / std
        self.is_scaled[col_ind] = True
        self.scaling[col_ind] = np.array([m, std])

    def standardize(self) -> None:
        """
        Standardizes all columns in the data.

        :return: None
        """
        for k in range(self.d):
            self.standardize_col(k)

    def visualize_nans(self, name_ext: str = ""):
        """
        Visualizes where the holes are in the different
        time series (columns) of the data.
        """
        nan_plot_dir = os.path.join(plot_dir, "NanPlots")
        create_dir(nan_plot_dir)
        s_name = os.path.join(nan_plot_dir, self.name)
        not_nans = np.logical_not(np.isnan(self.data))
        scaled = not_nans * np.arange(1, 1 + self.d, 1, dtype=np.int32)
        scaled[scaled == 0] = -1
        m = [{'description': d, 'dt': self.dt} for d in self.descriptions]
        plot_all(scaled, m,
                 use_time=False,
                 show=False,
                 title_and_ylab=["Nan plot", "Series"],
                 scale_back=False,
                 save_name=s_name + name_ext)


def generate_room_datasets():
    """
    Gather the right data and put it together.
    """

    # Get weather
    w_dataset = get_weather_data()

    # Get room data
    dfab_dataset_list = get_DFAB_heating_data()
    n_rooms = len(rooms)
    dfab_room_dataset_list = [dfab_dataset_list[i] for i in range(n_rooms)]

    # Heating water temperature
    dfab_heat_water_temp_ds = dfab_dataset_list[n_rooms]
    inlet_water_ds = dfab_heat_water_temp_ds[0]
    inlet_water_and_weather = w_dataset + inlet_water_ds

    out_ds_list = []

    # Single room datasets
    for ct, room_ds in enumerate(dfab_room_dataset_list):

        # Get name
        room_nr_str = room_ds.name[-2:]
        new_name = "Model_Room" + room_nr_str
        print("Processing", new_name)

        # Try loading from disk
        try:
            curr_out_ds = Dataset.loadDataset(new_name)
            out_ds_list += [curr_out_ds]
            continue
        except FileNotFoundError:
            pass

        # Extract datasets
        valves_ds = room_ds[1:4]
        room_temp_ds = room_ds[0]

        # Compute average valve data and put into dataset
        valves_avg = np.mean(valves_ds.data, axis=1)
        valves_avg_ds = Dataset(valves_avg,
                                valves_ds.dt,
                                valves_ds.t_init,
                                np.empty((1, 2), dtype=np.float32),
                                np.array([False]),
                                np.array(["Averaged valve open time."]))

        # Put all together
        full_ds = (inlet_water_and_weather + valves_avg_ds) + room_temp_ds
        full_ds.c_inds = np.array([3], dtype=np.int32)
        full_ds.p_inds = np.array([4], dtype=np.int32)

        # Add blinds
        if len(room_ds) == 5:
            blinds_ds = room_ds[4]
            full_ds = full_ds + blinds_ds

        # Save
        full_ds.name = new_name
        full_ds.save()
        out_ds_list += [full_ds]

    # Return all
    return out_ds_list


def generate_sin_cos_time_ds(other: Dataset) -> Dataset:
    """
    Generates a time dataset from the last two
    time series of another dataset.
    :param other: The other dataset.
    :return: A new dataset containing only the time series.
    """
    name = other.name + "_SinCosTime"

    try:
        return Dataset.loadDataset(name)
    except FileNotFoundError:
        pass
    # Construct Time dataset
    n_feat = other.d
    ds_sin_cos_time: Dataset = Dataset.copy(other[n_feat - 2: n_feat])
    ds_sin_cos_time.name = name
    ds_sin_cos_time.p_inds = np.array([0], dtype=np.int32)
    ds_sin_cos_time.c_inds = no_inds
    ds_sin_cos_time.save()
    return ds_sin_cos_time


#######################################################################################################
# Testing

# Test Data
TestData = DataStruct(id_list=[421100171, 421100172],
                      name="Test",
                      start_date='2019-08-08',
                      end_date='2019-08-09')


class TestDataSynthetic:
    """
    Synthetic and short dataset to be used for debugging.
    """

    def getData(self):
        # First Time series
        dict1 = {'description': "Synthetic Data Series 1: Base Series", 'unit': "Test Unit 1"}
        val_1 = np.array([1.0, 2.3, 2.3, 1.2, 2.3, 0.8])
        dat_1 = np.array([
            np.datetime64('2005-02-25T03:31'),
            np.datetime64('2005-02-25T03:39'),
            np.datetime64('2005-02-25T03:48'),
            np.datetime64('2005-02-25T04:20'),
            np.datetime64('2005-02-25T04:25'),
            np.datetime64('2005-02-25T04:30'),
        ], dtype='datetime64')

        # Second Time series
        dict2 = {'description': "Synthetic Data Series 2: Delayed and longer", 'unit': "Test Unit 2"}
        val_2 = np.array([1.0, 1.4, 2.1, 1.5, 3.3, 1.8, 2.5])
        dat_2 = np.array([
            np.datetime64('2005-02-25T03:51'),
            np.datetime64('2005-02-25T03:59'),
            np.datetime64('2005-02-25T04:17'),
            np.datetime64('2005-02-25T04:21'),
            np.datetime64('2005-02-25T04:34'),
            np.datetime64('2005-02-25T04:55'),
            np.datetime64('2005-02-25T05:01'),
        ], dtype='datetime64')

        # Third Time series
        dict3 = {'description': "Synthetic Data Series 3: Repeating Values", 'unit': "Test Unit 3"}
        val_3 = np.array([0.0, 1.4, 1.4, 0.0, 3.3, 3.3, 3.3, 3.3, 0.0, 3.3, 2.5])
        dat_3 = np.array([
            np.datetime64('2005-02-25T03:45'),
            np.datetime64('2005-02-25T03:53'),
            np.datetime64('2005-02-25T03:59'),
            np.datetime64('2005-02-25T04:21'),
            np.datetime64('2005-02-25T04:23'),
            np.datetime64('2005-02-25T04:25'),
            np.datetime64('2005-02-25T04:34'),
            np.datetime64('2005-02-25T04:37'),
            np.datetime64('2005-02-25T04:45'),
            np.datetime64('2005-02-25T04:55'),
            np.datetime64('2005-02-25T05:01'),
        ], dtype='datetime64')
        return [(val_1, dat_1), (val_2, dat_2), (val_3, dat_3)], [dict1, dict2, dict3]


TestData2 = TestDataSynthetic()


# Tests
def test_rest_client():
    """
    Tests the REST client by requesting test data,
    saving it locally, reading it locally and deleting
    it again.

    :return: None
    """

    # Load using REST api and locally
    t_dat = TestData
    t_dat.getData()
    t_dat.getData()

    # Remove data again
    fol = TestData.get_data_folder()
    for f in os.listdir(fol):
        file_path = os.path.join(fol, f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def get_data_test():
    name = "Test"

    # Test Sequence Cutting
    seq1 = np.array([1, 2, np.nan, 1, 2, np.nan, 3, 2, 1, 3, 4, np.nan, np.nan, 3, 2, 3, 1, 3, np.nan, 1, 2])
    seq2 = np.array([3, 4, np.nan, np.nan, 2, np.nan, 3, 2, 1, 3, 4, 7, np.nan, 3, 2, 3, 1, np.nan, np.nan, 3, 4])
    n = seq1.shape[0]
    all_dat = np.empty((n, 2), dtype=np.float32)
    all_dat[:, 0] = seq1
    all_dat[:, 1] = seq2
    print(all_dat)

    # Test
    streak = extract_streak(all_dat, 2, 2)
    print(streak)

    # Test Standardizing
    m = [{}, {}]
    all_dat, m = standardize(all_dat, m)
    print(all_dat)
    print(m)

    # Test Sequence Cutting
    c = find_rows_with_nans(all_dat)
    print(c)
    seq_inds = cut_into_fixed_len(c, 2, interleave=True)
    print(seq_inds)
    od = cut_data_into_sequences(all_dat, 2, interleave=True)
    print(od)

    # Test hole filling by interpolation
    test_ts = np.array([np.nan, np.nan, 1, 2, 3.5, np.nan, 4.5, np.nan])
    test_ts2 = np.array([1, 2, 3.5, np.nan, np.nan, 5.0, 5.0, np.nan, 7.0])
    fill_holes_linear_interpolate(test_ts, 1)
    fill_holes_linear_interpolate(test_ts2, 2)
    print(test_ts)
    print(test_ts2)

    # Test Outlier Removal
    test_ts3 = np.array(
        [1, 2, 3.5, np.nan, np.nan, 5.0, 5.0, 17.0, 5.0, 2.0, -1.0, np.nan, 7.0, 7.0, 17.0, np.nan, 20.0, 5.0, 6.0])
    print(test_ts3)
    remove_outliers(test_ts3, 5.0, [0.0, 100.0])
    print(test_ts3)

    # Show plots
    dt_mins = 15
    dat, m = TestData2.getData()

    # Clean data
    mod_dat = clean_data(dat[2], [0.0], 4, [3.3])

    # Interpolate
    [data1, dt_init1] = interpolate_time_series(dat[0], dt_mins)
    n_data = data1.shape[0]
    print(n_data, "total data points.")

    # Initialize np array for compact storage
    all_data = np.empty((n_data, 3), dtype=np.float32)
    all_data.fill(np.nan)

    # Add data
    add_time(all_data, dt_init1, 0, dt_mins)
    all_data[:, 1] = data1
    [data2, dt_init2] = interpolate_time_series(dat[1], dt_mins)

    print(data2)
    data2 = gaussian_filter_ignoring_nans(data2)
    print(data2)

    add_col(all_data, data2, dt_init1, dt_init2, 2, dt_mins)
    return all_data, m, name


def test_align():
    """
    Tests the alignment of two time series.
    """

    # Test data
    t_i1 = '2019-01-01 00:00:00'
    t_i2 = '2019-01-01 00:30:00'
    dt = 15
    ts_1 = np.array([1, 2, 2, 2, 3, 3], dtype=np.float32)
    ts_2 = np.array([2, 3, 3], dtype=np.float32)

    # Do tests
    test1 = align_ts(ts_1, ts_2, t_i1, t_i2, dt)
    print('Test 1:', test1)
    test2 = align_ts(ts_2, ts_1, t_i1, t_i2, dt)
    print('Test 2:', test2)
    test3 = align_ts(ts_1, ts_1, t_i1, t_i2, dt)
    print('Test 3:', test3)
    test4 = align_ts(ts_1, ts_1, t_i2, t_i1, dt)
    print('Test 4:', test4)
    test5 = align_ts(ts_2, ts_1, t_i2, t_i1, dt)
    print('Test 5:', test5)

    return


def test_dataset_artificially():
    """
    Constructs a small synthetic dataset and makes tests.
    """

    dat = np.array([1, 2, 3, 7,
                    1, 3, 4, 7,
                    1, 4, 5, 7,
                    1, 5, 6, 7], dtype=np.float32).reshape((4, 4))
    c_inds = np.array([2])
    p_inds = np.array([1])
    descs = np.array(["1", "2", "3", "4"])
    is_sc = np.array([False for _ in range(4)])
    sc = np.empty((4, 2), dtype=np.float32)

    dt = 15
    t_init = '2019-01-01 00:00:00'
    ds = Dataset(dat, dt, t_init, sc, is_sc, descs, c_inds, p_inds, "Test")

    ds.save()
    plot_dataset(ds, True, ["Test", "Fuck"])

    pass
