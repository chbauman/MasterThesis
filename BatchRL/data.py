
import os
import scipy
import pickle

import numpy as np
import pandas as pd

from ast import literal_eval
from datetime import datetime

from visualize import plot_time_series, plot_ip_time_series, \
    plot_single_ip_ts, plot_multiple_ip_ts, \
    plot_all, plot_single, preprocess_plot_path, \
    plot_multiple_time_series, plot_dataset
from restclient import DataStruct, save_dir
from util import *

# Data directories
processed_data_path = os.path.join(save_dir, "ProcessedSeries")
dataset_data_path = os.path.join(save_dir, "Datasets")

#######################################################################################################
# NEST Data

# UMAR Room Data
Room272Data = DataStruct(id_list = [42150280,
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
                        name = "Room272",
                        startDate='2017-01-01',
                        endDate='2019-12-31')

Room274Data = DataStruct(id_list = [42150281,
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
                        name = "Room274",
                        startDate='2017-01-01',
                        endDate='2019-12-31')

# DFAB Data
Room4BlueData = DataStruct(id_list = [421110054, # Temp
                                    421110023, # Valves
                                    421110024,
                                    421110029,
                                    421110209 # Blinds
                                    ],
                        name = "DFAB_Room41",
                        startDate='2017-01-01',
                        endDate='2019-12-31')

Room5BlueData = DataStruct(id_list = [421110072, # Temp
                                    421110038, # Valves
                                    421110043,
                                    421110044,
                                    421110219 # Blinds
                                    ],
                        name = "DFAB_Room51",
                        startDate='2017-01-01',
                        endDate='2019-12-31')

Room4RedData = DataStruct(id_list = [421110066, # Temp
                                    421110026, # Valves
                                    421110027,
                                    421110028, 
                                    ],
                        name = "DFAB_Room43",
                        startDate='2017-01-01',
                        endDate='2019-12-31')

Room5RedData = DataStruct(id_list = [421110084, # Temp
                                    421110039, # Valves
                                    421110040,
                                    421110041,
                                    ],
                        name = "DFAB_Room53",
                        startDate='2017-01-01',
                        endDate='2019-12-31')

DFAB_AddData = DataStruct(id_list = [421100168, # Vorlauf Temp
                                    421100170, # Rücklauf Temp
                                    ],
                        name = "DFAB_Extra",
                        startDate='2017-01-01',
                        endDate='2019-12-31')

DFAB_AllValves = DataStruct(id_list = [421110008, # First Floor
                                    421110009,
                                    421110010,
                                    421110011,
                                    421110012,
                                    421110013,
                                    421110014,
                                    421110023, # Second Floor,
                                    421110024,
                                    421110025,
                                    421110026,
                                    421110027,
                                    421110028,
                                    421110029,
                                    421110038, # Third Floor
                                    421110039, 
                                    421110040,
                                    421110041,
                                    421110042,
                                    421110043,
                                    421110044,
                                    ],
                        name = "DFAB_Valves",
                        startDate='2017-01-01',
                        endDate='2019-12-31')

rooms = [Room4BlueData, Room5BlueData, Room4RedData, Room5RedData]

# Weather Data
WeatherData = DataStruct(id_list = [3200000,
                                    3200002,
                                    3200008],
                        name = "Weather",
                        startDate='2019-01-01',
                        endDate='2019-12-31')

# Battery Data
BatteryData = DataStruct(id_list = [40200000,
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
                        name = "Battery",
                        startDate='2018-01-01',
                        endDate='2019-12-31')

#######################################################################################################
# Time Series Processing

def extract_date_interval(dates, vals, d1, d2):
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
    vals_new = vals[mask]
    dt_init_new = dates_new[0]
    return dates_new, vals_new, dt_init_new

def analyze_data(dat):
    """
    Analyzes the provided data.
    """

    vals, dates = dat
    n_data_p = len(vals)

    tot_time_span = dates[-1] - dates[0]
    print("Data ranges from ", dates[0], "to", dates[-1])
    

    t_diffs = dates[1:] - dates[:-1]
    max_t_diff = np.max(t_diffs)
    mean_t_diff = np.mean(t_diffs)
    print("Largest gap:", np.timedelta64(max_t_diff, 'D'), "or", np.timedelta64(max_t_diff, 'h'))
    print("Mean gap:", np.timedelta64(mean_t_diff, 'm'), "or", np.timedelta64(mean_t_diff, 's'))
    
    print("Positive differences:", np.all(t_diffs > np.timedelta64(0, 'ns')))
    return

def clean_data(dat, rem_vals = [], n_cons_least = 60, const_excepts = []):
    """
    Removes all values with a specified value 'rem_val'
    and removes all sequences where there are at 
    least 'n_cons_least' consecutive
    values having the exact same value. If the value 
    occurring multiple times is in 'const_excepts' then
    it is not removed.
    """

    vals, dates = dat
    tot_dat = vals.shape[0]

    # Make copy
    new_vals = np.copy(vals)
    new_dates = np.copy(dates)

    # Initialize
    prev_val = np.nan
    count = 0
    num_occ = 1
    consec_streak = False

    # Add cleaned values and dates
    for (v, d) in zip(vals, dates):

        if v not in rem_vals:

            # Monitor how many times the same value occurred
            if v == prev_val and v not in const_excepts:

                num_occ += 1
                if num_occ == n_cons_least:
                    consec_streak = True
                    count -= n_cons_least - 1
            else:
                consec_streak = False
                num_occ = 1

            # Add value if it has not occurred too many times
            if consec_streak == False:
            
                new_vals[count] = v
                new_dates[count] = d
                count += 1
                prev_val = v

        else:
            # Reset streak
            consec_streak = False
            num_occ = 1

    # Return clean data
    print(tot_dat - count, "data points removed.")
    return [new_vals[:count], new_dates[:count]]

def remove_out_interval(dat, interval = [0.0, 100]):
    """
    Removes values that do not lie within the interval.
    """    
    vals, dates = dat
    vals[vals > interval[1]] = np.nan
    vals[vals < interval[0]] = np.nan

def clip_to_interval(dat, interval = [0.0, 100]):
    """
    Clips the values of the time_series that are
    out of the interval to lie within.
    """
    vals, dates = dat
    vals[vals > interval[1]] = interval[1]
    vals[vals < interval[0]] = interval[0]

def floor_datetime_to_min(dt, mt):
    """
    Rounds deltatime64 dt down to mt minutes.
    In a really fucking cumbersome way.
    """
    assert 60 % mt == 0

    dt = np.array(dt, dtype='datetime64[s]')
    dt64 = np.datetime64(dt)
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    pdt = datetime.utcfromtimestamp(ts)
    mins = pdt.minute
    mins = mins % mt
    secs = pdt.second    
    dt -= np.timedelta64(secs, 's')
    dt -= np.timedelta64(mins, 'm')
    return dt

def interpolate_time_series(dat, dt_mins, lin_ip = False):
    """
    Interpolates the given time series
    to produce another one with equidistant timesteps
    and NaNs if values are missing.
    """

    # Unpack
    vals, dates = dat

    # Datetime of first and last data point
    start_dt = floor_datetime_to_min(dates[0], dt_mins)
    end_dt = floor_datetime_to_min(dates[-1], dt_mins)
    interv = np.timedelta64(dt_mins, 'm')
    n_ts = (end_dt - start_dt) // interv + 1
    print(n_ts, "Timesteps")

    # Initialize
    new_vals = np.empty((n_ts,), dtype = np.float32)
    new_vals.fill(np.nan)
    count = 0
    curr_val = 0
    last_dt = dates[0]
    last_val = vals[0]
    curr_val = (last_dt - start_dt) / interv * last_val

    # Loop over data points
    for ct, v in enumerate(vals[1:]):
        curr_dt = dates[ct + 1]
        curr_upper_lim = start_dt + (count + 1) * interv
        if curr_dt >= curr_upper_lim:
            if curr_dt <= curr_upper_lim + interv:
                # Next datetime in next interval
                curr_val += (curr_upper_lim - last_dt) / interv * v
                if not lin_ip:
                    new_vals[count] = curr_val
                else:
                    new_vals[count] = last_val + (v - last_val) * (curr_upper_lim - last_dt) / (curr_dt - last_dt)
                count += 1
                curr_val = (curr_dt - curr_upper_lim) / interv * v
            else:
                # Data missing!                
                curr_val += (curr_upper_lim - last_dt) / interv * last_val
                if not lin_ip:
                    new_vals[count] = curr_val
                else:
                    new_vals[count] = last_val
                count += 1
                n_data_missing = (curr_dt - curr_upper_lim) // interv
                print("Missing", n_data_missing, "data points :(")
                for k in range(n_data_missing):
                    new_vals[count] = np.nan
                    count += 1
                dtime_start_new_interv = curr_dt - curr_upper_lim - n_data_missing * interv
                curr_val = dtime_start_new_interv / interv * v

        else:
            # Next datetime still in same interval
            curr_val += (curr_dt - last_dt) / interv * v

        # Update
        last_dt = curr_dt
        last_val = v

    # Add last one
    curr_val += (end_dt + interv - curr_dt) / interv * v
    new_vals[count] = curr_val

    # Return
    return [new_vals, start_dt]
 
def add_col(full_dat_array, data, dt_init, dt_init_new, col_ind, dt_mins = 15):
    """
    Add time series as columd to data array at the right index.
    If the second time series exceeds the datatime range of the 
    first one it is cut to fit the first one. If it is too short
    the missing values are filled with NaNs.
    """
    
    n_data = full_dat_array.shape[0]
    n_data_new = data.shape[0]

    # Compute indices
    interv = np.timedelta64(dt_mins, 'm')
    offset_before = int(np.round((dt_init_new - dt_init) / interv))
    offset_after = n_data_new - n_data + offset_before
    dat_inds = [np.maximum(0, offset_before), n_data + np.minimum(0, offset_after)]
    new_inds = [np.maximum(0, -offset_before), n_data_new + np.minimum(0, -offset_after)]

    # Add to column
    full_dat_array[dat_inds[0]:dat_inds[1], col_ind] = data[new_inds[0]:new_inds[1]]
    return

def add_time(all_data, dt_init1, col_ind = 0, dt_mins = 15):
    """
    Adds the time as indices to the data,
    periodic with period one day.
    """

    n_data = all_data.shape[0]
    interv = np.timedelta64(dt_mins, 'm')
    n_ts_per_day = 24 * 60 / dt_mins
    t_temp_round = np.datetime64(dt_init1, 'D')
    start_t = (dt_init1 - t_temp_round) / interv
    for k in range(n_data):
        all_data[k, col_ind] = (start_t + k) % n_ts_per_day
    return

def fill_holes_linear_interpolate(time_series, max_width = 1):
    """
    Fills the holes of a equispaced time series
    with a width up to 'max_width' 
    by linearly interpolating between the previous and
    next data point.
    """

    n = time_series.shape[0]
    nan_bool = np.isnan(time_series)    

    # Return if there are no NaNs
    if np.sum(nan_bool) == 0:
        return

    # Neglect NaNs at beginning and end
    non_nans = np.where(nan_bool == False)[0]
    first_non_nan = non_nans[0]
    nan_bool[:non_nans[0]] = False
    nan_bool[non_nans[-1]:] = False

    # Find all indices with NaNs
    all_nans = np.argwhere(nan_bool)

    # Initialize itarators
    n_nans = all_nans.shape[0]
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

def remove_outliers(time_series, grad_clip = 100, clip_interv = None):
    """
    Removes data points that lie outside 
    the specified interval 'clip_int' and ones 
    with a large gradient.
    """

    # Helper functions
    def grad_fd(x1, x2):
        if x2 is None or x1 is None:
            return np.nan
        if np.isnan(x1) or np.isnan(x2):
            return np.nan        
        return x2 - x1

    def is_outlier(x, x_tm1, x_tp1 = None):
        g1 = grad_fd(x_tm1, x)
        g2 = grad_fd(x, x_tp1)
        if np.isnan(g1):            
            return True if np.absolute(g2) > 1.5 * grad_clip else False   
        if np.isnan(g2):            
            return True if np.absolute(g1) > 1.5 * grad_clip else False
        rej = np.absolute(g1) > grad_clip and np.absolute(g2) > grad_clip
        rej = rej and g1 * g2 < 0
        return rej

    def reject_outls(x, x_tm1, x_tp1 = None):
        if is_outlier(x, x_tm1, x_tp1):
            return np.nan
        return x

    # First and last values
    time_series[0] = reject_outls(time_series[0], time_series[1])
    time_series[-1] = reject_outls(time_series[-1], time_series[-2])
    
    # Iterate
    for ct, el in enumerate(time_series[1:-1]):
        if el != np.nan:
            # Remove large gradient outliers
            time_series[ct + 1] = reject_outls(el, 
                                          time_series[ct + 2], 
                                          time_series[ct])
            
            # Clip to interval
            if clip_interv is not None:
                if el < clip_interv[0] or el > clip_interv[1]:
                    time_series[ct + 1] = np.nan

    return 

def gaussian_filter_ignoring_nans(time_series, sigma = 2.0):
    """
    1-dimensional Gaussian Filtering ignoring 
    occurrences of NaNs. From:
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    """

    V = time_series.copy()
    V[np.isnan(time_series)] = 0
    VV = scipy.ndimage.filters.gaussian_filter1d(V, sigma=sigma)

    W = 0 * time_series.copy() + 1
    W[np.isnan(time_series)] = 0
    WW = scipy.ndimage.filters.gaussian_filter1d(W, sigma=sigma)

    Z = VV / WW
    Z[np.isnan(time_series)] = np.nan
    return Z

def pipeline_preps(orig_dat,
                   dt_mins,
                   all_data = None,
                   *,
                   dt_init = None,
                   row_ind = None,                   
                   clean_args = None,
                   clip_to_int_args = None,
                   remove_out_int_args = None,
                   rem_out_args = None,
                   hole_fill_args = None,
                   n_tot_cols = None,
                   gauss_sigma = None,
                   lin_ip = False):
    """
    Applies all the specified preprocessings to the
    given data.
    """
    modif_data = orig_dat

    # Clean Data
    if clean_args is not None:
        for k in clean_args:
            modif_data = clean_data(orig_dat, *k)            

    # Clip to interval
    if remove_out_int_args is not None:
        remove_out_interval(modif_data, remove_out_int_args)
            
    # Clip to interval
    if clip_to_int_args is not None:
        clip_to_interval(modif_data, clip_to_int_args)

    # Interpolate / Subsample
    [modif_data, dt_init_new] = interpolate_time_series(modif_data, dt_mins, lin_ip=lin_ip)

    # Remove Outliers
    if rem_out_args is not None:
        remove_outliers(modif_data, *rem_out_args)

    # Fill holes
    if hole_fill_args is not None:
        fill_holes_linear_interpolate(modif_data, hole_fill_args)

    # Gaussian Filtering
    if gauss_sigma is not None:
        modif_data = gaussian_filter_ignoring_nans(modif_data, gauss_sigma)

    if all_data is not None:
        if dt_init is None or row_ind is None:
            raise ValueError("Need to provide the initial time of the first series and the column index!")
            
        # Add to rest of data
        add_col(all_data, modif_data, dt_init, dt_init_new, row_ind, dt_mins)
    else:
        if n_tot_cols is None:
            print("Need to know the total number of columns!")
            raise ValueError("Need to know the total number of columns!")

        # Initialize np array for compact storage
        n_data = modif_data.shape[0]
        all_data = np.empty((n_data, n_tot_cols), dtype = np.float32)
        all_data.fill(np.nan)
        all_data[:, 0] = modif_data

    return all_data, dt_init_new

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
    f_mean = np.nanmean(data, axis = 0).reshape((1, n_feat))
    f_std = np.nanstd(data, axis = 0).reshape((1, n_feat))

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
    col_has_nan = np.empty((n, ), dtype = np.bool)
    col_has_nan.fill(False)

    for k in range(m):
        col_has_nan = np.logical_or(col_has_nan, np.isnan(all_data[:,k]))

    return col_has_nan

def cut_into_fixed_len(col_has_nan, seq_len = 20, interleave = True):
    """
    Cuts the time series into pieces of length 'seq_len'
    for training of RNN.
    """

    n = col_has_nan.shape[0]
    indices = np.arange(0, n)

    # Initialize and find first non-NaN
    max_n_seq = n if interleave else n // seq_len
    seqs = np.empty((seq_len, max_n_seq), dtype = np.int32)
    ct = np.where(col_has_nan == False)[0][0]
    seq_count = 0

    while True:
        # Find next NaN
        zers = np.where(col_has_nan[ct:] == True)[0]
        curr_seq_len = n - ct if zers.shape[0] == 0 else zers[0]

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
        nonzs = np.where(col_has_nan[ct:] == False)[0]

        # Break if none found
        if nonzs.shape[0] == 0:
            break
        ct += nonzs[0]

    # Return all found sequences
    return seqs[:, :seq_count]

def cut_data_into_sequences(all_data, seq_len, interleave = True):
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
    out_dat = np.empty((n_seqs, seq_len, n_feat), dtype = np.float32)

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
    rwn = np.int32(np.logical_not(find_rows_with_nans(all_data)))
    true_seq = np.empty((tot_s_len, ), dtype = np.int32)
    true_seq.fill(1)

    # Find sequences of length tot_s_len
    tmp = np.convolve(rwn, true_seq, 'valid')
    inds = np.where(tmp == tot_s_len)[0]
    last_seq_start = inds[-1]

    # Extract
    first_dat = all_data[:last_seq_start, :]
    streak_dat = all_data[last_seq_start:(last_seq_start + tot_s_len), :]
    return first_dat, streak_dat

def cut_and_split(dat, seq_len, streak_len):
    dat_train, dat_test = extract_streak(dat, streak_len, seq_len - 1)
    cut_train_dat = cut_data_into_sequences(dat_train, seq_len, interleave = True)
    cut_test_dat = cut_data_into_sequences(dat_test, seq_len, interleave = True)
    return cut_train_dat, cut_test_dat

#######################################################################################################
# Saving and Loading Processed Data

def save_processed_data(all_data, m, name):
    """
    Saves the processed data in numpy format.
    """
    create_dir(processed_data_path)

    # Get filename
    data_name = os.path.join(processed_data_path, name + "_data.npy")
    meat_data_name = os.path.join(processed_data_path, name + "_mdat.txt")
    np.save(data_name, all_data)
    with open(meat_data_name,'w') as data:
        data.write(str(m))
    return

def load_processed_data(name):
    """
    Loades the processed data in numpy format.
    """

    data_name = os.path.join(processed_data_path, name + "_data.npy")
    meat_data_name = os.path.join(processed_data_path, name + "_mdat.txt")

    if os.path.isfile(data_name) and os.path.isfile(meat_data_name):
        all_data = np.load(data_name)
        with open(meat_data_name, 'r') as data:
            contents = data.read()
            m = literal_eval(contents)
        return all_data, m, name
    else:
        print("Requested files do not exist!")
        return

def load_processed(Data):
    """
    Loads the processed data with the same name if 
    if exists, otherwise raises
    an Exception.
    """
    name = Data.name
    data = load_processed_data(name)
    if data is None:
        raise FileNotFoundError("Water temperature data could not be found.")
    return data

#######################################################################################################
# Full Data Retrieval and Preprocessing

def get_battery_data(save_plot = False, show = True, use_dataset = False):
    """
    Load and interpolate the battery data.
    """
    # Constants
    dt_mins = 15
    name = "Battery"

    # Plot files
    prep_plot_dir = os.path.join(preprocess_plot_path, name)
    create_dir(prep_plot_dir)
    plot_name_before = os.path.join(prep_plot_dir, "raw")
    plot_name_roi = os.path.join(prep_plot_dir, "strange")
    plot_name_after = os.path.join(prep_plot_dir, "processed")

    # Try loading data
    if use_dataset:
        try:
            loaded = Dataset.loadDataset(name)
        except FileNotFoundError:
            loaded = None
    else:
        loaded = load_processed_data(name)
    if loaded is not None:
        return loaded

    # Get data
    dat, m = BatteryData.getData()

    inds = [5, 17, 19, 20, 21, 22, 23, 28, 29, 30]

    soh = [20, 30] # Twice
    max_min_charge = [21, 22] # Idk, in kW
    inds_soc = [19, 23] # SoC and kWh, essentially the same
    soc_max_min = [28, 29] # Also kind of SoC, max and min, shorter time

    use_inds = [19, 17]
    n_feats = len(use_inds)

    # Plot
    x = [dat[i][1] for i in use_inds]
    y = [dat[i][0] for i in use_inds]
    m_used = [m[i] for i in use_inds]
    y_lab = '% / kW'
    t = "Raw Battery Data"
    if save_plot:
        plot_multiple_time_series(x, y, m_used, show = False, title_and_ylab = [t, y_lab], save_name = plot_name_before)

    # Extract ROI of data where it behaves strangely
    d1 = np.datetime64('2019-05-24T12:00')
    d2 = np.datetime64('2019-05-25T12:00')
    x_ext, y_ext = [[extract_date_interval(x[i], y[i], d1, d2)[k] for i in range(n_feats)] for k in range(2)]

    if save_plot:
        plot_multiple_time_series(x_ext, y_ext, m_used, 
                                  show = False, 
                                  title_and_ylab = ["Strange Battery Behavior", y_lab], 
                                  save_name = plot_name_roi)

    # SoC
    all_data, dt_init = pipeline_preps(dat[19], 
                                  dt_mins, 
                                  n_tot_cols = n_feats,
                                  clean_args=[([0.0], 24 * 60, [])],
                                  rem_out_args=(100, [0.0, 100.0]),
                                  lin_ip = True)

    # Active Power
    all_data, _ = pipeline_preps(dat[17], 
                                  dt_mins, 
                                  clean_args=[([], 6 * 60, [])],
                                  all_data=all_data,
                                  dt_init=dt_init,
                                  row_ind=1)
    
    # Metadata
    m_out = [m[19], m[17]]
    add_dt_and_tinit(m_out, dt_mins, dt_init)

    # Standardize
    all_data, m_out = standardize(all_data, m_out)

    if save_plot:
        # Plot
        plot_all(all_data, 
                 m_out, 
                 use_time = True, 
                 show = show, 
                 title_and_ylab = ['Processed Battery Data', y_lab], 
                 scale_back = True,
                 save_name = plot_name_after)

    # Save and return
    if use_dataset:
        bat_dataset = Dataset(all_data, m_out, name, c_inds = np.array([1]), p_inds = np.array([0]))
        bat_dataset.save()
        return bat_dataset
    save_processed_data(all_data, m_out, name)
    return all_data, m_out, name

def get_weather_data(save_plots = True):
    """
    Load and interpolate the weather data.
    """

    # Constants
    dt_mins = 15
    filter_sigma = 2.0
    fill_by_ip_max = 2
    name = "Weather"
    name +=  "" if filter_sigma is None else str(filter_sigma)

    # Plot files
    prep_plot_dir = os.path.join(preprocess_plot_path, name)
    create_dir(prep_plot_dir)
    plot_name_temp = os.path.join(prep_plot_dir, "Outside_Temp")
    plot_name_irrad = os.path.join(prep_plot_dir, "Irradiance")
    plot_name_temp_raw = os.path.join(prep_plot_dir, "Raw Outside_Temp")
    plot_name_irrad_raw = os.path.join(prep_plot_dir, "Raw Irradiance")

    # Try loading data
    try:
        loaded = Dataset.loadDataset(name)
        return loaded
    except FileNotFoundError:
        loaded = None

    # Initialize meta data dict list
    m_out = []

    # Weather data
    dat, m = WeatherData.getData()

    # Add Temperature
    all_data, dt_init = pipeline_preps(dat[0], 
                               dt_mins,
                               clean_args = [([], 30, [])],
                               rem_out_args = None,
                               hole_fill_args = fill_by_ip_max,
                               n_tot_cols = 6,
                               gauss_sigma = filter_sigma)
    add_dt_and_tinit(m, dt_mins, dt_init)
    m_out += [m[0]]

    if save_plots:
        plot_time_series(dat[0][1], dat[0][0], m = m[0], show = False, save_name = plot_name_temp_raw)
        plot_single(all_data[:, 0], 
                    m[0], 
                    use_time = True, 
                    show = False, 
                    title_and_ylab = ['Outside Temperature Processed', m[0]['unit']],
                    save_name = plot_name_temp)


    # Add Irradiance Data
    all_data, _ = pipeline_preps(dat[2], 
                               dt_mins,
                               all_data = all_data,
                               dt_init = dt_init,
                               row_ind = 2,
                               clean_args = [([], 3, [1300.0, 0.0]), ([], 60 * 20)],
                               rem_out_args = None,
                               hole_fill_args = fill_by_ip_max,
                               gauss_sigma = filter_sigma)
    m_out += [m[2]]

    if save_plots:
        plot_time_series(dat[2][1], dat[2][0], m = m[2], show = False, save_name = plot_name_irrad_raw)
        plot_single(all_data[:, 2], 
                    m[2], 
                    use_time = True, 
                    show = False, 
                    title_and_ylab = ['Irradiance Processed', m[2]['unit']],
                    save_name = plot_name_irrad)
    m_out += [m[2]]

    # Save and return
    no_inds = np.array([], dtype = np.int32)
    w_dataset = Dataset(all_data, m_out, name, c_inds = no_inds, p_inds = no_inds)
    w_dataset.save()
    return w_dataset

# Real Data, takes some time to run
# DEPRECATED
def get_heating_data(filter_sigma = None):
    """
    Load and interpolate all the necessary data.
    """

    # Constants
    dt_mins = 15
    fill_by_ip_max = 2
    name = "Room274AndWeather"
    name +=  "" if filter_sigma is None else str(filter_sigma)

    # Try loading data
    loaded = load_processed_data(name)
    if loaded is not None:
        return loaded

    # Initialize meta data dict list
    m_out = []

    # Weather data
    dat, m = WeatherData.getData()

    # Add Temperature
    all_data, dt_init = pipeline_preps(dat[0], 
                               dt_mins,
                               clean_args = [([], 30, [])],
                               rem_out_args = None,
                               hole_fill_args = fill_by_ip_max,
                               n_tot_cols = 6,
                               gauss_sigma = filter_sigma)
    m_out += [m[0]]

    # Add time
    add_time(all_data, dt_init, 1, dt_mins)
    m_out += [{'description': 'time of day', 'unit': str(dt_mins) + ' minutes'}]

    # Add Irradiance Data
    all_data, _ = pipeline_preps(dat[2], 
                               dt_mins,
                               all_data = all_data,
                               dt_init = dt_init,
                               row_ind = 2,
                               clean_args = [([], 3, [1300.0, 0.0]), ([], 60 * 20)],
                               rem_out_args = None,
                               hole_fill_args = fill_by_ip_max,
                               gauss_sigma = filter_sigma)
    m_out += [m[2]]

    # Room Data
    dat, m = Room274Data.getData()

    # Room Temperature
    m_out += [m[1]]
    all_data, _ = pipeline_preps(dat[1], 
                               dt_mins,
                               all_data = all_data,
                               dt_init = dt_init,
                               row_ind = 3,
                               clean_args = [([0.0], 10 * 60)],
                               rem_out_args = (4.5, [10, 100]),
                               hole_fill_args = fill_by_ip_max,
                               gauss_sigma = filter_sigma)
    
    # Windows
    win_dat, m_win = dat[11], m[11]
    m_out += [m_win]
    all_data, _ = pipeline_preps(win_dat, 
                               dt_mins,
                               all_data = all_data,
                               dt_init = dt_init,
                               row_ind = 4,
                               hole_fill_args = fill_by_ip_max,
                               gauss_sigma = filter_sigma)
 
    # Valve
    valve_dat, m_dat = dat[12], m[12]
    m_out += [m_dat]
    all_data, _ = pipeline_preps(valve_dat, 
                               dt_mins,
                               all_data = all_data,
                               dt_init = dt_init,
                               row_ind = 5,
                               hole_fill_args = 3,
                               gauss_sigma = filter_sigma)

    #dat, m = Room272Data.getData()

    for ct, e in enumerate(m_out):
        m_out[ct]['t_init'] = dt_init
        m_out[ct]['dt'] = dt_mins

    # Standardize, save and return
    all_data, m_out = standardize(all_data, m_out)
    save_processed_data(all_data, m_out, name)
    return all_data, m_out, name

def get_DFAB_heating_data(show_plots = False, use_dataset = False):

    data_list = []
    dt_mins = 15
    dfab_rooms_plot_path = os.path.join(preprocess_plot_path, "DFAB")
    create_dir(dfab_rooms_plot_path)

    ################################
    # Single Rooms
    for e in rooms:
        # Get name
        name = e.name

        # Try loading data
        if use_dataset:
            try:
                loaded = Dataset.loadDataset(name)
                data_list += [loaded]
                continue
            except FileNotFoundError:
                loaded = None
        else:
            # Try loading data
            loaded = load_processed_data(name)
            if loaded is not None:
                data_list += [loaded]
                continue

        data, m = e.getData()
        n_cols = len(data)

        # Temperature
        if show_plots:
            temp_plot_file_name = os.path.join(dfab_rooms_plot_path, name) + "_Raw_Temp"
            plot_time_series(data[0][1], data[0][0], m = m[0], show = False, save_name = temp_plot_file_name)
        all_data, dt_init = pipeline_preps(data[0], 
                                           dt_mins, 
                                           n_tot_cols = n_cols,
                                           clean_args = [([0.0], 24 * 60, [])],
                                           rem_out_args = (1.5, None),
                                           gauss_sigma = 5.0)
        add_dt_and_tinit(m, dt_mins, dt_init)
        if show_plots:
            temp_plot_file_name = os.path.join(dfab_rooms_plot_path, name) + "_Temp"
            plot_single(all_data[:, 0], 
                        m[0], 
                        use_time = True, 
                        show = False, 
                        title_and_ylab = ['Temperature Interpolated', m[0]['unit']],
                        save_name = temp_plot_file_name)

        # Valves
        for i in range(3):
            ind = i + 1
            if show_plots:
                valve_plot_file_name = os.path.join(dfab_rooms_plot_path, name + "_Raw_Valve" + str(ind))
                plot_time_series(data[ind][1], data[ind][0], m = m[ind], show = False, save_name = valve_plot_file_name)
            all_data, _ = pipeline_preps(data[ind], 
                                         dt_mins, 
                                         all_data = all_data,
                                         dt_init = dt_init,
                                         row_ind = ind,
                                         clean_args = [([], 30 * 24 * 60, [])])
            if show_plots:
                valve_plot_file_name = os.path.join(dfab_rooms_plot_path, name + "_Valve" + str(ind))
                plot_single(all_data[:, ind], 
                            m[ind], 
                            use_time = True, 
                            show = False, 
                            title_and_ylab = ['Valve ' + str(ind) + ' Interpolated', m[ind]['unit']],
                            save_name = valve_plot_file_name)

        # Blinds
        if n_cols == 5:
            ind = 4
            if show_plots:
                plot_file_name = os.path.join(dfab_rooms_plot_path, name + "_Raw_Blinds")
                plot_time_series(data[ind][1], data[ind][0], m = m[ind], show = False, save_name = plot_file_name)
            all_data, _ = pipeline_preps(data[ind],
                                         dt_mins,
                                         dt_init = dt_init,
                                         all_data = all_data,
                                         clip_to_int_args = [0.0, 100.0],
                                         clean_args = [([], 7 * 24 * 60, [])],
                                         row_ind = ind)
            if show_plots:
                plot_file_name = os.path.join(dfab_rooms_plot_path, name + "_Blinds")
                plot_single(all_data[:, ind], 
                            m[ind], 
                            use_time = True, 
                            show = False, 
                            title_and_ylab = ['Blinds Interpolated', m[ind]['unit']],
                            save_name = plot_file_name)

        # Standardize and save
        if use_dataset:
            dataset = Dataset(all_data, m, name, c_inds = np.array([1]), p_inds = np.array([0]))
            dataset.save()
            data_list += [dataset]
            continue
        all_data, m = standardize(all_data, m)
        save_processed_data(all_data, m, name)
        data_list += [all_data, m, name]

    ################################
    # General Heating Data

    # Get name
    name = DFAB_AddData.name

    # Try loading data
    if use_dataset:
        try:
            loaded = Dataset.loadDataset(name)
        except FileNotFoundError:
            loaded = None
    else:
        loaded = load_processed_data(name)

    if loaded is not None:
        data_list += [loaded]
    else:
        data, m = DFAB_AddData.getData()
        n_cols = len(data)

        # Temperature
        ind = 0
        if show_plots:
            plot_file_name = os.path.join(dfab_rooms_plot_path, name) + "_Raw_Temp_In"
            plot_time_series(data[ind][1], data[ind][0], m = m[ind], show = False, save_name = plot_file_name)
        all_data, dt_init = pipeline_preps(data[ind], 
                                           dt_mins, 
                                           n_tot_cols = n_cols,
                                           remove_out_int_args = [10, 50],
                                           gauss_sigma = 5.0)
        add_dt_and_tinit(m, dt_mins, dt_init)
        if show_plots:
            plot_file_name = os.path.join(dfab_rooms_plot_path, name) + "_Temp_In"
            plot_single(all_data[:, ind], 
                            m[ind], 
                            use_time = True, 
                            show = False, 
                            title_and_ylab = ['In Water Temperature Interpolated', m[ind]['unit']],
                            save_name = plot_file_name)

        ind = 1
        if show_plots:
            plot_file_name = os.path.join(dfab_rooms_plot_path, name) + "_Raw_Temp_Out"
            plot_time_series(data[ind][1], data[ind][0], m = m[ind], show = False, save_name = plot_file_name)
        all_data, _ = pipeline_preps(data[ind], 
                                     dt_mins, 
                                     all_data = all_data,
                                     dt_init = dt_init,
                                     row_ind = 1,
                                     remove_out_int_args = [10, 50],
                                     gauss_sigma = 5.0)
        if show_plots:
            plot_file_name = os.path.join(dfab_rooms_plot_path, name) + "_Temp_Out"
            plot_single(all_data[:, ind], 
                            m[ind], 
                            use_time = True,
                            show = False, 
                            title_and_ylab = ['Out Water Temperature Interpolated', m[ind]['unit']],
                            save_name = plot_file_name)

        # Standardize and save
        if use_dataset:
            all_data, m = standardize(all_data, m)
            no_inds = np.array([], dtype = np.int32)
            dataset = Dataset(all_data, m, name, c_inds = no_inds , p_inds = no_inds)
            dataset.save()
            data_list += [dataset]
        else:
            all_data, m = standardize(all_data, m)
            save_processed_data(all_data, m, name)
            data_list += [[all_data, m, name]]
    
    ################################
    # All Valves Together

    # Get name
    name = DFAB_AllValves.name

    # Try loading data
    if use_dataset:
        try:
            loaded = Dataset.loadDataset(name)
        except FileNotFoundError:
            loaded = None
    else:
        loaded = load_processed_data(name)

    if loaded is not None:
        data_list += [loaded]        
    else:
        data, m = DFAB_AllValves.getData()
        n_cols = len(data)

        # Valves
        for i in range(n_cols):
            ind = i
            if show_plots:
                addid_cols = m[i]['additionalColumns']

                plot_name = "Valve_" + addid_cols['Floor'] + "_" + addid_cols['Device Id']
                plot_path = os.path.join(dfab_rooms_plot_path, "Raw_" + plot_name)
                plot_time_series(data[ind][1], data[ind][0], m = m[ind], show = False, save_name = plot_path)
            if i == 0:
                all_data, dt_init = pipeline_preps(data[ind], 
                                                   dt_mins, 
                                                   n_tot_cols = n_cols,
                                                   clean_args = [([], 30 * 24 * 60, [])])
                add_dt_and_tinit(m, dt_mins, dt_init)
            else:
                all_data, _ = pipeline_preps(data[ind], 
                                             dt_mins, 
                                             all_data = all_data,
                                             dt_init = dt_init,
                                             row_ind = ind,
                                             clean_args = [([], 30 * 24 * 60, [])])
            if show_plots:
                plot_path = os.path.join(dfab_rooms_plot_path, plot_name)
                plot_single(all_data[:, ind], 
                            m[ind], 
                            use_time = True, 
                            show = False, 
                            title_and_ylab = ['Valve ' + str(ind) + ' Interpolated', m[ind]['unit']], save_name = plot_path)

        # Save
        if use_dataset:
            no_inds = np.array([], dtype = np.int32)
            c_inds = np.array(range(n_cols))
            dataset = Dataset(all_data, m, name, c_inds = c_inds , p_inds = no_inds)
            dataset.save()
            data_list += [dataset]
        else:
            save_processed_data(all_data, m, name)
            data_list += [[all_data, m, name]]

    return data_list

def compute_DFAB_energy_usage(show_plots = True):
    """
    Computes the energy usage for every room at DFAB
    using the valves data and the inlet and outlet water
    temperature difference.
    """
    
    # Load data
    w_dat, w_m, w_name = load_processed(DFAB_AddData)
    v_dat, v_m, v_name = load_processed(DFAB_AllValves)
    dt = w_m[0]['dt']

    t_init_w = w_m[0]['t_init']
    t_init_v = v_m[0]['t_init']

    # Load data from Dataset
    w_name = "DFAB_Extra"
    w_dataset = Dataset.loadDataset(w_name)
    w_dat = w_dataset.get_unscaled_data()
    t_init_w = w_dataset.t_init
    dt = w_dataset.dt

    v_name = "DFAB_Valves"
    v_dataset = Dataset.loadDataset(v_name)
    v_dat = v_dataset.data
    t_init_v = v_dataset.t_init

    # Scale back
    #w_dat[:,0] = add_mean_and_std(w_dat[:,0], w_m[0]['mean_and_std'])
    #w_dat[:,1] = add_mean_and_std(w_dat[:,1], w_m[1]['mean_and_std'])

    # Align data
    aligned_data, t_init_new = align_ts(v_dat, w_dat, t_init_v, t_init_w, dt)
    aligned_len = aligned_data.shape[0]
    w_dat = aligned_data[:, -2:]
    v_dat = aligned_data[:, :-2]

    # Room info
    room_dict = {0: "31", 1: "41", 2: "42", 3: "43", 4:"51", 5: "52", 6: "53"}
    valve_room_allocation = np.array(["31", "31", "31", "31", "31", "31", "31", # 3rd Floor
                                     "41", "41", "42", "43", "43", "43", "41", # 4th Floor
                                     "51", "51", "53", "53", "53", "52", "51", # 5th Floor
                                     ])
    n_rooms = len(room_dict)

    tot_n_vals_open = np.sum(v_dat, axis = 1)
    dTemp = w_dat[:, 0] - w_dat[:, 1]

    # Prepare output
    out_dat = np.empty((aligned_len, n_rooms), dtype = np.float32)
    m_list = []

    m_room = {'description': 'Energy consumption room',
                  'unit': 'TBD',
                  'dt': dt,
                  't_init': t_init_new}
    if show_plots:
        dfab_rooms_plot_path = os.path.join(preprocess_plot_path, "DFAB")
        w_plot_path = os.path.join(dfab_rooms_plot_path, w_name + "_WaterTemps")
        dw_plot_path = os.path.join(dfab_rooms_plot_path, w_name + "_WaterTempDiff")
        plot_dataset(w_dataset, show = False, 
                     title_and_ylab = ["Water Temps", "Temperature"],
                     save_name = w_plot_path)
        plot_single(dTemp, m_room, use_time = True, show = False,
                   title_and_ylab = ["Temperature Difference", "DT"], 
                   scale_back = False,
                   save_name = dw_plot_path)

    # Loop over rooms and compute energy
    for id, room_nr in room_dict.items():
        room_valves = v_dat[:, valve_room_allocation == room_nr]
        room_sum_valves = np.sum(room_valves, axis = 1)

        # Divide ignoring division by zero
        room_energy = dTemp * np.divide(room_sum_valves, 
                                tot_n_vals_open, 
                                out = np.zeros_like(room_sum_valves), 
                                where = tot_n_vals_open != 0)

        m_room['description'] = 'Room ' + room_nr
        if show_plots:
            tot_room_valves_plot_path = os.path.join(dfab_rooms_plot_path, "DFAB_" + m_room['description'] + "_Tot_Valves")
            room_energy_plot_path = os.path.join(dfab_rooms_plot_path, "DFAB_" + m_room['description'] + "_Tot_Energy")
            plot_single(room_energy,
                        m_room, 
                        use_time = True,
                        show = False,
                        title_and_ylab = ['Energy Consumption', 'Energy'],
                        save_name = room_energy_plot_path)
            plot_single(room_sum_valves,
                        m_room,
                        use_time = True,
                        show = False,
                        title_and_ylab = ['Sum All Valves', '0/1'],
                        save_name = tot_room_valves_plot_path)

        # Add data to output
        out_dat[:, id] = room_energy
        m_list += [m_room]

    pass

#######################################################################################################
# Dataset definition and generation

class Dataset():
    """
    This class contains all infos about a given dataset.
    """

    def __init__(self, all_data, m, name, c_inds, p_inds):
        """
        Constructor: Puts the important metadata from the
        dict m into member variables.
        """

        self.name = name

        # Dimensions
        self.n = all_data.shape[0]
        self.d = all_data.shape[1]
        self.n_c = c_inds.shape[0]
        self.n_p = p_inds.shape[0]

        # Metadata
        self.dt = m[0]['dt']
        self.t_init = m[0]['t_init']
        self.is_scaled = np.empty((self.d,), dtype = np.bool)
        self.is_scaled.fill(True)
        self.scaling = np.empty((self.d, 2), dtype = np.float32)
        self.descriptions = np.empty((self.d,), dtype = "U100")
        for ct, el in enumerate(m):
            desc = el['description']
            self.descriptions[ct] = desc
            m_a_s = el.get('mean_and_std')
            if m_a_s is not None:
                self.scaling[ct, 0] = m_a_s[0]
                self.scaling[ct, 1] = m_a_s[1]
            else:
                self.is_scaled[ct] = False

        # Actual data
        self.data = all_data
        self.c_inds = c_inds
        self.p_inds = p_inds

    def save(self):
        """
        Save the class object to a file.
        """
        create_dir(dataset_data_path)

        file_name = self.get_filename(self.name)
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
        pass

    def get_unscaled_data(self):
        """
        Adds the mean and std back to every column and returns
        the data.
        """
        data_out = np.copy(self.data)
        for k in range(self.d):
            if self.is_scaled[k]:
                data_out[:,k] = add_mean_and_std(data_out[:,k], self.scaling[k, :])

        return data_out

    @staticmethod
    def get_filename(name):
        return os.path.join(dataset_data_path, name) + '.pkl'

    @staticmethod
    def loadDataset(name):
        """
        Load a saved Dataset object.
        """
        f_name = Dataset.get_filename(name)
        if not os.path.isfile(f_name):
            raise FileNotFoundError("Dataset " + f_name + " does not exist.")
        with open(f_name, 'rb') as f:
            ds = pickle.load(f)
            return ds
    pass

def generateRoomDatasets():
    """
    Gather the right data and put it togeter
    """

    # Get weather
    w_dat, w_m, w_name = get_weather_data()
    w_tinit = w_m[0]['t_init']
    dt = w_m[0]['dt']

    # Get room data
    dfab_data = get_DFAB_heating_data()
    n_rooms = len(rooms)
    dfab_room_data_list = [dfab_data[i] for i in range(n_rooms)]

    # Heating water temperature
    dfab_water_temp = dfab_data[n_rooms]
    wt_dat, wt_m, wt_name = dfab_water_temp
    w_in_dat = wt_dat[:, 0]
    wt_tinit = wt_m[0]['t_init']

    # Single room datasets
    for ct, dat in enumerate(dfab_room_data_list):
        dat_room, m_room, name_room = dat
        n_cols = dat_room.shape[1]
        n = dat_room.shape[0]
        t_init = m_room[0]['t_init']

        # initialize
        dat_res = np.empty((n, n_cols - 2), dtype = dat_room.dtype)

        # Copy temp and blinds and average valve data
        dat_res[:, 0] = dat_room[:,0]
        dat_res[:, 1] = np.mean(dat_room[:, 1:4], axis = 1)
        if n_cols == 5:
            dat_res[:, 2] = dat_room[:, -1]

        # Add inlet water temperature and weather data
        dat_res, t_init_curr = align_ts(dat_res, w_in_dat, t_init, wt_tinit, dt)
        dat_res, t_init_curr = align_ts(dat_res, w_dat, t_init_curr, w_tinit, dt)

        # Add weather data
        m_out = [] + w_m
        m_out[0]['t_init'] = t_init_curr


        print(name_room)
        print(name_room)


    raise NotImplementedError("This still sucks, implement this shit already!")


#######################################################################################################
# Testing

# Test Data
TestData = DataStruct(id_list = [421100171, 421100172],
                      name = "Test",
                      startDate='2019-08-08',
                      endDate='2019-08-09')

class TestDataSynth:
     def getData(self):
         """
         Synthetic and short dataset to be used for debugging.
         """

         # First Time series
         dict1 = {}
         dict1['description'] = "Synthetic Data Series 1: Base Series"
         dict1['unit'] = "Test Unit 1"
         vals1 = np.array([1.0, 2.3, 2.3, 1.2, 2.3, 0.8])
         dats1 = np.array([
                np.datetime64('2005-02-25T03:31'),
                np.datetime64('2005-02-25T03:39'),
                np.datetime64('2005-02-25T03:48'),
                np.datetime64('2005-02-25T04:20'),
                np.datetime64('2005-02-25T04:25'),
                np.datetime64('2005-02-25T04:30'),
             ], dtype = 'datetime64')

         # Second Time series
         dict2 = {}
         dict2['description'] = "Synthetic Data Series 2: Delayed and longer"
         dict2['unit'] = "Test Unit 2"
         vals2 = np.array([1.0, 1.4, 2.1, 1.5, 3.3, 1.8, 2.5])
         dats2 = np.array([
                np.datetime64('2005-02-25T03:51'),
                np.datetime64('2005-02-25T03:59'),
                np.datetime64('2005-02-25T04:17'),
                np.datetime64('2005-02-25T04:21'),
                np.datetime64('2005-02-25T04:34'),
                np.datetime64('2005-02-25T04:55'),
                np.datetime64('2005-02-25T05:01'),
             ], dtype = 'datetime64')

         # Third Time series
         dict3 = {}
         dict3['description'] = "Synthetic Data Series 3: Repeating Values"
         dict3['unit'] = "Test Unit 3"
         vals3 = np.array([0.0, 1.4, 1.4, 0.0, 3.3, 3.3, 3.3, 3.3, 0.0, 3.3, 2.5])
         dats3 = np.array([
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
             ], dtype = 'datetime64')
         return ([(vals1, dats1), (vals2, dats2), (vals3, dats3)], [dict1, dict2, dict3])
TestData2 = TestDataSynth()

# Tests
def get_data_test():

    name = "Test"

    # Test Sequence Cutting
    seq1 = np.array([1,2, np.nan, 1, 2, np.nan, 3, 2, 1, 3, 4, np.nan, np.nan, 3, 2, 3, 1, 3, np.nan, 1, 2])
    seq2 = np.array([3,4, np.nan, np.nan, 2, np.nan, 3, 2, 1, 3, 4, 7, np.nan, 3, 2, 3, 1, np.nan, np.nan, 3, 4])
    n = seq1.shape[0]
    all_dat = np.empty((n, 2), dtype = np.float32)
    all_dat[:,0] = seq1
    all_dat[:,1] = seq2
    print(all_dat)

    # Test
    strk = extract_streak(all_dat, 2, 2)
    print(strk)

    # Test Standardizing
    m = [{}, {}]
    all_dat, m = standardize(all_dat, m)
    print(all_dat)
    print(m)

    # Test Sequence Cutting
    c = find_rows_with_nans(all_dat)
    print(c)
    seq_inds = cut_into_fixed_len(c, 2, interleave = True)
    print(seq_inds)
    od = cut_data_into_sequences(all_dat, 2, interleave = True)
    print(od)

    # Test hole filling by interpolation
    test_ts = np.array([np.nan, np.nan, 1, 2, 3.5, np.nan, 4.5, np.nan])
    test_ts2 = np.array([1, 2, 3.5, np.nan, np.nan, 5.0, 5.0, np.nan, 7.0])
    fill_holes_linear_interpolate(test_ts, 1)
    fill_holes_linear_interpolate(test_ts2, 2)
    print(test_ts)
    print(test_ts2)

    # Test Outlier Removal
    test_ts3 = np.array([1, 2, 3.5, np.nan, np.nan, 5.0, 5.0, 17.0, 5.0, 2.0, -1.0, np.nan, 7.0, 7.0, 17.0, np.nan, 20.0, 5.0, 6.0])
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
    all_data = np.empty((n_data, 3), dtype = np.float32)
    all_data.fill(np.nan)

    # Add data
    add_time(all_data, dt_init1, 0, dt_mins)
    all_data[:,1] = data1
    [data2, dt_init2] = interpolate_time_series(dat[1], dt_mins)

    print(data2)
    data2 = gaussian_filter_ignoring_nans(data2)
    print(data2)

    add_col(all_data, data2, dt_init1, dt_init2, 2, dt_mins)

    save_processed_data(all_data, m, name)
    d, m_new, name = load_processed_data(name)
    print(m_new[0])
    return all_data, m, name

def test_plotting_withDFAB_data():

    data_list = get_DFAB_heating_data()
    all_data, metadata, name = data_list[-1]
    data, m = DFAB_AddData.getData()

    # Plot test
    ind = 0
    plot_time_series(data[ind][1], data[ind][0], m = metadata[ind], show = False)

    plot_single_ip_ts(all_data[:, ind], 
               lab = 'Out Water Temp',
               show = False,
               mean_and_std = metadata[ind]['mean_and_std'], 
               use_time = False, 
               title_and_ylab = ['Single TS Plot Test', metadata[ind]['unit']],
               dt_mins = metadata[ind]['dt']
               )
    
    plot_single_ip_ts(all_data[:, ind], 
               lab = 'Out Water Temp', 
               show = False, 
               mean_and_std = metadata[ind]['mean_and_std'], 
               use_time = True,               
               title_and_ylab = ['Single TS with Dates Plot Test', metadata[ind]['unit']],
               dt_mins = metadata[ind]['dt'],
               dt_init_str = metadata[ind]['t_init']
               )

    m = metadata
    plot_multiple_ip_ts([all_data[:, 0], all_data[:, 1]],
                        lab_list = [m[i]['description'] for i in range(2)],
                        mean_and_std_list = [m[i]['mean_and_std'] for i in range(2)],
                        use_time = True,
                        timestep_offset_list = [-1, 1],
                        dt_init_str_list = [m[i]['t_init'] for i in range(2)],
                        show_last = False, 
                        title_and_ylab = ['Two TS with Dates Plot Test', m[ind]['unit']],
        )

    plot_all(all_data, m, show = True, title_and_ylab = ['Two TS with Dates High Level Plot Test', m[ind]['unit']])

def test_dataset_with_DFAB():
    name = 'Test'

    # Get part of DFAB data
    data_list = get_DFAB_heating_data()
    all_data, metadata, _ = data_list[-1]

    s = all_data.shape
    c_inds = np.array([s[1] - 1])
    p_inds = np.array([s[1] - 2])
    ds = Dataset(all_data, metadata, name, c_inds, p_inds)
    ds.save()

    ds_new = Dataset.loadDataset(name)

    compute_DFAB_energy_usage()

