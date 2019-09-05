
import os

import numpy as np
import pandas as pd

from datetime import datetime
from visualize import plot_time_series, plot_ip_time_series
from restclient import DataStruct

#######################################################################################################
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

# Weather Data
WeatherData = DataStruct(id_list = [3200000,
                                    3200002,
                                    3200008],
                        name = "Weather",
                        startDate='2018-01-01',
                        endDate='2019-12-31')

#######################################################################################################
# Time Series Processing

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
    #count -= 1
    print(tot_dat - count, "data points removed.")
    return [new_vals[:count], new_dates[:count]]

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

def interpolate_time_series(dat, dt_mins):
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
                new_vals[count] = curr_val
                count += 1
                curr_val = (curr_dt - curr_upper_lim) / interv * v
            else:
                # Data missing!                
                curr_val += (curr_upper_lim - last_dt) / interv * last_val
                new_vals[count] = curr_val
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

def pipeline_preps(orig_dat,
                   dt_mins,
                   all_data = None,
                   dt_init = None,
                   row_ind = None,
                   clean_args = None,
                   rem_out_args = None,
                   hole_fill_args = None,
                   n_tot_cols = None):
    """
    Applies all the specified preprocessings to the
    given data.
    """
    modif_data = orig_dat

    # Clean Data
    if clean_args is not None:
        for k in clean_args:
            modif_data = clean_data(orig_dat, *k)

    # Interpolate / Subsample
    [modif_data, dt_init_new] = interpolate_time_series(modif_data, dt_mins)

    # Remove Outliers
    if rem_out_args is not None:
        remove_outliers(modif_data, *rem_out_args)

    # Fill holes
    if hole_fill_args is not None:
        fill_holes_linear_interpolate(modif_data, hole_fill_args)

    if all_data is not None:
        if dt_init is None or row_ind is None:
            print("Need to provide the initial time of the first series and the column index!")
            return
        # Add to rest of data
        add_col(all_data, modif_data, dt_init, dt_init_new, row_ind, dt_mins)
    else:
        if n_tot_cols is None:
            print("Need to know the total number of columns!")
            return

        # Initialize np array for compact storage
        n_data = modif_data.shape[0]
        all_data = np.empty((n_data, n_tot_cols), dtype = np.float32)
        all_data.fill(np.nan)
        all_data[:, 0] = modif_data
    return all_data, dt_init_new

#######################################################################################################
# Preparing Data for model fitting

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

def cut_into_fixed_len(all_data, seq_len = 20, interleave = False):
    """
    Cuts the time series into pieces of length 'seq_len'
    for training of RNN.
    """


    if interleave:
        print("Not yet implemented!")

    n = all_data.shape[0]
    indices = np.arange(0, n)
    col_has_nan = find_rows_with_nans(all_data)

    # Initialize and find first non-NaN
    seqs = np.empty((seq_len, n // seq_len), dtype = np.float32)
    ct = np.where(col_has_nan == False)[0][0]
    seq_count = 0

    while True:
        # Find next NaN
        zers = np.where(col_has_nan[ct:] == True)[0]
        curr_seq_len = n - ct if zers.shape[0] == 0 else zers[0]
        n_seq_curr = curr_seq_len // seq_len

        # Add sequences
        for k in range(n_seq_curr):
            seqs[:, seq_count] = indices[(ct + k * seq_len):(ct + (k + 1) * seq_len)]
            seq_count += 1

        # Find next True
        ct += curr_seq_len
        nonzs = np.where(col_has_nan[ct:] == False)[0]

        # Break if none found
        if nonzs.shape[0] == 0:
            break
        ct += nonzs[0]

    # Return all found sequences
    return seqs[:, :seq_count]

#######################################################################################################
# Full Data Retrieval and Preprocessing

# Testing
def get_data_test():

    # Test Sequence Cutting
    seq1 = np.array([1,2, np.nan, 1, 2, np.nan, 3, 2, 1, 3, 4, np.nan, np.nan, 3, 2, 3, 1, 3, np.nan, 1, 2])
    seq2 = np.array([3,4, np.nan, np.nan, 2, np.nan, 3, 2, 1, 3, 4, 7, np.nan, 3, 2, 3, 1, np.nan, np.nan, 3, 4])
    n = seq1.shape[0]
    all_dat = np.empty((n, 2), dtype = np.float32)
    all_dat[:,0] = seq1
    all_dat[:,1] = seq2
    print(all_dat)
    c = find_rows_with_nans(all_dat)    
    print(c)
    seq_inds = cut_into_fixed_len(all_dat, 2)
    print(seq_inds)
    return "Fuck"

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
    plot_time_series(dat[0][1], dat[0][0], m[0], show = False)
    plot_time_series(dat[1][1], dat[1][0], m[1], show = False)
    plot_time_series(dat[2][1], dat[2][0], m[2], show = False)

    # Clean data
    mod_dat = clean_data(dat[2], [0.0], 4, [3.3])
    plot_time_series(mod_dat[1], mod_dat[0], m[2], show = True)

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
    add_col(all_data, data2, dt_init1, dt_init2, 2, dt_mins)

    return all_data, m

# Real Data, takes some time to run
def get_all_relevant_data(dt_mins = 15, fill_by_ip_max = 2):
    """
    Load and interpolate all the necessary data.
    """
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
                               n_tot_cols = 6)
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
                               hole_fill_args = fill_by_ip_max)
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
                               hole_fill_args = fill_by_ip_max)
    
    # Windows
    win_dat, m_win = dat[11], m[11]
    m_out += [m_win]
    all_data, _ = pipeline_preps(win_dat, 
                               dt_mins,
                               all_data = all_data,
                               dt_init = dt_init,
                               row_ind = 4,
                               hole_fill_args = fill_by_ip_max)
 
    # Valve
    valve_dat, m_dat = dat[12], m[12]
    m_out += [m_dat]
    all_data, _ = pipeline_preps(valve_dat, 
                               dt_mins,
                               all_data = all_data,
                               dt_init = dt_init,
                               row_ind = 5,
                               hole_fill_args = 3)

    #dat, m = Room272Data.getData()

    return all_data, m_out







    
