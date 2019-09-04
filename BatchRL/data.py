
import os

import numpy as np
import pandas as pd

from datetime import datetime
from visualize import plot_time_series
import restclient

class DataStruct:
    """
    Base Class for different sets of data columns
    defined by successive IDs.
    """
    def __init__(self,
                 id_list,
                 name, 
                 startDate='2019-01-01',
                 endDate='2019-12-31'):

        # Initialize values and client
        self.name = name
        self.startDate = startDate
        self.endDate = endDate

        # Convert elements of id_list to strings.
        for ct, el in enumerate(id_list):
            id_list[ct] = str(el)
        self.data_ids = id_list
        self.client_loaded = False
        pass

    def load_client(self):
        """
        Loads the REST client if not yet loaded.
        """
        if not self.client_loaded:
            self.REST = restclient.client()
            self.client_loaded = True

    def getData(self):
        """
        If the data is not found locally it is 
        retrieved from the SQL database, otherwise 
        the local data is read and returned.
        Returns (list((np.array(vals), np.array(timestamps))), list(dict()))
        """

        self.load_client()
        data_folder = self.REST.get_data_folder(self.name, self.startDate, self.endDate)
        if not os.path.isdir(data_folder):
            # Read from SQL database and write for later use
            ret_val, meta_data = self.REST.read(self.data_ids, 
                                     startDate = self.startDate, 
                                     endDate = self.endDate)
            if ret_val is None:
                return None
            self.REST.write_np(self.name)
        else:
            # Read locally
            ret_val, meta_data = self.REST.read_offline(self.name,  
                                             startDate = self.startDate, 
                                             endDate = self.endDate)
        return (ret_val, meta_data)

    pass

###########################################################################
# Test Data for debugging


# Test data, small part of the electricity data,
# for testing since access is faster.
TestData = DataStruct(
            id_list = [421100171, 421100172],
            name = "Test",
            startDate='2019-08-08',
            endDate='2019-08-09'
            )
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


###########################################################################
# UMAR Data
Room272Data = DataStruct(
            id_list = [42150280,
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
            endDate='2019-12-31'
            )


Room274Data = DataStruct(
            id_list = [42150281,
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
            endDate='2019-12-31'
            )

WeatherData = DataStruct(
            id_list = [3200000,
                       3200002,
                       3200008
                       ],
            name = "Weather",
            startDate='2018-01-01',
            endDate='2019-12-31'
            )



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


def floor_datetime_to_min(dt, mt):
    """
    Rounds deltatime64 dt down to mt minutes.
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
        all_data[k, 0] = (start_t + k) % n_ts_per_day
    return


def clean_data(dat, rem_vals = [], n_cons_least = 2, const_excepts = []):
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


def get_data_test():

    dt_mins = 15
    dat, m = TestData2.getData()
    plot_time_series(dat[0][1], dat[0][0], m[0], show = False)
    plot_time_series(dat[1][1], dat[1][0], m[1], show = False)
    plot_time_series(dat[2][1], dat[2][0], m[2], show = False)

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

def get_all_relevant_data(dt_mins = 15):
    """
    Load and interpolate all the necessary data.
    """
    # Initialize meta data dict list
    m_out = []

    # Weather data
    dat, m = WeatherData.getData()
    [amb_temp, dt_init] = interpolate_time_series(dat[0], dt_mins)
    n_data = amb_temp.shape[0]
    print(n_data, "total data points.")

    # Initialize np array for compact storage
    all_data = np.empty((n_data, 5), dtype = np.float32)
    all_data.fill(np.nan)

    # Add time and weather
    add_time(all_data, dt_init, 0, dt_mins)
    m_out += [{'description': 'time of day', 'unit': str(dt_mins) + ' minutes'}]
    all_data[:, 1] = amb_temp
    m_out += [m[0]]

    # Irradiance data
    [irradiance, dt_irr_init] = interpolate_time_series(dat[2], dt_mins)
    add_col(all_data, irradiance, dt_init, dt_irr_init, 2, dt_mins)
    m_out += [m[2]]

    # Room data
    dat, m = Room274Data.getData()
    [temp, dt_temp_init] = interpolate_time_series(dat[1], dt_mins)
    add_col(all_data, temp, dt_init, dt_temp_init, 3, dt_mins)
    m_out += [m[1]]

    #dat, m = Room272Data.getData()

    return all_data, m_out







    
