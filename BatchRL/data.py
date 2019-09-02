
import os

import numpy as np
import pandas as pd

from datetime import datetime

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

# Test data, small part of the electricity data,
# for testing since access is faster.
TestData = DataStruct(
            id_list = [421100171, 421100172],
            name = "Test",
            startDate='2019-08-08',
            endDate='2019-08-09'
            )



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

    interpolate_time_series(dat, 15)

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
    count = 0
    curr_val = 0
    last_dt = dates[0]
    last_val = vals[0]
    curr_val = (last_dt - start_dt) / interv * last_val

    for ct, v in enumerate(vals):

        curr_dt = dates[ct]
        curr_upper_lim = start_dt + count * interv
        if curr_dt > curr_upper_lim:
            if curr_dt <= curr_upper_lim + interv:
                # Next datetime in next interval
                curr_val += (curr_upper_lim - last_dt) / interv * v
                new_vals[count] = curr_val
                count += 1
                curr_val = (curr_dt - curr_upper_lim) / interv * v
            else:
                # Data missing!
                print("Missing data")
                curr_val += (curr_upper_lim - last_dt) / interv * last_val
                count += 1
                n_data_missing = (curr_dt - curr_upper_lim) // interv
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


    return [new_vals, start_dt]

    



    
