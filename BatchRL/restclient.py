###################################################################################################
# Name: rest client
# Version: 0.1
# Description: This client provides a possible solution if you want to
# connect to our REST API of the nest database from your local computer


# Activities:                                   Author:                         Date:
# Initial comment                               RK                              20190425
# Modified timestamp                            RK                              20190529
# Fixed conversion to datetime, added local 
# data storage and added password GUI.          CB                              20190821
# Added meta data retrieval                     CB                              20190904
###################################################################################################

import os
import time
from ast import literal_eval
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import requests

USE_CL: bool = True
if not USE_CL:
    from pw_gui import get_pw
else:
    from pw_cl import get_pw

# Where to put the local copy of the data
save_dir: str = '../Data/'


def get_data_folder(name: str, start_date: str, end_date: str) -> str:
    """
    Defines the naming of the data directory given
    the name and the dates.
    """
    ext = start_date + "__" + end_date + "__"
    data_dir = os.path.join(save_dir, ext + name)
    return data_dir


def read_offline(name: str, start_date: str = '2019-01-01', end_date: str = '2019-12-31') -> Tuple[List, List]:
    """
    Read numpy data that has already been created.
    """

    # Get folder name
    data_dir = get_data_folder(name, start_date, end_date)

    # Count files
    ct = 0
    for f in os.listdir(data_dir):
        file_path = os.path.join(data_dir, f)
        if f[:5] == "dates":
            ct += 1

    # Loop over files in directory and insert data into lists
    val_list = [None] * ct
    ts_list = [None] * ct
    meta_list = [None] * ct
    for k in range(ct):
        val_list[k] = np.load(os.path.join(data_dir, "values_" + str(k) + ".npy"))
        ts_list[k] = np.load(os.path.join(data_dir, "dates_" + str(k) + ".npy"))
        with open(os.path.join(data_dir, "meta_" + str(k) + ".txt"), 'r') as data:
            contents = data.read()
            meta_list[k] = literal_eval(contents)

    # Transform to list of pairs and return
    list_of_tuples = list(zip(val_list, ts_list))
    return list_of_tuples, meta_list


class Client(object):
    """
    Client for data retrieval from local disk or
    from SQL data base of NEST. Once loaded from the
    server, it can be stored to and reloaded from
    the local disk.
    """

    def __init__(self,
                 domain: str = 'nest.local',
                 url: str = 'https://visualizer.nestcollaboration.ch/Backend/api/v1/datapoints/'):
        """
        Initialize parameters and empty data containers.
        :param domain: Use default.
        :param url: Use default.
        """
        self.domain = domain
        self.url = url
        self.save_dir = save_dir
        self.start_date = None
        self.end_date = None
        self.np_data = []
        self.meta_data = []
        self.auth = None

    def read(self, df_data: List[str],
             start_date: str = '2019-01-01',
             end_date: str = '2019-12-31'
             ) -> Optional[Tuple[List, List]]:
        """
        Reads data defined by the list of column IDs df_data
        that was acquired between startDate and endDate.
        :param df_data: List of IDs in string format.
        :param start_date: Starting date in string format.
        :param end_date: End date in string format.
        :return: (List[(Values, Dates)], List[Metadata])
        """

        # https://github.com/brandond/requests-negotiate-sspi/blob/master/README.md
        from requests_negotiate_sspi import HttpNegotiateAuth

        self.start_date = start_date
        self.end_date = end_date

        # Check Login
        username, pw = get_pw()
        self.auth = HttpNegotiateAuth(domain=self.domain,
                                      username=username,
                                      password=pw)
        s = requests.Session()
        try:
            # This fails if the username exists but the password
            # is wrong, but not if the username does not exist?!!
            r = s.get(url=self.url, auth=self.auth)
        except TypeError as e:
            print("Login failed, invalid password!")
            return None
        except Exception as e:
            print(e)
            print("A problem occurred!")
            return None

        # Check if login valid.
        if r.status_code != requests.codes.ok:
            print("Login failed, invalid username!")
            return None
        print(time.ctime() + ' REST client login successful.')

        # Iterate over column IDs
        for ct, column in enumerate(df_data):
            url = self.url + column
            meta_data = s.get(url=url).json()
            self.meta_data += [meta_data]
            url += '/timeline?startDate=' + start_date + '&endDate=' + end_date
            df = pd.DataFrame(data=s.get(url=url).json())

            # Convert to Numpy
            values = df.loc[:, "value"].to_numpy()
            ts = pd.to_datetime(df.loc[:, "timestamp"])
            ts = ts.to_numpy(dtype=np.datetime64)
            self.np_data += [(values, ts)]
            print("Added column", ct + 1, "with ID", column)

        print(time.ctime() + ' REST client data acquired')
        return self.np_data, self.meta_data

    def write_np(self, name: str, overwrite: bool = False):
        """
        Writes the read data in numpy format
        to files.
        """

        print("Writing Data to local disk.")

        # Create directory
        if self.start_date is None:
            print("Read data first!!")
            return
        data_dir = get_data_folder(name, self.start_date, self.end_date)
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        elif overwrite:
            print("Not implemented, remove manually and try again.")
        else:
            print("Directory already exists.")
            return

        # Loop over data and save column-wise
        for ct, data_tup in enumerate(self.np_data):
            v, t = data_tup
            v_name = os.path.join(data_dir, 'values_' + str(ct) + '.npy')
            np.save(v_name, v)
            d_name = os.path.join(data_dir, 'dates_' + str(ct) + '.npy')
            np.save(d_name, t)
            meta_name = os.path.join(data_dir, 'meta_' + str(ct) + '.txt')
            with open(meta_name, 'w') as data:
                data.write(str(self.meta_data[ct]))
        return


class DataStruct:
    """
    Base Class for different sets of data columns
    defined by successive IDs, a name and a date range.
    """

    def __init__(self,
                 id_list: List[int],
                 name: str,
                 start_date: str = '2019-01-01',
                 end_date: str = '2019-12-31'):
        """
        Initialize DataStruct.
        :param id_list: IDs of the data series.
        :param name: Name of the collection of data series.
        :param start_date: Begin of time interval.
        :param end_date: End of time interval.
        """
        # Initialize values
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.client_loaded = False
        self.REST = None

        # Convert elements of id_list to strings.
        self.data_ids = [str(e) for e in id_list]

    def load_client(self) -> None:
        """
        Loads the REST client if not yet loaded.
        :return: None
        """
        if not self.client_loaded:
            self.REST = Client()
            self.client_loaded = True

    def getData(self) -> Optional[Tuple[List, List]]:
        """
        If the data is not found locally it is
        retrieved from the SQL database, otherwise
        the local data is read and returned.
        :return: (List[(values: np.ndarray, timestamps: np.ndarray)], List[metadata: Dict])
        """

        self.load_client()
        data_folder = get_data_folder(self.name, self.start_date, self.end_date)
        if not os.path.isdir(data_folder):
            # Read from SQL database and write for later use
            ret_val, meta_data = self.REST.read(self.data_ids,
                                                start_date=self.start_date,
                                                end_date=self.end_date)
            if ret_val is None:
                return None
            self.REST.write_np(self.name)
        else:
            # Read locally
            ret_val, meta_data = read_offline(self.name,
                                              start_date=self.start_date,
                                              end_date=self.end_date)
        return ret_val, meta_data
    pass


def example():
    """
    Example usage of above code.
    :return: None
    """
    # Example data.
    test_data = DataStruct(
        id_list=[421100171, 421100172],
        name="Test",
        start_date='2019-08-08',
        end_date='2019-08-09'
    )

    # Get data from SQL 
    data, metadata = test_data.getData()

    # Get data corresponding to first ID (421100171)
    values, timestamps = data[0]

    # Do something with the data
    # Add your code here...
    return
