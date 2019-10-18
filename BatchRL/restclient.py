"""REST client for data retrieval from NEST database.

This module can be used to download and save data
from the NEST database. Returns the values and the
timesteps in numpy format for each defined data series.
The following example shows how to use this module.

Example usage::

    # Define data.
    test_data = DataStruct(
        id_list=[421100171, 421100172],
        name="Test",
        start_date='2019-08-08',
        end_date='2019-08-09'
    )

    # Get data from SQL database
    data, metadata = test_data.getData()

    # Get data corresponding to first ID (421100171)
    values, timestamps = data[0]

    # Do something with the data
    # Add your code here...
    print(values, timestamps)

Written by Christian Baumann ans Ralf Knechtle @ Empa, 2019
"""

import os
import time
from ast import literal_eval
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import requests

USE_CL: bool = True  #: Whether to use the command line for the login.
if not USE_CL:
    from pw_gui import get_pw
else:
    from pw_cl import get_pw

#: Where to put the local copy of the data.
save_dir: str = '../Data/'


def _get_data_folder(name: str, start_date: str, end_date: str) -> str:
    """
    Defines the naming of the data directory given
    the name and the dates.

    Args:
        name: Name of data.
        start_date: Start of data collection.
        end_date: End of data collection.

    Returns:
        Full path of data defined by params.
    """
    ext = start_date + "__" + end_date + "__"
    data_dir = os.path.join(save_dir, ext + name)
    return data_dir


class _Client(object):
    """Client for data retrieval.

    Reads from local disk if it already exists or else
    from SQL data base of NEST. Once loaded from the
    server, it can be stored to and reloaded from
    the local disk.
    """

    _DOMAIN: str = 'nest.local'
    _URL: str = 'https://visualizer.nestcollaboration.ch/Backend/api/v1/datapoints/'

    def __init__(self,
                 name,
                 start_date: str = '2019-01-01',
                 end_date: str = '2019-12-31'):
        """
        Initialize parameters and empty data containers.

        :param name: Name of the data.
        :param start_date: Starting date in string format.
        :param end_date: End date in string format.
        """
        self.save_dir = save_dir
        self.start_date = start_date
        self.end_date = end_date
        self.np_data = []
        self.meta_data = []
        self.auth = None
        self.name = name

    def read(self, df_data: List[str]) -> Optional[Tuple[List, List]]:
        """
        Reads data defined by the list of column IDs df_data
        that was acquired between startDate and endDate.

        Args:
            df_data: List of IDs in string format.

        Returns:
            (List[(Values, Dates)], List[Metadata])
        """

        # https://github.com/brandond/requests-negotiate-sspi/blob/master/README.md
        from requests_negotiate_sspi import HttpNegotiateAuth

        # Check Login
        username, pw = get_pw()
        self.auth = HttpNegotiateAuth(domain=self._DOMAIN,
                                      username=username,
                                      password=pw)
        s = requests.Session()
        try:
            # This fails if the username exists but the password
            # is wrong, but not if the username does not exist?!!
            r = s.get(url=self._URL, auth=self.auth)
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
            url = self._URL + column
            meta_data = s.get(url=url).json()
            self.meta_data += [meta_data]
            url += '/timeline?startDate=' + self.start_date + '&endDate=' + self.end_date
            df = pd.DataFrame(data=s.get(url=url).json())

            # Convert to Numpy
            values = df.loc[:, "value"].to_numpy()
            ts = pd.to_datetime(df.loc[:, "timestamp"])
            ts = ts.to_numpy(dtype=np.datetime64)
            self.np_data += [(values, ts)]
            print("Added column {} with ID {}.".format(ct + 1, column))

        print(time.ctime() + ' REST client data acquired')
        return self.np_data, self.meta_data

    def read_offline(self) -> Tuple[List, List]:
        """Read numpy and text data that has already been created.

        Returns:
             values, dates and metadata.
        """

        # Get folder name
        data_dir = _get_data_folder(self.name, self.start_date, self.end_date)

        # Count files
        ct = 0
        for f in os.listdir(data_dir):
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

    def write_np(self, overwrite: bool = False):
        """
        Writes the read data in numpy format
        to files.

        Args:
            overwrite: Whether to overwrite existing data with same name.

        Returns:
            None
        """
        name = self.name
        print("Writing Data to local disk.")

        # Create directory
        if self.start_date is None:
            raise ValueError("Read data first!!")
        data_dir = _get_data_folder(name, self.start_date, self.end_date)
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        elif overwrite:
            raise NotImplementedError("Not implemented, remove manually and try again.")
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
    """Main Class for data retrieval from NEST database.

    The data is defined when initializing the class
    by a list of IDs, a name and a date range.
    The method `getData` then retrieves the data when needed.
    Once read, the data is cached in `save_dir` for faster
    access if read again.
    """

    def __init__(self,
                 id_list: List[int],
                 name: str,
                 start_date: str = '2019-01-01',
                 end_date: str = '2019-12-31'):
        """Initialize DataStruct.

        Args:
            id_list: IDs of the data series.
            name: Name of the collection of data series.
            start_date: Begin of time interval.
            end_date: End of time interval.
        """
        # Initialize values
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.REST = _Client(self.name, self.start_date, self.end_date)

        # Convert elements of id_list to strings.
        self.data_ids = [str(e) for e in id_list]

    def get_data_folder(self) -> str:
        """Returns path to data.

        Returns the path of the directory where
        the data has been / will be stored.

        Returns:
            Full path to directory of data.
        """
        return _get_data_folder(self.name, self.start_date, self.end_date)

    def getData(self) -> Optional[Tuple[List, List]]:
        """Get the data associated with the DataStruct

        If the data is not found locally it is
        retrieved from the SQL database and saved locally, otherwise
        the local data is read and returned.

        Returns:
            (List[(values: np.ndarray, timestamps: np.ndarray)], List[metadata: Dict])
        """

        data_folder = self.get_data_folder()
        if not os.path.isdir(data_folder):
            # Read from SQL database and write for later use
            ret_val, meta_data = self.REST.read(self.data_ids)
            if ret_val is None:
                return None
            self.REST.write_np()
        else:
            # Read locally
            ret_val, meta_data = self.REST.read_offline()
        return ret_val, meta_data


def example():
    """Example usage of REST client.

    Shows you how to use the `DataStruct` class
    to define the data and retrieve it.

    Returns:
        None
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
    print(values, timestamps)
