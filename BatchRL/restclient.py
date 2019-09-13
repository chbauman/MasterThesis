#########################################################################################################################
# Name: rest client
# Version: 0.1
# Description: This client provides a possible solution if you want to connect to our REST API of the nest database from your local computer


# Activities:                                   Author:                         Date:
# Initial comment                               RK                              20190425
# Modified timestamp                            RK                              20190529
# Fixed conversion to datetime, added local 
# data storage and added password GUI.          CB                              20190821
# Added meta data retrieval                     CB                              20190904

########################################################################################################################

import requests
import time
import os

from ast import literal_eval

##https://github.com/brandond/requests-negotiate-sspi/blob/master/README.md
from requests_negotiate_sspi import HttpNegotiateAuth 

import pandas as pd
import numpy as np

use_cl = True
if not use_cl:
    from pw_gui import getPW
else:
    from pw_cl import getPW

# Where to put the local copy of the data
save_dir = '../Data/'


class client(object):
    """
    Client for data retrieval from local disk or
    from SQL data base of NEST. Once loaded from the
    server, it can be stored to and reloaded from
    the local disk.
    """
    def __init__(self, 
                 domain='nest.local', 
                 url='https://visualizer.nestcollaboration.ch/Backend/api/v1/datapoints/'):
        """
        Initialize parameters and empty data containers.
        """
        self.domain = domain
        self.url = url
        self.save_dir = save_dir
        self.startDate = None
        self.np_data = []
        self.meta_data = []
        pass

    def read(self, df_data=[], startDate='2019-01-01', endDate='2019-12-31'):
        """
        Reads data defined by the list of column IDs df_data
        that was acquired between startDate and endDate.
        """

        self.startDate = startDate
        self.endDate = endDate

        # Check Login
        log_data = getPW()
        self.username = log_data[0]
        self.password = log_data[1]
        self.auth = HttpNegotiateAuth(domain=self.domain, 
                                      username=self.username, 
                                      password=self.password)
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
            print("A problem ocurred")
            return None

        # Check if login valid.
        if r.status_code != requests.codes.ok:
            print("Login failed, invalid username!")
            return None
        print(time.ctime() + ' REST client login successfull')

        # Iterate over column IDs
        for ct, column in enumerate(df_data):
            url = self.url + column 
            meta_data = s.get(url=url).json()
            self.meta_data += [meta_data]
            url += '/timeline?startDate=' + startDate + '&endDate=' + endDate
            df = pd.DataFrame(data=s.get(url=url).json())

            # Convert to Numpy
            vals = df.loc[:, "value"].to_numpy()
            ts = pd.to_datetime(df.loc[:, "timestamp"])
            ts = ts.to_numpy(dtype=np.datetime64)
            self.np_data += [(vals, ts)]
            print("Added column", ct + 1, "with ID", column)

        print(time.ctime() + ' REST client data acquired')
        return self.np_data, self.meta_data

    def get_data_folder(self, name, startDate, endDate):
        """
        Defines the naming of the data directory given
        the name and the dates.
        """
        ext = startDate + "__" + endDate + "__"
        data_dir = os.path.join(save_dir, ext + name)
        return data_dir

    def write_np(self, name, overwrite = False):
        """
        Writes the read data in numpy format
        to files.
        """

        print("Writing Data to local disk.")

        # Create directory
        if self.startDate is None:
            print("Read data first!!")
            return
        data_dir = self.get_data_folder(name, self.startDate, self.endDate)
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        elif overwrite:
            print("Not implemented, remove manually and try again.")
        else:
            print("Directory already exists.")
            return

        # Loop over data and save columnwise
        for ct, data_tup in enumerate(self.np_data):
            v, t = data_tup
            v_name = os.path.join(data_dir, 'values_' + str(ct) + '.npy')
            np.save(v_name, v)
            d_name = os.path.join(data_dir, 'dates_' + str(ct) + '.npy')
            np.save(d_name, t)
            meta_name = os.path.join(data_dir, 'meta_' + str(ct) + '.txt')
            with open(meta_name,'w') as data:
                data.write(str(self.meta_data[ct]))

        return

    def read_offline(self, name, startDate='2019-01-01', endDate='2019-12-31'):
        """
        Read numpy data that has already been created.
        """

        # Get folder name
        data_dir = self.get_data_folder(name, startDate, endDate)

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
        return (list_of_tuples, meta_list)



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
            self.REST = client()
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


def example():

    ###########################################################################
    # Example data.
    TestData = DataStruct(
                id_list = [421100171, 421100172],
                name = "Test",
                startDate='2019-08-08',
                endDate='2019-08-09'
                )

    # Get data from SQL 
    data, metadata = TestData.getData()

    # Get data corresponding to first ID (421100171)
    vals, timestamps = data[0]

    # Do something with the data
    # Add your code here...
    pass