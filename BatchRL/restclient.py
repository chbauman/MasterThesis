#########################################################################################################################
# Name: rest client
# Version: 0.1
# Description: This client provides a possible solution if you want to connect to our REST API of the nest database from your local computer


# Activities:                                   Author:                         Date:
# Initial comment                               RK                              20190425
# Modified timestamp                            RK                              20190529

########################################################################################################################

import requests
from requests_negotiate_sspi import HttpNegotiateAuth ##https://github.com/brandond/requests-negotiate-sspi/blob/master/README.md
import pandas as pd
import numpy as np
import time
import os

import wx
from pw_gui import getPW

save_dir = '../Data/'


class client(object):
    def __init__(self, username, password, 
                 domain='nest.local', 
                 url='https://visualizer.nestcollaboration.ch/Backend/api/v1/datapoints/',
                 ):
        """
        Initialize parameters and empty data containers
        and check login credentials.
        """
        self.username = username
        self.password = password
        self.domain = domain
        self.url = url
        self.save_dir = save_dir
        self.startDate = None

        self.df_data = None
        self.np_data = []

        self.auth = HttpNegotiateAuth(domain=self.domain, username=self.username, password=self.password)
        print(time.ctime() + ' REST client initialized')


    def read(self, df_data=[], startDate='2019-01-01', endDate='2019-12-31'):

        self.startDate = startDate
        self.endDate = endDate

        s = requests.Session()
        r = s.get(url=self.url, auth=self.auth)
        if r.status_code != requests.codes.ok:
            print(r.status_code)
        else:
            print(time.ctime() + ' REST client login successfull')
            for ct, column in enumerate(df_data):
                df = pd.DataFrame(data=s.get(url=self.url + column +'/timeline?startDate='+ startDate + '&endDate=' + endDate).json())

                # Convert to Numpy
                vals = df.loc[:, "value"].to_numpy()
                ts = pd.to_datetime(df.loc[:, "timestamp"]).to_numpy(dtype=np.datetime64)
                self.np_data += [(vals, ts)]
                
                df.columns =['value_' + str(ct), column + "_" + str(ct)]
                if self.df_data is None:
                    # Initialize
                    self.df_data = df
                else:                        
                    # Add column and timestamp
                    self.df_data = pd.concat([self.df_data, df], axis=1, sort=False)

                print("Added column", ct + 1, "with ID", column)

            print(time.ctime() + ' REST client data acquired')
            return self.df_data

    def get_data_folder(self, name, startDate, endDate):
        ext = startDate + "__" + endDate + "__"
        data_dir = os.path.join(save_dir, ext + name)
        return data_dir

    def write_np(self, name, overwrite = False):
        """
        Writes the read data in numpy format
        to files.
        """

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

        return

    def read_offline(self, name, startDate='2019-01-01', endDate='2019-12-31'):
        """
        Read numpy data that has already been created.
        """

        data_dir = self.get_data_folder(name, startDate, endDate)
        val_list = []
        ts_list = []

        # Loop over files in directory
        for f in os.listdir(folder):
            file_path = os.path.join(folder, f)
            nparr = np.load(file_path)
            if f[:5] == "dates":
                ts_list += [nparr]
            elif f[:6] == "values":
                val_list = [nparr]
            else:
                print("Unknown File Name!!!")

        return val_list, ts_list




