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
import time

import wx
from pw_gui import getPW


class client(object):
    def __init__(self, username, password, domain='nest.local', url='https://visualizer.nestcollaboration.ch/Backend/api/v1/datapoints/'):
        self.username = username
        self.password = password
        self.domain = domain
        self.url = url
        self.df_data = None

        self.auth = HttpNegotiateAuth(domain=self.domain, username=self.username, password=self.password)
        print(time.ctime() + ' REST client initialized')


    def read(self, df_data=[], startDate='2016-10-01', endDate='2018-10-02'):

        s = requests.Session()
        r = s.get(url=self.url, auth=self.auth)
        if r.status_code != requests.codes.ok:
            print(r.status_code)
        else:
            print(time.ctime() + ' REST client login successfull')
            for ct, column in enumerate(df_data):
                df = pd.DataFrame(data=s.get(url=self.url + column +'/timeline?startDate='+ startDate + '&endDate=' + endDate).json())
                df.columns =['value_' + str(ct), column + "_" + str(ct)]
                #df['timestamp'] = df['timestamp'].astype('datetime64[m]')
                if self.df_data is None:
                    # Initialize
                    self.df_data = df
                else:                        
                    # Add column and timestamp
                    self.df_data = pd.concat([self.df_data, df], axis=1, sort=False)

                print("Added column", ct, "with ID", column)

            print(time.ctime() + ' REST client data acquired')
            return self.df_data





