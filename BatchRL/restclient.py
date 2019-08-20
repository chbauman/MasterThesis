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
        self.df_data = pd.DataFrame({'timestamp': []})

        self.auth = HttpNegotiateAuth(domain=self.domain, username=self.username, password=self.password)
        print(time.ctime() + ' REST client initialized')


    def read(self, df_data=pd.DataFrame(columns=[]),startDate='2016-10-01',endDate='2018-10-02'):

        s = requests.Session()
        r = s.get(url=self.url, auth=self.auth)
        if r.status_code != requests.codes.ok:
            print(r.status_code)
        else:
            print(time.ctime() + ' REST client login successfull')
            for column in df_data:
                try:
                    df = pd.DataFrame(data=s.get(url=self.url + column +'/timeline?startDate='+ startDate + '&endDate=' + endDate).json())
                    df.columns =['timestamp', column]
                    df['timestamp'] = df['timestamp'].astype('datetime64[m]')
                    self.df_data = pd.merge(self.df_data, df, how='outer', on='timestamp')
                except Exception as e:
                    print(e)
            print(time.ctime() + ' REST client data acquired')
            return self.df_data





