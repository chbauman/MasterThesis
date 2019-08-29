
import os

import numpy as np
import pandas as pd

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
