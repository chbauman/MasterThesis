
import os

import numpy as np
import pandas as pd


import restclient

class DataStruct:
    """
    Base Class for different sets of data columns
    defined by successive IDs
    """
    def __init__(self, 
                 data_desc, 
                 base_id, 
                 name, 
                 startDate='2019-01-01',
                 endDate='2019-12-31'):

        # Initialize values and client
        self.data_desc = data_desc
        self.base_id = base_id
        self.name = name
        self.startDate = startDate
        self.endDate = endDate

        # Check if base_id is a list of ints or just an int,
        # in that case create the list with the subsequent ints.
        self.num_cols = len(data_desc)
        try:
            for ct, el in enumerate(base_id):
                base_id[ct] = str(el)
            self.data_ids = base_id
        except TypeError as te:
            self.data_ids = [str(base_id + i) for i in range(self.num_cols)]
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
            data_desc = [
                "alarm active",
                "th. power total",
            ],
            base_id = 421100171,
            name = "Test",
            startDate='2019-08-08',
            endDate='2019-08-09'
            )

# Electricity consumption data
ElData = DataStruct(
            data_desc = [
                "alarm active",
                "el. active power total",
                "el. active power consumpted",
                "el. active power L1",
                "el. reactive power L1",
                "el. cos Phi L1",
                "el. current L1",
                "el. voltage L1",
                "el. active power L2",
                "el. reactive power L2",
                "el. cos Phi L2",
                "el. current L2",
                "el. voltage L2",
                "el. active power L3",
                "el. reactive power L3",
                "el. cos Phi L3",
                "el. current L3",
                "el. voltage L3",
            ],
            base_id = 42190138,
            name = "Electric",
            startDate='2019-01-01',
            endDate='2019-12-31'
            )

# Heating data
HeatingData = DataStruct(
            data_desc = [
                "alarm active",
                "th. power total",
                "th. energy total heating",
                "volume flow",
                "temperature high",
                "temperature low",
            ],
            base_id = 421100171,
            name = "Heat-U33M1-P890",
            startDate='2019-01-01',
            endDate='2019-12-31'
            )

# Valve data
ValveData = DataStruct(data_desc=[
                            "research mode approval",
                            "research mode confirmation",
                            "research mode status",
                            "alarm active",
                            "position relative",
                            "research mode approval",
                            "research mode confirmation",
                            "research mode status",
                            "alarm active",
                            "position relative",
                            "research mode approval",
                            "research mode confirmation",
                            "research mode status",
                            "alarm active",
                            "position relative",
                            "research mode approval",
                            "research mode confirmation",
                            "research mode status",
                            "alarm active",
                            "position relative",
                            "research mode approval",
                            "research mode confirmation",
                            "research mode status",
                            "alarm active",
                            "position relative",
                            ],
                       base_id=[421100138 + i for i in range(5)] \
                             + [421100194 + i for i in range(5)] \
                             + [421100143 + i for i in range(5)] \
                             + [421100199 + i for i in range(5)] \
                             + [421100148 + i for i in range(5)],
                       name="Valve-U33M1,U33N1-Y720,Y721,Y750",
                       startDate='2019-01-01',
                       endDate='2019-12-31'    
    )


