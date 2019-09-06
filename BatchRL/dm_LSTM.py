
import numpy as np

import keras

from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense

from base_dynamics_model import BaseDynamicsModel


class LSTM_DM(BaseDynamicsModel):
    """
    Simple LSTM used for training a dynamics model.
    """
    def __init__(self, seq_len, n_feats, hidden_sizes = [20], out_dim = 1):

        # Store parameters
        self.seq_len = seq_len
        self.n_feats = n_feats
        self.hidden_sizes = hidden_sizes
        self.out_dim = out_dim

        # Build model
        self.build_model()

    def build_model(self):
        """
        Builds the keras LSTM model.
        """

        # Initialize
        n_lstm = len(self.hidden_sizes)
        model = Sequential()        

        # Add layers
        for k in range(n_lstm - 1):
            in_sh = (self.seq_len, self.n_feats) if k == 0 else None
            ret_seq = k != n_lstm - 1
            model.add(LSTM(self.hidden_sizes[k],  
                           input_shape=in_sh, 
                           return_sequences=ret_seq))

        # Output layer
        #model.add(TimeDistributed(Dense(self.out_dim, activation=None)))
        model.add(Dense(self.out_dim, activation=None))
        model.compile(loss='mse',
                      optimizer='adam')
        model.summary()
        self.m = model

    def fit(self, data):
        pass

    def predict(self, data):
        pass

    pass
