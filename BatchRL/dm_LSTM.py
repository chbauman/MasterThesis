
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
    def __init__(self, train_seq_len, n_feats, hidden_sizes = [20, 20], n_iter_max = 10000, out_dim = 1):

        # Store parameters
        self.train_seq_len = train_seq_len
        self.n_feats = n_feats
        self.hidden_sizes = np.array(hidden_sizes, dtype = np.int32)
        self.n_iter_max = n_iter_max
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
        for k in range(n_lstm):
            in_sh = (self.train_seq_len, self.n_feats) if k == 0 else (-1, -1)
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
        """
        Fit the model.
        """
               
        # Prepare the data
        d_shape = data.shape
        seq_len_data = d_shape[1]
        input_data = data[:, :-1, :]
        #output_data = data[:, 1:, 3]
        #output_data.reshape((d_shape[0], seq_len_data - 1, 1))
        output_data = data[:, -1, 3]

        # Fit model
        self.m.fit(input_data, output_data, 
                   epochs = self.n_iter_max, 
                   initial_epoch = 0,
                   batch_size = 128,
                   validation_split = 0.1)

    def predict(self, data):
        pass

    pass
