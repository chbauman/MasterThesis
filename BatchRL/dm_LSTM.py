
import numpy as np

import keras

from keras import backend as K
from keras.models import Sequential
from keras.layers import GRU, LSTM, TimeDistributed, Dense, GaussianNoise

from base_dynamics_model import BaseDynamicsModel
from keras_layers import Input
from time_series import AR_Model

class BaseRNN_DM(BaseDynamicsModel):
    """
    Simple LSTM used for training a dynamics model.
    """
    def __init__(self, 
                 train_seq_len, 
                 n_feats, 
                 hidden_sizes = [20, 20], 
                 n_iter_max = 10000, 
                 out_dim = 1, 
                 name = 'baseRNN',
                 *,
                 gru = False, 
                 input_noise_std = None,
                 use_AR = False):

        # Store parameters
        self.train_seq_len = train_seq_len
        self.n_feats = n_feats
        self.hidden_sizes = np.array(hidden_sizes, dtype = np.int32)
        self.n_iter_max = n_iter_max
        self.name = name
        self.out_dim = out_dim
        self.gru = gru
        self.input_noise_std = input_noise_std
        self.use_AR = use_AR

        # Build model
        self.build_model()

    def build_model(self):
        """
        Builds the keras LSTM model.
        """

        # Initialize
        n_lstm = len(self.hidden_sizes)
        model = Sequential()
        model.add(Input(input_shape=(self.train_seq_len, self.n_feats)))

        # Add noise layer
        if self.input_noise_std is not None:
            model.add(GaussianNoise(self.input_noise_std))

        # Add layers
        rnn = GRU if self.gru else LSTM
        for k in range(n_lstm):
            ret_seq = k != n_lstm - 1
            model.add(rnn(self.hidden_sizes[k],
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
        Fit the model if it hasn't been fitted before.
        Otherwise load the trained model.
        """

        loaded = self.load_if_exists(self.m, self.name)
        if not loaded:
            self.deb("Fitting Model...")
               
            # Prepare the data
            input_data, output_data = self.prepare_data(data)

            # Fit model
            self.m.fit(input_data, output_data,
                       epochs = self.n_iter_max,
                       initial_epoch = 0,
                       batch_size = 128,
                       validation_split = 0.1)

            self.m.save_weights(self.get_path(self.name))
        else:
            self.deb("Restored trained model")

        # Save disturbance parameters
        reds = self.get_residuals(data)
        if self.use_AR:
            self.dist_mod = AR_Model(lag = 4)
            self.dist_mod.fit(reds)
            self.init_pred = np.zeros((4,))
        self.res_std = np.std(reds)
        

    def predict(self, data, prepared = False):
        """
        Predicts a batch of sequences.
        """

        input_data = np.copy(data)

        # Prepare the data
        if not prepared:
            input_data, _ = self.prepare_data(data)

        # Predict
        preds = self.m.predict(input_data)
        preds = preds.reshape((-1,))
        return preds

    def disturb(self, n):
        """
        Returns a sample of noise of length n.
        """
        if self.use_AR:
            next = self.dist_mod.predict(self.init_pred)
            self.init_pred[:-1] = self.init_pred[1:]
            self.init_pred[-1] = next
            return next

        return np.random.normal(0, self.res_std, n)

    pass
