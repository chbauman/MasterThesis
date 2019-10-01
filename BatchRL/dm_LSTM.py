
import numpy as np

import keras

from keras import backend as K
from keras.models import Sequential
from keras.layers import GRU, LSTM, TimeDistributed, Dense, GaussianNoise
from keras.optimizers import Adam
from keras.utils import plot_model
from functools import partial

from base_dynamics_model import BaseDynamicsModel
from keras_layers import SeqInput
from time_series import AR_Model
from visualize import plot_train_history

from util import *

def weighted_loss(y_true, y_pred, weights):
    return K.mean(K.abs(y_true - y_pred) * weights)

class BaseRNN_DM(BaseDynamicsModel):
    """
    Simple LSTM used for training a dynamics model.
    """
    def __init__(self, 
                 data,
                 hidden_sizes = [20, 20], 
                 n_iter_max = 10000, 
                 name = 'baseRNN',
                 *,
                 weight_vec = None,
                 gru = False, 
                 input_noise_std = None,
                 use_AR = False,
                 residual_learning = False,
                 lr = 0.001):

        super(BaseRNN_DM, self).__init__()

        # Store data
        self.data = data
        self.train_seq_len = self.data.seq_len - 1
        self.n_feats = self.data.d        
        self.out_dim = self.data.d  - self.data.n_c

        # Store parameters        
        self.hidden_sizes = np.array(hidden_sizes, dtype = np.int32)
        self.n_iter_max = n_iter_max
        self.gru = gru
        self.input_noise_std = input_noise_std
        self.use_AR = use_AR
        self.weight_vec = weight_vec
        self.res_learn = residual_learning
        self.lr = lr
        self.name = self.constr_name(name)

        # Build model
        self.build_model()

    def constr_name(self, name):
        """
        Constructs the name of the network.
        """
        ds_pt = '_DATA_' + self.data.name
        arch = '_L' + '-'.join(map(str, self.hidden_sizes))
        gru_str = '' if not self.gru else '_GRU'
        ep_s = '_E' + str(self.n_iter_max)
        lrs = '_LR' + str(self.lr)
        return name + ds_pt + ep_s + arch + lrs + gru_str

    def build_model(self):
        """
        Builds the keras LSTM model.
        """

        # Initialize
        n_lstm = len(self.hidden_sizes)
        model = Sequential()
        model.add(SeqInput(input_shape=(self.train_seq_len, self.n_feats)))

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
        if self.res_learn:
            # TODO
            # Add input to output
            pass


        if self.weight_vec is not None:
            weights_tensor = Input(shape=(self.out_dim,))
            loss = partial(weighted_loss, weights=weights_tensor)
        else:
            loss = 'mse'

        optim = Adam(lr = self.lr)
        model.compile(loss=loss,
                      optimizer='adam')
        model.summary()
        self.m = model
        #pth = self.get_plt_path("Model.png")
        #plot_model(model, to_file=pth)

    def fit(self):
        """
        Fit the model if it hasn't been fitted before.
        Otherwise load the trained model.
        """

        loaded = self.load_if_exists(self.m, self.name)
        if not loaded:
            self.deb("Fitting Model...")
               
            # Prepare the data
            input_data, output_data = self.data.get_prepared_data('train_val')

            # Fit and save model
            h = self.m.fit(input_data, output_data,
                           epochs = self.n_iter_max,
                           initial_epoch = 0,
                           batch_size = 128,
                           validation_split = self.data.val_perc)
            pth = self.get_plt_path("TrainHist")
            plot_train_history(h, pth)
            create_dir(self.model_path)
            self.m.save_weights(self.get_path(self.name))
        else:
            self.deb("Restored trained model")

    def predict(self, input_data):
        """
        Predicts a batch of sequences.
        """

        n = input_data.shape[0]

        # Predict
        preds = self.m.predict(input_data)
        preds = preds.reshape((n, -1))
        return preds


    pass
