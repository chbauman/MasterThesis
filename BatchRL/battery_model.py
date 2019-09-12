
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Multiply, Add

from base_dynamics_model import BaseDynamicsModel
from keras_layers import SeqInput
from keras_util import getMLPModel

class BatteryModel(BaseDynamicsModel):
    """
    The model of the battery:
    s_t: SoC at time t
    Model: s_{t+1} = s_t + \eta(s_t) p_{t+1}
    p_{t+1}: average charging power from time t to t+1 (cotrol input)
    """

    def __init__(self, mlp_layers = [10, 10, 10], n_iter = 10000):
        
        self.n_iter = n_iter

        # Input tensors
        s_inp = Input(shape=(1,))
        p_inp = Input(shape=(1,))

        # Define model
        mlp = getMLPModel(mlp_layers, 1)
        m_s = mlp(s_inp)
        out = Add(Multiply(m_s, p_inp), s_inp)
        self.m = Model(inputs=[s_inp, p_inp], outputs=out)
        self.m.compile(loss='mse', optimizer='adam')

    def prepare_data(self, data, diff = False):
        """
        shape of data: (n, 2, 2)
        """
        n = data.shape[0]

        inp = np.empty((n, 2), dtype = np.float32)
        outp = np.empty((n), dtype = np.float32)

        outp = data[:, 0, 1]
        inp[:,0] = data[:, 0, 0]
        inp[:,1] = data[:, 1, 1]

        return inp, outp

    def fit(self, data):

        i, o = self.prepare_data(data)
        self.m.fit(i, o, epochs = self.n_iter)

    def predict(self, data, prepared = False):
        pass

    def disturb(self, n):
        """
        Returns a sample of noise of length n.
        """
        pass

    pass