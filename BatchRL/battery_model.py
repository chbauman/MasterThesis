
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Multiply, Add, Input

from base_dynamics_model import BaseDynamicsModel
from keras_layers import SeqInput
from keras_util import getMLPModel
from visualize import scatter_plot

class BatteryModel(BaseDynamicsModel):
    """
    The model of the battery:
    s_t: SoC at time t
    Model: s_{t+1} = s_t + \eta(s_t) p_{t+1}
    p_{t+1}: average charging power from time t to t+1 (cotrol input)
    """


    def __init__(self, mlp_layers = [10, 10, 10], n_iter = 10):
        
        
        super(BatteryModel, self).__init__(0)
        self.n_iter = n_iter

        # Input tensors
        s_inp = Input(shape=(1,))
        p_inp = Input(shape=(1,))

        # Define model
        mlp = getMLPModel(mlp_layers, 1, ker_reg = 0.0001)
        m_s = mlp(s_inp)
        mul = Multiply()([m_s, p_inp])
        out = Add()([mul, s_inp])
        self.m = Model(inputs=[s_inp, p_inp], outputs=out)
        self.m.compile(loss='mse', optimizer='adam')

    def prepare_dat_bat(self, indat, outdat):
        """
        shape of indat: (n, 1, 2)
        shape of outdat: (n, )
        """

        n = indat.shape[0] - 1
        inp = np.empty((n, 2), dtype = np.float32)

        #outp = data[:, 1, 0]
        inp[:,0] = indat[:-1, 0, 0]
        inp[:,1] = indat[1:, 0, 1]

        return inp, outdat[:-1]

    def fit(self, data):

        i, o = self.prepare_data(data)
        i, o = self.prepare_dat_bat(i, o)
        self.m.fit([i[:,0], i[:,1]], o, 
                   epochs = self.n_iter,
                   validation_split = 0.1)
        self.analyze_bat_model(data)


    def predict(self, data, prepared = False):

        prepdata = data
        if not prepared:
            prepdata, _ = self.prepare_data(data)

        return self.m.predict(prepdata)

    def disturb(self, n):
        """
        Returns a sample of noise of length n.
        """
        pass

    def analyze_bat_model(self, data):

        ds = data[:, 1, 0] - data[:, 0, 0]
        p = data[:, 1, 1]

        scatter_plot(p, ds)

    pass