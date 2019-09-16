
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Multiply, Add, Input

from base_dynamics_model import BaseDynamicsModel
from keras_layers import SeqInput
from keras_util import getMLPModel
from visualize import scatter_plot, plot_ip_time_series

class BatteryModel(BaseDynamicsModel):
    """
    The model of the battery:
    s_t: SoC at time t
    Model: s_{t+1} = s_t + \eta(s_t) p_{t+1}
    p_{t+1}: average charging power from time t to t+1 (cotrol input)
    """


    def __init__(self, mlp_layers = [10, 10, 10], n_iter = 2):
        
        
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

    def fit(self, data, m = None):

        self.m_dat = m

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

        n = data.shape[0]
        start_ind = 0 # n // 2
        end_ind = n
        ds = data[start_ind:end_ind, 1, 0] - data[start_ind:end_ind, 0, 0]
        p = data[start_ind:end_ind, 1, 1]

        mas_ds = self.m_dat[0]['mean_and_std']
        mas_p = self.m_dat[1]['mean_and_std']
        labs = {'title': 'Battery Model Data', 'xlab': 'Active Power', 'ylab': r'$\Delta$ SoC'}

        d_soc_std = mas_ds[1]

        scatter_plot(p, ds, lab_dict = labs, 
                     m_and_std_x = mas_p,
                     m_and_std_y = [0.0, d_soc_std])

        plot_ip_time_series([ds, p], 
                            lab = ['SoC', 'Active Power'], 
                            show = True, 
                            m = self.m_dat,
                            mean_and_stds = [mas_ds, mas_p],
                            use_time = True)

    pass