
import numpy as np

from keras.models import Model
from keras.layers import Multiply, Add, Input

from base_dynamics_model import BaseDynamicsModel
from keras_util import getMLPModel
from visualize import scatter_plot, model_plot_path
from util import *

class BatteryModel(BaseDynamicsModel):
    """
    The model of the battery:
    s_t: SoC at time t
    Model: s_{t+1} = s_t + \eta(s_t) p_{t+1}
    p_{t+1}: average charging power from time t to t+1 (cotrol input)
    """

    def __init__(self, dataset, mlp_layers = [10, 10, 10], n_iter = 2):
        
        
        super(BatteryModel, self).__init__(0)
        self.n_iter = n_iter

        self.plot_path = os.path.join(model_plot_path, "BatteryModel")
        create_dir(self.plot_path)

        # Save dataset
        self.data = dataset

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
        start_ind = 0 
        end_ind = n
        ds = data[start_ind:end_ind, 1, 0] - data[start_ind:end_ind, 0, 0]
        p = data[start_ind:end_ind, 1, 1]

        m_a_s = self.data.scaling
        mas_ds = m_a_s[0]
        mas_p = m_a_s[1]
        labs = {'title': 'Battery Model', 'xlab': 'Active Power [kW]', 'ylab': r'$\Delta$ SoC [%]'}
        d_soc_std = mas_ds[1]

        # Fit linear Model
        fitted_ds = fit_linear_1d(p, ds, p)
        mask = np.logical_or(ds > fitted_ds - 0.35, p < 0)
        masked_p = p[mask]
        masked_ds = ds[mask]
        n_mask = masked_p.shape[0]

        # Fit pw. linear model: y = \alpha_1 + \alpha_2 * x * \alpha_3 * max(0, x) 
        ls_mat = np.empty((n_mask, 3), dtype = np.float32)
        ls_mat[:, 0] = 1
        ls_mat[:, 1] = masked_p
        ls_mat[:, 2] = masked_p
        ls_mat[:, 2][ls_mat[:, 2] < 0] = 0
        a1, a2, a3 = np.linalg.lstsq(ls_mat, masked_ds, rcond=None)[0]
        x_pw_line = np.array([np.min(p), 0, np.max(p)], dtype = np.float32)
        y_pw_line = a1 + a2 * x_pw_line + a3 * np.maximum(0, x_pw_line)

        # Plot before and after
        before_ppath = os.path.join(self.plot_path, "WithOutliers")
        after_ppath = os.path.join(self.plot_path, "Cleaned")
        scatter_plot(p, ds, lab_dict = labs,
                     m_and_std_x = mas_p,
                     m_and_std_y = [0.0, d_soc_std],
                     show = False, 
                     add_line = True,
                     save_name = before_ppath)
        scatter_plot(masked_p, masked_ds, lab_dict = labs, 
                     m_and_std_x = mas_p,
                     m_and_std_y = [0.0, d_soc_std],
                     show = False, 
                     add_line = True, 
                     custom_line = [x_pw_line, y_pw_line],
                     custom_label = 'PW Linear Fit',
                     save_name = after_ppath)

