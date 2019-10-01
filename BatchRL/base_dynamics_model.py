
import os

import numpy as np

from keras.models import load_model
from abc import ABC, abstractmethod

from visualize import plot_ip_time_series, plot_multiple_ip_ts, plot_dataset,\
    model_plot_path
from data import Dataset
from util import * 

class BaseDynamicsModel(ABC):
    """
    This class describes the interface of a ML-based
    dynamics model.
    """

    def __init__(self, out_ind = 3):

        self.out_indx = out_ind

    debug = True
    model_path = "../Models/Dynamics/"

    @abstractmethod
    def fit(self, data, m = None):
        pass

    @abstractmethod
    def predict(self, data, prepared = False):
        pass

    def get_path(self, name):
        return self.model_path + name + ".h5"

    def load_if_exists(self, m, name):
        """
        Loads a keras model if it exists.
        Returns true if it could be loaded, else False.
        """
        full_path = self.get_path(name)

        if os.path.isfile(full_path):
            m.load_weights(full_path)
            return True
        return False
    
    def model_disturbance(self, data_str = 'train_val'):
        """
        Models the uncertainties in the model.
        """

        # Compute residuals
        input_data, output_data = self.data.get_prepared_data(data_str)
        resids = self.predict(input_data) - output_data
        #print(resids)
        N_LAG = 4

        if self.use_AR:
            # Fit an AR process for each output dimension
            self.dist_mod = [AR_Model(lag = N_LAG).fit(resids[:, k]) for k in range(self.out_dim)]
            self.init_pred = np.zeros((N_LAG, self.out_dim))
        self.res_std = np.std(resids, axis = 0)
        self.modeled_disturbance = True

    def disturb(self):
        """
        Returns a sample of noise of length n.
        """
        try:
            md = self.modeled_disturbance
        except AttributeError as e:
            raise AttributeError('Need to model the disturbance first.')

        if self.use_AR:
            next = np.empty((self.out_dim,), dtype = np.float32)
            for k in range(self.out_dim):
                next[k] = self.dist_mod.predict(self.init_pred[:, k])      
                
            self.init_pred[:-1, :] = self.init_pred[1:, :]
            self.init_pred[-1, :] = next
            return next

        return np.random.normal(0, 1, self.out_dim) * self.res_std

    def n_step_predict(self, prepared_data, n, pred_inds, return_all_preds = False, disturb_pred = False, diff = False):
        """
        Applies the model n times and returns the 
        predictions.
        """

        if diff:
            raise NotImplementedError("Differentiating data is not implemented")

        in_data, out_data = prepared_data

        # Get shapes
        n_samples = in_data.shape[0]
        seq_len = in_data.shape[1]
        n_out = n_samples - n + 1
        d = len(pred_inds)

        # Reserve output array
        all_preds = None
        if return_all_preds:
            all_preds = np.empty((n_out, n, d))

        curr_in_data = in_data[:n_out]
        curr_out_data = out_data[:n_out]
        for k in range(n):

            # Predict
            curr_preds = self.predict(curr_in_data)
            if disturb_pred:
                curr_preds += self.disturb()

            if return_all_preds:
                all_preds[:, k] = curr_preds

            # Construct next data
            curr_in_data[:, :-1, :] = curr_in_data[:, 1:, :]
            curr_in_data[:, -1, :] = in_data[k:(n_out + k), -1, :]
            curr_in_data[:, -1, pred_inds] = curr_preds[:, pred_inds]
        
        if return_all_preds:
            return all_preds
        return curr_preds

    def get_plt_path(self, name):
        """
        Specifies the path of the plot with name 'name'
        where it should be saved. If there is not a directory
        for the current model, it is created.
        """
        dir_name = os.path.join(model_plot_path, self.name)
        create_dir(dir_name)
        return os.path.join(dir_name, name)

    def analyze(self, diff = False):
        """
        Analyzes the trained model
        """

        print("Analyzing model")
        d = self.data

        # Prepare the data
        input_data, output_data = d.get_prepared_data('test')
        s = input_data.shape        
        p_ind = d.p_inds_prep[0]
        orig_p_ind = d.p_inds[0]
        tr = output_data[:, p_ind]

        # Plot data
        plot_data = np.empty((s[0], 2), dtype = np.float32)
        plot_data[:, 1] = tr
        desc = d.descriptions[orig_p_ind]
        scals = np.array(repl(d.scaling[orig_p_ind], 2))
        is_scd = np.array(repl(d.is_scaled[orig_p_ind], 2), dtype = np.bool)
        analysis_ds = Dataset(plot_data,
                              d.dt,
                              d.t_init,
                              scals,
                              is_scd,
                              ['Prediction', 'Ground Truth'])


        # One step predictions
        preds = self.predict(input_data)
        p = preds[:, p_ind]
        plot_data[:, 0] = p
        title_and_ylab = ['15-Min Ahead Predictions', desc]
        plot_dataset(analysis_ds,
                     show = False,
                     title_and_ylab = title_and_ylab,
                     save_name = self.get_plt_path('15minAhead'))

        # One hour predictions (4 steps)
        one_h_pred = self.n_step_predict([input_data, output_data], 4, d.p_inds_prep)
        print(one_h_pred)
        print(one_h_pred.shape)
        return

        # One hour predictions (4 steps)
        one_h_pred = self.n_step_predict(week_data, 4, diff=diff)
        m['description'] = '1h Ahead Predictions'
        plot_ip_time_series([one_h_pred, output_data[3:]], lab = ['predictions', 'truth'], m = m, show = True)

        # 5 hour predictions (20 steps)
        one_h_pred = self.n_step_predict(week_data, 20, diff=diff)
        m['description'] = '5h Ahead Predictions'
        plot_ip_time_series([one_h_pred, output_data[19:]], lab = ['predictions', 'truth'], m = m, show = True)

        # One-week prediction
        full_pred = self.n_step_predict(week_data, s[0], return_all_preds=True, diff=diff)
        full_pred_noise = self.n_step_predict(week_data, s[0], return_all_preds=True, disturb_pred = True)
        print("Prediction Shape", full_pred.shape)
        full_pred = np.reshape(full_pred, (-1,))
        full_pred_noise = np.reshape(full_pred_noise, (-1,))
        init_data = week_data[0, :-1, self.out_indx]
        m['description'] = 'Evolution'
        plot_ip_time_series([full_pred, output_data, full_pred_noise], lab = ['predictions', 'truth', 'noisy prediction'], m = m, show = True, init = init_data)

        pass

    def get_residuals(self, data):
        """
        Computes the residuals using the fitted model.
        """
        input_data, output_data = self.prepare_data(data)
        preds = self.predict(input_data, prepared = True)
        return output_data - preds

    def deb(self, *args):
        """
        Prints Debug Info to console.
        """
        if self.debug:
            print(*args)
