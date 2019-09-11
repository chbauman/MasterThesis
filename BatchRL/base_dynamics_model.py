
import os

import numpy as np

from keras.models import load_model
from abc import ABC, abstractmethod

from visualize import plot_ip_time_series
 
class BaseDynamicsModel(ABC):
    """
    This class describes the interface of a ML-based
    dynamics model.
    """

    out_indx = 3
    debug = True
    model_path = "../Models/Dynamics/"

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, data, prepared = False):
        pass

    @abstractmethod
    def disturb(self, n):
        """
        Returns a sample of noise of length n.
        """
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
    
    def prepare_data(self, data, diff = False):
        """ 
        Prepare the data
        """

        d_shape = data.shape
        seq_len_data = d_shape[1]
        input_data = np.copy(data[:, :-1, :])
        #output_data = data[:, 1:, 3]
        #output_data.reshape((d_shape[0], seq_len_data - 1, 1))
        output_data = np.copy(data[:, -1, self.out_indx])
        if diff:
            output_data -= np.copy(data[:, -2, self.out_indx])

        if self.debug:
            print("Input shape", input_data.shape)
            print("Output shape", output_data.shape)

        return input_data, output_data

    def n_step_predict(self, data, n, return_all_preds = False, disturb_pred = False, diff = False):
        """
        Applies the model n times and returns the 
        predictions.
        """

        s = data.shape
        seq_len = s[1]
        n_data = s[0]
        n_out = n_data - n + 1
        
        all_preds = None
        if return_all_preds:
            all_preds = np.empty((n_out, n))

        input_data, output_data = self.prepare_data(np.copy(data), diff=diff)
        curr_in_data = input_data[:n_out]
        curr_out_data = output_data[:n_out]
        for k in range(n):

            # Predict
            curr_preds = self.predict(curr_in_data, prepared = True)
            if disturb_pred:
                curr_preds += self.disturb(curr_preds.shape[0])

            if return_all_preds:
                all_preds[:, k] = curr_preds

            # Construct next data
            curr_in_data[:, :-1, :] = curr_in_data[:, 1:, :]
            curr_in_data[:, -1, :] = input_data[k:(n_out + k), -1, :]
            curr_in_data[:, -1, self.out_indx] = curr_preds

        if return_all_preds:
            return all_preds
        return curr_preds

    def analyze(self, week_data, diff = False):
        """
        Analyzes the trained model
        """

        print("Analyzing model")
        s = week_data.shape
        input_data, output_data = self.prepare_data(week_data)

        # One step predictions
        preds = self.predict(week_data)
        preds = preds.reshape((-1,))
        er = preds - output_data
        m = {'description': '15-Min Ahead Predictions', 'unit': 'Scaled Temperature'}
        plot_ip_time_series([preds, output_data], lab = ['predictions', 'truth'], m = m, show = True)        

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
