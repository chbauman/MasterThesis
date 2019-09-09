from abc import ABC, abstractmethod

import numpy as np

from visualize import plot_ip_time_series
 
class BaseDynamicsModel(ABC):
    """
    This class describes the interface of a ML-based
    dynamics model.
    """

    out_indx = 3
    
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, data, prepared = False):
        pass

    def prepare_data(self, data):
        """ 
        Prepare the data
        """

        d_shape = data.shape
        seq_len_data = d_shape[1]
        input_data = data[:, :-1, :]
        #output_data = data[:, 1:, 3]
        #output_data.reshape((d_shape[0], seq_len_data - 1, 1))
        output_data = data[:, -1, self.out_indx]

        print("Input shape", input_data.shape)
        print("Output shape", output_data.shape)

        return input_data, output_data

    def n_step_predict(self, data, n, return_all_preds = False):
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

        input_data, output_data = self.prepare_data(data)
        curr_in_data = input_data[:n_out]
        curr_out_data = output_data[:n_out]
        for k in range(n):

            # Predict
            curr_preds = self.predict(curr_in_data, prepared = True).reshape((-1,))
            if return_all_preds:
                all_preds[:, k] = curr_preds

            # Construct next data
            curr_in_data[:, :-1, :] = curr_in_data[:, 1:, :]
            curr_in_data[:, -1, :] = input_data[k:(n_out + k), -1, :]
            curr_in_data[:, -1, self.out_indx] = curr_preds

        if return_all_preds:
            return all_preds
        return curr_preds

    def analyze(self, week_data):
        """
        Analyzes the trained model
        """

        print("Analyzing model")
        s = week_data.shape
        input_data, output_data = self.prepare_data(week_data)

        # One step predictions
        preds = self.predict(week_data).reshape((-1,))
        er = preds - output_data

        print(er.shape)
        m = {'description': '15-Min Ahead Predictions', 'unit': 'Scaled Temperature'}
        plot_ip_time_series([preds, output_data], lab = ['predictions', 'truth'], m = m, show = True)        

        # One hour predictions (4 steps)
        one_h_pred = self.n_step_predict(week_data, 4)
        m['description'] = '1h Ahead Predictions'
        plot_ip_time_series([one_h_pred, output_data[3:]], lab = ['predictions', 'truth'], m = m, show = True)        

        # One-week prediction
        full_pred = self.n_step_predict(week_data, s[0], return_all_preds=True)
        print(full_pred.shape)
        full_pred = np.reshape(full_pred, (-1,))
        m['description'] = 'Evolution'
        plot_ip_time_series([full_pred, output_data], lab = ['predictions', 'truth'], m = m, show = True)        
        pass

