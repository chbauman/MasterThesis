from abc import ABC, abstractmethod

from visualize import plot_ip_time_series
 
class BaseDynamicsModel(ABC):
    """
    This class describes the interface of a ML-based
    dynamics model.
    """
    
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, data):
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
        output_data = data[:, -1, 3]

        print("Input shape", input_data.shape)
        print("Output shape", output_data.shape)

        return input_data, output_data

    def n_step_predict(self, data, n):
        """
        Applies the model n times and returns the 
        predictions.
        """

        s = data.shape
        seq_len = s[1]
        n_data = s[0]
        n_out = n_data - n
        input_data, output_data = self.prepare_data(data)


    def analyze(self, week_data):
        """
        Analyzes the trained model
        """

        print("Analyzing model")
        input_data, output_data = self.prepare_data(week_data)

        # One step predictions
        preds = self.predict(week_data).reshape((-1,))
        er = preds - output_data

        print(er.shape)
        m = {'description': '15-Min Ahead Predictions', 'unit': 'Scaled Temperature'}
        plot_ip_time_series([preds, output_data], lab = ['predictions', 'truth'], m = m, show = True)        

        # One-week prediction
        pass

