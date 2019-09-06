
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from base_dynamics_model import BaseDynamicsModel


class GPR_DM(BaseDynamicsModel):

    def __init__(self, validation = 0.1):
        self.val = validation
        pass

    def prepare_data(self, data):
        """
        Prepares the data for usage with the estimator.
        """
        d_shape = data.shape
        n = d_shape[0]
        out_data = data[:, -1, 3]
        in_data = np.reshape(data, (n, -1))
        return in_data, out_data

    def fit(self, data):
        """
        Fit the model.
        """
        # Prepare the data
        d_shape = data.shape
        self.n = d_shape[0]
        i, o = self.prepare_data(data)
        self.output_data = o
        self.input_data = i

        # Split for validation and training
        self.n_train = int(self.val * self.n)
        self.n_val = self.n - self.n_train
        self.out_train = self.output_data[:self.n_train]
        self.out_val = self.output_data[self.n_train:]
        self.in_train = self.input_data[:self.n_train]
        self.in_val = self.input_data[self.n_train:]

        # Init and fit GP
        self.gpr = GaussianProcessRegressor(n_restarts_optimizer = 10)
        self.gpr.fit(self.in_train, self.out_train)

    def predict(self, data):
        """
        Predict outcome for new data.
        """
        in_d, out_d = self.prepare_data(data)
        return self.gpr.predict(in_d)

    def mse_error_pred(self, X, y):
        """
        Returns the MSE of the true labels 'y' and the
        predicted values from the ffitted GP.
        """
        return np.mean((y - self.gpr.predict(X)) ** 2)

    def analyze(self):

        print("Training Error")
        print(self.mse_error_pred(self.in_train, self.out_train))
        print("Validation Error")
        print(self.mse_error_pred(self.in_val, self.out_val))

    pass
