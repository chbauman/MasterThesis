
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from base_dynamics_model import BaseDynamicsModel


class GPR_DM(BaseDynamicsModel):

    def __init__(self, validation = 0.1):
        self.val = validation
        pass

    def prepare_data_gp(self, data):
        """
        Prepares the data for usage with the estimator.
        """
        i, o = self.prepare_data(data)
        n = data.shape[0]
        in_data = np.reshape(i, (n, -1))
        return in_data, o

    def fit(self, data):
        """
        Fit the model.
        """
        # Prepare the data
        d_shape = data.shape
        self.n = d_shape[0]
        i, o = self.prepare_data_gp(data)
        self.output_data = o
        self.input_data = i

        # Split for validation and training
        self.n_train = int(self.val * self.n)
        self.n_val = self.n - self.n_train
        self.out_train = self.output_data[:self.n_train]
        self.out_val = self.output_data[self.n_train:]
        self.in_train = self.input_data[:self.n_train, :]
        self.in_val = self.input_data[self.n_train:, :]

        # Init and fit GP
        self.deb("Train Input Shape", self.in_train.shape)
        self.deb("Train Output Shape", self.out_train.shape)
        self.gpr = GaussianProcessRegressor(alpha = 1.0, n_restarts_optimizer = 10)
        self.gpr.fit(self.in_train, self.out_train)
        self.analyze_gp()

    def predict(self, data, prepared = False, disturb = False):
        """
        Predict outcome for new data.
        """
        if disturb:
            print("Not implemented")
        n = data.shape[0]
        in_d = None
        if not prepared:
            in_d, _ = self.prepare_data_gp(data)
        else:
            in_d = data.reshape((n, -1))
        #print("Predict Input Shape", in_d.shape)
        return self.gpr.predict(in_d)

    def mse_error_pred(self, X, y):
        """
        Returns the MSE of the true labels 'y' and the
        predicted values from the ffitted GP.
        """
        return np.mean((y - self.gpr.predict(X)) ** 2)

    def analyze_gp(self):
        """
        Analyze generalization performance.
        """
        print("Training Error")
        print(self.mse_error_pred(self.in_train, self.out_train))
        print("Validation Error")
        print(self.mse_error_pred(self.in_val, self.out_val))

    def disturb(self, n):
        """
        Returns a sample of noise of length n.
        """
        return np.random.normal(0, self.res_std, n)

    pass