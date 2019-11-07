import numpy as np
from hyperopt import hp, fmin, rand
from sklearn.gaussian_process import GaussianProcessRegressor

from base_dynamics_model import BaseDynamicsModel


class GPR_DM(BaseDynamicsModel):

    def __init__(self,
                 alpha=2.0,
                 validation=0.1,
                 use_diff_data=True):

        super(GPR_DM, self).__init__()

        self.alpha = alpha
        self.val = validation
        self.use_diff_data = use_diff_data
        pass

    def prepare_data_gp(self, data):
        """
        Prepares the data for usage with the estimator.
        """
        i, o = self.prepare_data(data, diff=self.use_diff_data)
        n = data.shape[0]
        in_data = np.reshape(i, (n, -1))
        return in_data, o

    def fit(self, data, m=None):
        """
        Fit the model.
        """
        self.m_dat = m

        # Prepare the data
        d_shape = data.shape
        self.n = d_shape[0]
        self.seq_len = d_shape[1]
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

        # Fitness function for the hyperparameter optimization
        def gpr_fit_fun(alpha):
            self.gpr = GaussianProcessRegressor(alpha=alpha, n_restarts_optimizer=10)
            self.gpr.fit(self.in_train, self.out_train)
            return self.mse_error_pred(self.in_val, self.out_val)

        # Define space and find best hyperparameters
        space = hp.loguniform('alpha', -2, 5)
        self.best = fmin(gpr_fit_fun, space, algo=rand.suggest, max_evals=2)['alpha']
        print("Best alpha: ", self.best)

        # Fit again with best
        self.gpr = GaussianProcessRegressor(alpha=self.best, n_restarts_optimizer=10)
        self.gpr.fit(self.in_train, self.out_train)
        self.analyze_gp()

        # Save disturbance parameters
        reds = self.get_residuals(data)
        self.res_std = np.std(reds)

    def predict(self, data, prepared=False, disturb=False):
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
        # print("Predict Input Shape", in_d.shape)
        res = self.gpr.predict(in_d)
        if self.use_diff_data:
            ind_resh = np.reshape(in_d, (n, self.seq_len - 1, -1))
            res += ind_resh[:, -1, self.out_indx]
        return res

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
