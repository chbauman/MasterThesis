import os

import numpy as np

from keras.models import load_model
from abc import ABC, abstractmethod

from time_series import AR_Model
from visualize import plot_dataset, model_plot_path
from data import Dataset
from util import *


def get_plot_ds(s, tr: np.ndarray, d: Dataset, orig_p_ind: np.ndarray) -> Dataset:
    """
    Creates a dataset with truth time series tr and parameters
    from the original dataset d. Intended for plotting after
    the first column of the data is set to the predicted series.

    :param s: Shape of data of dataset.
    :param tr: Ground truth time series.
    :param d: Original dataset.
    :param orig_p_ind: Prediction index relative to the original dataset.
    :return: Dataset with ground truth series as second series.
    """
    plot_data = np.empty((s[0], 2), dtype=np.float32)
    plot_data[:, 1] = tr
    scaling = np.array(repl(d.scaling[orig_p_ind], 2))
    is_scd = np.array(repl(d.is_scaled[orig_p_ind], 2), dtype=np.bool)
    analysis_ds = Dataset(plot_data,
                          d.dt,
                          d.t_init,
                          scaling,
                          is_scd,
                          ['Prediction', 'Ground Truth'])
    return analysis_ds


class BaseDynamicsModel(ABC):
    """
    This class describes the interface of a ML-based
    (partial) dynamics model.
    """

    # Constants
    N_LAG: int = 4
    debug: bool = True
    model_path: str = "../Models/Dynamics/"

    # Member variables
    data: Dataset
    pred_inds: np.ndarray
    name: str

    def __init__(self, ds: Dataset, name: str, pred_indices: np.ndarray = None):
        """
        Constructor for the base of every dynamics model.
        If pred_indices is None, all series are predicted.

        :param ds: Dataset containing all the data.
        :param name: Name of the model.
        :param pred_indices: Indices specifying the series in the data that the model predicts.
        """

        # Set members
        self.data = ds
        self.name = name
        if pred_indices is None:
            pred_indices = np.arange(ds.d - ds.n_c)
        self.pred_inds = pred_indices

        self.out_dim = None
        self.use_AR = None
        self.dist_mod = None
        self.init_pred = None
        self.res_std = None
        self.modeled_disturbance = None

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, in_data):
        pass

    def get_path(self, name: str) -> str:
        """
        Returns the path where the model parameters
        are stored. Used for keras models only.

        :param name: Model name.
        :return: Model parameter file path.
        """
        return self.model_path + name + ".h5"

    def load_if_exists(self, m, name: str) -> bool:
        """
        Loads a keras model if it exists.
        Returns true if it could be loaded, else False.

        :param m: Model to be loaded.
        :param name: Name of model.
        :return: True if model could be loaded else False.
        """
        full_path = self.get_path(name)

        if os.path.isfile(full_path):
            m.load_weights(full_path)
            return True
        return False

    def model_disturbance(self, data_str: str = 'train_val'):
        """
        Models the uncertainties in the model
        by matching the distribution of the residuals.
        """

        # Compute residuals
        residuals = self.get_residuals(data_str)

        if self.use_AR:
            # Fit an AR process for each output dimension
            self.dist_mod = [AR_Model(lag=self.N_LAG).fit(residuals[:, k]) for k in range(self.out_dim)]
            self.init_pred = np.zeros((self.N_LAG, self.out_dim))
        self.res_std = np.std(residuals, axis=0)
        self.modeled_disturbance = True

    def disturb(self):
        """
        Returns a sample of noise of length n.
        """

        # Check if disturbance model was fitted
        try:
            md = self.modeled_disturbance
            if md is None:
                raise AttributeError
        except AttributeError as e:
            raise AttributeError('Need to model the disturbance first: {}'.format(e))

        # Compute next noise
        if self.use_AR:
            next_noise = np.empty((self.out_dim,), dtype=np.float32)
            for k in range(self.out_dim):
                next_noise[k] = self.dist_mod.predict(self.init_pred[:, k])

            self.init_pred[:-1, :] = self.init_pred[1:, :]
            self.init_pred[-1, :] = next_noise
            return next_noise
        return np.random.normal(0, 1, self.out_dim) * self.res_std

    def n_step_predict(self, prepared_data, n: int, pred_inds, *,
                       return_all_predictions: bool = False,
                       disturb_pred: bool = False,
                       predict_all: bool = False,
                       diff: bool = False):
        """
        Applies the model n times and returns the 
        predictions.
        TODO: Implement or remove the diff == True case.
        """

        if diff:
            raise NotImplementedError("Differentiating data is not implemented")

        in_data, out_data = copy_arr_list(prepared_data)

        # Get shapes
        n_samples = in_data.shape[0]
        seq_len = in_data.shape[1]
        n_feat = out_data.shape[1]
        n_out = n_samples - n + 1
        if predict_all:
            pred_inds = np.arange(n_feat)

        # Do checks
        if n < 1:
            raise ValueError("n: ({}) has to be larger than 0!".format(n))
        if n_out <= 0:
            raise ValueError("n: ({}) too large".format(n))

        # Initialize values and reserve output array
        all_pred = None
        if return_all_predictions:
            all_pred = np.empty((n_out, n, n_feat))
        curr_in_data = np.copy(in_data[:n_out])
        curr_pred = None

        # Predict continuously
        for k in range(n):
            # Predict
            curr_pred = self.predict(np.copy(curr_in_data))
            if disturb_pred:
                curr_pred += self.disturb()
            if return_all_predictions:
                all_pred[:, k] = np.copy(curr_pred)

            # Construct next data
            curr_in_data[:, :-1, :] = np.copy(curr_in_data[:, 1:, :])
            if not predict_all:
                curr_in_data[:, -1, :n_feat] = np.copy(out_data[k:(n_out + k), :])
                if k != n - 1:
                    curr_in_data[:, -1, n_feat:] = np.copy(in_data[(k + 1):(n_out + k + 1), -1, n_feat:])
                else:
                    curr_in_data[:, -1, n_feat:] = 0
            curr_in_data[:, -1, pred_inds] = np.copy(curr_pred[:, pred_inds])

        # Return
        if return_all_predictions:
            return all_pred
        return curr_pred

    def get_plt_path(self, name: str) -> str:
        """
        Specifies the path of the plot with name 'name'
        where it should be saved. If there is not a directory
        for the current model, it is created.

        :param name: Name of the plot.
        :return: Full path of the plot file.
        """
        dir_name = os.path.join(model_plot_path, self.name)
        create_dir(dir_name)
        return os.path.join(dir_name, name)

    def const_nts_plot(self, predict_data, n_list: List[int], ext: str = '', *,
                       predict_all: bool = False):
        """
        Creates a plot that shows the performance of the
        trained model when predicting a fixed number of timesteps into 
        the future.
        """

        d = self.data
        in_d, out_d = predict_data
        s = in_d.shape
        p_ind = d.p_inds_prep[0]
        orig_p_ind = d.p_inds[0]
        tr = np.copy(out_d[:, p_ind])
        dt = d.dt

        # Plot for all n
        analysis_ds = get_plot_ds(s, tr, d, orig_p_ind)
        desc = d.descriptions[orig_p_ind]
        for n_ts in n_list:
            curr_ds = Dataset.copy(analysis_ds)
            time_str = mins_to_str(dt * n_ts)
            one_h_pred = self.n_step_predict(copy_arr_list(predict_data), n_ts, d.p_inds_prep,
                                             predict_all=predict_all)
            curr_ds.data[(n_ts - 1):, 0] = np.copy(one_h_pred[:, p_ind])
            curr_ds.data[:(n_ts - 1), 0] = np.nan
            title_and_ylab = [time_str + ' Ahead Predictions', desc]
            plot_dataset(curr_ds,
                         show=False,
                         title_and_ylab=title_and_ylab,
                         save_name=self.get_plt_path(time_str + 'Ahead' + ext))

    def one_week_pred_plot(self, dat_test, ext: str = None,
                           predict_all: bool = False):
        """
        Makes a plot by continuously predicting with
        the fitted model and comparing it to the ground
        truth.

        :param predict_all: Whether to predict all the state variables.
        :param dat_test: Data to use for making plots.
        :param ext: String extension for the filename.
        :return: None
        """

        ext = "_" if ext is None else "_" + ext
        d = self.data
        in_dat_test, out_dat_test = dat_test
        s = in_dat_test.shape
        p_ind = d.p_inds_prep[0]
        orig_p_ind = d.p_inds[0]
        tr = np.copy(out_dat_test[:, p_ind])

        # Plot data
        analysis_ds = get_plot_ds(s, tr, d, orig_p_ind)
        desc = d.descriptions[orig_p_ind]

        # Continuous prediction
        full_pred = self.n_step_predict([in_dat_test, out_dat_test], s[0], d.p_inds_prep,
                                        return_all_predictions=True,
                                        predict_all=predict_all)
        analysis_ds.data[:, 0] = np.copy(full_pred[0, :, p_ind])
        title_and_ylab = ['1 Week Continuous Predictions', desc]
        plot_dataset(analysis_ds,
                     show=False,
                     title_and_ylab=title_and_ylab,
                     save_name=self.get_plt_path('OneWeek' + ext))

        if predict_all:
            n_pred_f = self.data.d - self.data.n_c
            for k in range(n_pred_f):
                k_true = k + np.sum(self.data.c_inds <= k)
                print(k, ": k, k_true =", k_true)
                analysis_ds.data[:, 0] = np.copy(full_pred[0, :, k])
                analysis_ds.data[:, 1] = np.copy(out_dat_test[:, k])
                analysis_ds.scaling = np.array(repl(d.scaling[k_true], 2))
                analysis_ds.is_scaled = np.array(repl(d.is_scaled[k_true], 2))
                curr_p_inds = np.array([k_true], dtype=np.int32)
                analysis_ds.p_inds = curr_p_inds
                desc = d.descriptions[k_true]
                title_and_ylab = ['1 Week Continuous Predictions', desc]
                print(desc)
                plot_dataset(analysis_ds,
                             show=False,
                             title_and_ylab=title_and_ylab,
                             save_name=self.get_plt_path('OneWeek_' + str(k) + "_" + ext))
        pass

    def analyze(self, diff=False) -> None:
        """
        Analyzes the trained model and makes some
        plots.
        """

        if diff:
            raise NotImplementedError("Not fucking implemented!")

        print("Analyzing model {}".format(self.name))
        d = self.data

        # Prepare the data
        # dat_test = d.get_prepared_data('test')
        dat_train = d.get_prepared_data('train_streak')
        dat_val = d.get_prepared_data('val_streak')

        # Plot for fixed number of time-steps
        val_copy = copy_arr_list(dat_val)
        train_copy = copy_arr_list(dat_train)
        self.const_nts_plot(val_copy, [1, 4, 20], ext='Validation')
        self.const_nts_plot(train_copy, [4, 20], ext='Train')

        # Plot for continuous predictions
        self.one_week_pred_plot(copy_arr_list(dat_val), "Validation")
        self.one_week_pred_plot(copy_arr_list(dat_train), "Train")
        self.one_week_pred_plot(copy_arr_list(dat_val), "Validation_All", predict_all=True)
        self.one_week_pred_plot(copy_arr_list(dat_train), "Train_All", predict_all=True)

    def get_residuals(self, data_str: str):
        """
        Computes the residuals using the fitted model.

        :param data_str: String defining which part of the data to use.
        :return: Residuals.
        """
        input_data, output_data = self.data.get_prepared_data(data_str)
        residuals = self.predict(input_data) - output_data
        return residuals

    def deb(self, *args) -> None:
        """
        Prints Debug Info to console.

        :param args: Arguments as for print() function.
        :return: None
        """
        if self.debug:
            print(*args)
