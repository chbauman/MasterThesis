import os

import numpy as np

from keras.models import load_model
from abc import ABC, abstractmethod

from time_series import AR_Model
from visualize import plot_dataset, model_plot_path
from data import Dataset
from util import *


def get_plot_ds(s, tr: Optional[np.ndarray], d: Dataset, orig_p_ind: np.ndarray) -> Dataset:
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
    if tr is not None:
        plot_data[:, 1] = tr
    scaling = np.array(repl(d.scaling[orig_p_ind], 2))
    scaling = scaling.reshape((-1, 2))
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
    p_pred_inds: np.ndarray
    name: str
    plot_path: str
    n_pred_full: int
    n_pred: int

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

        # Indices
        if pred_indices is None:
            pred_indices = np.arange(ds.d - ds.n_c)
        self.pred_inds = pred_indices
        self.p_pred_inds = ds.to_prepared(pred_indices)

        self.n_pred = len(pred_indices)
        self.n_pred_full = ds.d - ds.n_c

        self.plot_path = os.path.join(model_plot_path, name)
        create_dir(self.plot_path)

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

    def n_step_predict(self, prepared_data: Sequence, n: int, *,
                       pred_ind: int = None,
                       return_all_predictions: bool = False,
                       disturb_pred: bool = False) -> np.ndarray:
        """
        Applies the model n times and returns the
        predictions.

        :param prepared_data: Data to predict.
        :param n: Number of timesteps to predict.
        :param pred_ind: Which series to predict, all if None.
        :param return_all_predictions: Whether to return intermediate predictions.
        :param disturb_pred: Whether to apply a disturbance to the prediction.
        :return: The predictions.
        """

        in_data, out_data = copy_arr_list(prepared_data)

        # Get shapes
        n_pred = len(self.pred_inds)
        n_tot = self.data.d
        n_samples = in_data.shape[0]
        n_feat = n_tot - self.data.n_c
        n_out = n_samples - n + 1

        # Prepare indices
        if pred_ind is None:
            # Predict all series in pred_inds
            orig_pred_inds = np.copy(self.pred_inds)
            out_inds = np.arange(n_pred)
        else:
            # Predict pred_ind'th series only
            mod_pred_ind = self.pred_inds[pred_ind]
            orig_pred_inds = np.array([mod_pred_ind], dtype=np.int32)
            out_inds = np.array([pred_ind], dtype=np.int32)
        prep_pred_inds = self.data.to_prepared(orig_pred_inds)

        # Do checks
        if n < 1:
            raise ValueError("n: ({}) has to be larger than 0!".format(n))
        if n_out <= 0:
            raise ValueError("n: ({}) too large".format(n))

        # Initialize values and reserve output array
        all_pred = None
        if return_all_predictions:
            all_pred = np.empty((n_out, n, n_pred))
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
            curr_in_data[:, -1, :n_feat] = np.copy(out_data[k:(n_out + k), :])
            if k != n - 1:
                curr_in_data[:, -1, n_feat:] = np.copy(in_data[(k + 1):(n_out + k + 1), -1, n_feat:])
            else:
                curr_in_data[:, -1, n_feat:] = 0
            curr_in_data[:, -1, prep_pred_inds] = np.copy(curr_pred[:, out_inds])

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
        dir_name = self.plot_path
        return os.path.join(dir_name, name)

    def const_nts_plot(self, predict_data, n_list: List[int], ext: str = '', *,
                       predict_ind: int = None) -> None:
        """
        Creates a plot that shows the performance of the
        trained model when predicting a fixed number of timesteps into
        the future.
        If predict_ind is None, all series that can be
        predicted are predicted simultaneously and each predicted
        series is plotted individually.

        :param predict_data: The string specifying the data to predict.
        :param n_list: The list of number of timesteps into the future.
        :param ext: Extension for the name of the plot.
        :param predict_ind: Which series to predict.
        :return: None
        """
        d = self.data
        in_d, out_d = predict_data
        s = in_d.shape
        dt = d.dt

        # Plot for all n
        if predict_ind is not None:
            orig_p_ind = np.copy(self.pred_inds[predict_ind])
            p_ind = d.to_prepared(np.array([orig_p_ind]))[0]
            tr = np.copy(out_d[:, p_ind])

            analysis_ds = get_plot_ds(s, tr, d, orig_p_ind)
            desc = d.descriptions[orig_p_ind]
            for n_ts in n_list:
                curr_ds = Dataset.copy(analysis_ds)
                time_str = mins_to_str(dt * n_ts)
                one_h_pred = self.n_step_predict(copy_arr_list(predict_data), n_ts,
                                                 pred_ind=predict_ind)
                curr_ds.data[(n_ts - 1):, 0] = np.copy(one_h_pred[:, predict_ind])
                curr_ds.data[:(n_ts - 1), 0] = np.nan
                title_and_ylab = [time_str + ' Ahead Predictions', desc]
                plot_dataset(curr_ds,
                             show=False,
                             title_and_ylab=title_and_ylab,
                             save_name=self.get_plt_path(time_str + 'Ahead' + ext))
        else:
            n_pred = len(self.pred_inds)
            for n_ts in n_list:
                # Predict
                full_pred = self.n_step_predict([in_d, out_d], n_ts,
                                                pred_ind=predict_ind)
                time_str = mins_to_str(dt * n_ts)

                # Plot all
                for k in range(n_pred):
                    # Construct dataset and plot
                    k_orig = self.pred_inds[k]
                    k_prep = self.data.to_prepared(np.array([k_orig]))[0]
                    k_orig_arr = np.array([k_orig])
                    new_ds = get_plot_ds(s, np.copy(out_d[:, k_prep]), d, k_orig_arr)
                    desc = d.descriptions[k_orig]
                    new_ds.data[(n_ts - 1):, 0] = np.copy(full_pred[:, k])
                    new_ds.data[:(n_ts - 1), 0] = np.nan
                    title_and_ylab = [time_str + ' Ahead Predictions', desc]
                    plot_dataset(new_ds,
                                 show=False,
                                 title_and_ylab=title_and_ylab,
                                 save_name=self.get_plt_path(time_str + 'Ahead_' + str(k) + "_" + ext))

    def one_week_pred_plot(self, dat_test, ext: str = None,
                           predict_ind: int = None):
        """
        Makes a plot by continuously predicting with
        the fitted model and comparing it to the ground
        truth. If predict_ind is None, all series that can be
        predicted are predicted simultaneously and each predicted
        series is plotted individually.

        :param predict_ind: The index of the series to be plotted. If None, all series are plotted.
        :param dat_test: Data to use for making plots.
        :param ext: String extension for the filename.
        :return: None
        """

        ext = "_" if ext is None else "_" + ext
        d = self.data
        in_dat_test, out_dat_test = dat_test
        s = in_dat_test.shape

        # Continuous prediction
        full_pred = self.n_step_predict([in_dat_test, out_dat_test], s[0],
                                        pred_ind=predict_ind,
                                        return_all_predictions=True)

        if predict_ind is None:
            n_pred = len(self.pred_inds)
            for k in range(n_pred):
                # Construct dataset and plot
                k_orig = self.pred_inds[k]
                k_prep = self.data.to_prepared(np.array([k_orig]))[0]
                k_orig_arr = np.array([k_orig])
                new_ds = get_plot_ds(s, np.copy(out_dat_test[:, k_prep]), d, k_orig_arr)
                new_ds.data[:, 0] = np.copy(full_pred[0, :, k])
                desc = d.descriptions[k_orig]
                title_and_ylab = ['1 Week Continuous Predictions', desc]
                plot_dataset(new_ds,
                             show=False,
                             title_and_ylab=title_and_ylab,
                             save_name=self.get_plt_path('OneWeek_' + str(k) + "_" + ext))
        else:
            # Construct dataset and plot
            orig_p_ind = self.pred_inds[predict_ind]
            analysis_ds = get_plot_ds(s, None, d, np.array([orig_p_ind]))
            orig_p_ind = self.pred_inds[predict_ind]
            p_ind = d.to_prepared(np.array([orig_p_ind]))[0]
            desc = d.descriptions[orig_p_ind]
            analysis_ds.data[:, 0] = np.copy(full_pred[0, :, predict_ind])
            analysis_ds.data[:, 1] = np.copy(out_dat_test[:, p_ind])
            title_and_ylab = ['1 Week Continuous Predictions', desc]
            plot_dataset(analysis_ds,
                         show=False,
                         title_and_ylab=title_and_ylab,
                         save_name=self.get_plt_path('OneWeek' + ext))

    def analyze(self) -> None:
        """
        Analyzes the trained model and makes some
        plots.

        :return: None
        """

        print("Analyzing model {}".format(self.name))
        d = self.data

        # Prepare the data
        # dat_test = d.get_prepared_data('test')
        dat_train = d.get_prepared_data('train_streak')
        dat_val = d.get_prepared_data('val_streak')

        # Plot for fixed number of time-steps
        val_copy = copy_arr_list(dat_val)
        train_copy = copy_arr_list(dat_train)
        self.const_nts_plot(val_copy, [1, 4, 20], ext='Validation_All')
        self.const_nts_plot(train_copy, [1, 4, 20], ext='Train_All')

        # Plot for continuous predictions
        self.one_week_pred_plot(copy_arr_list(dat_val), "Validation_All")
        self.one_week_pred_plot(copy_arr_list(dat_train), "Train_All")

        # Make more predictions
        # n_pred = len(self.pred_inds)
        # if n_pred > 1:
        #     for k in range(n_pred):
        #         ext = str(k) + "__"
        #         self.one_week_pred_plot(copy_arr_list(dat_val), ext + "Validation",
        #                                 predict_ind=k)
        #         self.one_week_pred_plot(copy_arr_list(dat_train), ext + "Train",
        #                                 predict_ind=k)

    def get_residuals(self, data_str: str):
        """
        Computes the residuals using the fitted model.

        :param data_str: String defining which part of the data to use.
        :return: Residuals.
        """
        input_data, output_data = self.data.get_prepared_data(data_str)
        prep_inds = self.data.to_prepared(self.pred_inds)
        residuals = self.predict(input_data) - output_data[:, prep_inds]
        return residuals

    def deb(self, *args) -> None:
        """
        Prints Debug Info to console.

        :param args: Arguments as for print() function.
        :return: None
        """
        if self.debug:
            print(*args)
