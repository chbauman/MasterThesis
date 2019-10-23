from abc import ABC, abstractmethod
from data import Dataset, get_test_ds
from time_series import AR_Model
from util import *
from visualize import plot_dataset, model_plot_path


def get_plot_ds(s, tr: Optional[np.ndarray], d: Dataset, orig_p_ind: np.ndarray,
                n_offs: int = 0) -> Dataset:
    """
    Creates a dataset with truth time series tr and parameters
    from the original dataset d. Intended for plotting after
    the first column of the data is set to the predicted series.

    Args:
        n_offs: Number of time steps to offset t_init in the new dataset.
        s: Shape of data of dataset.
        tr: Ground truth time series.
        d: Original dataset.
        orig_p_ind: Prediction index relative to the original dataset.

    Returns:
        Dataset with ground truth series as second series.
    """
    plot_data = np.empty((s[0], 2), dtype=np.float32)
    if tr is not None:
        plot_data[:, 1] = tr
    scaling, is_scd = d.get_scaling_mul(orig_p_ind[0], 2)
    actual_dt = d.t_init if n_offs == 0 else d.get_shifted_t_init(n_offs + d.seq_len - 1)
    analysis_ds = Dataset(plot_data,
                          d.dt,
                          actual_dt,
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
    N_LAG: int = 6
    debug: bool = True
    model_path: str = "../Models/Dynamics/"
    use_AR: bool = True

    #: Dataset containing all the data
    data: Dataset
    out_inds: np.ndarray
    p_out_inds: np.ndarray
    in_indices: np.ndarray
    p_in_indices: np.ndarray
    name: str  #: Name of the model
    plot_path: str  #: Path to the plot folder
    n_pred_full: int
    n_pred: int  #: Number of dimensions of the prediction

    # Disturbance variables
    modeled_disturbance: bool = False
    res_std: Arr = None
    dist_mod = None
    init_pred: np.ndarray = None

    def __init__(self, ds: Dataset, name: str,
                 out_indices: np.ndarray = None,
                 in_indices: np.ndarray = None):
        """
        Constructor for the base of every dynamics model.
        If out_indices is None, all series are predicted.
        If in_indices is None, all series are used as input to the model.

        :param ds: Dataset containing all the data.
        :param name: Name of the model.
        :param out_indices: Indices specifying the series in the data that the model predicts.
        :param in_indices: Indices specifying the series in the data that the model takes as input.
        """

        # Set dataset
        self.data = ds

        # Indices
        out_inds_str = ""
        if out_indices is None:
            out_indices = ds.from_prepared(np.arange(ds.d - ds.n_c))
        else:
            out_inds_str = "_Out" + '-'.join(map(str, out_indices))
        self.out_inds = out_indices
        self.p_out_inds = ds.to_prepared(out_indices)
        ds.check_inds(out_indices, True)

        in_inds_str = ""
        if in_indices is None:
            in_indices = ds.from_prepared(np.arange(ds.d))
        else:
            in_inds_str = "_In" + '-'.join(map(str, in_indices))
        self.in_indices = in_indices
        self.p_in_indices = ds.to_prepared(in_indices)
        ds.check_inds(in_indices, True)

        # name
        self.name = ds.name + out_inds_str + in_inds_str + "_MODEL_" + name

        self.n_pred = len(out_indices)
        self.n_pred_full = ds.d - ds.n_c

        self.plot_path = os.path.join(model_plot_path, self.name)
        create_dir(self.plot_path)

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def predict(self, in_data):
        pass

    def init_1day(self, day_data: np.ndarray) -> None:
        """
        Initializer for models that need more previous data than seq_len
        time steps.

        :param day_data: The data of one day to initialize model.
        :return: None
        """
        pass

    def get_fit_data(self, data_name: str = "train"):
        """
        Returns the required data for fitting the model
        taking care of choosing the right series by indexing.

        Args:
            data_name: The string specifying which portion of the data to use.

        Returns:
            The input and output data for supervised learning.
        """
        in_dat, out_dat, n = self.data.get_split(data_name)
        res_in_dat = in_dat[:, :, self.p_in_indices]
        res_out_dat_out = out_dat[:, self.p_out_inds]
        return res_in_dat, res_out_dat_out

    def get_path(self, name: str) -> str:
        """
        Returns the path where the model parameters
        are stored. Used for keras models only.

        Args:
            name: Model name.

        Returns:
            Model parameter file path.
        """
        return self.model_path + name + ".h5"

    def load_if_exists(self, m, name: str) -> bool:
        """
        Loads the keras model if it exists.
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

    def model_disturbance(self, data_str: str = 'train'):
        """
        Models the uncertainties in the model
        by matching the distribution of the residuals.
        """

        # Compute residuals
        residuals = self.get_residuals(data_str)
        self.modeled_disturbance = True

        if self.use_AR:
            # Fit an AR process for each output dimension
            self.dist_mod = [AR_Model(lag=self.N_LAG) for _ in range(self.n_pred)]
            for k, d in enumerate(self.dist_mod):
                d.fit(residuals[:, k])
            self.reset_disturbance()

        self.res_std = np.std(residuals, axis=0)

    def disturb(self) -> np.ndarray:
        """
        Returns a sample of noise of length n.

        Returns:
            Numpy array of disturbances.

        Raises:
            AttributeError: If the disturbance model was not fitted before.
        """

        # Check if disturbance model was fitted
        if not self.modeled_disturbance:
            raise AttributeError('Need to model the disturbance first!')

        # Compute next noise
        if self.use_AR:
            next_noise = np.empty((self.n_pred,), dtype=np.float32)
            for k in range(self.n_pred):
                next_noise[k] = self.dist_mod[k].predict(self.init_pred[:, k])

            self.init_pred[:-1, :] = self.init_pred[1:, :]
            self.init_pred[-1, :] = next_noise
            return next_noise
        return np.random.normal(0, 1, self.n_pred) * self.res_std

    def reset_disturbance(self) -> None:
        """Resets the disturbance to zero.

        Returns:
            None
        """
        self.init_pred = np.zeros((self.N_LAG, self.n_pred), dtype=np.float32)

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
        :raises ValueError: If n < 0 or n too large or if the prepared data
            does not have the right shape.
        """

        in_data, out_data = copy_arr_list(prepared_data)

        # Get shapes
        n_pred = len(self.out_inds)
        n_tot = self.data.d
        n_samples = in_data.shape[0]
        n_feat = n_tot - self.data.n_c
        n_out = n_samples - n + 1

        # Prepare indices
        if pred_ind is None:
            # Predict all series in out_inds
            orig_pred_inds = np.copy(self.out_inds)
            out_inds = np.arange(n_pred)
        else:
            # Predict pred_ind'th series only
            mod_pred_ind = self.out_inds[pred_ind]
            orig_pred_inds = np.array([mod_pred_ind], dtype=np.int32)
            out_inds = np.array([pred_ind], dtype=np.int32)
        prep_pred_inds = self.data.to_prepared(orig_pred_inds)

        # Do checks
        if n < 1:
            raise ValueError("n: ({}) has to be larger than 0!".format(n))
        if n_out <= 0:
            raise ValueError("n: ({}) too large".format(n))
        if in_data.shape[0] != out_data.shape[0]:
            raise ValueError("Shape mismatch of prepared data.")
        if in_data.shape[-1] != n_tot or out_data.shape[-1] != n_feat:
            raise ValueError("Not the right number of dimensions in prepared data!")

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
                       predict_ind: int = None, n_ts_off: int = 0) -> None:
        """
        Creates a plot that shows the performance of the
        trained model when predicting a fixed number of timesteps into
        the future.
        If predict_ind is None, all series that can be
        predicted are predicted simultaneously and each predicted
        series is plotted individually.

        Args:
            predict_data: The string specifying the data to predict.
            n_list: The list of number of timesteps into the future.
            ext: Extension for the name of the plot.
            predict_ind: Which series to predict.
            n_ts_off: Number of time steps to shift the initial time for correct plotting.

        Returns:
            None
        """
        d = self.data
        in_d, out_d = predict_data
        s = in_d.shape
        dt = d.dt

        # Plot for all n
        if predict_ind is not None:
            orig_p_ind = np.copy(self.out_inds[predict_ind])
            p_ind = d.to_prepared(np.array([orig_p_ind]))[0]
            tr = np.copy(out_d[:, p_ind])

            analysis_ds = get_plot_ds(s, tr, d, orig_p_ind, n_ts_off)
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
            n_pred = len(self.out_inds)
            for n_ts in n_list:
                # Predict
                full_pred = self.n_step_predict([in_d, out_d], n_ts,
                                                pred_ind=predict_ind)
                time_str = mins_to_str(dt * n_ts)

                # Plot all
                for k in range(n_pred):
                    # Construct dataset and plot
                    k_orig = self.out_inds[k]
                    k_prep = self.data.to_prepared(np.array([k_orig]))[0]
                    k_orig_arr = np.array([k_orig])
                    new_ds = get_plot_ds(s, np.copy(out_d[:, k_prep]), d, k_orig_arr, n_ts_off)
                    desc = d.descriptions[k_orig]
                    new_ds.data[(n_ts - 1):, 0] = np.copy(full_pred[:, k])
                    new_ds.data[:(n_ts - 1), 0] = np.nan
                    title_and_ylab = [time_str + ' Ahead Predictions', desc]
                    plot_dataset(new_ds,
                                 show=False,
                                 title_and_ylab=title_and_ylab,
                                 save_name=self.get_plt_path(time_str + 'Ahead_' + str(k) + "_" + ext))

    def one_week_pred_plot(self, dat_test, ext: str = None,
                           predict_ind: int = None,
                           n_ts_off: int = 0) -> None:
        """
        Makes a plot by continuously predicting with
        the fitted model and comparing it to the ground
        truth. If predict_ind is None, all series that can be
        predicted are predicted simultaneously and each predicted
        series is plotted individually.

        Args:
            predict_ind: The index of the series to be plotted. If None, all series are plotted.
            dat_test: Data to use for making plots.
            ext: String extension for the filename.
            n_ts_off: Time step offset of data used.

        Returns: None
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
            n_pred = len(self.out_inds)
            for k in range(n_pred):
                # Construct dataset and plot
                k_orig = self.out_inds[k]
                k_orig_arr = np.array([k_orig])
                k_prep = self.data.to_prepared(k_orig_arr)[0]
                new_ds = get_plot_ds(s, np.copy(out_dat_test[:, k_prep]), d, k_orig_arr, n_ts_off)
                new_ds.data[:, 0] = np.copy(full_pred[0, :, k])
                desc = d.descriptions[k_orig]
                title_and_ylab = ['1 Week Continuous Predictions', desc]
                plot_dataset(new_ds,
                             show=False,
                             title_and_ylab=title_and_ylab,
                             save_name=self.get_plt_path('OneWeek_' + str(k) + "_" + ext))
        else:
            # Construct dataset and plot
            orig_p_ind = self.out_inds[predict_ind]
            analysis_ds = get_plot_ds(s, None, d, np.array([orig_p_ind]), n_ts_off)
            orig_p_ind = self.out_inds[predict_ind]
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
        plots using the fitted model and the streak data.

        Returns:
            None
        """

        print("Analyzing model {}".format(self.name))
        d = self.data

        # Prepare the data
        # dat_test = d.get_prepared_data('test')
        # dat_train = d.get_prepared_data('train_streak')
        # dat_val = d.get_prepared_data('val_streak')
        dat_train = d.get_streak('train')
        dat_val = d.get_streak('val')

        n_train = dat_train[2]
        n_val = dat_val[2]
        dat_train = [dat_train[0], dat_train[1]]
        dat_val = [dat_val[0], dat_val[1]]

        # Plot for fixed number of time-steps
        val_copy = copy_arr_list(dat_val)
        train_copy = copy_arr_list(dat_train)
        self.const_nts_plot(val_copy, [1, 4, 20], ext='Validation_All', n_ts_off=n_val)
        self.const_nts_plot(train_copy, [1, 4, 20], ext='Train_All', n_ts_off=n_train)

        # Plot for continuous predictions
        self.one_week_pred_plot(copy_arr_list(dat_val), "Validation_All", n_ts_off=n_val)
        self.one_week_pred_plot(copy_arr_list(dat_train), "Train_All", n_ts_off=n_train)

        # Make more predictions
        # n_pred = len(self.out_inds)
        # if n_pred > 1:
        #     for k in range(n_pred):
        #         ext = str(k) + "__"
        #         self.one_week_pred_plot(copy_arr_list(dat_val), ext + "Validation",
        #                                 predict_ind=k)
        #         self.one_week_pred_plot(copy_arr_list(dat_train), ext + "Train",
        #                                 predict_ind=k)

    def analyze_6_days(self) -> None:
        """
        Analyzes this model using the 7 day streaks.

        :return: None
        """
        d = self.data
        n = 60 * 24 // d.dt

        # Prepare the data
        # dat_test = d.get_prepared_data('test')
        # dat_train_in, dat_train_out, _ = d.get_prepared_data('train_streak')
        # dat_val_in, dat_val_out, _ = d.get_prepared_data('val_streak')
        dat_train_in, dat_train_out, _ = d.get_streak('train')
        dat_val_in, dat_val_out, _ = d.get_streak('val')
        n_feat = dat_train_out.shape[-1]

        # Extract first day
        dat_train_init = dat_train_out[:n].reshape((1, -1, n_feat))
        dat_val_init = dat_val_out[:n].reshape((1, -1, n_feat))

        # Collect rest
        dat_train_used = dat_train_in[n:], dat_train_out[n:]
        dat_val_used = dat_val_in[n:], dat_val_out[n:]

        # Plot for continuous predictions
        self.init_1day(dat_train_init)
        self.one_week_pred_plot(copy_arr_list(dat_train_used), "6d_Train_All")
        self.init_1day(dat_val_init)
        self.one_week_pred_plot(copy_arr_list(dat_val_used), "6d_Validation_All")

    def analyze_disturbed(self,
                          ext: str = None,
                          data_str: str = "val",
                          n_trials: int = 25) -> None:
        """
            Makes a plot by continuously predicting with
            the fitted model and comparing it to the ground
            truth. If predict_ind is None, all series that can be
            predicted are predicted simultaneously and each predicted
            series is plotted individually.

            Args:
                data_str: The string specifying the data to use.
                n_trials: Number of predictions with noise to average.
                ext: String extension for the filename.

            Returns: None
            """

        # Model the disturbance
        self.model_disturbance("train")

        # Get the data
        d = self.data
        dat_val = d.get_streak(data_str)
        in_dat_test, out_dat_test, n_ts_off = dat_val

        # Predict without noise
        s = in_dat_test.shape
        full_pred = self.n_step_predict([in_dat_test, out_dat_test], s[0],
                                        pred_ind=None,
                                        return_all_predictions=True)

        # Predict with noise
        s_pred = full_pred.shape
        all_noise_preds = np.empty((n_trials, s_pred[1], s_pred[2]), dtype=full_pred.dtype)
        for k in range(n_trials):
            all_noise_preds[k] = self.n_step_predict([in_dat_test, out_dat_test], s[0],
                                                     pred_ind=None,
                                                     return_all_predictions=True,
                                                     disturb_pred=True)
        mean_noise_preds = np.mean(all_noise_preds, axis=0)
        std_noise_preds = np.std(all_noise_preds, axis=0)

        # Build dataset
        plot_data = np.empty((s_pred[1], 5), dtype=np.float32)
        actual_dt = d.get_shifted_t_init(n_ts_off)
        descs = ['Prediction',
                 'Ground Truth',
                 'Mean Noisy Prediction',
                 'Noisy Prediction +2 STD.',
                 'Noisy Prediction -2 STD.']
        ext = "_" if ext is None else "_" + ext

        for k in range(self.n_pred):
            # Construct dataset and plot
            k_orig = self.out_inds[k]
            k_prep = self.data.to_prepared(np.array([k_orig]))[0]
            scaling, is_scd = d.get_scaling_mul(k_orig, 5)
            plot_data[:, 0] = np.copy(full_pred[0, :, k])
            plot_data[:, 1] = np.copy(out_dat_test[:, k_prep])
            mean_pred = mean_noise_preds[:, k]
            std_pred = std_noise_preds[:, k]
            plot_data[:, 2] = np.copy(mean_pred)
            plot_data[:, 3] = np.copy(mean_pred + 2 * std_pred)
            plot_data[:, 4] = np.copy(mean_pred - 2 * std_pred)

            desc = d.descriptions[k_orig]
            title_and_ylab = ['1 Week Predictions', desc]
            analysis_ds = Dataset(plot_data,
                                  d.dt,
                                  actual_dt,
                                  scaling,
                                  is_scd,
                                  descs)
            plot_dataset(analysis_ds,
                         show=False,
                         title_and_ylab=title_and_ylab,
                         save_name=self.get_plt_path('OneWeek_WithNoise_' + str(k) + "_" + ext))

    def hyper_obj(self, n_ts: int = 16, series_ind: Arr = None) -> float:
        """Defines the objective for the hyperparameter optimization.

        Uses multistep prediction to define the performance for
        the hyperparameter optimization. Objective needs to be minimized.

        Args:
            n_ts: Number of timesteps to predict.
            series_ind: The indices of the series to predict.

        Returns:
            The numerical value of the objective.
        """
        d = self.data

        # Transform indices and get data
        if isinstance(series_ind, (int, float)):
            series_ind = d.to_prepared(np.array([series_ind]))
        elif series_ind is None:
            series_ind = self.p_out_inds
        in_d, out_d, _ = d.get_streak('val')
        tr = np.copy(out_d[:, series_ind])

        # Predict and compute residuals
        one_h_pred = self.n_step_predict(copy_arr_list([in_d, out_d]), n_ts,
                                         pred_ind=None)
        residuals = tr[(n_ts - 1):] - one_h_pred[:, series_ind]
        return float(np.sum(residuals * residuals))

    def get_residuals(self, data_str: str):
        """
        Computes the residuals using the fitted model.

        :param data_str: String defining which part of the data to use.
        :return: Residuals.
        """
        input_data, output_data, n = self.data.get_split(data_str)
        prep_inds = self.data.to_prepared(self.out_inds)
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


##########################################################################
# Testing stuff
class TestModel(BaseDynamicsModel):
    """Dummy dynamics model class for testing.

    Does not fit anything. Works only with datasets
    that have exactly 3 series to predict and one
    control variable series.
    """
    n_prediction: int = 0
    n_pred: int = 3

    def __init__(self,
                 ds: Dataset):
        name = "TestModel"
        super(TestModel, self).__init__(ds, name)
        self.use_AR = False

        if ds.n_c != 1:
            raise ValueError("Only one control variable allowed!!")
        if ds.d - 1 != self.n_pred:
            raise ValueError("Dataset needs exactly 3 non-controllable state variables!")

    def fit(self) -> None:
        pass

    def predict(self, in_data: np.ndarray) -> np.ndarray:

        rel_out_dat = -0.9 * in_data[:, -1, :self.n_pred]
        # rel_out_dat[:, 1] = -0.9 * rel_out_dat[:, 1]
        # rel_out_dat[:, 2] = -0.9 * rel_out_dat[:, 2]

        self.n_prediction += 1
        return rel_out_dat


def construct_test_ds(n: int = 201) -> Dataset:
    # Define dataset
    dat = np.empty((n, 4), dtype=np.float32)
    dat[:, 0] = np.arange(n)
    dat[:, 1] = np.reciprocal(1.0 + np.arange(n))
    dat[:, 2] = np.reciprocal(1.0 + np.arange(n))
    dat[:, 3] = 1
    c_inds = np.array([3])
    ds = get_test_ds(dat, c_inds, dt=60 * 6, name="SyntheticModelData")
    ds.seq_len = 8
    ds.val_percent = 0.3
    ds.split_data()
    return ds


def test_dyn_model() -> None:
    """Tests the dynamic model base class with the TestModel.

    Returns:
        None

    Raises:
        AssertionError: If a test fails.
    """

    # Define dataset
    n = 201
    ds = construct_test_ds(n)

    # Compute sizes
    n_val = n - int((1.0 - ds.val_percent) * n)
    n_train = n - 2 * n_val
    n_train_seqs = n_train - ds.seq_len + 1
    n_streak = 7 * 24 * 60 // ds.dt
    n_streak_offset = n_train_seqs - n_streak

    # Check sizes
    dat_in_train, dat_out_train, _ = ds.get_split("train")
    if len(dat_in_train) != n_train_seqs:
        raise AssertionError("Something baaaaad!")
    dat_in, dat_out, n_str = ds.get_streak("train")
    if n_streak_offset != n_str:
        raise AssertionError("Streak offset fucking wrong!!")

    # Find time of initial output
    n_streak_output_offset = n_streak_offset + ds.seq_len - 1
    dt_offset = n_mins_to_np_dt(ds.dt * n_streak_output_offset)
    t_init_streak = str_to_np_dt(ds.t_init) + dt_offset

    # Initialize, fit and analyze model
    test_mod = TestModel(ds)
    test_mod.fit()
    test_mod.analyze()

    # Test prediction and residuals
    sample_pred_inp = ds.data[:(ds.seq_len - 1)]
    sample_pred_inp = sample_pred_inp.reshape((1, sample_pred_inp.shape[0], -1))
    test_mod.predict(sample_pred_inp)
    mod_out_test = test_mod.predict(dat_in_train)
    res = test_mod.get_residuals("train")
    if not np.allclose(mod_out_test - dat_out_train, res):
        raise AssertionError("Residual computation wrong!!")

    print("First point in week plot should be at: {}".format(t_init_streak))

    print("Dynamic model test passed :)")
