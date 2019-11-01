from base_dynamics_model import BaseDynamicsModel
from util import *
from visualize import scatter_plot, model_plot_path
from data import Dataset


class BatteryModel(BaseDynamicsModel):
    """
    The model of the battery:
    :math:`s_t`: SoC at time t
    Model:
    :math:`s_{t+1} = s_t + \\eta(s_t) p_{t+1}`,
    :math:`p_{t+1}`:
    average charging power from time t to t+1 (control input)
    """

    def __init__(self, dataset: Dataset):
        """
        Initializes the battery model with the specified
        dataset.

        :param dataset: Dataset with data to fit modes.
        """
        super(BatteryModel, self).__init__(dataset, dataset.name, None)
        self.params = None

    def fit(self) -> None:
        """
        Calls analyze_bat_model, there the
        actual model fitting happens.

        :return: None
        """
        self.analyze_bat_model()

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model
        on the provided data.

        :param in_data: Prepared data.
        :return: Predictions
        """
        p = np.copy(in_data[:, -1, 1])
        s_t = np.copy(in_data[:, -1, 0])
        a1, a2, a3 = self.params

        s_tp1 = s_t + a1 + a2 * p + a3 * np.maximum(0, p)
        return s_tp1.reshape((-1, 1))

    def disturb(self):
        """
        Returns a sample of noise of length n.
        """
        raise NotImplementedError("Disturbance for battery model not implemented!")

    def analyze_bat_model(self):
        """
        This is basically the fit method, but it also
        does some data analysis and makes battery data specific some plots.

        :return: None
        """
        # Get data
        d = self.data
        dat_in, dat_out, inds = d.get_split("train_val")
        print(dat_in.shape)
        scale = np.copy(d.scaling)
        scale[0, 0] = 0.0

        # Extract data
        p = np.copy(dat[1:, 1])
        ds = np.copy(dat[1:, 0] - dat[:-1, 0])

        # Remove nans
        not_nans = np.logical_not(np.logical_or(np.isnan(p), np.isnan(ds)))
        p = p[not_nans]
        ds = ds[not_nans]

        # Plot data
        labs = {'title': 'Battery Model', 'xlab': 'Active Power [kW]', 'ylab': r'$\Delta$ SoC [%]'}
        before_plt_path = os.path.join(self.plot_path, "WithOutliers")
        scatter_plot(p, ds, lab_dict=labs,
                     show=False,
                     m_and_std_x=scale[1],
                     m_and_std_y=scale[0],
                     add_line=True,
                     save_name=before_plt_path)

        # Fit linear Model and filter out outliers
        fitted_ds = fit_linear_1d(p, ds, p)
        mask = np.logical_or(ds > fitted_ds - 0.35, p < -1.0)
        masked_p = p[mask]
        masked_ds = ds[mask]
        n_mask = masked_p.shape[0]

        # Fit pw. linear model: $y = \alpha_1 + \alpha_2 * x * \alpha_3 * max(0, x)$
        ls_mat = np.empty((n_mask, 3), dtype=np.float32)
        ls_mat[:, 0] = 1
        ls_mat[:, 1] = masked_p
        ls_mat[:, 2] = masked_p
        ls_mat[:, 2][ls_mat[:, 2] < 0] = 0
        self.params = np.linalg.lstsq(ls_mat, masked_ds, rcond=None)[0]
        a1, a2, a3 = self.params
        x_pw_line = np.array([np.min(p), 0, np.max(p)], dtype=np.float32)
        y_pw_line = a1 + a2 * x_pw_line + a3 * np.maximum(0, x_pw_line)

        # Plot model
        after_plt_path = os.path.join(self.plot_path, "Cleaned")
        scatter_plot(masked_p, masked_ds, lab_dict=labs,
                     show=False,
                     add_line=False,
                     m_and_std_x=scale[1],
                     m_and_std_y=scale[0],
                     custom_line=[x_pw_line, y_pw_line],
                     custom_label='PW Linear Fit',
                     save_name=after_plt_path)
