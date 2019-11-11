from dynamics.base_model import BaseDynamicsModel
from data import Dataset
from util.visualize import scatter_plot
from util.util import *


class BatteryModel(BaseDynamicsModel):
    """Dynamics model of the battery.

    The model of the battery:
    :math:`s_t`: SoC at time t
    Model:
    :math:`s_{t+1} = s_t + \\eta(s_t) p_{t+1}`,
    :math:`p_{t+1}`:
    average charging power from time t to t+1 (control input)
    """
    params: np.ndarray = None  #: Parameters of the pw linear model.

    def __init__(self, dataset: Dataset):
        """Initializes the battery model with the specified dataset.

        Args:
            dataset: Dataset with data to fit modes.
        """
        super().__init__(dataset, dataset.name, None)

    def fit(self) -> None:
        """Fits the battery model.

        Calls `analyze_bat_model`, there the
        actual model fitting happens.
        """
        self.analyze_bat_model()

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model on the provided data.

        Args:
            in_data: Prepared data.

        Returns:
            Predictions
        """
        p = np.copy(in_data[:, -1, 1])
        s_t = np.copy(in_data[:, -1, 0])

        s_tp1 = s_t + self._eval_at(p)
        return s_tp1.reshape((-1, 1))

    def disturb(self):
        """Returns a sample of noise.
        """
        return 0

    def _eval_at(self, p):
        """Evaluates the model for a given active power `p`."""
        a1, a2, a3 = self.params
        return a1 + a2 * p + a3 * np.maximum(0, p)

    def analyze_bat_model(self) -> None:
        """This is basically the fit method, but it also
        does some data analysis and makes some battery data specific plots.
        """
        # Get data
        d = self.data
        dat = d.split_dict["train_val"].get_rel_data()
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

        # Fit pw. linear model: $y = \alpha_1 + \alpha_2 * x * \alpha_3 * max(0, x)$
        def feat_fun(x: float):
            return np.array([1.0, x, max(0.0, x)])
        params = fit_linear_bf_1d(masked_p, masked_ds, feat_fun)
        self.params = params
        x_pw_line = np.array([np.min(p), 0, np.max(p)], dtype=np.float32)
        y_pw_line = self._eval_at(x_pw_line)

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
