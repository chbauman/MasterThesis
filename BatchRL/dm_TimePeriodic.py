from base_dynamics_model import BaseDynamicsModel
from util import *
from data import Dataset


class Periodic1DayModel(BaseDynamicsModel):
    """
    The 1-day periodic model that predicts the values as the ones
    from the last day.
    """

    # Member variables
    hist: np.ndarray
    pred_t: int = 0

    def __init__(self, d: Dataset, exo_inds: np.ndarray = None, alpha: float = 1.0):
        """
        Initializes model.

        :param d: Dataset to use.
        :param exo_inds: The indices of the series that will be predicted.
        :param alpha: The decay parameter alpha.
        """
        name = "1DayPeriodic_Alpha" + str(alpha)
        super(Periodic1DayModel, self).__init__(d, name, exo_inds)

        # Save parameters
        self.data = d
        self.n: int = 60 * 24 // d.dt
        self.alpha: float = alpha

    def init_1day(self, day_data: np.ndarray) -> None:
        """
        Sets the history and resets time.

        :param day_data: New history data to use.
        :return: None
        """
        self.pred_t = 0
        self.hist = day_data

    def fit(self) -> None:
        """
        No need to fit anything.

        :return: None
        """
        pass

    def curr_alpha(self, t: int) -> float:
        """
        The decay function:
        :math:`\\alpha_t = \\e^{-\\alpha t} \\in [0, 1]`

        :param t: Time variable.
        :return: Current weight of the input.
        """
        return np.exp(-self.alpha * t)

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """
        Make predictions converging to the data in self.hist
        when predicting for multiple times in a row.
        Starts with predicting the last input.

        :param in_data: Prepared data.
        :return: New prediction.
        """

        # Update time
        self.pred_t = (self.pred_t + 1) % self.n

        # Make prediction
        curr_in = in_data[:, -1, :]
        curr_h = self.hist[:, self.pred_t, :]
        curr_a = self.curr_alpha(self.pred_t)
        curr_out = curr_a * curr_in + (1.0 - curr_a) * curr_h
        return curr_out

    def disturb(self):
        """
        Returns a sample of noise of length n.
        """
        raise NotImplementedError("Disturbance for naive model not implemented!")

    def analyze_6_days(self) -> None:
        """
        Analyzes this model using the 7 day streaks.

        :return: None
        """
        d = self.data
        n = 60 * 24 // d.dt

        # Prepare the data
        # dat_test = d.get_prepared_data('test')
        dat_train_in, dat_train_out = d.get_prepared_data('train_streak')
        dat_val_in, dat_val_out = d.get_prepared_data('val_streak')
        n_feat = dat_train_out.shape[-1]

        # Extract first day
        dat_train_init = dat_train_out[:n].reshape((1, -1, n_feat))
        dat_val_init = dat_val_out[:n].reshape((1, -1, n_feat))

        # Collect rest
        dat_train_used = dat_train_in[n:], dat_train_out[n:]
        dat_val_used = dat_val_in[n:], dat_val_out[n:]

        # Plot for continuous predictions
        self.init_1day(dat_train_init)
        self.one_week_pred_plot(copy_arr_list(dat_train_used), "Train_All")
        self.init_1day(dat_val_init)
        self.one_week_pred_plot(copy_arr_list(dat_val_used), "Validation_All")
