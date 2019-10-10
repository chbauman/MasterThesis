from base_dynamics_model import BaseDynamicsModel
from util import *
from visualize import model_plot_path
from data import Dataset


class Periodic1DayModel(BaseDynamicsModel):
    """
    The periodic model that predicts the values as the ones
    from the last day.
    """

    # Member variables
    hist: np.ndarray
    pred_n: int = 0

    def __init__(self, d: Dataset, exo_inds: np.ndarray, alpha: float = 1.0):

        name = d.name + "_1DayPeriodic_Alpha" + str(alpha)
        super(Periodic1DayModel, self).__init__(d, name, exo_inds)

        # Save parameters
        self.data = d
        self.n_feat: int = len(self.pred_indices)
        self.n: int = 60 * 24 // self.d.dt

    def init_1day(self, day_data: np.ndarray) -> None:

        self.pred_n = 0
        self.hist = day_data

    def fit(self) -> None:
        """
        No need to fit anything.

        :return: None
        """
        pass

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """
        Make predictions by just returning the last input.

        :param in_data: Prepared data.
        :return:
        """

        # Initialize history
        if self.pred_n == 0:
            s = in_data.shape
            if s[0] != 1:
                raise NotImplementedError("Not implemented for multiple predictions at once!")
            seq_len = s[1]
            self.hist = np.empty((self.n, seq_len, self.n_feats))

        # Update history
        self.hist[:-1, :, :] = self.hist[1:, :, :]
        self.hist[-1:] = in_data
        self.pred_n += 1

        # Make prediction
        if self.pred_n >= self.n:
            return in_data[-self.n: (1 - self.n), -1, :]
        return in_data[:, -1, :]

    def disturb(self):
        """
        Returns a sample of noise of length n.
        """
        raise NotImplementedError("Disturbance for naive model not implemented!")
