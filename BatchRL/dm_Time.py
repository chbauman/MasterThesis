from base_dynamics_model import BaseDynamicsModel
from util import *
from visualize import model_plot_path
from data import Dataset


class SCTimeModel(BaseDynamicsModel):
    """
    The naive model that predicts the last
    input seen.
    """

    def __init__(self, dataset: Dataset, time_ind: int = None):
        """
        Initialize the Sine-Cosine-Time model.
        It predicts the next values given only the previous
        values of the sine and the cosine of the time.

        :param dataset: Dataset containing two time series, sin(time) and cos(time).
        :param time_ind: Specifying which column holds the sin(t) series.
                        The cos(t) series is assumed to be in column time_ind + 1.
        """
        # Compute indices and name
        name = dataset.name + "_Exact"
        if time_ind is None:
            time_ind = dataset.d - 2
        if time_ind > dataset.d - 2 or time_ind < 0:
            raise IndexError("Time index out of range.")
        inds = np.array([time_ind, time_ind + 1], dtype=np.int32)
        super(SCTimeModel, self).__init__(dataset, name, inds)

        # Save parameters
        self.dx = 2 * np.pi / (24 * 60 / dataset.dt)

        # Scaling parameters
        s_ind, c_ind = self.out_inds
        if dataset.is_scaled[s_ind] != dataset.is_scaled[c_ind]:
            raise AttributeError("Be fucking consistent with the scaling!")
        self.is_scaled = dataset.is_scaled[s_ind] and dataset.is_scaled[c_ind]
        self.s_scale, self.c_scale = dataset.scaling[s_ind], dataset.scaling[c_ind]
        self.s_ind_prep, self.c_ind_prep = dataset.to_prepared(self.out_inds)

    def fit(self) -> None:
        """
        No need to fit anything, model is deterministic
        and exact up to numerical errors.

        :return: None
        """
        return

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """
        Compute the next sin(t) and cos(t) value given the
        values at the last timestep.

        :param in_data: Prepared data.
        :return: Same as input
        """

        in_sh = in_data.shape
        s_ind, c_ind = self.s_ind_prep, self.c_ind_prep

        # Get previous values
        s = np.copy(in_data[:, -1, s_ind])
        c = np.copy(in_data[:, -1, c_ind])

        # Scale back
        if self.is_scaled:
            s = add_mean_and_std(s, self.s_scale)
            c = add_mean_and_std(c, self.c_scale)

        # Compute new
        x = np.arccos(c)
        x = np.where(s < 0, -x, x) + self.dx
        s_new = np.sin(x)
        c_new = np.cos(x)

        # Evaluate and scale
        if self.is_scaled:
            s_new = rem_mean_and_std(s_new, self.s_scale)
            c_new = rem_mean_and_std(c_new, self.c_scale)

        # Concatenate and return
        out_dat = np.empty((in_sh[0], 2), dtype=in_data.dtype)
        out_dat[:, 0] = s_new
        out_dat[:, 1] = c_new
        return out_dat
