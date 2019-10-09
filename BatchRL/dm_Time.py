from base_dynamics_model import BaseDynamicsModel
from util import *
from visualize import model_plot_path
from data import Dataset


class SCTimeModel(BaseDynamicsModel):
    """
    The naive model that predicts the last
    input seen.
    """

    def __init__(self, dataset: Dataset):
        super(SCTimeModel, self).__init__()

        self.name = dataset.name + "_Exact"
        self.plot_path = os.path.join(model_plot_path, self.name)
        create_dir(self.plot_path)

        # Save dataset
        self.data = dataset
        self.nc = dataset.n_c

    def fit(self) -> None:
        """
        No need to fit anything.
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

        # Get previous values
        s = np.copy(in_data[:, -1, 0])
        c = np.copy(in_data[:, -1, 1])

        # Scale back
        if self.data.is_scaled[0]:
            s = add_mean_and_std(s, self.data.scaling[0])
        if self.data.is_scaled[1]:
            c = add_mean_and_std(c, self.data.scaling[1])

        # Compute new
        dx = 2 * np.pi / (24 * 60 / self.data.dt)
        x = np.arccos(c)
        x = np.where(s < 0, -x, x)

        print(s, " true ", c)
        print(np.sin(x), " comp ", np.cos(x))

        x += dx
        s_new = np.sin(x)
        c_new = np.cos(x)

        # Evaluate and scale
        if self.data.is_scaled[0]:
            s_new = rem_mean_and_std(s_new, self.data.scaling[0])
        if self.data.is_scaled[1]:
            c_new = rem_mean_and_std(c_new, self.data.scaling[1])

        # Concatenate and return
        res = np.concatenate([s_new.reshape((-1, 1)), c_new.reshape((-1, 1))], axis=1)
        return res

    def disturb(self):
        """
        Returns a sample of noise of length n.
        """
        raise NotImplementedError("Disturbance for naive model not implemented!")
