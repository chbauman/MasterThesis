from base_dynamics_model import BaseDynamicsModel
from util import *
from visualize import model_plot_path
from data import Dataset


class ConstModel(BaseDynamicsModel):
    """
    The naive model that predicts the last
    input seen.
    """

    def __init__(self, dataset: Dataset):
        name = dataset.name + "_Naive"
        super(ConstModel, self).__init__(dataset, name, None)

        # Save data
        self.nc = dataset.n_c

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
        :return: Same as input
        """
        return in_data[:, -1, :-self.nc]

    def disturb(self):
        """
        Returns a sample of noise of length n.
        """
        raise NotImplementedError("Disturbance for naive model not implemented!")
