from base_dynamics_model import BaseDynamicsModel
from util import *
from data import Dataset


class ConstModel(BaseDynamicsModel):
    """The naive model that predicts the last input seen.
    """

    def __init__(self, dataset: Dataset, pred_inds: np.ndarray = None):
        """
        Initializes the constant model. All series specified
        by prep_inds are predicted by the last seen value.

        :param dataset: Dataset containing the data.
        :param pred_inds: Indices of series to predict.
        """
        name = "Naive"
        super(ConstModel, self).__init__(dataset, name, pred_inds)

        # Save data
        self.nc = dataset.n_c

    def fit(self) -> None:
        """No need to fit anything."""
        pass

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """
        Make predictions by just returning the last input.

        :param in_data: Prepared data.
        :return: Same as input
        """
        return in_data[:, -1, self.out_inds]
