import numpy as np

from data_processing.dataset import Dataset
from dynamics.base_model import BaseDynamicsModel


class LinearModel(BaseDynamicsModel):
    """The naive model that predicts the last input seen.

    """

    name: str = "Linear"  #: Base name of model.
    n_out: int  #: Number of series that are predicted.

    def __init__(self, dataset: Dataset, **kwargs):
        """Initializes the constant model.

        All series specified by prep_inds are predicted by the last seen value.

        Args:
            dataset: Dataset containing the data.
            kwargs: Kwargs for base class, e.g. `in_indices`.
        """

        # Init base class
        super().__init__(dataset, self.name, **kwargs)

        # Save data
        self.n_out = len(self.out_inds)
        self.nc = dataset.n_c

    def fit(self, verbose: int = 0) -> None:
        """Fit linear model."""
        print("Model const, no fitting needed!")
        raise NotImplementedError("Implement this!!")

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Make predictions by applying the linear model.

        Args:
            in_data: Prepared data.

        Returns:
            The predictions.
        """
        return np.array(0)
