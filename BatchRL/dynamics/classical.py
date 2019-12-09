import numpy as np

from data_processing.dataset import Dataset
from dynamics.base_model import BaseDynamicsModel
from util.util import train_decorator


class LinearModel(BaseDynamicsModel):
    """The naive model that predicts the last input seen.

    """

    name: str = "Linear"  #: Base name of model.
    n_out: int  #: Number of series that are predicted.

    def __init__(self, dataset: Dataset, residual: bool = True, **kwargs):
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
        self.residual_learning = residual

    def fit(self, verbose: int = 0) -> None:
        """Fit linear model."""

        # Prepare the data
        input_data, output_data = self.get_fit_data('train', residual_output=self.residual_learning)

        in_sh = input_data.shape
        first_sh, last_sh = in_sh[0], in_sh[-1]
        input_data_2d = input_data.reshape((first_sh, -1))

        raise NotImplementedError("Implement this!!")

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Make predictions by applying the linear model.

        Args:
            in_data: Prepared data.

        Returns:
            The predictions.
        """
        return np.array(0)
