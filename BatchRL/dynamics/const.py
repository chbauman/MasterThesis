from abc import ABC

import numpy as np

from dynamics.base_model import BaseDynamicsModel
from data_processing.data import Dataset


class NoDisturbanceModel(BaseDynamicsModel, ABC):
    """Interface for models without a disturbance."""

    def model_disturbance(self, data_str: str = 'train'):
        """No need to model, no disturbance used."""
        self.modeled_disturbance = True

    def disturb(self) -> np.ndarray:
        """No disturbance, model is exact."""
        return np.zeros((self.n_pred,), dtype=np.float32)


class ConstModel(BaseDynamicsModel):
    """The naive model that predicts the last input seen.

    If the input series to the model are not specified, the
    same as the output series are taken.
    If there are more input than output series, the output
    consists of the last observation of the first few input series.
    """

    name: str = "Naive"  #: Base name of model.
    n_out: int  #: Number of series that are predicted.

    def __init__(self, dataset: Dataset, pred_inds: np.ndarray = None, **kwargs):
        """Initializes the constant model.

        All series specified by prep_inds are predicted by the last seen value.

        Args:
            dataset: Dataset containing the data.
            pred_inds: Indices of series to predict, all if None.
            kwargs: Kwargs for base class, e.g. `in_indices`.
        """
        # Set in_indices to pred_inds if not specified.
        in_inds = kwargs.get('in_indices')
        if in_inds is None:
            kwargs['in_indices'] = pred_inds
        else:
            if len(in_inds) < len(pred_inds):
                raise ValueError("Need at least as many input series as output series!")

        # Init base class
        super().__init__(dataset, self.name, pred_inds, **kwargs)

        # Save data
        self.n_out = len(self.out_inds)
        self.nc = dataset.n_c

    def fit(self) -> None:
        """No need to fit anything."""
        pass

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Make predictions by just returning the last input.

        Args:
            in_data: Prepared data.

        Returns:
            Same as input
        """
        return np.copy(in_data[:, -1, :self.n_out])
