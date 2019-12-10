import pickle

import numpy as np

from data_processing.dataset import Dataset
from dynamics.base_model import BaseDynamicsModel
from util.numerics import check_shape


class SKLoader:
    """Wrapper class for sklearn models to be used as a Keras model
    in terms of saving and loading parameters.

    """
    def __init__(self, skl_mod):
        self.skl_mod = skl_mod

    def load_weights(self, full_path: str):
        params = pickle.load(open(full_path, "rb"))
        self.skl_mod.set_params(**params)

    def save(self, path: str) -> None:
        params = self.skl_mod.get_params()
        pickle.dump(params, open(path, "wb"))


class SKLearnModel(BaseDynamicsModel):
    """The naive model that predicts the last input seen.

    """

    name: str = "Linear"  #: Base name of model.
    n_out: int  #: Number of series that are predicted.

    def __init__(self, dataset: Dataset, skl_model, residual: bool = True, **kwargs):
        """Initializes the constant model.

        All series specified by prep_inds are predicted by the last seen value.

        Args:
            dataset: Dataset containing the data.
            kwargs: Kwargs for base class, e.g. `in_inds`.
        """

        # Init base class
        super().__init__(dataset, self.name, **kwargs)

        # Save model
        self.m = SKLoader(skl_model)

        # Save data
        self.n_out = len(self.out_inds)
        self.nc = dataset.n_c
        self.residual_learning = residual

        # Fitting model
        self.is_fitted = False
        self.skl_mod = skl_model

    def fit(self, verbose: int = 0) -> None:
        """Fit linear model."""

        # Check if already fitted
        if self.is_fitted:
            if verbose:
                print("Already fitted!")
            return

        # Prepare the data
        input_data, output_data = self.get_fit_data('train', residual_output=self.residual_learning)
        in_sh = input_data.shape
        first_sh, last_sh = in_sh[0], in_sh[-1]
        input_data_2d = input_data.reshape((first_sh, -1))

        # Fit
        self.skl_mod.fit(input_data_2d, output_data)
        self.is_fitted = True

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Make predictions by applying the linear model.

        Args:
            in_data: Prepared data.

        Returns:
            The predictions.
        """
        check_shape(in_data, (-1, -1, -1))

        # Add previous state contribution
        prev_state = self._extract_output(in_data)

        # Flatten
        sh = in_data.shape
        in_data_res = in_data.reshape((sh[0], -1))
        prev_state_res = prev_state.reshape((sh[0], -1))

        # Predict
        p = self.skl_mod.predict(in_data_res)
        return prev_state_res + p
