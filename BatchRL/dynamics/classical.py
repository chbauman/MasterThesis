import pickle

import numpy as np

from data_processing.dataset import Dataset
from dynamics.base_model import BaseDynamicsModel
from util.numerics import check_shape
from util.util import train_decorator, make_param_ext, param_dict_to_name


class SKLoader:
    """Wrapper class for sklearn models to be used as a Keras model
    in terms of saving and loading parameters.

    Enables the use of `train_decorator` with the fit() method.
    """
    def __init__(self, skl_mod, parent: 'SKLearnModel'):
        self.skl_mod = skl_mod
        self.p = parent

    def load_weights(self, full_path: str):
        with open(full_path, "rb") as f:
            mod = pickle.load(f)
        self.skl_mod = mod
        self.p.skl_mod = mod

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.skl_mod, f)


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
        # Construct meaningful name
        params = skl_model.get_params()
        ext = param_dict_to_name(params)
        name = skl_model.__class__.__name__ + ext
        if kwargs.get('name'):
            # I need the walrus!
            name = kwargs['name']

        # Init base class
        super().__init__(dataset, name, **kwargs)

        # Save model
        self.m = SKLoader(skl_model, self)

        # Save data
        self.n_out = len(self.out_inds)
        self.nc = dataset.n_c
        self.residual_learning = residual

        # Fitting model
        self.is_fitted = False
        self.skl_mod = skl_model

    @train_decorator()
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
