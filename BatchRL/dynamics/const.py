from abc import ABC

from dynamics.base_model import BaseDynamicsModel
from util.util import *
from data import Dataset


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
            pred_inds: Indices of series to predict.
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


# Testing models.

class ConstTestModel(ConstModel, NoDisturbanceModel):
    """Const Test model without a disturbance."""

    name: str = "NaiveNoDisturb"  #: Base name of model.


class ConstSeriesTestModel(NoDisturbanceModel):
    """Test model that predicts a possibly different constant value for each series.

    Can take any number of input series and output series.
    """

    def __init__(self,
                 ds: Dataset,
                 pred_val_list: Union[Num, List[Num]],
                 predict_input_check: np.ndarray = None,
                 **kwargs):
        """Constructor

        If `pred_val_list` is a number, it will be used as prediction
        for all series that are predicted.

        Args:
            ds: The dataset.
            pred_val_list: The list with the values that will be predicted.
            predict_input_check: If not None, the input series have to have these values.
            **kwargs: Kwargs for the base class, e.g. `out_indices` or `in_indices`.
        """
        name = f"ConstModel"
        super().__init__(ds, name, **kwargs)

        # Check if the list contains the right number of elements if it is a list.
        if isinstance(pred_val_list, List):
            if not len(pred_val_list) == self.n_pred:
                raise ValueError("Not the right number of values specified!")

        self.predict_input_check = predict_input_check
        self.values = np.array(pred_val_list)
        self.use_AR = False

    def fit(self) -> None:
        pass

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Predicts a constant value for each series.

        Raises:
            AssertionError: If `predict_input_check` is not None and
                the input does not match.
        """
        in_sh = in_data.shape
        if self.predict_input_check is not None:
            pc = self.predict_input_check
            assert len(pc) == in_sh[-1], "Predictions do not have the right shape!"
            exp_inp = np.ones(in_sh) * pc
            assert np.array_equal(exp_inp, in_data), "Input incorrect!!!"
        preds = np.ones((in_sh[0], self.n_pred), dtype=np.float32) * self.values
        return preds
