from base_dynamics_model import BaseDynamicsModel
from util import *
from data import Dataset


class ConstModel(BaseDynamicsModel):
    """The naive model that predicts the last input seen.
    """

    def __init__(self, dataset: Dataset, pred_inds: np.ndarray = None):
        """Initializes the constant model.

        All series specified by prep_inds are predicted by the last seen value.

        Args:
            dataset: Dataset containing the data.
            pred_inds: Indices of series to predict.
        """
        name = "Naive"
        super(ConstModel, self).__init__(dataset, name, pred_inds)

        # Save data
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
        return in_data[:, -1, self.out_inds]


class ConstSeriesTestModel(BaseDynamicsModel):
    """Test model that predicts a possibly different constant value for each series.

    Can take any number of input series and output series.
    """

    def __init__(self,
                 ds: Dataset,
                 pred_val_list: Union[Num, List[Num]],
                 **kwargs):
        """Constructor

        If `pred_val_list` is a number, it will be used as prediction
        for all series that are predicted.

        Args:
            ds: The dataset.
            pred_val_list: The list with the values that will be predicted.
            **kwargs: Kwargs for the base class, e.g. `out_indices` or `in_indices`.
        """
        name = f"ConstModel"
        super().__init__(ds, name, **kwargs)

        # Check if the list contains the right number of elements if it is a list.
        if isinstance(pred_val_list, List):
            if not len(pred_val_list) == self.n_pred:
                raise ValueError("Not the right number of values specified!")

        self.values = np.array(pred_val_list)
        self.use_AR = False

    def fit(self) -> None:
        pass

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Predicts a constant value for each series."""
        in_sh = in_data.shape
        preds = np.ones((in_sh[0], self.n_pred), dtype=np.float32) * self.values
        return preds
