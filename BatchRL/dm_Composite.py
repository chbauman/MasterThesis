from base_dynamics_model import BaseDynamicsModel
from util import *
from data import Dataset


class CompositeModel(BaseDynamicsModel):
    """The composite model, combining multiple models.

    All models need to be based on the same dataset.
    """

    model_list: List[BaseDynamicsModel]

    def __init__(self, dataset: Dataset, model_list: List[BaseDynamicsModel], new_name: str = None):
        """Initialize the Composite model.

        All individual model need to be initialized with the same dataset!

        Args:
            dataset: Dataset.
            model_list: A list of dynamics models defined for the same dataset.
            new_name: The name to give to this model, default produces very long names.

        Raises:
            ValueError: If the model in list do not have access to `dataset`.
        """
        # Compute name and check datasets
        name = dataset.name + "Composite"
        for m in model_list:
            name += "_" + m.name
            if m.data != dataset:
                raise ValueError("Model {} needs to model the same dataset as the Composite model.".format(m.name))
        if new_name is not None:
            name = new_name

        # Collect indices
        all_out_inds = np.concatenate([m.out_inds for m in model_list])
        if has_duplicates(all_out_inds):
            raise ValueError("Predicting one or more series multiple times.")

        super(CompositeModel, self).__init__(dataset, name, all_out_inds, None)

        # Save models
        self.model_list = model_list

    def init_1day(self, day_data: np.ndarray) -> None:
        """Calls the same function on all models in list.

        Args:
            day_data: The data for the initialization.
        """
        for m in self.model_list:
            m.init_1day(day_data)

    def fit(self) -> None:
        """Fits all the models."""
        for m in self.model_list:
            m.fit()

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """Aggregated prediction by predicting with all models.

        Args:
            in_data: Prepared data.

        Returns:
            Aggregated predictions.
        """

        in_sh = in_data.shape
        out_dat = np.empty((in_sh[0], self.n_pred), dtype=in_data.dtype)

        # Predict with all the models
        curr_ind = 0
        for m in self.model_list:
            n_pred_m = m.n_pred
            in_inds = m.p_in_indices
            pred_in_dat = in_data[:, :, in_inds]
            preds = m.predict(pred_in_dat)
            out_dat[:, curr_ind: (curr_ind + n_pred_m)] = preds
            curr_ind += n_pred_m

        return out_dat

    def disturb(self):
        """Returns a sample of noise.
        """
        seq_len = self.data.seq_len - 1
        out_dat = np.empty((self.n_pred,), dtype=np.float32)

        # Disturb with all the models
        curr_ind = 0
        for m in self.model_list:
            n_pred_m = m.n_pred
            out_dat[curr_ind: (curr_ind + n_pred_m)] = m.disturb()
            curr_ind += n_pred_m

        return out_dat

    def model_disturbance(self, data_str: str = 'train'):
        """Models the disturbances for all sub-models."""
        for m in self.model_list:
            m.model_disturbance(data_str)
        self.modeled_disturbance = True
