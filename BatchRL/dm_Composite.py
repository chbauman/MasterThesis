from base_dynamics_model import BaseDynamicsModel
from util import *
from data import Dataset


class CompositeModel(BaseDynamicsModel):
    """
    The composite model, containing multiple other models.
    """

    def __init__(self, dataset: Dataset, model_list: List[BaseDynamicsModel] = None):
        """
        Initialize the Composite model. All individual model
        need to be initialized with the same dataset!

        :param dataset: Dataset.
        :param model_list: A list of dynamics models defined for the same dataset.
        """
        # Compute name and check datasets
        name = dataset.name + "Composite"
        for m in model_list:
            name += "_" + m.name
            if m.data != dataset:
                raise ValueError("Model {} needs to model the same dataset as the Composite model.".format(m.name))

        # Collect indices
        all_out_inds = np.concatenate([m.out_inds for m in model_list])
        if has_duplicates(all_out_inds):
            raise ValueError("Predicting one or more series multiple times.")

        super(CompositeModel, self).__init__(dataset, name, all_out_inds, None)

        # Save parameters
        self.model_list = model_list

    def fit(self) -> None:
        """
        Fits all the models.

        :return: None
        """

        for m in self.model_list:
            m.fit()

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        """
        Aggregated prediction by predicting with all models.

        :param in_data: Prepared data.
        :return: Aggregated predictions.
        """

        in_sh = in_data.shape
        out_dat = np.empty((in_sh[0], in_sh[1], self.n_pred))

        for m in self.model_list:
            in_inds = m.p_in_indices
            out_inds = m.p_out_inds
            preds = m.predict(in_data[:, :, in_inds])
            out_dat[:, :, out_inds] = preds

        return out_dat

    def disturb(self):
        """
        Returns a sample of noise of length n.
        """
        raise NotImplementedError("Disturbance for naive model not implemented!")
