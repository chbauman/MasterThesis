from base_dynamics_model import BaseDynamicsModel, construct_test_ds
from data import Dataset
from dm_Const import ConstSeriesTestModel, ConstModel, ConstTestModel
from util import *


class CompositeModel(BaseDynamicsModel):
    """The composite model, combining multiple models.

    All models need to be based on the same dataset.
    """

    model_list: List[BaseDynamicsModel]

    def __init__(self, dataset: Dataset, model_list: List[BaseDynamicsModel], new_name: str = None):
        """Initialize the Composite model.

        All individual model need to be initialized with the same dataset!

        Args:
            dataset: The common `Dataset`.
            model_list: A list of dynamics models defined for the same dataset.
            new_name: The name to give to this model, default produces very long names.

        Raises:
            ValueError: If the model in list do not have access to `dataset` or if
                any series is predicted by multiple models.
        """
        # Compute name and check datasets
        name = dataset.name + "Composite"
        for m in model_list:
            name += "_" + m.name
            if m.data != dataset:
                raise ValueError("Model {} needs to model the same dataset as the Composite model.".format(m.name))
        if new_name is not None:
            name = new_name

        # Collect indices and initialize base class.
        n_pred_full = dataset.d - dataset.n_c
        all_out_inds = np.concatenate([m.out_inds for m in model_list])
        if has_duplicates(all_out_inds):
            raise ValueError("Predicting one or more series multiple times.")
        out_inds = dataset.from_prepared(np.arange(n_pred_full))
        super().__init__(dataset, name, out_inds, None)

        # We allow only full models, i.e. when combined, the models have to predict
        # all series except for the controlled ones.
        if self.n_pred != n_pred_full or len(all_out_inds) != n_pred_full:
            raise ValueError("You need to predict all non-control series!")

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
        # Get shape of prediction
        in_sh = in_data.shape
        out_dat = np.empty((in_sh[0], self.n_pred), dtype=in_data.dtype)

        # Predict with all the models
        for m in self.model_list:
            in_inds = m.p_in_indices
            out_inds = m.p_out_inds
            pred_in_dat = in_data[:, :, in_inds]
            preds = m.predict(pred_in_dat)
            out_dat[:, out_inds] = preds

        return out_dat

    def disturb(self):
        """Returns a sample of noise.
        """
        out_dat = np.empty((self.n_pred,), dtype=np.float32)

        # Disturb with all the models
        curr_ind = 0
        for m in self.model_list:
            n_pred_m = m.n_pred
            out_inds = m.p_out_inds
            out_dat[out_inds] = m.disturb()
            curr_ind += n_pred_m

        return out_dat

    def model_disturbance(self, data_str: str = 'train'):
        """Models the disturbances for all sub-models."""
        for m in self.model_list:
            m.model_disturbance(data_str)
        self.modeled_disturbance = True


##########################################################################
# Testing stuff


def test_composite():
    # Define datasets
    dataset = construct_test_ds(200)
    ds_1 = construct_test_ds(200, c_series=1)

    # Define individual models
    m1 = ConstSeriesTestModel(dataset, pred_val_list=1.0, out_indices=np.array([0, 2], dtype=np.int32))
    m2 = ConstSeriesTestModel(dataset, pred_val_list=2.0, out_indices=np.array([1], dtype=np.int32))
    m3 = ConstSeriesTestModel(dataset,
                              pred_val_list=[1.0],
                              out_indices=np.array([1], dtype=np.int32),
                              in_indices=np.array([1], dtype=np.int32))
    m4 = ConstSeriesTestModel(dataset,
                              pred_val_list=[0.0, 2.0],
                              out_indices=np.array([0, 2], dtype=np.int32),
                              in_indices=np.array([1, 2, 3], dtype=np.int32))
    m5 = ConstTestModel(dataset, pred_inds=np.array([1], dtype=np.int32))
    m6 = ConstTestModel(dataset,
                        pred_inds=np.array([0, 2], dtype=np.int32),
                        in_indices=np.array([1, 2, 3], dtype=np.int32))
    m7 = ConstTestModel(ds_1,
                        pred_inds=np.array([0, 2], dtype=np.int32),
                        in_indices=np.array([1, 2, 3], dtype=np.int32))
    m8 = ConstTestModel(ds_1,
                        pred_inds=np.array([3], dtype=np.int32),
                        in_indices=np.array([1, 3], dtype=np.int32))

    # Define composite models
    mc1 = CompositeModel(dataset, [m1, m2], new_name="CompositeTest1")
    mc2 = CompositeModel(dataset, [m3, m4], new_name="CompositeTest2")
    mc3 = CompositeModel(dataset, [m5, m6], new_name="CompositeTest3")
    mc4 = CompositeModel(ds_1, [m7, m8], new_name="CompositeTest4")

    # Test predictions
    in_data = np.reshape(np.arange(2 * 5 * 4), (2, 5, 4))
    in_data_2 = np.ones((2, 5, 4), dtype=np.float32) * np.arange(4)
    exp_out_data = np.ones((2, 3), dtype=np.float32)
    exp_out_data[:, 1] = 2.0
    assert np.array_equal(exp_out_data, mc1.predict(in_data)), "Composite model prediction wrong!"
    exp_out_2 = np.ones((2, 3), dtype=np.float32) * np.arange(3.0)
    assert np.array_equal(exp_out_2, mc2.predict(in_data)), "Composite model prediction wrong!"
    assert np.array_equal(exp_out_2, mc2.predict(in_data_2)), "Composite model prediction wrong!"
    exp_out_3 = np.copy(exp_out_2)
    exp_out_3[:, 0] = 1.0
    assert np.array_equal(exp_out_3, mc3.predict(in_data_2)), "Composite model prediction wrong!"
    exp_out_4 = np.copy(exp_out_2)
    exp_out_4[..., 1:3] += 1
    act_out = mc4.predict(in_data_2)
    assert np.array_equal(exp_out_4, act_out), "Composite model prediction wrong!"

    print("Composite model test passed! :)")
