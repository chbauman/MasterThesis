from dynamics.base_model import BaseDynamicsModel, construct_test_ds
from data import Dataset
from dynamics.const import ConstSeriesTestModel, ConstTestModel
from util.util import *


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

        # Reset the indices, since we do not want to permute twice!
        self.p_in_indices = np.arange(dataset.d)

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
    n = 200
    dataset = construct_test_ds(200)
    dataset.data = np.ones((n, 4)) * np.arange(4)
    dataset.split_data()
    ds_1 = construct_test_ds(200, c_series=1)
    ds_1.data = np.ones((n, 4)) * np.arange(4)
    ds_1.split_data()

    # Define individual models
    inds_1 = np.array([1], dtype=np.int32)
    inds_2 = np.array([2], dtype=np.int32)
    inds_02 = np.array([0, 2], dtype=np.int32)
    inds_123 = np.array([1, 2, 3], dtype=np.int32)
    m1 = ConstSeriesTestModel(dataset, pred_val_list=1.0, out_indices=np.array([0, 2], dtype=np.int32))
    m2 = ConstSeriesTestModel(dataset, pred_val_list=2.0, out_indices=inds_1)
    m3 = ConstSeriesTestModel(dataset,
                              pred_val_list=[1.0],
                              out_indices=inds_1,
                              in_indices=inds_1)
    m4 = ConstSeriesTestModel(dataset,
                              pred_val_list=[0.0, 2.0],
                              out_indices=inds_02,
                              in_indices=inds_123)
    m5 = ConstTestModel(dataset, pred_inds=inds_1)
    m6 = ConstTestModel(dataset,
                        pred_inds=inds_02,
                        in_indices=inds_123)
    m7 = ConstTestModel(ds_1,
                        pred_inds=inds_02,
                        in_indices=np.array([1, 2], dtype=np.int32))
    m8 = ConstTestModel(ds_1,
                        pred_inds=np.array([3], dtype=np.int32),
                        in_indices=inds_1)
    m9 = ConstSeriesTestModel(ds_1,
                              pred_val_list=[2.0],
                              predict_input_check=inds_2,
                              out_indices=inds_2,
                              in_indices=inds_2)
    m10 = ConstSeriesTestModel(ds_1,
                               pred_val_list=[0.0, 3.0],
                               predict_input_check=inds_123,
                               out_indices=np.array([0, 3], dtype=np.int32),
                               in_indices=inds_123)

    # Define composite models
    mc1 = CompositeModel(dataset, [m1, m2], new_name="CompositeTest1")
    mc2 = CompositeModel(dataset, [m3, m4], new_name="CompositeTest2")
    mc3 = CompositeModel(dataset, [m5, m6], new_name="CompositeTest3")
    mc4 = CompositeModel(ds_1, [m7, m8], new_name="CompositeTest4")
    mc5 = CompositeModel(ds_1, [m9, m10], new_name="CompositeTest4")

    # Test predictions
    sh = (2, 5, 4)
    sh_out = (2, 3)
    sh_tot = tot_size(sh)
    in_data = np.reshape(np.arange(sh_tot), sh)
    in_data_2 = np.ones(sh, dtype=np.float32) * np.arange(4)
    exp_out_data = np.ones((2, 3), dtype=np.float32)
    exp_out_data[:, 1] = 2.0
    assert np.array_equal(exp_out_data, mc1.predict(in_data)), "Composite model prediction wrong!"
    exp_out_2 = np.ones((2, 3), dtype=np.float32) * np.arange(3.0)
    assert np.array_equal(exp_out_2, mc2.predict(in_data)), "Composite model prediction wrong!"
    assert np.array_equal(exp_out_2, mc2.predict(in_data_2)), "Composite model prediction wrong!"
    exp_out_3 = np.copy(exp_out_2)
    exp_out_3[:, 0] = 1.0
    assert np.array_equal(exp_out_3, mc3.predict(in_data_2)), "Composite model prediction wrong!"
    exp_out_4 = np.ones(sh_out) * np.array([3, 1, 3])
    act_out = mc4.predict(in_data_2)
    assert np.array_equal(exp_out_4, act_out), "Composite model prediction wrong!"

    # Test analyze()
    m9.analyze(plot_acf=False)
    mc5.analyze(plot_acf=False)

    print("Composite model test passed! :)")
