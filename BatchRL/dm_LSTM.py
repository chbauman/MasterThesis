from functools import partial
from typing import Dict, Optional

from hyperopt import hp
from hyperopt.pyll import scope as ho_scope
from keras import backend as K
from keras.layers import GRU, LSTM, Dense, Input, Add, Concatenate, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model
from tensorflow.python import debug as tf_debug

from base_hyperopt import HyperOptimizableModel
from data import Dataset, get_test_ds
from data import SeriesConstraint
from keras_layers import ConstrainedNoise, FeatureSlice, ExtractInput, IdRecurrent, IdDense, \
    get_multi_input_layer_output
from util import *
from visualize import plot_train_history


def weighted_loss(y_true, y_pred, weights):
    """
    Returns the weighted MSE between y_true and y_pred.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param weights: Weights.
    :return: Weighted MSE.
    """
    return K.mean((y_true - y_pred) * (y_true - y_pred) * weights)


def constr_name(name: str,
                hidden_sizes: Sequence,
                n_iter_max: int,
                lr: float,
                gru: bool = False,
                res_learn: bool = True,
                weight_vec: np.ndarray = None,
                input_noise_std: float = None) -> str:
    """
    Constructs the name of the network.

    :param name: Base name
    :param hidden_sizes: Layer size list or tuple
    :param n_iter_max: Number of iterations
    :param lr: Learning rate
    :param gru: Whether to use GRU units
    :param res_learn: Whether to use residual learning
    :param weight_vec: Weight vector for weighted loss
    :param input_noise_std: Standard deviation of input noise.
    :return: String combining all these parameters.
    """
    arch = '_L' + '-'.join(map(str, hidden_sizes))
    ep_s = '_E' + str(n_iter_max)
    lrs = '_LR' + "{:.4g}".format(lr)
    n_str = '' if input_noise_std is None else '_N' + "{:.4g}".format(input_noise_std)
    gru_str = '' if not gru else '_GRU'
    res_str = '' if not res_learn else '_RESL'
    w_str = '' if weight_vec is None else '_W' + '-'.join(map(str, weight_vec))
    return name + ep_s + arch + lrs + n_str + gru_str + res_str + w_str


class RNNDynamicModel(HyperOptimizableModel):
    """Simple RNN used for training a dynamics model.

    All parameters are defined in `__init__`. Implements
    the `HyperOptimizableModel` for hyperparameter optimization.
    """

    train_seq_len: int  #: Length of sequences used for training.
    n_feats: int  #: Number of input features in train data.

    def get_space(self) -> Dict:
        """
        Defines the hyper parameter space.

        Returns:
            Dict specifying the hyper parameter space.
        """
        hp_space = {
            'n_layers': ho_scope.int(hp.quniform('n_layers', low=1, high=4, q=1)),
            'n_neurons': ho_scope.int(hp.quniform('n_neurons', low=5, high=100, q=5)),
            'n_iter_max': ho_scope.int(hp.quniform('n_iter_max', low=5, high=100, q=5)),
            'gru': hp.choice('gru', [True, False]),
            'lr': hp.loguniform('lr', low=-5 * np.log(10), high=1 * np.log(10)),
            'input_noise_std': hp.loguniform('input_noise_std', low=-6 * np.log(10), high=-1 * np.log(10)),
        }
        return hp_space

    def conf_model(self, hp_sample: Dict) -> 'HyperOptimizableModel':
        """
        Returns a model of the same type with the same output and input
        indices and the same constraints, but other hyper parameters.

        Args:
            hp_sample: Sample of hyper parameters to initialize the model with.

        Returns:
            New model.
        """
        n_layers = hp_sample['n_layers']
        n_units = hp_sample['n_neurons']
        hidden_sizes = [n_units for _ in range(n_layers)]
        init_kwargs = {key: hp_sample[key] for key in hp_sample if key not in ['n_layers', 'n_neurons']}
        new_mod = RNNDynamicModel(self.data,
                                  in_inds=self.in_indices,
                                  out_inds=self.out_inds,
                                  constraint_list=self.constraint_list,
                                  hidden_sizes=hidden_sizes,
                                  verbose=0,
                                  **init_kwargs)
        return new_mod

    def hyper_objective(self) -> float:
        """Uses the hyperparameter objective from the base class.

        Returns:
            Objective loss.
        """
        return self.hyper_obj()

    def __init__(self,
                 data: Dataset,
                 hidden_sizes: Sequence[int] = (20, 20),
                 n_iter_max: int = 10000,
                 name: str = 'baseRNN',
                 *,
                 in_inds: np.ndarray = None,
                 out_inds: np.ndarray = None,
                 weight_vec: Optional[np.ndarray] = None,
                 gru: bool = False,
                 input_noise_std: Optional[float] = None,
                 residual_learning: bool = True,
                 lr: float = 0.001,
                 constraint_list: Sequence[SeriesConstraint] = None,
                 verbose: int = 1):

        """Constructor, defines all the network parameters.

        :param name: Base name
        :param data: Dataset
        :param hidden_sizes: Layer size list or tuple
        :param out_inds: Prediction indices
        :param in_inds: Input indices
        :param n_iter_max: Number of iterations
        :param lr: Learning rate
        :param gru: Whether to use GRU units
        :param residual_learning: Whether to use residual learning
        :param weight_vec: Weight vector for weighted loss
        :param input_noise_std: Standard deviation of input noise.
        :param constraint_list: The constraints on the data series.
        :param verbose: The verbosity level, 0, 1 or 2.
        """

        name = constr_name(name, hidden_sizes,
                           n_iter_max, lr, gru,
                           residual_learning, weight_vec, input_noise_std)
        super(RNNDynamicModel, self).__init__(data, name, out_inds, in_inds, verbose)

        # Store data
        self.train_seq_len = self.data.seq_len - 1
        self.n_feats = self.data.d

        # Store parameters
        self.constraint_list = constraint_list
        self.hidden_sizes = np.array(hidden_sizes, dtype=np.int32)
        self.n_iter_max = n_iter_max
        self.gru = gru
        self.input_noise_std = input_noise_std
        self.weight_vec = weight_vec
        self.res_learn = residual_learning
        self.lr = lr

        # Build model
        self.m = self._build_model()

    def _build_model(self, debug: bool = False) -> Any:
        """Builds the keras LSTM model and returns it.

        The parameters how to build it were passed to `__init__`.

        Returns:
             The built keras model.
        """

        # Initialize
        n_lstm = len(self.hidden_sizes)
        model = Sequential(name="rnn")
        input_shape_dict = {'input_shape': (self.train_seq_len, self.n_feats)}

        # Add noise layer
        if self.input_noise_std is not None:
            model.add(ConstrainedNoise(self.input_noise_std, consts=self.constraint_list,
                                       name="constrain_input",
                                       **input_shape_dict))
            input_shape_dict = {}

        # Add layers
        rnn = GRU if self.gru else LSTM
        for k in range(n_lstm):
            ret_seq = k != n_lstm - 1
            if debug:
                model.add(IdRecurrent(return_sequences=ret_seq))
            else:
                model.add(rnn(int(self.hidden_sizes[k]),
                              return_sequences=ret_seq,
                              name="rnn_layer_{}".format(k),
                              **input_shape_dict))
            input_shape_dict = {}

        # Output layer
        # model.add(TimeDistributed(Dense(self.n_pred, activation=None)))
        if self.constraint_list is not None:
            out_constraints = [self.constraint_list[i] for i in self.out_inds]
        else:
            out_constraints = None
        out_const_layer = ConstrainedNoise(0, consts=out_constraints,
                                           is_input=False,
                                           name="constrain_output")
        if debug:
            model.add(IdDense(n=self.n_pred))
        else:
            model.add(Dense(self.n_pred, activation=None, name="dense_reduce"))

        if self.res_learn:
            # Add last non-control input to output
            seq_input = Input(shape=(self.train_seq_len, self.n_feats),
                              name="input_sequences")
            m_out = model(seq_input)
            slicer = FeatureSlice(self.p_out_inds, name="get_previous_output")
            last_input = slicer(seq_input)
            final_out = Add(name="add_previous_state")([m_out, last_input])
            final_out = out_const_layer(final_out)
            model = Model(inputs=seq_input, outputs=final_out)
        else:
            model.add(out_const_layer)

        if self.weight_vec is not None:
            k_constants = K.constant(self.weight_vec)
            fixed_input = Input(tensor=k_constants)
            seq_input = Input(shape=(self.train_seq_len, self.n_feats))
            model = Model(inputs=seq_input, outputs=model(seq_input))
            self.loss = partial(weighted_loss, weights=fixed_input)
        else:
            self.loss = 'mse'

        return model

    def _plot_model(self, model: Any, name: str = "Model.png",
                    expand: bool = True):
        """Plots keras model."""
        pth = self.get_plt_path(name)
        plot_model(model, to_file=pth,
                   show_shapes=True,
                   expand_nested=expand,
                   dpi=500)

    def fit(self) -> None:
        """
        Fit the model if it hasn't been fitted before.
        Otherwise load the trained model.

        TODO: Compute more accurate val_percent for fit method!

        Returns:
             None
        """

        loaded = self.load_if_exists(self.m, self.name)
        if not loaded:
            if self.verbose:
                self.deb("Fitting Model...")

            # Define optimizer and compile
            opt = Adam(lr=self.lr)
            self.m.compile(loss=self.loss, optimizer=opt)
            if self.verbose:
                self.m.summary()
            if not EULER:
                self._plot_model(self.m)

            # Prepare the data
            input_data, output_data = self.get_fit_data('train_val')

            # Fit and save model
            h = self.m.fit(input_data, output_data,
                           epochs=self.n_iter_max,
                           initial_epoch=0,
                           batch_size=128,
                           validation_split=self.data.val_percent,
                           verbose=self.verbose)
            pth = self.get_plt_path("TrainHist")
            plot_train_history(h, pth)
            create_dir(self.model_path)
            self.m.save_weights(self.get_path(self.name))
        else:
            self.deb("Restored trained model")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Predicts a batch of sequences using the fitted model.

        :param input_data: 3D numpy array with input data.
        :return: 2D numpy array with the predictions.
        """

        n = input_data.shape[0]

        # Predict
        predictions = self.m.predict(input_data)
        predictions = predictions.reshape((n, -1))
        return predictions


class RNNDynamicOvershootModel(RNNDynamicModel):
    m: Any = None  #: The base model used for prediction.
    overshoot_model: Any = None  #: The overshoot model used for training.
    n_overshoot: int
    DEBUG: bool = True

    def __init__(self, n_overshoot: int = 10, decay_rate: float = 1.0, **kwargs):
        """Initialize model

        Args:
            n_overshoot: The number of timesteps to consider in overshoot model.
            kwargs: Kwargs for super class.
        """
        super(RNNDynamicOvershootModel, self).__init__(**kwargs)

        self.decay_rate = decay_rate
        self.n_overshoot = n_overshoot
        self.pred_seq_len = self.train_seq_len
        self.tot_train_seq_len = n_overshoot + self.pred_seq_len

        self._build()

    def _build(self) -> None:
        """Builds the keras model."""

        # Build base model.
        b_mod = self._build_model(self.DEBUG)
        self.m = b_mod

        # Build train model.
        inds = self.p_out_inds
        tot_len = self.tot_train_seq_len

        def res_lay(k_ind: int):
            return Reshape((1, self.n_pred), name=f"reshape_{k_ind}")

        def copy_mod(k_ind: int):
            m = self.m
            ip = Input(rem_first(m.input_shape))
            out = m(ip)
            new_mod = Model(inputs=ip, outputs=out, name=f"base_model_{k_ind}")
            return new_mod

        # Define input
        full_input = Input((tot_len, self.n_feats))

        first_out = ExtractInput(inds, self.pred_seq_len, 0,
                                 name="first_extraction")(full_input)
        first_out = copy_mod(0)(first_out)
        all_out = [res_lay(0)(first_out)]
        for k in range(self.n_overshoot - 1):
            first_out = ExtractInput(inds, self.pred_seq_len, k + 1,
                                     name=f"extraction_{k + 1}")([full_input, first_out])
            first_out = copy_mod(k + 1)(first_out)
            all_out += [res_lay(k + 1)(first_out)]

        # Concatenate all outputs and make model
        full_out = Concatenate(axis=-2, name="final_concatenate")(all_out)
        train_mod = Model(inputs=full_input, outputs=full_out)

        # Define loss and compile
        if self.decay_rate != 1.0:
            ww = np.ones((self.n_overshoot, 1), dtype=np.float32)
            ww = np.repeat(ww, self.n_pred, axis=-1)
            for k in range(self.n_overshoot):
                ww[k] *= self.decay_rate ** k
            k_constants = K.constant(ww)
            fixed_input = Input(tensor=k_constants)
            loss = partial(weighted_loss, weights=fixed_input)
        else:
            loss = 'mse'
        opt = Adam(lr=self.lr)
        train_mod.compile(loss=loss, optimizer=opt)
        if self.verbose:
            train_mod.summary()

        # Plot and save
        if not EULER:
            self._plot_model(train_mod, "TrainModel.png", expand=False)
        self.train_mod = train_mod

    def fit(self) -> None:
        """Fits the model using the data from the dataset.

        TODO: Use decorator for 'loading if existing' to avoid duplicate
            code!

        Returns:
            None
        """
        loaded = self.load_if_exists(self.m, self.name)
        if not loaded or self.DEBUG:
            if self.verbose:
                self.deb("Fitting Model...")

            fit_data, _ = self.data.split_dict['train_val'].get_sequences(self.tot_train_seq_len)
            out_data = fit_data[:, self.pred_seq_len:, self.p_out_inds]
            h = self.train_mod.fit(x=fit_data, y=out_data,
                                   epochs=self.n_iter_max,
                                   initial_epoch=0,
                                   batch_size=128,
                                   validation_split=self.data.val_percent,
                                   verbose=self.verbose)

            pth = self.get_plt_path("TrainHist")
            plot_train_history(h, pth)
            create_dir(self.model_path)
            self.m.save_weights(self.get_path(self.name))
        else:
            self.deb("Restored trained model")
    pass


def test_rnn_models():
    """Tests the RNN model classes.

    Raises:
        AssertionError: If a test fails.
    """

    # Create dataset for testing
    n, n_feat = 100, 10
    dat = np.arange(n * n_feat).reshape((n, n_feat))
    c_inds = np.array([4, 6], dtype=np.int32)
    ds = get_test_ds(dat, c_inds, name="RNNTestDataset")
    ds.seq_len = 6
    train_s_len = ds.seq_len - 1
    ds.split_data()

    # Define model
    p_inds = np.array([0, 1, 3], dtype=np.int32)
    test_kwargs = {'hidden_sizes': (10,),
                   'n_iter_max': 1,
                   'input_noise_std': 0.001,
                   'lr': 0.01,
                   'residual_learning': True,
                   'weight_vec': None,
                   'out_inds': p_inds,
                   'constraint_list': None,
                   }
    n_over = 3
    mod_test_overshoot = RNNDynamicOvershootModel(n_overshoot=n_over,
                                                  data=ds,
                                                  name="DebugOvershoot",
                                                  **test_kwargs)
    full_sam_seq_len = n_over + train_s_len

    # Check sequence lengths
    assert full_sam_seq_len == mod_test_overshoot.tot_train_seq_len, "Train seq len mismatch!"
    assert train_s_len == mod_test_overshoot.train_seq_len, "Seq len mismatch!"

    # Check model debug output
    lay_input = np.array([dat[:full_sam_seq_len]])
    test_mod = mod_test_overshoot.train_mod
    test_base_mod = mod_test_overshoot.m
    l_out = get_multi_input_layer_output(test_mod, lay_input, learning_phase=0)
    l_out_base = get_multi_input_layer_output(test_base_mod, lay_input[:, :train_s_len, :], learning_phase=0)
    assert np.allclose(l_out_base[0], l_out[0, 0]), "Model output not correct!"

    # Find intermediate layer output
    ex1_out = first_out = None
    for k in test_mod.layers:
        if k.name == 'extraction_1':
            ex1_out = k.output
        if k.name == 'first_extraction':
            first_out = k.output
    assert ex1_out is not None and first_out is not None, "Layers not found!"
    first_ex_out = Model(inputs=test_mod.inputs, outputs=first_out)
    l_out_first = get_multi_input_layer_output(first_ex_out, lay_input, learning_phase=0)
    assert np.allclose(lay_input[0, :train_s_len], l_out_first), "First extraction incorrect!"
    sec_ex_out = Model(inputs=test_mod.inputs, outputs=ex1_out)
    exp_out = np.copy(lay_input[:, 1:(1 + train_s_len), :])
    exp_out[0, -1, p_inds] = l_out[0, 0]
    l_out_sec = get_multi_input_layer_output(sec_ex_out, lay_input, learning_phase=0)
    assert np.allclose(exp_out, l_out_sec), "Second extraction incorrect!"

    # Try fitting
    mod_test_overshoot.fit()

    print("RNN models test passed :)")
