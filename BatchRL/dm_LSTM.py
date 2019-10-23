from functools import partial
from typing import Dict

from hyperopt import hp
from hyperopt.pyll import scope as ho_scope
from keras import backend as K
from keras.layers import GRU, LSTM, Dense, Input, Add, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam

from base_hyperopt import HyperOptimizableModel
from data import Dataset
from data import SeriesConstraint
from keras_layers import SeqInput, ConstrainedNoise
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
    """
    Simple LSTM used for training a dynamics model.
    """

    def get_space(self) -> Dict:
        """
        Defines the hyper parameter space.

        Returns:
            Dict specifying the hyper parameter space.
        """
        hp_space = {
            'n_layers': ho_scope.int(hp.quniform('n_layers', low=1, high=4, q=1)),
            'n_neurons': ho_scope.int(hp.quniform('n_neurons', low=5, high=100, q=5)),
            'n_iter_max': ho_scope.int(hp.quniform('n_iter_max', low=5, high=10, q=5)),
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
        """
        Uses the hyperparameter objective from the base class.

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

        """
        Constructor, defines all the network parameters.

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
        self.m = None
        self._build_model()

    def _build_model(self) -> None:
        """
        Builds the keras LSTM model and saves it
        to self.

        Returns:
             None
        """

        # Initialize
        n_lstm = len(self.hidden_sizes)
        model = Sequential()
        model.add(SeqInput(input_shape=(self.train_seq_len, self.n_feats)))

        # Add noise layer
        if self.input_noise_std is not None:
            model.add(ConstrainedNoise(self.input_noise_std, consts=self.constraint_list))

        # Add layers
        rnn = GRU if self.gru else LSTM
        for k in range(n_lstm):
            ret_seq = k != n_lstm - 1
            model.add(rnn(int(self.hidden_sizes[k]),
                          return_sequences=ret_seq))

        # Output layer
        # model.add(TimeDistributed(Dense(self.n_pred, activation=None)))
        if self.constraint_list is not None:
            out_constraints = [self.constraint_list[i] for i in self.out_inds]
        else:
            out_constraints = None
        out_const_layer = ConstrainedNoise(0, consts=out_constraints, is_input=False)
        model.add(Dense(self.n_pred, activation=None))
        if self.res_learn:
            # Add last non-control input to output
            seq_input = Input(shape=(self.train_seq_len, self.n_feats))
            m_out = model(seq_input)
            slicer = Lambda(lambda x: x[:, -1, :self.n_pred])
            last_input = slicer(seq_input)
            final_out = Add()([m_out, last_input])
            final_out = out_const_layer(final_out)
            model = Model(inputs=seq_input, outputs=final_out)
        else:
            model.add(out_const_layer)

        if self.weight_vec is not None:
            k_constants = K.constant(self.weight_vec)
            fixed_input = Input(tensor=k_constants)
            seq_input = Input(shape=(self.train_seq_len, self.n_feats))
            model = Model(inputs=seq_input, outputs=model(seq_input))
            loss = partial(weighted_loss, weights=fixed_input)
        else:
            loss = 'mse'

        opt = Adam(lr=self.lr)
        model.compile(loss=loss, optimizer=opt)
        if self.verbose:
            model.summary()
        self.m = model
        # pth = self.get_plt_path("Model.png")
        # plot_model(model, to_file=pth)

    def fit(self) -> None:
        """
        Fit the model if it hasn't been fitted before.
        Otherwise load the trained model.

        Returns:
             None
        """

        loaded = self.load_if_exists(self.m, self.name)
        if not loaded:
            if self.verbose:
                self.deb("Fitting Model...")

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
        """
        Predicts a batch of sequences using the fitted model.

        :param input_data: 3D numpy array with input data.
        :return: 2D numpy array with the predictions.
        """

        n = input_data.shape[0]

        # Predict
        predictions = self.m.predict(input_data)
        predictions = predictions.reshape((n, -1))
        return predictions
