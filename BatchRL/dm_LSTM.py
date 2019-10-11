import keras

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import GRU, LSTM, TimeDistributed, Dense, GaussianNoise, Input, Add, Lambda
from keras.optimizers import Adam
from keras.utils import plot_model
from functools import partial

from base_dynamics_model import BaseDynamicsModel
from keras_layers import SeqInput
from time_series import AR_Model
from visualize import plot_train_history
from data import Dataset
from util import *


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
    lrs = '_LR' + str(lr)
    n_str = '' if input_noise_std is None else '_N' + str(input_noise_std)
    gru_str = '' if not gru else '_GRU'
    res_str = '' if not res_learn else '_RESL'
    w_str = '' if weight_vec is None else '_W' + '-'.join(map(str, weight_vec))
    return name + ep_s + arch + lrs + n_str + gru_str + res_str + w_str


class RNNDynamicModel(BaseDynamicsModel):
    """
    Simple LSTM used for training a dynamics model.
    """

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
                 use_ar_process: bool = False,
                 residual_learning: bool = False,
                 lr: float = 0.001):

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
        """
        name = constr_name(name, hidden_sizes,
                           n_iter_max, lr, gru,
                           residual_learning, weight_vec, input_noise_std)
        super(RNNDynamicModel, self).__init__(data, name, out_inds, in_inds)

        # Store data
        self.train_seq_len = self.data.seq_len - 1
        self.n_feats = self.data.d
        self.out_dim = self.n_pred

        # Store parameters
        self.hidden_sizes = np.array(hidden_sizes, dtype=np.int32)
        self.n_iter_max = n_iter_max
        self.gru = gru
        self.input_noise_std = input_noise_std
        self.use_AR = use_ar_process
        self.weight_vec = weight_vec
        self.res_learn = residual_learning
        self.lr = lr

        # Build model
        self.m = None
        self.build_model()

    def build_model(self) -> None:
        """
        Builds the keras LSTM model and saves it
        to self.

        :return: None
        """

        # Initialize
        n_lstm = len(self.hidden_sizes)
        model = Sequential()
        model.add(SeqInput(input_shape=(self.train_seq_len, self.n_feats)))

        # Add noise layer
        if self.input_noise_std is not None:
            model.add(GaussianNoise(self.input_noise_std))

        # Add layers
        rnn = GRU if self.gru else LSTM
        for k in range(n_lstm):
            ret_seq = k != n_lstm - 1
            model.add(rnn(self.hidden_sizes[k],
                          return_sequences=ret_seq))

        # Output layer
        # model.add(TimeDistributed(Dense(self.out_dim, activation=None)))
        model.add(Dense(self.out_dim, activation=None))
        if self.res_learn:
            # Add last non-control input to output
            seq_input = Input(shape=(self.train_seq_len, self.n_feats))
            m_out = model(seq_input)
            slicer = Lambda(lambda x: x[:, -1, :self.out_dim])
            last_input = slicer(seq_input)
            final_out = Add()([m_out, last_input])
            model = Model(inputs=seq_input, outputs=final_out)

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
        model.summary()
        self.m = model
        # pth = self.get_plt_path("Model.png")
        # plot_model(model, to_file=pth)

    def fit(self) -> None:
        """
        Fit the model if it hasn't been fitted before.
        Otherwise load the trained model.

        :return: None
        """

        loaded = self.load_if_exists(self.m, self.name)
        if not loaded:
            self.deb("Fitting Model...")

            # Prepare the data
            input_data, output_data = self.get_fit_data('train_val')

            # Fit and save model
            h = self.m.fit(input_data, output_data,
                           epochs=self.n_iter_max,
                           initial_epoch=0,
                           batch_size=128,
                           validation_split=self.data.val_percent)
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
