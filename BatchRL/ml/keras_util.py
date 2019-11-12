import os
from typing import Sequence, Union

from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2


KerasModel = Union[Sequential, Model]


def soft_update_params(model_to_update, other, lam: float = 1.0) -> None:
    """
    Soft parameter update:
    :math:`\\theta' = \\lambda * \\theta + (1 - \\lambda) * \\theta'`

    :param model_to_update: Model where the parameters will be updated.
    :param other: Model where the parameters of the model should be updated to.
    :param lam: Factor determining how much the parameters are updated.
    :return: None
    """
    params = other.get_weights()

    if lam == 1.0:
        model_to_update.set_weights(params)
        return
    else:
        orig_params = model_to_update.get_weights()
        for ct, el in enumerate(orig_params):
            orig_params[ct] = (1.0 - lam) * el + lam * params[ct]
        model_to_update.set_weights(orig_params)


def max_loss(y_true, y_pred):
    """
    Loss independent of the true labels for
    optimization without it, i.e. maximize y_pred
    directly.

    :param y_true: True labels, not used here.
    :param y_pred: Output for maximization.
    :return: -y_pred since this will be minimized.
    """
    return -y_pred


def getMLPModel(mlp_layers: Sequence = (20, 20), out_dim: int = 1,
                trainable: bool = True,
                dropout: bool = False,
                bn: bool = False,
                ker_reg: float = 0.01):
    """Returns a sequential MLP keras model.

    Args:
        mlp_layers: The numbers of neurons per layer.
        out_dim: The output dimension.
        trainable: Whether the parameters should be trainable.
        dropout: Whether to use dropout.
        bn: Whether to use batch normalization.
        ker_reg: Kernel regularization weight.

    Returns:
        Sequential keras MLP model.
    """
    model = Sequential()
    if bn:
        model.add(BatchNormalization(trainable=trainable, name="bn0"))

    # Add layers
    n_fc_layers = len(mlp_layers)
    for i in range(n_fc_layers):
        next_layer = Dense(mlp_layers[i],
                           activation='relu',
                           trainable=trainable,
                           kernel_regularizer=l2(ker_reg),
                           name="dense" + str(i))
        model.add(next_layer)
        if bn:
            model.add(BatchNormalization(trainable=trainable, name="bn" + str(i + 1)))
        if dropout:
            model.add(Dropout(0.2))

    # Reduce to 1D
    last = Dense(out_dim, activation=None, trainable=trainable, name="last_dense")
    model.add(last)
    return model


class KerasBase:
    """Base class for keras models.

    Provides an interface for saving and loading models.
    """

    model_path: str = "../Models/Dynamics/"
    m: KerasModel

    def save_model(self, m, name: str) -> None:
        """Saves a keras model.

        Args:
            m: Keras model.
            name: Name of the model.
        """
        m.save(self.get_path(name))

    def load_if_exists(self, m, name: str) -> bool:
        """Loads the keras model if it exists.

        Returns true if it could be loaded, else False.

        Args:
            m: Keras model to be loaded.
            name: Name of model.

        Returns:
             True if model could be loaded else False.
        """
        full_path = self.get_path(name)

        if os.path.isfile(full_path):
            m.load_weights(full_path)
            return True
        return False

    def get_path(self, name: str) -> str:
        """
        Returns the path where the model parameters
        are stored. Used for keras models only.

        Args:
            name: Model name.

        Returns:
            Model parameter file path.
        """
        return os.path.join(self.model_path, name + ".h5")