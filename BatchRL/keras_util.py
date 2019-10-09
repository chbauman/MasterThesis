from typing import Sequence

from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2


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
    """
    Returns a sequential MLP keras model.

    :param mlp_layers: The numbers of neurons per layer.
    :param out_dim: The output dimension.
    :param trainable: Whether the parameters should be trainable.
    :param dropout: Whether to use dropout.
    :param bn: Whether to use batch normalization.
    :param ker_reg: Kernel regularization weight.
    :return: Keras MLP model.
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
