import numpy as np
import tensorflow as tf

from keras import backend as K

from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Activation, Flatten, Dropout, Input, RepeatVector, \
    Lambda, Subtract, BatchNormalization, Layer, Concatenate


def soft_update_params(model_to_update, other, lam = 1.0):
    """
    Soft parameter update, \theta' = \lambda * \theta + (1 - \lambda) * \theta'
    """
    params = other.get_weights();

    if lam == 1.0:
        model_to_update.set_weights(params)
        return
    else:
        orig_params = model_to_update.get_weights();
        for ct, el in enumerate(orig_params):
            orig_params[ct] = (1.0 - lam) * el + lam * params[ct]
        model_to_update.set_weights(orig_params)
    return


def max_loss(y_true, y_pred):
    """
    Loss independent of the true labels for 
    optimization without it, i.e. maximize y_pred
    directly.
    """
    return -y_pred


def getMLPModel(mlp_layers=[20, 20], out_dim = 1, trainable = True, dropout = False, bn = False, ker_reg = 0.01):
    """
    Returns a sequential MLP keras model.
    """
    model = Sequential()
    if bn:
        model.add(BatchNormalization(trainable=trainable, name = "bn0"))

    # Add layers
    n_fc_layers = len(mlp_layers)
    for i in range(n_fc_layers):
        next = Dense(mlp_layers[i],
                          activation='relu', 
                          trainable=trainable, 
                          kernel_regularizer=l2(ker_reg), 
                          name = "dense" + str(i))
        model.add(next)
        if bn:
            model.add(BatchNormalization(trainable=trainable, name = "bn" + str(i + 1)))
        if dropout:
            model.add(Dropout(0.2))

    # Reduce to 1D
    last = Dense(out_dim, activation=None, trainable=trainable, name = "lastdense")
    model.add(last)
    return model