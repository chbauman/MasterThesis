import numpy as np
import tensorflow as tf

from util import *
from keras_util import *
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dropout, Input, RepeatVector, \
    Lambda, Subtract, BatchNormalization, Layer, Concatenate, GaussianNoise, namedtuple
from data import SeriesConstraint


class ReduceMax2D(Layer):

    def __init__(self, axis=0, **kwargs):
        self.axis = axis
        super(ReduceMax2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReduceMax2D, self).build(input_shape)

    def call(self, x):
        return K.max(x, axis=self.axis + 1)

    def compute_output_shape(self, input_shape):
        if self.axis == 0:
            return input_shape[0], input_shape[2]
        else:
            return input_shape[0], input_shape[1]


class ReduceProbabilisticSoftMax2D(Layer):

    def __init__(self, axis=0, beta=1.0, regular=0.001, **kwargs):
        self.axis = axis
        self.beta = beta
        self.regular = regular
        super(ReduceProbabilisticSoftMax2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReduceProbabilisticSoftMax2D, self).build(input_shape)

    def call(self, x):
        x_sh = x.shape
        act_axis = self.axis + 1
        mu = K.mean(x, axis=act_axis)
        s = K.std(x, axis=act_axis)
        x_standardized = (x - mu) / (s + self.regular)
        x_std_res = K.reshape(x_standardized, (-1, x_sh[1]))
        indices = tf.random.multinomial(self.beta * x_std_res, 1)
        return K.reshape(indices, (-1, x_sh[1], x_sh[2]))

    def compute_output_shape(self, input_shape):
        if self.axis == 0:
            return input_shape[0], input_shape[2]
        else:
            return input_shape[0], input_shape[1]


class ReduceArgMax2D(Layer):

    def __init__(self, axis=0, **kwargs):
        self.axis = axis
        super(ReduceArgMax2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReduceArgMax2D, self).build(input_shape)

    def call(self, x):
        return K.argmax(x, axis=self.axis + 1)

    def compute_output_shape(self, input_shape):
        if self.axis == 0:
            return input_shape[0], input_shape[2]
        else:
            return input_shape[0], input_shape[1]


class OneHot(Layer):
    """
    Assuming input of shape (None, 1). 
    """

    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        super(OneHot, self).__init__(**kwargs)

    def build(self, input_shape):
        super(OneHot, self).build(input_shape)

    def call(self, x):
        one_h_enc = K.one_hot(x, self.num_classes)
        return K.reshape(one_h_enc, (-1, self.num_classes))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.num_classes


class PrepInput(Layer):
    """
    Uses the states input only for matching the size.
    """

    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        super(PrepInput, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PrepInput, self).build(input_shape)

    def call(self, x):
        tf_constant = K.arange(self.num_classes)
        one_hot_enc = K.one_hot(tf_constant, self.num_classes)
        one_hot_enc_p1 = K.reshape(one_hot_enc, (1, self.num_classes, self.num_classes))
        all_encodings = K.tile(one_hot_enc_p1, (K.shape(x)[0], 1, 1))
        return K.reshape(all_encodings, (K.shape(x)[0], self.num_classes, self.num_classes))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.num_classes, self.num_classes


class SeqInput(Layer):
    """
    Dummy Layer, it lets you specify the input shape
    when used as a first layer in a Sequential model.
    """

    def __init__(self, **kwargs):
        super(SeqInput, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class ConstrainedNoise(Layer):
    """
    Constrained noise layer.
    """

    consts: Sequence[SeriesConstraint]  #: Sequence of constraints
    input_noise_std: float  #: The std of the Gaussian noise to add

    def __init__(self, input_noise_std: float = 0, consts: Sequence[SeriesConstraint] = None, **kwargs):
        super(ConstrainedNoise, self).__init__(**kwargs)
        self.input_noise_std = input_noise_std
        self.consts = consts

    def build(self, input_shape):
        pass

    def call(self, x):
        if self.input_noise_std == 0:
            return x
        gn_layer = GaussianNoise(self.input_noise_std)
        if self.consts is None:
            return gn_layer(x)
        else:
            noise_x = gn_layer(x)
            n_feats = len(self.consts)

            # Split features
            feature_tensors = [noise_x[:, :, ct:(ct + 1)] for ct in range(n_feats)]
            for ct, c in enumerate(self.consts):
                if c[0] is None:
                    continue
                elif c[0] == 'interval':
                    iv = c[1]
                    feature_tensors[ct] = K.clip(feature_tensors[ct], iv[0], iv[1])
                elif c[0] == 'exact':
                    feature_tensors[ct] = x[:, :, ct:(ct + 1)]
                else:
                    raise ValueError("Constraint type {} not supported!!".format(c[0]))

            # Concatenate again
            return K.concatenate(feature_tensors, axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape


def test_layers() -> None:
    """
    Test the custom layers.

    :return: None
    :raises AssertionError: If a test fails.
    """
    seq_input = np.array([
        [[1, 2, 3, 4], [2, 3, 4, 5]],
        [[5, 5, 5, 5], [3, 3, 3, 3]],
    ])
    output = np.array([
        [2, 3],
        [3, 3],
    ])

    in_shape = seq_input.shape
    out_shape = output.shape
    batch_size, seq_len, n_in_feat = in_shape
    n_out_feat = out_shape[1]

    # Initialize model
    model = Sequential()

    # Test SeqInput
    inp_layer = SeqInput(input_shape=(seq_len, n_in_feat))
    model.add(inp_layer)
    inp = model.input

    out = model.output
    k_fun = K.function([inp, K.learning_phase()], [out])
    layer_out = k_fun([seq_input, 1.0])
    if not np.allclose(layer_out, seq_input):
        raise AssertionError("SeqInput layer not implemented correctly!!")

    # Test Constraint Layer
    consts = [
        SeriesConstraint('interval', [0.0, 1.0]),
        SeriesConstraint(),
        SeriesConstraint('exact'),
        SeriesConstraint('exact'),
            ]
    noise_level = 5.0
    const_layer = ConstrainedNoise(noise_level, consts)
    model.add(const_layer)
    out = model.output
    k_fun = K.function([inp, K.learning_phase()], [out])
    layer_out = k_fun([seq_input, 1.0])[0]
    if not np.allclose(layer_out[:, :, 2:], seq_input[:, :, 2:]):
        raise AssertionError("Exact constraint in Constrained Noise layer not implemented correctly!!")
    if not check_in_range(layer_out[:, :, 0], 0.0, 1.00001):
        raise AssertionError("Interval constraint in Constrained Noise layer not implemented correctly!!")

    print("Keras Layers tests passed :)")

