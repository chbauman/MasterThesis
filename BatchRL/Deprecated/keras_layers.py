"""Custom keras layers.

Define your custom keras layers here.
There is also a function that tests the layers
for some example input.
"""
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer


class ReduceMax2D(Layer):

    def __init__(self, axis=0, **kwargs):
        self.axis = axis
        super(ReduceMax2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReduceMax2D, self).build(input_shape)

    def call(self, x, **kwargs):
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

    def call(self, x, **kwargs):
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

    def call(self, x, **kwargs):
        return K.argmax(x, axis=self.axis + 1)

    def compute_output_shape(self, input_shape):
        if self.axis == 0:
            return input_shape[0], input_shape[2]
        else:
            return input_shape[0], input_shape[1]


class OneHot(Layer):
    """Assuming input of shape (None, 1)."""

    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        super(OneHot, self).__init__(**kwargs)

    def build(self, input_shape):
        super(OneHot, self).build(input_shape)

    def call(self, x, **kwargs):
        one_h_enc = K.one_hot(x, self.num_classes)
        return K.reshape(one_h_enc, (-1, self.num_classes))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.num_classes


class PrepInput(Layer):
    """Uses the states input only for matching the size."""

    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        super(PrepInput, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PrepInput, self).build(input_shape)

    def call(self, x, **kwargs):
        tf_constant = K.arange(self.num_classes)
        one_hot_enc = K.one_hot(tf_constant, self.num_classes)
        one_hot_enc_p1 = K.reshape(one_hot_enc, (1, self.num_classes, self.num_classes))
        all_encodings = K.tile(one_hot_enc_p1, (K.shape(x)[0], 1, 1))
        return K.reshape(all_encodings, (K.shape(x)[0], self.num_classes, self.num_classes))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.num_classes, self.num_classes
