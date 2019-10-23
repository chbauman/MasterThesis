import tensorflow as tf
# from tensorflow.keras import backend as K
from keras import backend as K
from keras.layers import Layer, GaussianNoise

from data import SeriesConstraint
from keras_util import *
from util import *


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
        """
        Initializes the layer.

        Args:
            **kwargs: kwargs for super.
        """
        super(SeqInput, self).__init__(**kwargs)

    def call(self, x):
        """
        Returns `x` unchanged.

        Args:
            x: Input tensor.

        Returns:
            `x` unchanged.
        """
        return x

    def compute_output_shape(self, input_shape):
        """
        The shape stays the same.

        Args:
            input_shape: The shape of the input.

        Returns:
            Same as input.
        """
        return input_shape


class ConstrainedNoise(Layer):
    """
    Constrained noise layer.
    Note that the clipping will be active during testing.
    """

    consts: Sequence[SeriesConstraint]  #: Sequence of constraints
    input_noise_std: float  #: The std of the Gaussian noise to add
    is_input: bool

    def __init__(self, input_noise_std: float = 0,
                 consts: Sequence[SeriesConstraint] = None,
                 is_input: bool = True,
                 **kwargs):
        """
        Adds Gaussian noise with mean 0 and std as specified
        and then applies the constraints.

        Args:
            input_noise_std: The level of noise.
            consts: The list of constraints.
            is_input: Whether it is applied to an input tensor (3D)
                or an output tensor (2D).
            **kwargs: Layer kwargs.
        """
        super(ConstrainedNoise, self).__init__(**kwargs)
        self.input_noise_std = input_noise_std
        self.consts = consts
        self.is_input = is_input

    def call(self, x):
        """
        Builds the layer given the input x.

        Args:
            x: The input to the layer.

        Returns:
            The output of the layer satisfying the constraints.
        """
        x_modify = x

        # Add noise if std > 0
        if self.input_noise_std > 0:
            gn_layer = GaussianNoise(self.input_noise_std)
            x_modify = gn_layer(x_modify)

        # Enforce constraints
        if self.consts is not None:
            noise_x = x_modify
            n_feats = len(self.consts)

            # Split features
            if self.is_input:
                feature_tensors = [noise_x[:, :, ct:(ct + 1)] for ct in range(n_feats)]
            else:
                feature_tensors = [noise_x[:, ct:(ct + 1)] for ct in range(n_feats)]
            for ct, c in enumerate(self.consts):
                if c[0] is None:
                    continue
                elif c[0] == 'interval':
                    iv = c[1]
                    feature_tensors[ct] = K.clip(feature_tensors[ct], iv[0], iv[1])
                elif c[0] == 'exact' and self.input_noise_std > 0:
                    feature_tensors[ct] = x[:, :, ct:(ct + 1)]
                else:
                    raise ValueError("Constraint type {} not supported!!".format(c[0]))

            # Concatenate again
            x_modify = K.concatenate(feature_tensors, axis=-1)

        return x_modify

    def compute_output_shape(self, input_shape):
        """
        The shape stays the same.

        Args:
            input_shape: The shape of the input.

        Returns:
            Same as input.
        """
        return input_shape


class FeatureSlice(Layer):
    """Extracts specified features from tensor.

    TODO: Make it more efficient by considering not single slices but
        multiple consecutive ones.
    """

    slicing_indices: np.ndarray  #: The array with the indices.
    n_feats: int  #: The number of selected features.
    n_dims: int  #: The number of dimensions of the input tensor.
    return_last_seq: bool  #: Whether to only return the last slice of each sequence.

    def __init__(self, s_inds: np.ndarray,
                 n_dims: int = 3,
                 return_last_seq: bool = True,
                 **kwargs):
        """Initialize layer.

        Args:
            s_inds: The array with the indices.
            n_dims: The number of dimensions of the input tensor.
            **kwargs: The kwargs for super(), e.g. `name`.
        """
        super(FeatureSlice, self).__init__(**kwargs)
        self.slicing_indices = s_inds
        self.n_feats = len(s_inds)
        self.n_dims = n_dims
        self.return_last_seq = return_last_seq

        if n_dims == 2:
            raise NotImplementedError("Not implemented for 2D tensors!")

    def call(self, x):
        """
        Builds the layer given the input x. Selects the features
        specified in `slicing_indices` and concatenates them.

        Args:
            x: The input to the layer.

        Returns:
            The output of the layer containing the slices.
        """
        s = -1 if self.return_last_seq else slice(None)
        feature_tensors = [x[:, s, ct:(ct + 1)] for ct in self.slicing_indices]
        return K.concatenate(feature_tensors, axis=-1)

    def compute_output_shape(self, input_shape):
        """
        The shape only changes in the feature dimension.

        Args:
            input_shape: The shape of the input.

        Returns:
            Same as input with the last dimension changed.
        """
        s = input_shape
        if self.return_last_seq:
            return s[0], self.n_feats
        return s[0], s[1], self.n_feats


def get_test_layer_output(layer: Layer, np_input, learning_phase: float = 1.0):
    """Test a keras layer.

    Builds a model with only the layer given and
    returns the output when given `np.input` as input.

    Args:
        layer: The keras layer.
        np_input: The input to the layer.
        learning_phase: Whether learning is active or not.

    Returns:
        The layer output.
    """
    # Construct sequential model with only one layer
    m = Sequential()
    m.add(layer)
    out, inp = m.output, m.input
    k_fun = K.function([inp, K.learning_phase()], [out])
    layer_out = k_fun([np_input, learning_phase])[0]
    return layer_out


def test_layers() -> None:
    """
    Test the custom layers.

    Returns: None

    Raises:
        AssertionError: If a test fails.
    """
    # Define data
    seq_input = np.array([
        [[1, 2, 3, 4], [2, 3, 4, 5]],
        [[5, 5, 5, 5], [3, 3, 3, 3]],
    ])
    output = np.array([
        [2, 3],
        [3, 3],
    ])

    # Get shapes
    in_shape = seq_input.shape
    out_shape = output.shape
    batch_size, seq_len, n_in_feat = in_shape
    n_out_feat = out_shape[1]

    # Test SeqInput
    inp_layer = SeqInput(input_shape=(seq_len, n_in_feat))
    layer_out = get_test_layer_output(inp_layer, seq_input, 1.0)
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
    const_layer = ConstrainedNoise(noise_level, consts, input_shape=(seq_len, n_in_feat))
    layer_out = get_test_layer_output(const_layer, seq_input, 1.0)
    layer_out_test = get_test_layer_output(const_layer, seq_input, 0.0)
    if not np.allclose(layer_out[:, :, 2:], seq_input[:, :, 2:]):
        raise AssertionError("Exact constraint in Constrained Noise layer not implemented correctly!!")
    if not check_in_range(layer_out[:, :, 0], 0.0, 1.00001):
        raise AssertionError("Interval constraint in Constrained Noise layer not implemented correctly!!")
    if not np.allclose(layer_out_test[:, :, 1:], seq_input[:, :, 1:]):
        raise AssertionError("Noise layer during testing still active!!")
    print("Keras Layers tests passed :)")

    # Test FeatureSlice layer
    indices = [0, 2]
    lay = FeatureSlice(np.array(indices), input_shape=(seq_len, n_in_feat))
    layer_out = get_test_layer_output(lay, seq_input)
    if not np.array_equal(layer_out, seq_input[:, -1, indices]):
        raise AssertionError("FeatureSlice layer not working!!")
