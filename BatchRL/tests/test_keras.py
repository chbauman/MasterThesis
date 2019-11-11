from unittest import TestCase

import numpy as np

from keras_layers import get_test_layer_output, SeqInput


class TestKeras(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define shapes
        self.seq_shape = (2, 6, 4)
        self.seq_len_red = 2
        self.b_s, self.seq_len_test, self.n_feats = self.seq_shape
        self.seq_shape_red = (self.b_s, self.seq_len_red, self.n_feats)

        # Define the data
        self.seq_input = np.arange(self.b_s * self.seq_len_red * self.n_feats).reshape(self.seq_shape_red)
        self.seq_input_long = np.arange(self.b_s * self.seq_len_test * self.n_feats).reshape(self.seq_shape)
        self.feat_indices = np.array([0, 2], dtype=np.int32)
        self.n_feats_chosen = len(self.feat_indices)
        self.output = -1. * np.arange(self.b_s * self.n_feats_chosen).reshape((self.b_s, self.n_feats_chosen))
        self.id_1 = np.array([[1, 2, 3]])
        self.id_2 = np.array([[2, 2, 2]])

    def test_seq_input(self):
        inp_layer = SeqInput(input_shape=(self.seq_len_red, self.n_feats))
        layer_out = get_test_layer_output(inp_layer, self.seq_input, 1.0)
        self.assertTrue(np.allclose(layer_out, self.seq_input), "SeqInput layer not implemented correctly!!")

    pass
