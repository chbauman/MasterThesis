import numpy as np

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Input, RepeatVector, Lambda, Subtract

from keras_layers import ReduceMax2D, OneHot, PrepInput


class NFQI:

    def __init__(self,
                 state_dim,
                 nb_actions,
                 mlp_layers=[20, 20],
                 discount_factor=0.99,
                 separate_target_network=False,
                 target_network_update_freq=None,
                 lr=0.01,
                 max_iters=20000,
                 max_q_predicted = 100000):

        
        self.state_dim = state_dim
        self.nb_actions = nb_actions
        self.mlp_layers = mlp_layers
        self.discount_factor = discount_factor
         
        self.create_optimization_target()
        
    def create_Q_model(self, a_t, s_t, mlp_layers=[20, 20], trainable=True):
        """
        Initialize an NFQ network.
        """

        a_s_t = keras.layers.concatenate([a_t, s_t], axis=-1)

        # Add layers
        n_fc_layers = len(mlp_layers)
        for i in range(n_fc_layers):
            a_s_t = Dense(mlp_layers[i], activation='relu', trainable=trainable)(a_s_t)

        # Reduce to 1D
        out = Dense(1, activation=None, trainable=trainable)(a_s_t)
        return out

    def create_optimization_target(self):

        # State and action inputs
        s_t = Input(shape=(self.state_dim,))
        s_tp1 = Input(shape=(self.state_dim,))
        a_t = Input(shape=(self.nb_actions,))

        # Get Q-network
        q_out = self.create_Q_model(a_t, s_t, self.mlp_layers, trainable = True)

        # Target Q-network
        s_tp1_tar = RepeatVector(self.nb_actions)(s_tp1);
        a_t_tar = PrepInput(self.nb_actions)(s_tp1)        
        q_out_tar = self.create_Q_model(a_t_tar, s_tp1_tar, self.mlp_layers, trainable = False)
        max_q = ReduceMax2D(0)(q_out_tar)
        lam_max_q = Lambda(lambda x: x * self.discount_factor)(max_q)

        # Optimization Target
        opt_target = Subtract()([q_out, lam_max_q])

        # Define model
        model = Model(inputs=[s_t, a_t, s_tp1], outputs=opt_target)
        model.compile(optimizer='rmsprop', loss='mse')
        model.summary()

        return model

    #target_model.set_weights(model.get_weights()) 