import numpy as np

import keras
from keras import backend as K
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Input, RepeatVector, Lambda, Subtract, BatchNormalization

from keras_layers import ReduceMax2D, ReduceArgMax2D, OneHot, PrepInput


class NFQI:

    def __init__(self,
                 state_dim,
                 nb_actions,
                 mlp_layers=[20, 20, 20],
                 discount_factor=0.99,
                 target_network_update_freq=10,
                 lr=0.02,
                 max_iters=300):

        """
        Init function, stores all required parameters.
        """
        self.state_dim = state_dim
        self.nb_actions = nb_actions
        self.mlp_layers = mlp_layers
        self.discount_factor = discount_factor
        self.target_network_update_freq = target_network_update_freq
        self.lr = lr
        self.max_iters = max_iters
         
        self.opt_model = self.create_optimization_target()
        
    def create_Q_model(self, a_t, s_t, mlp_layers=[20, 20], trainable=True):
        """
        Initialize an NFQ network.
        """

        a_s_t = keras.layers.concatenate([a_t, s_t], axis=-1)

        # Add layers
        n_fc_layers = len(mlp_layers)
        for i in range(n_fc_layers):
            a_s_t = Dense(mlp_layers[i], activation='relu', trainable=trainable)(a_s_t)
            a_s_t = BatchNormalization(trainable=trainable)(a_s_t)
            a_s_t = Dropout(0.2)(a_s_t)


        # Reduce to 1D
        out = Dense(1, activation=None, trainable=trainable)(a_s_t)
        return out

    def create_optimization_target(self):
        """
        Creates Q(s_t, a_t) - \gamma \max_a Q(s_tp1, a) as Keras model.
        """

        # State and action inputs
        s_t = Input(shape=(self.state_dim,))
        s_tp1 = Input(shape=(self.state_dim,))
        a_t = Input(shape=(1,), dtype = np.int32)

        # Build Q-network
        a_t_one_hot = OneHot(self.nb_actions)(a_t)
        print(a_t_one_hot)
        q_out = self.create_Q_model(a_t_one_hot, s_t, self.mlp_layers, trainable = True)
        self.Q_net = Model(inputs=[s_t, a_t], outputs=q_out)

        # Target Q-network
        s_tp1_tar = RepeatVector(self.nb_actions)(s_tp1);
        a_t_tar = PrepInput(self.nb_actions)(s_tp1)        
        q_out_tar = self.create_Q_model(a_t_tar, s_tp1_tar, self.mlp_layers, trainable = False)
        max_q = ReduceMax2D(0)(q_out_tar)
        argmax_q = ReduceArgMax2D(0)(q_out_tar)
        self.greedy_policy = Model(inputs=s_tp1, outputs=argmax_q)
        self.target_Q_net = Model(inputs=s_tp1, outputs=max_q)

        lam_max_q = Lambda(lambda x: x * self.discount_factor)(max_q)        

        # Optimization Target
        opt_target = Subtract()([q_out, lam_max_q])

        # Define model
        model = Model(inputs=[s_t, a_t, s_tp1], outputs=opt_target)
        optim = RMSprop(lr=self.lr)
        model.compile(optimizer=optim, loss='mse')
        model.summary()

        return model

    def fit(self, D_s, D_a, D_r, D_s_prime):
        """
        Fit the Q-function.
        """
               
        num_targ_net_update = self.max_iters // self.target_network_update_freq

        for k in range(num_targ_net_update):   
            
            # Copy parameters to target network
            self.target_Q_net.set_weights(self.Q_net.get_weights())

            # Fit model with constant target network
            curr_init_epoch = k * self.target_network_update_freq
            self.opt_model.fit([D_s, D_a, D_s_prime], D_r, 
                               epochs = curr_init_epoch + self.target_network_update_freq, 
                               initial_epoch = curr_init_epoch,
                               validation_split = 0.1)

        pass

    def get_policy(self):
        """
        Returns the fitted policy.
        """
        def policy(s_t):
            s_t = np.reshape(s_t, (1, -1))
            action = self.greedy_policy.predict(s_t)
            return action[0][0]
        
        return policy