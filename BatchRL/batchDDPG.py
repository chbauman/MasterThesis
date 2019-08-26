
import numpy as np

import keras
from keras import backend as K
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Input, RepeatVector, \
                Lambda, Subtract, BatchNormalization
from keras.regularizers import l2


from keras_layers import ReduceMax2D, ReduceArgMax2D, OneHot, PrepInput, \
                ReduceProbabilisticSoftMax2D
from keras_util import *



class bDDPG:
    """
    Implements the algorithm described in:
        Deterministic Policy Gradient Algorithms
        by: David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller
    But applied offline, i.e. with the collected past data as replay memory
    and without sampling new trajectories.
    Further also using deep neural networks as function approximators
    for the policy and the action value function.
    """

    def __init__(self,
                 state_dim,
                 action_dim = 1,
                 mlp_layers=[50, 50, 50],
                 discount_factor=0.9,
                 use_diff_target_net = True,
                 target_network_update_freq=3,
                 lr=0.0001,
                 max_iters=200,
                 tau = 0.2,
                 ):

        """
        Init function, stores all required parameters.
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mlp_layers = mlp_layers
        self.discount_factor = discount_factor
        self.use_diff_target_net = use_diff_target_net
        self.target_network_update_freq = target_network_update_freq
        self.lr = lr
        self.tau = tau
        self.max_iters = max_iters
         
        # Create model
        self.opt_model = self.create_optimization_target()

    def create_optimization_target(self):
        """
        Creates Q(s_t, a_t) - \gamma Q'(s_tp1, \mu'(s_tp1)) as Keras model.
        Also: Q(s_t, \mu(s_t)) for policy updates.
        """

        # State and action inputs, continuous actions
        s_t = Input(shape=(self.state_dim,))
        s_tp1 = Input(shape=(self.state_dim,))
        a_t = Input(shape=(self.action_dim,))        

        # Build models
        self.Q_net = getMLPModel(self.mlp_layers, trainable = True)
        self.const_Q_net = getMLPModel(self.mlp_layers, trainable = True)
        self.Q_tar_net = getMLPModel(self.mlp_layers, trainable = False)
        self.Pol_net = getMLPModel(self.mlp_layers, trainable = True)
        self.Pol_tar_net = getMLPModel(self.mlp_layers, trainable = False)

        # Get outputs
        pol_tar_s_tp1 = self.Pol_tar_net(s_tp1)
        q_tar_args = keras.layers.concatenate([pol_tar_s_tp1, s_tp1], axis=-1)
        q_tar_eval = self.Q_tar_net(q_tar_args)
        a_s_t = keras.layers.concatenate([a_t, s_t], axis=-1)
        q_eval = self.Q_net(a_s_t)
        pol_s_t = self.Pol_net(s_t)
        pol_s_t_s_t = keras.layers.concatenate([pol_s_t, s_t], axis=-1)
        q_const_eval_pol = self.const_Q_net(pol_s_t_s_t)

        # Q-Function update
        lam_max_q = Lambda(lambda x: x * self.discount_factor)(q_tar_eval) 
        q_target = Subtract()([q_eval, lam_max_q])
        q_update_model = Model(inputs=[s_t, a_t, s_tp1], outputs=q_const_eval_pol)
        optim = RMSprop(lr=self.lr)
        q_update_model.compile(optimizer=optim, loss='mse')
        q_update_model.summary()

        # Policy Improvement
        pol_model = Model(inputs=[s_t], outputs=q_target)
        pol_model.compile(optimizer=optim, loss=max_loss)
        pol_model.summary()

        # The final policy
        self.fitted_policy = Model(inputs=s_t, outputs=pol_s_t)

        return [q_update_model, pol_model]

    def fit(self, D_s, D_a, D_r, D_s_prime):
        """
        Fit the Q-function.
        """
               
        num_targ_net_update = self.max_iters // self.target_network_update_freq

        # Initialize target networks as trainable ones
        self.Q_tar_net.set_weights(self.Q_net.get_weights())
        self.Pol_tar_net.set_weights(self.Pol_net.get_weights())

        for k in range(num_targ_net_update):   
            
            # Copy parameters to target network
            if self.use_diff_target_net:
                self.target_Q_net.set_weights(self.Q_net.get_weights())

            # Q-update with constant target network
            curr_init_epoch = k * self.target_network_update_freq
            self.opt_model[0].fit([D_s, D_a, D_s_prime], D_r, 
                               epochs = curr_init_epoch + self.target_network_update_freq, 
                               initial_epoch = curr_init_epoch,
                               batch_size = 128,
                               validation_split = 0.1)
            
            # Copy params to const Q-net
            self.const_Q_net.set_weights(self.Q_net.get_weights())

            # Policy update
            self.opt_model[1].fit([D_s], D_r, 
                               epochs = curr_init_epoch + self.target_network_update_freq, 
                               initial_epoch = curr_init_epoch,
                               batch_size = 128,
                               validation_split = 0.1)

            # Soft parameter update
            soft_update_params(self.Q_tar_net, self.Q_net, self.tau)
            soft_update_params(self.Pol_tar_net, self.Pol_net, self.tau)

        pass

    def get_policy(self):
        """
        Returns the fitted policy.
        """
        def policy(s_t):
            s_t = np.reshape(s_t, (1, -1))
            action = self.fitted_policy.predict(s_t)
            return action
        
        return policy
