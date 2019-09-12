


from keras.models import Sequential, Model
from keras.layers import Dense, Multiply

from base_dynamics_model import BaseDynamicsModel
from keras_layers import SeqInput
from keras_util import getMLPModel

class BatteryModel(BaseDynamicsModel):
    """
    The model of the battery:
    s_t: SoC at time t
    Model: s_{t+1} = \eta(s_t) p_{t+1}
    p_{t+1}: average charging power from time t to t+1 (cotrol input)
    """

    def __init__(self, mlp_layers = [10, 10, 10]):
        
        # Input tensors
        s_inp = Input(shape=(1,))
        p_inp = Input(shape=(1,))

        # Define model
        mlp = getMLPModel(mlp_layers, 1)
        m_s = mlp(s_inp)
        out = Multiply(m_s, p_inp)
        self.m = Model(inputs=[s_inp, p_inp], outputs=out)

        pass

    def fit(self, data):
        pass

    def predict(self, data, prepared = False):
        pass

    def disturb(self, n):
        """
        Returns a sample of noise of length n.
        """
        pass

    pass