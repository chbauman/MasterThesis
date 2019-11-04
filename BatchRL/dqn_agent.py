from keras import Input, Model
from keras.layers import Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from base_agent import AgentBase
from dynamics_envs import FullRoomEnv
from keras_util import getMLPModel
from visualize import plot_rewards


class DQNBaseAgent(AgentBase):

    def __init__(self, env: FullRoomEnv):

        # Initialize super class
        super().__init__(env)

        # Build Q-function model.
        nb_actions = env.nb_actions
        n_state_vars = env.m.n_pred
        inputs = Input(shape=(1, n_state_vars))
        flat_inputs = Flatten()(inputs)
        model = getMLPModel(out_dim=nb_actions)
        model = Model(inputs=inputs, outputs=model(flat_inputs))
        # model.summary()

        # Configure and compile our agent.
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        self.dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                            policy=policy,
                            gamma=0.9,
                            train_interval=3,
                            target_model_update=1000)
        self.dqn.compile(Adam(lr=1e-5), metrics=['mae'])

    def fit(self) -> None:

        # Fit and plot rewards
        hist = self.dqn.fit(self.env, nb_steps=3000, visualize=False, verbose=1)
        train_plot = self.env.get_plt_path("test")
        plot_rewards(hist, train_plot)

        # self.dqn.save_weights('dqn_{}_weights.h5f'.format(env.m.name), overwrite=True)
        # dqn.test(env, nb_episodes=5, visualize=True)

    def get_action(self, state):
        pass
