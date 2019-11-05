import gym
from keras import Input, Model, Sequential
from keras.layers import Flatten, Concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.agents.dqn import DQNAgent, NAFAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.random import OrnsteinUhlenbeckProcess

from base_agent import AgentBase
from dynamics_envs import FullRoomEnv
from keras_util import getMLPModel
from visualize import plot_rewards
from util import *


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
                            train_interval=100,
                            target_model_update=500)
        self.dqn.compile(Adam(lr=1e-5), metrics=['mae'])

    def fit(self) -> None:
        # Fit and plot rewards
        hist = self.dqn.fit(self.env, nb_steps=100000, visualize=False, verbose=1)
        train_plot = self.env.get_plt_path("test")
        plot_rewards(hist, train_plot)

        # self.dqn.save_weights('dqn_{}_weights.h5f'.format(env.m.name), overwrite=True)
        # dqn.test(env, nb_episodes=5, visualize=True)

    def get_action(self, state):
        return self.dqn.forward(state)


class NAFBaseAgent(AgentBase):

    def __init__(self, env: FullRoomEnv):
        # Initialize super class
        super().__init__(env)

        # Build Q-function model.
        nb_actions = env.nb_actions
        n_state_vars = env.m.n_pred

        # V model
        inputs = Input(shape=(1, n_state_vars))
        flat_inputs = Flatten()(inputs)
        v_model = getMLPModel(out_dim=1)
        v_model = Model(inputs=inputs, outputs=v_model(flat_inputs))

        # Mu model
        inputs = Input(shape=(1, n_state_vars))
        flat_inputs = Flatten()(inputs)
        m_model = getMLPModel(out_dim=nb_actions)
        m_model = Model(inputs=inputs, outputs=m_model(flat_inputs))

        # L model
        n_out_l = (nb_actions * nb_actions + nb_actions) // 2
        action_input = Input(shape=(nb_actions,), name='action_input')
        state_inputs = Input(shape=(1, n_state_vars))
        flat_inputs = Flatten()(state_inputs)
        x = Concatenate()([action_input, flat_inputs])
        l_model = getMLPModel(out_dim=n_out_l)
        l_model = Model(inputs=[action_input, state_inputs], outputs=l_model(x))

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
        self.agent = NAFAgent(nb_actions=nb_actions, V_model=v_model, L_model=l_model, mu_model=m_model,
                              memory=memory, nb_steps_warmup=100, random_process=random_process,
                              gamma=.99, target_model_update=1e-3)
        self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    def fit(self) -> None:
        # Fit and plot rewards
        hist = self.agent.fit(self.env, nb_steps=100000, visualize=False, verbose=1)
        train_plot = self.env.get_plt_path("test")
        plot_rewards(hist, train_plot)

        # self.dqn.save_weights('dqn_{}_weights.h5f'.format(env.m.name), overwrite=True)
        # dqn.test(env, nb_episodes=5, visualize=True)

    def get_action(self, state):
        return self.agent.forward(state)


class DDPGBaseAgent(AgentBase):

    def __init__(self, env: FullRoomEnv):
        # Initialize super class
        super().__init__(env)

        # Build Q-function model.
        nb_actions = env.nb_actions
        n_state_vars = env.m.n_pred

        # Get the environment and extract the number of actions.
        self.env = env

        # Next, we build a very simple model.
        actor = Sequential()
        actor.add(Flatten(input_shape=(1, n_state_vars)))
        actor.add(getMLPModel(out_dim=nb_actions))

        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1, n_state_vars), name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])
        x = getMLPModel(out_dim=1)(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
        self.agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                               memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                               random_process=random_process, gamma=.99, target_model_update=1e-3)
        opt = Adam(lr=.001, clipnorm=1.0)
        self.agent.compile(opt, metrics=['mae'])

    def fit(self) -> None:
        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        self.agent.fit(self.env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=200)

        # Finally, evaluate our algorithm for 5 episodes.
        self.agent.test(self.env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
        # Fit and plot rewards
        hist = self.agent.fit(self.env, nb_steps=100000, visualize=False, verbose=1)
        train_plot = self.env.get_plt_path("test")
        plot_rewards(hist, train_plot)

        # self.dqn.save_weights('dqn_{}_weights.h5f'.format(env.m.name), overwrite=True)
        # dqn.test(env, nb_episodes=5, visualize=True)

    def get_action(self, state):
        return self.agent.forward(state)
