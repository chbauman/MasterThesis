"""A few keras RL agents.

Based on the agents of the keras-rl library, the agents
here are basically wrappers of those adding functionality
to work with the present framework.
"""
from keras import Input, Model, Sequential
from keras.layers import Flatten, Concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.agents.dqn import DQNAgent, NAFAgent
from rl.core import Agent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.random import OrnsteinUhlenbeckProcess

from agents.base_agent import AgentBase
from envs.dynamics_envs import FullRoomEnv, RLDynEnv
from ml.keras_layers import ClipByValue
from ml.keras_util import getMLPModel, KerasBase
from util.visualize import plot_rewards
from util.util import *


class KerasBaseAgent(AgentBase, KerasBase):
    """The interface for all keras-rl agent wrappers."""

    m: Agent  #: The keras-rl agent.
    model_path: str = "../Models/RL/"  #: Where to store the model parameters.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs.get('name') is None:
            print("Please provide a name for the agent!")

    def get_action(self, state):
        """Use the keras-rl model to get an action."""
        return self.m.forward(state)


class DQNBaseAgent(KerasBaseAgent):

    def __init__(self, env: FullRoomEnv):
        # Initialize super class
        name = "DQN"
        super().__init__(env=env, name=name)

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
        self.m = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                          policy=policy,
                          gamma=0.9,
                          train_interval=100,
                          target_model_update=500)
        self.m.compile(Adam(lr=1e-5), metrics=['mae'])

    @train_decorator(True)
    def fit(self) -> None:
        # Fit and plot rewards
        hist = self.m.fit(self.env, nb_steps=100000, visualize=False, verbose=1)
        train_plot = self.env.get_plt_path("test")
        plot_rewards(hist, train_plot)
        # dqn.test(env, nb_episodes=5, visualize=True)


class NAFBaseAgent(KerasBaseAgent):
    """This does not work!

    TODO: Fix this!
    """

    def __init__(self, env: FullRoomEnv):
        # Initialize super class
        name = "NAF"
        super().__init__(env=env, name=name)
        print("Why don't you work??????")

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
        self.m = NAFAgent(nb_actions=nb_actions, V_model=v_model, L_model=l_model, mu_model=m_model,
                          memory=memory, nb_steps_warmup=100, random_process=random_process,
                          gamma=.99, target_model_update=1e-3)
        self.m.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    def fit(self) -> None:
        # Fit and plot rewards
        hist = self.m.fit(self.env, nb_steps=100000, visualize=False, verbose=1)
        train_plot = self.env.get_plt_path("test")
        plot_rewards(hist, train_plot)


class DDPGBaseAgent(KerasBaseAgent):
    """The wrapper of the keras-rl DDPG agent.

    Suited for continuous action and state space.
    Range of allowed actions can be specified.
    """
    def __init__(self, env: RLDynEnv,
                 n_steps: int = 50000,
                 lr: float = 0.001,
                 gamma: float = 0.9,
                 layers: Sequence[int] = (50, 50),
                 reg: float = 0.01,
                 action_range: Sequence = None):

        """Constructor.

        Args:
            env: The underlying environment.
            n_steps: The number of steps to train.
            lr: The base learning rate.
            gamma: The discount factor.
            layers: The layer architecture of the MLP for the actor and the critic network.
            reg: The regularization factor for the networks.
            action_range: The range of the actions the actor can take.
        """
        # Find unique name based on parameters.
        param_ex_list = [("N", n_steps),
                         ("LR", lr),
                         ("GAM", gamma),
                         ("L", layers),
                         ("REG", reg),
                         ("AR", action_range)]
        name = "DDPG_" + env.name + make_param_ext(param_ex_list)

        # Initialize super class.
        super().__init__(env=env, name=name)

        # Save reference to env and extract relevant dimensions.
        self.env = env
        self.nb_actions = env.nb_actions
        self.n_state_vars = env.m.n_pred

        # Training parameters
        self.n_steps = n_steps
        self.lr = lr
        self.gamma = gamma

        # Network parameters
        self.layers = layers
        self.reg = reg
        if action_range is not None:
            assert len(action_range) == 2, "Fucking retarded?"
        self.action_range = action_range

        # Build the model.
        self._build_agent_model()

    def _build_agent_model(self) -> None:
        """Builds the Keras model of the agent."""
        # Build actor model
        actor = Sequential()
        actor.add(Flatten(input_shape=(1, self.n_state_vars)))
        actor.add(getMLPModel(mlp_layers=self.layers,
                              out_dim=self.nb_actions,
                              ker_reg=self.reg))

        # Clip actions to desired interval
        if self.action_range is not None:
            clip_layer = ClipByValue(self.action_range[0], self.action_range[1])
            actor.add(clip_layer)

        # Build critic model
        action_input = Input(shape=(self.nb_actions,), name='action_input')
        observation_input = Input(shape=(1, self.n_state_vars), name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])
        x = getMLPModel(mlp_layers=self.layers, out_dim=1, ker_reg=self.reg)(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)

        # Configure and compile the agent.
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=.15, mu=0., sigma=.3)
        self.m = DDPGAgent(nb_actions=self.nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                           memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                           random_process=random_process, gamma=self.gamma, target_model_update=1e-3)
        opt = Adam(lr=self.lr, clipnorm=1.0)
        self.m.compile(opt, metrics=['mae'])

    def load_if_exists(self, m, name: str) -> bool:
        """Loads the keras model if it exists.

        Returns true if it could be loaded, else False.
        Overrides the function in `KerasBase`, but in this
        case there are two models to load.

        Args:
            m: Keras-rl agent model to be loaded.
            name: Name of model.

        Returns:
             True if model could be loaded else False.
        """
        full_path = self.get_path(name)
        path_actor = full_path[:-3] + "_actor.h5"
        path_critic = full_path[:-3] + "_critic.h5"

        if os.path.isfile(path_actor) and os.path.isfile(path_critic):
            m.load_weights(full_path)
            return True
        return False

    def save_model(self, m, name: str) -> None:
        """Saves a keras model.

        Needs to be overridden here since the keras-rl
        `DDPGAgent` class does not have a `save` method.

        Args:
            m: Keras-rl agent model.
            name: Name of the model.
        """
        m.save_weights(self.get_path(name))

    @train_decorator(True)
    def fit(self) -> None:
        """Fit the agent using the environment.

        Makes a plot of the rewards received during the training.
        """
        # Fit and plot rewards
        hist = self.m.fit(self.env, nb_steps=self.n_steps, visualize=False, verbose=1, nb_max_episode_steps=200)
        train_plot = self.env.get_plt_path(self.name + "_train_rewards")
        plot_rewards(hist, train_plot)