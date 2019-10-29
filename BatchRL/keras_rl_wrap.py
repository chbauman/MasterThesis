import numpy as np
import gym
from keras import Input, Model

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from keras_util import getMLPModel
from visualize import plot_train_history, plot_rewards


def test_env(env):

    # Next, we build a very simple model.
    nb_actions = env.nb_actions
    inputs = Input(shape=(1, 7))
    flat_inputs = Flatten()(inputs)
    model = getMLPModel(out_dim=nb_actions)
    model = Model(inputs=inputs, outputs=model(flat_inputs))
    model.summary()

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-5), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    hist = dqn.fit(env, nb_steps=2000, visualize=False, verbose=1)
    plot_rewards(hist, "test")
    print(hist)
    print("hoi")
    return
    # plot_train_history(hist, val=True)

    # After training is done, we save the final weights.
    # dqn.save_weights('dqn_{}_weights.h5f'.format(env.m.name), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)
