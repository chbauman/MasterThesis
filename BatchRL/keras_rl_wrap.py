from keras import Input, Model
from keras.layers import Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from keras_util import getMLPModel
from visualize import plot_rewards


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
                   policy=policy,
                   gamma=0.9,
                   train_interval=3,
                   target_model_update=1000)
    dqn.compile(Adam(lr=1e-5), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    hist = dqn.fit(env, nb_steps=3000, visualize=False, verbose=1)
    train_plot = env.get_plt_path("test")
    plot_rewards(hist, train_plot)
    print("hoi")
    return

    # After training is done, we save the final weights.
    # dqn.save_weights('dqn_{}_weights.h5f'.format(env.m.name), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)
