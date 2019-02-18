import gym
import time
import numpy as np
from dqn_functions import *

game = 'Breakout-v0'
env = gym.make(game)
n_actions = env.action_space.n
agent_history_length = 4 # number of frames the agent sees when acting
atari_shape = (105,80,agent_history_length)

dqn = tf.keras.models.Sequential([ # dqn, with as many outputs as actions
    tf.keras.layers.Conv2D(filters = 16, kernel_size = (8,8),\
        strides=(4,4), activation='relu', input_shape=atari_shape),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (4,4),\
        strides=(2,2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(n_actions)
])

rms_opti = tf.keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
dqn.compile(optimizer=rms_opti,loss='logcosh')

test_dqn(game, .05, dqn, agent_history_length)
