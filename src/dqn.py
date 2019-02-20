import numpy as np
import tensorflow as tf
import gym
import time
import random
import matplotlib.pyplot as plt
from dqn_functions import *
from display import *

"""Deep Q Network Algorithm"""

## Initializations : environment
game = 'Breakout-v0'
env = gym.make(game) # environment
n_actions = env.action_space.n
# DQN
agent_history_length = 4 # number of frames the agent sees when acting
atari_shape = (agent_history_length,105,80)

dqn = tf.keras.models.Sequential([ # dqn, with as many outputs as actions
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (8,8), strides=(4,4), \
        activation=tf.nn.relu, input_shape=atari_shape, data_format='channels_first'),
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (4,4),\
        strides=(2,2), activation=tf.nn.relu),
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),\
        strides=(1,1), activation=tf.nn.relu),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(n_actions)
])

rms_opti = tf.keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
dqn.compile(optimizer=rms_opti,loss='logcosh')

# miscellanous initializations of variables or hyperparameters
max_memory = 3*10**5 # max size of replay_memory
memory_start_size = 5*10**4 # amount of transitions in memory before using it
max_epoch = 10**2
reload_model = 10**4 # frame frequency of nn parameters reloading
gamma = .99 # discount factor
batch_size = 32 # amount of elements sampled from the replay_memory
(min_decay, no_decay_threshold) = (.1, 10**6)
update_freq = 4 # actions taken before learning on a batch
epoch_size = 5*10**4 # number of frames (training batches) within an epoch

# Results display variables
replay_memory = [] # replay memory to learn smoothly from the past
frame = 0 # frame number, throughout the whole, main loop.
i_episode = 0
epoch_record = [] # average scores per epoch

## Main loop
while len(epoch_record) < max_epoch:
    # init observation
    print("\tEp : ", i_episode)
    observation = preprocess(env.reset())
    done = False
    #Game loop
    while not done:
        # update decaying exploration parameter
        epsilon = decay_epsilon(frame, min_decay, no_decay_threshold)

        # take action; or act randomly if memory is too small
        if (frame > memory_start_size):
            action = eps_greedy(epsilon, n_actions, dqn, replay_memory, \
                                agent_history_length)
        else : action = random_action(n_actions)

        # env step; data cleaning
        observation, reward, done, info = env.step(action)
        observation = preprocess(observation)
        reward = np.sign(reward)

        # Replay memory. Discard the obs_old idea since it's above in memory
        replay_memory.append((observation, action, reward, done))
        if len(replay_memory) > max_memory:
            replay_memory.pop(0)

        # old parameters recording
        if (frame % reload_model == 0):
            print("Reloading model")
            old_dqn = copy_model(dqn) # recording of the NN's old weights

        # learning
        if (len(replay_memory) > memory_start_size) and (frame % update_freq == 0):
            mini_batch = extract_mini_batch(replay_memory, batch_size, \
                                            agent_history_length)
            train_dqn(dqn, old_dqn, mini_batch, gamma)

        frame += 1

        if (frame % epoch_size == 0):
            print_info(frame, i_episode, len(epoch_record), len(replay_memory), max_memory, epsilon)
            epoch_record.append(test_dqn(game, .05, dqn, agent_history_length))
            graph(epoch_record, 'Average score per epoch')
    i_episode += 1

keep_playing(game, .05, dqn, agent_history_length)
