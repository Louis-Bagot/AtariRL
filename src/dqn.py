import numpy as np
import tensorflow as tf
import gym
import time
import random
import matplotlib.pyplot as plt
from dqn_functions import *
from display import *
from wrappers2 import wrap_dqn

graph(0, "somex", "somey", "test display", "PlotTester")
"""Deep Q Network Algorithm"""
new_algo = True
## Initializations : environment
game_name = 'Pong'
game = game_name + 'NoFrameskip-v4'
env = gym.make(game) # environment
env = wrap_dqn(env)
n_actions = env.action_space.n
# DQN
o = env.reset()
agent_history_length = 4 # number of frames the agent sees when acting
atari_shape = (agent_history_length, 84,84)
print(atari_shape)

dqn = init_DQN2(atari_shape,n_actions) if new_algo\
 else init_DQN(atari_shape, n_actions)

# miscellanous initializations of variables or hyperparameters
max_memory = 3*10**5 # max size of replay_memory
memory_start_size = 10**4 # amount of transitions in memory before using it
epoch_size = 5*10**4 # number of frames (training batches) within an epoch
max_epoch = 10**2
reload_model = 10**4 # frame frequency of nn parameters reloading
gamma = .99 # discount factor
batch_size = 32 # amount of elements sampled from the replay_memory
(min_decay, no_decay_threshold) = (.1, 10**6)
test_explo = 0.05
update_freq = 4 # actions taken before learning on a batch

# Results display variables
replay_memory = [] # replay memory to learn smoothly from the past
frame = 0 # frame number, throughout the whole, main loop.
i_episode = 0
epoch_record = [] # average scores per epoch

## Main loop
while len(epoch_record) < max_epoch:
    # init observation
    observation = env.reset().squeeze(axis=2)
    done = False
    #Game loop
    while not done:
        # update decaying exploration parameter
        epsilon = decay_epsilon(frame, min_decay, no_decay_threshold)

        # take action; or act randomly if memory is too small
        if (frame > memory_start_size):
            action = eps_greedy(epsilon, n_actions, dqn,replay_memory,\
                                agent_history_length, new_algo)
        else : action = random_action(n_actions)

        # env step
        observation, reward, done, info = env.step(action)

        # Replay memory. Discard the obs_old idea since it's above in memory
        replay_memory.append((observation.squeeze(axis=2), action, reward, done))
        if len(replay_memory) > max_memory:
            del replay_memory[0]

        # old parameters recording
        if (frame % reload_model == 0):
            print("Reloading model")
            old_dqn = copy_model(dqn) # recording of the NN's old weights

        # learning
        if (len(replay_memory) > memory_start_size) and (frame % update_freq == 0):
            mini_batch = extract_mini_batch(replay_memory, batch_size * update_freq, \
                                            agent_history_length)
            if new_algo:
                train_dqn2(dqn, old_dqn, mini_batch, gamma, n_actions)
            else :
                train_dqn(dqn, old_dqn, mini_batch, gamma)

        frame += 1

        if (frame % epoch_size == 0):
            print_info(frame, i_episode, len(epoch_record), len(replay_memory), max_memory, epsilon)
            epoch_record.append(test_dqn(game, test_explo, dqn, agent_history_length, new_algo))
            graph(epoch_record,'Epoch ('+str(epoch_size)+' frames)','Mean cumulated reward', 'Agent Performance (mean score) at '+game_name, game_name)
    i_episode += 1

keep_playing(game, .05, dqn, agent_history_length, new_algo)
