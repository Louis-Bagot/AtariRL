import gym
import time
import numpy as np
from dqn_functions import *

game = 'BreakoutDeterministic-v4'
env = gym.make(game)
n_actions = env.action_space.n
frame = 0
replay_memory = [] # replay memory to learn smoothly from the past
done = False
max_memory = 200
memory_start_size = 100
i_episode = 0

while (frame < 1000):
    cumul_score = 0
    observation = preprocess(env.reset())
    done = False
    dones = [done]
    while not done:
        action = env.action_space.sample()
        #env.render()
        # env step
        observation, reward, done, info = env.step(action)
        observation = preprocess(observation)
        replay_memory.append((observation,action,reward,done))
        if (len(replay_memory) > max_memory):
            replay_memory.pop(0)
        frame += 1
        cumul_score += reward
    i_episode += 1


agent_history_length = 4 # number of frames the agent sees when acting
atari_shape = (agent_history_length,105,80)

dqn = tf.keras.models.Sequential([ # dqn, with as many outputs as actions
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (8,8), strides=(4,4), \
        activation='relu', input_shape=atari_shape, data_format='channels_first'),
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (4,4),\
        strides=(2,2), activation='relu'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3),\
        strides=(1,1), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(n_actions)
])
rms_opti = tf.keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
dqn.compile(optimizer=rms_opti,loss='logcosh')

test_dqn(game, .05, dqn, agent_history_length)

batch_size = 32
mini_batch = extract_mini_batch(replay_memory, batch_size, agent_history_length)
gamma = .99
train_dqn(dqn, dqn, mini_batch, gamma)
