import numpy as np;
import tensorflow as tf;
import gym;
import time;
import random;
from dqn_functions import decay_epsilon

"""Deep Q Network Algorithm"""


## Initializations
env = gym.make('Breakout-ram-v0') # environment
n_actions = env.action_space.n;
dqn = tf.keras.models.Sequential([ # dqn, with as many outputs as actions
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation=tf.nn.relu),
tf.keras.layers.Dense(128, activation=tf.nn.relu),
tf.keras.layers.Dense(n_actions, activation=tf.nn.softmax)
])
dqn.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

# miscellanous initializations of variables or hyperparameters
replay_memory = [] # replay memory to learn smoothly from the past
max_memory = 10*5 # max size of replay_memory
max_frames = 10**7
max_episodes = 10**3
dqn_old = dqn # recording of the NN's old weights (learning stabilization)
reload_model = 4*10**4
gamma = .9 # discount factor
frameskip = 4 # number of frames during which to spam the same action (not learn)
batch_size = 32 # amount of elements sampled from the replay_memory
frame = 0 # frame number, throughout the whole, main loop.
(min_decay, no_decay_threshold) = (.1, 10**6)

## Main loop
for i_episode in range(max_episodes):
    # init observation
    observation = env.reset()
    done = False

    #Game loop
    while not done:
        # update decaying exploration parameter
        epsilon = decay_epsilon(frame, min_decay, no_decay_threshold)
        env.render()
        action = eps_greedy(epsilon, n_actions, dqn, observation)
        obs_old = observation
        observation, reward, done, info = env.step(action)
        if len(replay_memory) > max_memory:
            replay_memory.pop(0)
        replay_memory.append((obs_old, action, reward, observation))
        if len(replay_memory) > batch_size:
            mini_batch = random.choice(replay_memory, batch_size)
            dqn.train_on_batch()


        frame += 1

        if done:
            print("Episode finished after {} timesteps".format(t+1))

    if frame > max_frames:
        break
