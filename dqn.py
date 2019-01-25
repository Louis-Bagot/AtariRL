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
memory_start_size = 100 # amount of transitions in memory before learning from it
max_frames = 10**7
max_episodes = 10**3
dqn_old = dqn # recording of the NN's old weights (learning stabilization)
reload_model = 4*10**4
gamma = .95 # discount factor
frameskip = 4 # number of frames during which to spam the same action (not learn)
batch_size = 32 # amount of elements sampled from the replay_memory
frame = 0 # frame number, throughout the whole, main loop.
(min_decay, no_decay_threshold) = (.1, 10**6)

## Main loop
for i_episode in range(max_episodes):
    # init observation
    observation = env.reset()/255
    done = False

    #Game loop
    while not done:
        # update decaying exploration parameter
        epsilon = decay_epsilon(frame, min_decay, no_decay_threshold)
        env.render()
        action = eps_greedy(epsilon, n_actions, dqn, observation)
        obs_old = observation
        observation, reward, done, info = env.step(action)
        observation /= 255
        if len(replay_memory) > max_memory:
            replay_memory.pop(0)

        replay_memory.append((obs_old, action, reward, observation))
        if len(replay_memory) > memory_start_size:
            mini_batch = np.array(random.choice(replay_memory, batch_size))
            # 0: state, 1: action, 2: reward, 3: state'

            td_target = mini_batch[:,2] + gamma*
            dqn.train_on_batch(mini_batch[:,0], td_target)


        frame += 1

        if done:
            print("Episode finished after {} timesteps".format(frame+1))

    if frame > max_frames:
        break
