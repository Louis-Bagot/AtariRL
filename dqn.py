import numpy as np
import tensorflow as tf
import gym
import time
import random
import matplotlib.pyplot as plt
from dqn_functions import *

"""Deep Q Network Algorithm"""

## Initializations
env = gym.make('Breakout-ram-v0') # environment
n_actions = env.action_space.n
print(n_actions)
dqn = tf.keras.models.Sequential([ # dqn, with as many outputs as actions
    tf.keras.layers.Dropout(.3),
    tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(128,)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(n_actions)
])
rms_opti = tf.keras.optimizers.RMSprop(lr=0.0002)
dqn.compile(optimizer=rms_opti,
            loss='mean_squared_error',
            metrics=['accuracy'])

# miscellanous initializations of variables or hyperparameters
replay_memory = [] # replay memory to learn smoothly from the past
max_memory = 10*5 # max size of replay_memory
memory_start_size = 100 # amount of transitions in memory before learning from it
max_frames = 10**9
max_episodes = 10**3
reload_model = 10**4 # frame frequency of nn parameters reloading
gamma = .95 # discount factor
batch_size = 32 # amount of elements sampled from the replay_memory
frame = 0 # frame number, throughout the whole, main loop.
(min_decay, no_decay_threshold) = (.1, 10**6)

render_freq = 10**2 # frequency of episode render to follow evolution

# Results display variables
q_record = [] # [max_a Q(s,a) value] over time (frames)
score_record = [] # total episode score over time

## Main loop
for i_episode in range(max_episodes):
    # init observation
    observation = env.reset()/255
    done = False
    print("Epsiode ", i_episode)
    print("\t Frame ", frame)
    cumul_score = 0
    #Game loop
    while not done:
        # update decaying exploration parameter
        epsilon = decay_epsilon(frame, min_decay, no_decay_threshold)
        # watching a game
        if (i_episode % render_freq == 0):
            env.render()
            time.sleep(.05)
        action = eps_greedy(epsilon, n_actions, dqn, observation)


        # env step
        obs_old = observation
        observation, reward, done, info = env.step(action)
        observation = observation.astype(float)
        observation /= 255
        cumul_score += reward
        q_record.append(np.max(dqn.predict(np.array([observation]))))

        replay_memory.append((obs_old, action, reward, observation))
        if len(replay_memory) > max_memory:
            replay_memory.pop(0)

        # old parameters recording
        if (frame % reload_model == 0):
            old_dqn = dqn # recording of the NN's old weights
        # learning
        if (len(replay_memory) > memory_start_size):
            mini_batch = np.array(random.choice(replay_memory, batch_size))
            # 0: state, 1: action, 2: reward, 3: state'
            train_dqn(dqn, old_dqn, mini_batch[0], mini_batch[1], mini_batch[2], \
                mini_batch[3], gamma)
        frame += 1

    score_record.append(cumul_score)
    if frame > max_frames:
        break

plt.plot(q_record)
plt.ylabel('max Q value')
plt.show()

plt.plot(score_record)
plt.ylabel('episode score')
plt.show()
