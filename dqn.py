import numpy as np;
import tensorflow as tf;
import gym;
import time;

"""Deep Q Network Algorithm"""


## Initializations
env = gym.make('Breakout-ram-v0') # environment
dqn = tf.keras.models.Sequential([ # dqn, with as many outputs as actions
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation=tf.nn.relu),
tf.keras.layers.Dense(128, activation=tf.nn.relu),
tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax)
])
dqn.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

# miscellanous initializations of variables or hyperparameters
replay_memory = [] # replay memory to learn smoothly from the past
max_memory = 10*6 # max size of replay_memory
max_episodes =
model_old = model # recording of the NN's old weights (learning stabilization)
reload_model = 4*10**4
epsilon = 1 # decaying probability to act randomly
gamma = .9 # discount factor
frameskip = 4 # number of frames during which to spam the same action (not learn)
batch_size = 32 # amount of elements sampled from the replay_memory
frame = 0 # frame number, throughout the whole


## Main loop
for i_episode in range(10):
    # init observation
    observation = env.reset()
    done = False

    #Game loop
    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        frame += 1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
