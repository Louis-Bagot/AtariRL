"""
In order to initalize dqn's weights, we train an autoencoder to learn a latent
space which act as feature extraction for the dqn. The encoder part of the
autoencoder will be the first layers of the dqn.
"""
import gym
from dqn_functions import *
from scipy.misc import imresize
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def create_autoencoder(input_shape, latent_space=512):
    return tf.keras.models.Sequential([ # dqn, with as many outputs as actions
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape, data_format='channels_last'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(16, (3, 3), strides=(2,2), activation='relu', padding='same'),

        # Flatten encoding for visualization
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(latent_space),
        tf.keras.layers.Reshape((8, 16, 16)),

        # Decoder Layers,
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(96, (3, 3), activation='sigmoid', padding='same'),
        tf.keras.layers.Conv2D(input_shape[0], (3, 3), activation='sigmoid', padding='same',data_format='channels_last')

    ])




game = 'BreakoutDeterministic-v4'
env = gym.make(game) # environment
n_actions = env.action_space.n
agent_history_length = 4 # number of frames the agent sees when acting
atari_shape = (128,96,agent_history_length)

#
#
# # Encoder part
# frames_input = tf.keras.layers.Input(atari_shape, name='frames')
# actions_input = tf.keras.layers.Input((n_actions,), name='mask')
#
# # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
# conv_1 = tf.keras.layers.Conv2D(
#     16, (8, 8), strides=(4, 4), activation=tf.nn.relu, data_format="channels_first"
# )(frames_input)
# # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
# conv_2 = tf.keras.layers.Conv2D(
#     32, (4, 4), strides=(2, 2), activation=tf.nn.relu
# )(conv_1)
# # Flattening the second convolutional layer.
# conv_flattened = tf.keras.layers.Flatten()(conv_2)
# # "The final hidden layer is fully-connected and consists of 256 rectifier units."
# hidden = tf.keras.layers.Dense(256, activation=tf.nn.relu)(conv_flattened)
# # "The output layer is a fully-connected linear layer with a single output for each valid action."
# output = tf.keras.layers.Dense(n_actions)(hidden)
# # Finally, we multiply the output by the mask
# filtered_output = tf.keras.layers.Multiply()([output,actions_input])
#
# dqn = tf.keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)
#
# dqn.summary()
#
# mine = tf.keras.models.Sequential([ # dqn, with as many outputs as actions
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=atari_shape, data_format='channels_first'),
#     tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
#     tf.keras.layers.Conv2D(16, (3, 3), strides=(2,2), activation='relu', padding='same'),
#
#     # Flatten encoding for visualization
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(256),
#         tf.keras.layers.Dense(4)
#     ])
#
# mine.summary()

autoencoder = create_autoencoder(atari_shape, latent_space=512)
#autoencoder.summary()
rms_opti = tf.keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
autoencoder.compile(optimizer='adam',loss='mse')


observation = preprocess(env.reset())
observation = imresize(observation,(128,96))
# plt.imshow(observation)
# plt.show()
## Play forever
while True:
    # init observation
    observation = preprocess(env.reset())
    done = False
    cumul_score = 0 #episode total score
    one_instance = [] # one instance is composed of agent_history_length images
    batch = []
    batch_size = 32
    #Game loop
    while not done:
        action = random_action(n_actions)
        #env.render()
        #time.sleep(.05)
        observation, reward, done, info = env.step(action)
        observation = preprocess(observation)
        one_instance.append(imresize(observation,(128,96)))

        # Fitting
        if len(batch) == batch_size:
            one_instance = np.array(one_instance).reshape(128,96,4)
            batch = np.array(batch)
            print(one_instance.shape)
            print(batch.shape)
            autoencoder.fit(batch, batch, epochs=1, batch_size=batch_size)
            batch = []

        # Create instance
        if len(one_instance) == agent_history_length:
            batch.append(np.array(one_instance).reshape(128,96,4))
            one_instance.pop(0) # Remove one instance
