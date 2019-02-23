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
    encoder =  tf.keras.models.Sequential()
    #, input_shape=input_shape, data_format='channels_first'
     # dqn, with as many outputs as actions
    encoder.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first'))
    encoder.add(tf.keras.layers.Dropout(0.3))
    encoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    encoder.add(tf.keras.layers.Dropout(0.3))
    encoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(2,2), activation='relu', padding='same'))

    # Flatten encoding for visualization
    encoder.add(tf.keras.layers.Flatten())
    encoder.add(tf.keras.layers.Dropout(0.3))
    encoder.add(tf.keras.layers.Dense(latent_space))

    decoder = tf.keras.models.Sequential([
        #tf.keras.layers.Input((latent_space,)),
        tf.keras.layers.Reshape((8, 12, 16)),

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
        tf.keras.layers.Conv2D(input_shape[0], (3, 3), activation='sigmoid', padding='same',data_format='channels_first')
    ])

    input_layer = tf.keras.layers.Input(input_shape)
    encoder = encoder(input_layer)
    autoencoder = decoder(encoder)



    return tf.keras.models.Model(input_layer, encoder), tf.keras.models.Model(input_layer, autoencoder)




game = 'BreakoutDeterministic-v4'
env = gym.make(game) # environment
n_actions = env.action_space.n
agent_history_length = 4 # number of frames the agent sees when acting
atari_shape = (agent_history_length, 96, 96)


encoder, autoencoder = create_autoencoder(atari_shape, latent_space=512)

autoencoder.summary()
encoder.summary()

rms_opti = tf.keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
autoencoder.compile(optimizer='adam',loss='mse')

# Check autoencoder
X = np.random.randn(32,atari_shape[0],atari_shape[1], atari_shape[2])
autoencoder.fit(X,X, epochs=1, batch_size=4, verbose=1)

observation = preprocess(env.reset())
observation = imresize(observation,(atari_shape[1],atari_shape[2]))
# plt.imshow(observation)
# plt.show()

counter = 0
## Play forever
while True:
    counter +=1

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
        one_instance.append(imresize(observation,(atari_shape[1],atari_shape[2])))

        # Fitting
        if len(batch) == batch_size:
            batch = np.array(batch)
            #print(one_instance.shape)
            print(batch.shape)
            autoencoder.fit(batch, batch, epochs=1, batch_size=batch_size, verbose=1)
            batch = []

        # Create instance
        if len(one_instance) == agent_history_length:
            batch.append(one_instance)
            one_instance.pop(0) # Remove one instance

        # Saving encoder weights
        if counter % 10 == 0:
            encoder.save('encoder.h5', overwrite=True)
