import tensorflow as tf
import numpy as np


class Alpha4():
    """Class for the A4 agent"""
    def __init__(self, c4_shape, n_actions):
        self.nn = self.init_A4(c4_shape, n_actions)

    def init_A4(self, c4_shape,n_actions):
        """Returns the network roughly adapted for C4"""
        # With the functional API we need to define the inputs.
        board_input = tf.keras.layers.Input(c4_shape, name='board')

        # The first hidden layer convolves 16 4×4 filters with stride 1 with the input image and applies a rectifier nonlinearity.
        conv_1 = tf.keras.layers.Conv2D(16, (4, 4), strides=(1, 1), activation=tf.nn.relu)(board_input)
        # Flattening the convolutional layer.
        conv_flattened = tf.keras.layers.Flatten()(conv_1)
        # The final hidden layer is fully-connected and consists of 64 rectifier units.
        hidden = tf.keras.layers.Dense(64, activation=tf.nn.relu)(conv_flattened)
        # 2 outputs : policy (ie distribution over moves) and vf (-1:1)
        output_policy = tf.keras.layers.Dense(n_actions, activation=tf.nn.softmax, name='policy')(hidden)
        output_value = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name='value_fun')(hidden)

        nn = tf.keras.models.Model(inputs=board_input, outputs=[output_policy, output_value])
        optimizer = tf.keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        nn.compile(optimizer, loss='mse')
        return nn

    def predict(self,input):
        return self.nn.predict(input)
