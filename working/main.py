TRAIN = False
TEST = True

ENV_NAME = 'BreakoutDeterministic-v4'
ENV_NAME = 'PongDeterministic-v4'  
# You can increase the learning rate to 0.00025 in Pong for quicker results

import os
import random
import gym
import tensorflow as tf
import numpy as np
import imageio
from skimage.transform import resize
from ProcessFrame import ProcessFrame
