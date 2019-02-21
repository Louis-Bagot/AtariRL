from C4Env import C4Env
from Alpha4 import Alpha4
import numpy as np
import time

env = C4Env()
for i_episode in range(2):
    observation = env.reset()
    done = False
    while not done:
        action = env.random_move()
        observation, reward, done, info = env.step(action)

env.render()
l = env.legal_moves()
print(l)
a4 = Alpha4(observation.shape, env.n_actions)
p = a4.predict(np.array([observation]))
print(p)
print(p[0]*l)
