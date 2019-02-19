from C4Env import C4Env
import time

env = C4Env()
for i_episode in range(2):
    observation = env.reset()
    done = False
    while not done:
        action = env.random_move()
        observation, reward, done, info = env.step(action)
