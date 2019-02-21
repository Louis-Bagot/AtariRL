from TicTacEnv import TicTacEnv
import time

env = TicTacEnv()
for i_episode in range(2):
    observation = env.reset()
    done = False
    while not done:
        action = env.random_move()
        env.render()
        observation, reward, done, info = env.step(action)
