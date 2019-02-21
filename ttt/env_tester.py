from TicTacEnv import TicTacEnv
from Random_agent import Random_agent
from MCTS import MCTS_agent
import time
import numpy as np

env = TicTacEnv()
agent1 = MCTS_agent()
agent1 = Random_agent()
agent2 = Random_agent()

reward_rec = []
for i_episode in range(20):
    observation = env.reset()
    done = False
    while not done:
        # agent plays
        if env.player == 1:
            action = agent1.play(env.make_copy())
        else:
            action = agent2.play(env.make_copy())
        # env step
        observation, reward, done, info = env.step(action)
    reward_rec.append(reward)

print("Results : ", reward_rec)
print("Mean : ", np.mean(reward_rec))
